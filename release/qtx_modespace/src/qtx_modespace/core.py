"""Device, lead self-energies, sparse / block-tridiagonal solvers and the contour slab density matrix.

obc="beyn" (periodized lead, QuaTrEx's model) / "beyn-cell" (lead cell as given): one lead model for every z, the
spectral OBC of obc_beyn.BeynLead (Beyn on the primitive cell, Woodbury face fixed point; real axis with the incoming
Bloch modes as injection, complex z with the decaying modes), numpy or the GPU, no qttools. Cached as
bsig<p|c>_<Re z>_<Im z>.npz. The default ("auto") is unchanged below: qttools on the real axis, folded lead on contours.

Lead OBC on the real axis (|Im z| <= Device.qtt_eta; the transmission, FGMRES and exact solves at real E): qttools'
spectral OBC (QuaTrEx) by default, Sigma at E = Re z and the flux-normalized incoming Bloch modes as injection
(one column per propagating channel, as QuaTrEx's QTBM). qttools is found on sys.path or in $QTTOOLS_SRC (a
QuaTrEx src folder; needs mpi4py, numba). When it does not import (or fails at an energy) the folded lead below is
used at E + i fallback_eta with the Gamma eigenchannels as injection. Complex z (contours): always the folded lead.

Device: the device blocks of a read_quatrex dict are the slabs; the lead blocks enter through Sigma_L, Sigma_R.
Lead Sigma(z): the lead-cell interior (orbitals not coupled to the neighbouring cells) is folded onto the coupled
faces F, A_eff(z) = A_FF - A_FI A_II^-1 A_IF with A = z S - H, then Lopez-Sancho decimation of the folded chain
(complex Hermitian H, any S).
QTX_SIGMA_PREC=mixed: decimation in complex64 (half the memory), then Newton refinement of g = (A00 - C g C')^-1 with the
residual in complex128 on the CPU and the correction (a Stein equation, doubling) in complex64 (sancho_mixed).
"""
import glob, os, sys, time, warnings
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
try:
    import pypardiso
except ImportError:          # pragma: no cover
    pypardiso = None

from . import gpu


def log(*a):
    print(*a, flush=True)


def savez_atomic(f, **arrays):
    """np.savez to a temporary file next to f, then os.replace: concurrent readers (other ranks sharing a cache on
    Lustre) see either no file or the whole file."""
    tmp = f"{f}.{os.getpid()}.tmp.npz"
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, f)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_text_atomic(f, text):
    """Path.write_text through a temporary file and os.replace (see savez_atomic)."""
    tmp = f"{f}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, f)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class CSolver:
    """Factorization of a complex sparse matrix: PARDISO on the real-equivalent [[Ar, -Ai], [Ai, Ar]] when pypardiso
    is installed, else SuperLU. Each solve does n_ref refinement steps, then GMRES preconditioned by the factorization
    on the columns still above tol (PARDISO perturbs small pivots)."""

    def __init__(self, A, n_ref=2, tol=1e-12, maxit=60):
        A = sp.csr_matrix(A)
        self.A, self.n, self.n_ref, self.tol, self.maxit = A, A.shape[0], n_ref, tol, maxit
        self.res = 0.0
        if pypardiso is None:
            self.lu = spla.splu(A.astype(complex).tocsc())
            return
        Ar, Ai = sp.csr_matrix(A.real), sp.csr_matrix(A.imag)
        self.M = sp.bmat([[Ar, -Ai], [Ai, Ar]], format="csr")
        self.ps = pypardiso.PyPardisoSolver(mtype=11)
        self.ps.factorize(self.M)

    def factor_solve(self, B):
        if pypardiso is None:
            return self.lu.solve(B.astype(complex))
        Z = self.ps.solve(self.M, np.vstack([B.real, B.imag])).reshape(2 * self.n, -1)
        return Z[:self.n] + 1j * Z[self.n:]

    def solve(self, B):
        B = np.asarray(B)
        one = B.ndim == 1
        B = B.reshape(len(B), -1).astype(complex)
        X = self.factor_solve(B)
        for _ in range(self.n_ref):
            X += self.factor_solve(B - self.A @ X)
        bn = np.maximum(np.linalg.norm(B, axis=0), 1e-300)
        res = np.linalg.norm(B - self.A @ X, axis=0) / bn
        for c in np.where(res > self.tol)[0]:
            X[:, c] = gmres_right(lambda v: self.A @ v, lambda v: self.factor_solve(v[:, None])[:, 0],
                                  B[:, c], X[:, c], self.tol, self.maxit)
            res[c] = np.linalg.norm(B[:, c] - self.A @ X[:, c]) / bn[c]
        self.res = float(res.max()) if len(res) else 0.0
        return X[:, 0] if one else X

    def free(self):
        if pypardiso is not None:
            self.ps.free_memory(everything=True)


def gmres_right(mv, prec, b, x0, tol, maxit):
    """GMRES on A x = b from x0, right preconditioner prec ~ A^-1; stops at |b - A x| <= tol |b|."""
    r0 = b - mv(x0)
    bn, be = np.linalg.norm(b), np.linalg.norm(r0)
    if be <= tol * bn:
        return x0
    V = np.zeros((len(b), maxit + 1), complex); Z = np.zeros((len(b), maxit), complex)
    H = np.zeros((maxit + 1, maxit), complex)
    V[:, 0] = r0 / be
    for j in range(maxit):
        Z[:, j] = prec(V[:, j])
        w = mv(Z[:, j])
        for _ in range(2):
            h = V[:, :j + 1].conj().T @ w; H[:j + 1, j] += h; w -= V[:, :j + 1] @ h
        H[j + 1, j] = np.linalg.norm(w)
        e1 = np.zeros(j + 2, complex); e1[0] = be
        y = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)[0]
        if np.linalg.norm(H[:j + 2, :j + 1] @ y - e1) <= tol * bn or H[j + 1, j] == 0:
            break
        V[:, j + 1] = w / H[j + 1, j]
    return x0 + Z[:, :j + 1] @ y


def sancho_both(A00, A01, A10, tol=1e-11, maxit=300, dtype=complex):
    """Lopez-Sancho decimation with on-site A00 = z S00 - H00, couplings A01 (cell n -> n+1), A10 (n+1 -> n).
    Returns the surface Green's functions gL (last cell of the left-extending chain) and gR (first cell of the
    right-extending chain). numpy or cupy arrays (the result lives where the input does)."""
    xp = gpu.xp_of(A00)
    e = A00.astype(dtype)
    esR, esL = e.copy(), e.copy()
    a, b = A01.astype(dtype), A10.astype(dtype)
    n = len(e)
    for _ in range(maxit):
        X = gpu.solve(e, xp.hstack([a, b]))
        ga, gb = X[:, :n], X[:, n:]
        agb, bga = a @ gb, b @ ga
        esR -= agb
        esL -= bga
        e -= agb + bga
        a, b = -a @ ga, -b @ gb
        if max(float(xp.abs(a).max()), float(xp.abs(b).max())) < tol:
            break
    else:
        log(f"  sancho_both: no convergence after {maxit} steps")
    return gpu.inv(esL), gpu.inv(esR)


def newton_refine(g, A00, C, Cp, tol=1e-11, max_ref=6, on_gpu=None):
    """Newton refinement of a surface Green's function g = (A00 - C g C')^-1 (g_L: C = A10, C' = A01; g_R: C = A01,
    C' = A10) from a low-precision start g (e.g. complex64 decimation or complex64 qttools OBC). Per step the residual
    R = I - X g, X = A00 - C g C', in complex128 on the CPU, and the correction d - P d Q = g R (P = g C, Q = C' g) by
    doubling in complex64 (GPU when on_gpu, default gpu.on); stops at |R|_max <= tol or when it stops contracting.
    Returns (G complex128, residual history)."""
    on_gpu = gpu.on if on_gpu is None else on_gpu
    c64 = np.complex64
    to = (lambda M: gpu.asarray(M, c64)) if on_gpu else (lambda M: np.asarray(M, c64))
    back = gpu.asnumpy if on_gpu else np.asarray
    G = np.asarray(back(g) if not isinstance(g, np.ndarray) else g).astype(complex)
    c, cp_ = to(C), to(Cp)
    xp = gpu.xp_of(c)
    eye = np.eye(len(A00), dtype=complex)
    res = []
    for it in range(max_ref + 1):
        R = eye - (A00 - C @ (G @ Cp)) @ G
        res.append(float(np.abs(R).max()))
        if res[-1] <= tol or it == max_ref:
            break
        if not np.isfinite(res[-1]) or (it and res[-1] > 0.5 * res[-2]) or res[-1] > 0.5:
            break                     # correction not contracting (|Im z| too small for complex64)
        gd = to(G)
        Y = gd @ to(R)
        del R
        P, Q = gd @ c, cp_ @ gd
        del gd
        for _ in range(100):                                    # Y = sum_k P^k (g R) Q^k by doubling
            T = P @ (Y @ Q)
            Y += T
            tm, ym = float(xp.abs(T).max()), float(xp.abs(Y).max())
            if not np.isfinite(tm) or tm <= 1e-8 * ym:
                break
            P, Q = P @ P, Q @ Q
        del P, Q, T
        G += back(Y).astype(complex)
        del Y
    return G, res


def sancho_mixed(A00, A01, A10, tol=1e-11, max_ref=6, on_gpu=None, info=None, fallback_tol=1e-9):
    """sancho_both in complex64, then Newton refinement per side to |I - X g|_max <= tol, X = A00 - C g C'
    (g_L: C = A10, C' = A01; g_R: C = A01, C' = A10). Residual and update in complex128 on the CPU (numpy inputs);
    the correction solves the Stein equation  d - P d Q = g R,  P = g C, Q = C' g  (spectral radii < 1: the decaying
    Bloch factors) by doubling in complex64, on the GPU when on_gpu (default gpu.on). Returns numpy (g_L, g_R).
    Conditioning ~ 1/|Im z|: on the real axis (E + 1e-6 i) complex64 cannot converge (residual 0.1 on the 6 nm lead,
    2026-11-18); when the residual stays above fallback_tol the complex128 decimation runs on the CPU."""
    on_gpu = gpu.on if on_gpu is None else on_gpu
    c64 = np.complex64
    to = (lambda M: gpu.asarray(M, c64)) if on_gpu else (lambda M: np.asarray(M, c64))
    back = gpu.asnumpy if on_gpu else np.asarray
    t0 = time.time()
    gs = [back(g) for g in sancho_both(to(A00), to(A01), to(A10), tol=1e-7, dtype=c64)]
    t_s = time.time() - t0
    out, hist = [], []
    for g, (C, Cp) in zip(gs, ((A10, A01), (A01, A10))):
        G, res = newton_refine(g, A00, C, Cp, tol=tol, max_ref=max_ref, on_gpu=on_gpu)
        out.append(G)
        hist.append(res)
    fallback = not all(r[-1] <= fallback_tol for r in hist)
    if fallback:                                  # complex128 decimation on the CPU (no GPU memory)
        log(f"  sancho_mixed: residual L {hist[0][-1]:.1e}, R {hist[1][-1]:.1e}; "
            f"complex128 decimation on the CPU instead")
        out = sancho_both(np.asarray(A00), np.asarray(A01), np.asarray(A10))
    if info is not None:
        info.update(sancho_s=t_s, total_s=time.time() - t0, res_L=hist[0], res_R=hist[1], fallback=fallback)
    return out[0], out[1]


class FoldedLead:
    """Periodic lead cell H00, S00 with coupling H01, S01 (cell n -> n+1). Faces F = orbitals of H01 / S01; the
    interior I is folded onto F; g_L, g_R live on F."""

    def __init__(self, H00, H01, S00, S01):
        H00, H01, S00, S01 = (sp.csr_matrix(M) for M in (H00, H01, S00, S01))
        P01 = abs(H01) + abs(S01)
        F = np.union1d(np.unique(P01.nonzero()[0]), np.unique(P01.nonzero()[1]))
        I = np.setdiff1d(np.arange(H00.shape[0]), F)
        self.F, self.I = F, I
        self.HFF, self.SFF = H00[F][:, F].toarray(), S00[F][:, F].toarray()
        self.HFI, self.SFI = H00[F][:, I].toarray(), S00[F][:, I].toarray()
        self.HIF, self.SIF = H00[I][:, F].toarray(), S00[I][:, F].toarray()
        self.HII, self.SII = H00[I][:, I].tocsr(), S00[I][:, I].tocsr()
        self.H01, self.S01 = H01[F][:, F].toarray(), S01[F][:, F].toarray()

    precision = os.environ.get("QTX_SIGMA_PREC", "double")      # "double" or "mixed" (sancho_mixed)
    mixed_on_gpu = os.environ.get("QTX_SIGMA_MIXED_DEVICE", "auto") != "cpu"

    def surface_gfs(self, z):
        """(g_L, g_R) on F at complex z (numpy). GPU mode: dense LU of the interior and the decimation on the GPU."""
        if self.precision == "mixed":
            return self._surface_gfs_mixed(z)
        if gpu.on:
            return self._surface_gfs_gpu(z)
        return sancho_both(self._aeff_cpu(z), z * self.S01 - self.H01, z * self.S01.conj().T - self.H01.conj().T)

    def _aeff_cpu(self, z):
        Aeff = z * self.SFF - self.HFF
        if len(self.I):
            cs = CSolver(z * self.SII - self.HII)
            X = cs.solve(z * self.SIF - self.HIF)
            cs.free()
            Aeff = Aeff - (z * self.SFI - self.HFI) @ X
        return Aeff

    def _surface_gfs_mixed(self, z):
        info = {}
        out = sancho_mixed(self._aeff_cpu(z), z * self.S01 - self.H01, z * self.S01.conj().T - self.H01.conj().T,
                           on_gpu=gpu.on and self.mixed_on_gpu, info=info)
        self.last_info = info
        return out

    def _surface_gfs_gpu(self, z):
        d = getattr(self, "_dev", None)
        if d is None:                                   # device copies of the dense face blocks, made once
            d = self._dev = {k: gpu.asarray(getattr(self, k)) for k in
                             ("HFF", "SFF", "HFI", "SFI", "HIF", "SIF", "H01", "S01")}
            d["HII"], d["SII"] = gpu.csr(self.HII, np.complex128), gpu.csr(self.SII, np.complex128)
        Aeff = z * d["SFF"] - d["HFF"]
        if len(self.I):
            AII = (z * d["SII"] - d["HII"]).toarray()
            lu = gpu.lu_factor(AII)
            X = gpu.lu_solve(lu, (z * d["SIF"] - d["HIF"]).astype(complex))
            del lu, AII
            Aeff = Aeff - (z * d["SFI"] - d["HFI"]) @ X
        gL, gR = sancho_both(Aeff, z * d["S01"] - d["H01"], z * d["S01"].conj().T - d["H01"].conj().T)
        return gpu.asnumpy(gL), gpu.asnumpy(gR)


_QTT = None
QTT_ERROR = ""                      # why qttools_obc() returned None


def qttools_obc():
    """(Spectral, Full) from qttools (QuaTrEx), or None when it does not import or runs on cupy."""
    global _QTT, QTT_ERROR
    if _QTT is None:
        _QTT = False
        # qttools picks its array module from QTX_ARRAY_MODULE and defaults to cupy when cupy is installed (the gpu
        # extra); the OBC wrapper below works on numpy arrays, so ask for numpy unless the user set it explicitly
        os.environ.setdefault("QTX_ARRAY_MODULE", "numpy")
        src = os.environ.get("QTTOOLS_SRC")
        for path in (None, src) if src else (None,):
            if path and path not in sys.path:
                sys.path.append(path)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from qttools import xp
                    from qttools.boundary_conditions.obc.spectral import Spectral
                    from qttools.nevp import Full
            except ImportError as e:
                QTT_ERROR = f"qttools does not import: {e}"
                continue
            except SyntaxError as e:     # qttools uses PEP 695 `type X[T] = ...`: Python >= 3.12
                QTT_ERROR = (f"qttools needs Python >= 3.12 (running {sys.version.split()[0]}): "
                             f"{e.msg} in {e.filename}:{e.lineno}")
                break
            _QTT = (Spectral, Full) if xp is np else False
            QTT_ERROR = "" if _QTT else QTT_ERROR
            if not _QTT:
                QTT_ERROR = "qttools runs on cupy (xp is not numpy)"
            break
    return _QTT or None


class QttLead:
    """Lead cell (H00, S00, coupling H01, S01 to the next cell along +x) through qttools' spectral OBC at real E.
    sections = primitive cells per lead cell (the NEVP runs on one primitive cell). Per side: g = surface Green's
    function of the semi-infinite lead (cell next to the device included), u = lead-cell amplitudes of the
    flux-normalized incoming modes (device source A_dl u, B B^H = Gamma on the device face). perm: lead-cell orbital
    order with the primitive cells contiguous along +x and in the same internal order; qttools reads the periodic
    layer from the outermost primitive cell, cells numbered inwards (+x for the left lead, reversed for the right)."""

    def __init__(self, H00, H01, S00, S01, sections=1, perm=None, nevp=None, beyn=None, **opts):
        """nevp (default $QTX_QTT_NEVP or "auto"):
          "beyn"       qttools Beyn on the primitive cell in complex128: contour integral of T(z)^-1 over an annulus,
                       QuaTrEx defaults r_o 10, r_i 0.8 per lead cell, 20 points, max_decay 1.5 ln r_o; beyn = dict
                       overriding r_o, r_i, m_0 (default m / 2, doubled up to m when the rank saturates),
                       num_quad_points;
          "beyn-mixed" Beyn in complex64 + eigenpairs refined in complex128 (_MixedBeyn); g, fixed-point steps and the
                       injection stay complex128. 2808a lead (2026-11-18): g within 1e-11 of "beyn", same channels;
          "auto"       "beyn", or "beyn-mixed" when the memory estimate of "beyn" (beyn_memory) exceeds 80% of the free
                       memory (GPU in GPU mode, else host);
          "full"       dense companion eig of the primitive-cell polynomial ((blocks - 1) m: 6 240 for 2808a, 78 s;
                       25k for the 12 nm lead, hours)."""
        Spectral, Full = qttools_obc()
        n = H00.shape[0]
        assert n % sections == 0, f"lead cell {n} not divisible into {sections}"
        m = n // sections
        p = np.arange(n) if perm is None else np.asarray(perm)
        self.p = {"L": p, "R": p.reshape(sections, m)[::-1].ravel()}
        self.H00, self.H01, self.S00, self.S01 = (np.asarray(sp.csr_matrix(M)[p][:, p].toarray(), complex)
                                                  for M in (H00, H01, S00, S01))
        blk = lambda M, i, j: M[i * m:(i + 1) * m, j * m:(j + 1) * m]
        for nm, M in (("H", self.H00), ("S", self.S00)):
            d = max([abs(blk(M, i, i) - blk(M, 0, 0)).max() for i in range(sections)] +
                    [abs(blk(M, i, i + 1) - blk(M, 0, 1)).max() for i in range(sections - 1)])
            if d > 1e-2 * max(abs(M).max(), 1):
                raise RuntimeError(f"lead cell is not {sections} identical primitive cells (max |d{nm}| {d:.2e})")
            if d > 1e-8:
                log(f"  qttools lead: primitive cells of the lead cell differ by up to {d:.1e} in {nm}; "
                    f"the OBC uses the outermost cell (as QuaTrEx)")
        kw = dict(num_ref_iterations=4)
        self.sections = sections
        nevp = nevp or os.environ.get("QTX_QTT_NEVP", "auto")
        assert nevp in ("auto", "beyn", "beyn-mixed", "full"), nevp
        b = dict(r_o=10.0, r_i=0.8, m_0=max(64, m // 2), num_quad_points=20)
        b.update(beyn or {})
        if nevp == "auto":
            need = beyn_memory(m, n, b["num_quad_points"])
            free = _free_memory()
            nevp = "beyn" if need["beyn"] <= 0.8 * free else "beyn-mixed"
            log(f"  qttools lead: Beyn complex128 needs ~{need['beyn'] / 2**30:.1f} GB, complex64 + refinement "
                f"~{need['beyn-mixed'] / 2**30:.1f} GB, {free / 2**30:.1f} GB free ({'GPU' if gpu.on else 'host'}) "
                f"-> {nevp}")
        self.nevp_kind = nevp
        if nevp in ("beyn", "beyn-mixed"):
            from qttools.nevp import Beyn
            kw["max_decay"] = 1.5 * np.log(b["r_o"])
            kw.update(opts)
            wrap = _MixedBeyn if nevp == "beyn-mixed" else _RankBeyn
            self.nevp = wrap(Beyn(r_o=b["r_o"] ** (1 / sections), r_i=b["r_i"] ** (1 / sections), m_0=int(b["m_0"]),
                                  num_quad_points=int(b["num_quad_points"])))
            self.beyn = b
        else:
            kw.update(opts)
            self.nevp = Full()
        self.solver = Spectral(self.nevp, block_sections=sections, **kw)

    def modes(self, E, side, dtype=complex, refine=False, info=None):
        """dtype=np.complex64: the NEVP, mode selection and qttools' own refinement in single precision (half the
        memory); refine=True then Newton-refines g in complex128 on the CPU (newton_refine) with the same (periodized)
        blocks. The injected modes keep the precision of dtype."""
        t0 = time.time()
        a00 = E * self.S00 - self.H00
        a01 = E * self.S01 - self.H01
        a10 = E * self.S01.conj().T - self.H01.conj().T
        a_ji, a_ij = (a10, a01) if side == "L" else (a01, a10)
        r = np.argsort(self.p["L"])[self.p[side]]
        a_xx = tuple(M[np.ix_(r, r)][None] for M in (a_ji, a00, a_ij))
        if self.sections > 1:          # exactly periodic lead from the outermost primitive cell (as QuaTrEx)
            from qttools.toeplitz.toeplitz import periodize_layer
            a_xx = tuple(np.ascontiguousarray(x) for x in periodize_layer(a_xx, self.sections))
        a_ji = a_xx[0][0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g, b = self.solver(tuple(x.astype(dtype) for x in a_xx), "", return_injected=True)
        bad = [str(x.message) for x in w if "recursion error" in str(x.message)]
        if bad:
            raise RuntimeError(f"qttools OBC {side} at E = {E}: {bad[0]}")
        g = g[0].astype(complex)
        t1 = time.time()
        res = None
        if refine or info is not None:          # info without refine: only the residual of g
            G, res = newton_refine(g, a_xx[1][0], a_ji, a_xx[2][0], max_ref=6 if refine else 0, on_gpu=False)
            g = G if refine else g
        if info is not None:
            info.update(qtt_s=t1 - t0, refine_s=time.time() - t1, res=res)
        u = g @ (a_ji @ b[0].astype(complex))
        ip = np.argsort(self.p[side])
        return g[np.ix_(ip, ip)], u[ip]


def nevp_residuals(blocks, ws, vs):
    """Relative residuals |T(w) v| / (sum_j |B_j|_F |w|^(j - off) |v|) of eigenpairs of the qttools polynomial."""
    B = [np.asarray(b, complex) for b in blocks]
    off = len(B) // 2
    ws = np.asarray(ws, complex)
    vs = np.asarray(vs, complex)
    num = np.zeros(vs.shape, complex)
    den = np.zeros(len(ws))
    for j, b in enumerate(B):
        nb = np.linalg.norm(b)
        if nb == 0:
            continue
        p = ws ** (j - off)
        num += (b @ vs) * p
        den += nb * np.abs(p)
    return np.linalg.norm(num, axis=0) / (den * np.linalg.norm(vs, axis=0))


def refine_nevp_modes(blocks, ws, vs, steps=2, cluster_tol=1e-4, dup_tol=1e-6, pre_tol=1e-3, post_tol=1e-10,
                      stats=None):
    """Refine eigenpairs (w, v) of the polynomial T(w) = sum_j B_j w^(j - len(B)//2) (qttools block order, lowest
    power first) in complex128 from a low-precision start (complex64 Beyn). Eigenvalues within cluster_tol (relative)
    are refined together: per step one LU of T(w_mean), block inverse iteration X = T^-1 T'(w_mean) V, Rayleigh-Ritz on
    span(X) (small generalized companion problem), keeping the Ritz values closest to the cluster. Exactly repeated
    pairs (|dw| <= dup_tol |w|) are merged by an SVD of their vectors. Returns (ws, vs) complex128, columns unit norm.
    Residual filters (nevp_residuals): pairs above pre_tol before the refinement are dropped (complex64 Beyn keeps
    hundreds of noise directions above its eps-scaled SVD cut: 2808a lead 618 of 624 vs 167 in complex128), and
    pairs above post_tol after it."""
    B = [np.asarray(b, complex) for b in blocks]
    off = len(B) // 2
    nz = [j for j, b in enumerate(B) if np.abs(b).max() > 0]
    j0, j1 = nz[0], nz[-1]
    T = lambda w: sum(b * w ** (j - off) for j, b in enumerate(B) if j0 <= j <= j1)
    dT = lambda w: sum((j - off) * b * w ** (j - off - 1) for j, b in enumerate(B) if j0 <= j <= j1 and j != off)
    ws = np.asarray(ws, complex)
    vs = np.asarray(vs, complex)
    r0 = nevp_residuals(B, ws, vs)
    ok = r0 <= pre_tol
    if stats is not None:
        stats.update(n_in=len(ws), res_in=np.quantile(r0, [0, 0.25, 0.5, 0.75, 1]).tolist() if len(r0) else [],
                     n_pre=int(ok.sum()))
    ws, vs = ws[ok], vs[:, ok]
    order = np.argsort(np.abs(ws) + 1e-3 * np.angle(ws))
    ws, vs = ws[order], vs[:, order]
    groups, cur = [], [0]
    for i in range(1, len(ws)):
        if abs(ws[i] - ws[cur[0]]) <= cluster_tol * abs(ws[cur[0]]):
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    W, V = [], []
    for g in groups:
        Vg, wbar = vs[:, g], ws[g].mean()
        lam = ws[g]
        for _ in range(steps):
            lu = sla.lu_factor(T(wbar), check_finite=False)
            Q, _ = np.linalg.qr(sla.lu_solve(lu, dT(wbar) @ Vg, check_finite=False))
            k, d = Q.shape[1], j1 - j0
            small = [Q.conj().T @ B[j] @ Q for j in range(j0, j1 + 1)]    # sum_i small_i w^i (times w^(j0 - off))
            A = np.zeros((k * d, k * d), complex)
            Bm = np.eye(k * d, dtype=complex)
            A[:-k, k:] = np.eye(k * (d - 1))
            for i in range(d):
                A[-k:, i * k:(i + 1) * k] = -small[i]
            Bm[-k:, -k:] = small[d]
            lw, Z = sla.eig(A, Bm, check_finite=False)
            ok = np.isfinite(lw)
            lw, Z = lw[ok], Z[:, ok]
            sel = np.argsort(np.abs(lw - wbar))[:k]
            lam, Vg = lw[sel], Q @ Z[:k, sel]
            Vg = Vg / np.linalg.norm(Vg, axis=0)
            wbar = lam.mean()
        W.append(lam); V.append(Vg)
    W, V = np.concatenate(W), np.concatenate(V, axis=1)
    r1 = nevp_residuals(B, W, V)
    ok = r1 <= post_tol
    if stats is not None:
        stats.update(res_out=np.quantile(r1, [0, 0.5, 1]).tolist() if len(r1) else [], n_post=int(ok.sum()))
    W, V = W[ok], V[:, ok]
    keep_w, keep_v, used = [], [], np.zeros(len(W), bool)          # merge repeated eigenpairs
    for i in range(len(W)):
        if used[i]:
            continue
        same = np.where(~used & (np.abs(W - W[i]) <= dup_tol * abs(W[i])))[0]
        used[same] = True
        if len(same) == 1:
            keep_w.append(W[i:i + 1]); keep_v.append(V[:, i:i + 1])
            continue
        U, sv, _ = np.linalg.svd(V[:, same], full_matrices=False)
        r = int((sv > 1e-6 * sv[0]).sum())
        keep_w.append(np.full(r, W[same].mean())); keep_v.append(U[:, :r])
    W, V = np.concatenate(keep_w), np.concatenate(keep_v, axis=1)
    if stats is not None:
        stats.update(n_out=len(W))
    return W, V


class _MixedBeyn:
    """qttools Beyn in complex64 on the primitive cell, eigenpairs refined in complex128 (refine_nevp_modes); the
    caller (Spectral) gets complex128 modes and builds g, its fixed-point refinement and the injection in complex128.
    .rank = modes Beyn found, .nmodes = after refinement / merging, .t = (beyn s, refine s)."""

    def __init__(self, b):
        self.b, self.rank, self.nmodes, self.t, self.stats = b, None, None, None, {}

    def __call__(self, a_xx):
        t0 = time.time()
        ws, vs = self.b(tuple(x.astype(np.complex64) for x in a_xx))
        ws, vs = ws[0], vs[0]
        k = np.abs(ws) > 0
        self.rank = int(k.sum())
        t1 = time.time()
        self.stats = {}
        w, v = refine_nevp_modes([x[0] for x in a_xx], ws[k], vs[:, k], stats=self.stats)
        self.nmodes, self.t = len(w), (t1 - t0, time.time() - t1)
        return w[None], v[None]


class _RankBeyn:
    """qttools Beyn with the subspace rank of the last call (eigenvalues found in the annulus) in .rank. A saturated
    subspace (rank = m_0 < m) doubles m_0 (up to m) and repeats the call; m_0 then stays at the larger value."""

    def __init__(self, b):
        self.b, self.rank = b, None

    def __call__(self, a_xx):
        d = a_xx[0].shape[-1]
        while True:
            ws, vs = self.b(a_xx)
            self.rank = int((abs(ws) > 0).sum())
            if self.rank < self.b.m_0 or self.b.m_0 >= d:
                return ws, vs
            log(f"  qttools Beyn: subspace saturated (rank {self.rank} = m_0); m_0 -> {min(d, 2 * self.b.m_0)}")
            self.b.m_0 = min(d, 2 * self.b.m_0)


def beyn_memory(m, n, nq):
    """Rough peak bytes of the qttools spectral OBC with Beyn, primitive cell m, lead cell n, nq quadrature points:
    contour inverses at the outer and inner points (2 nq m^2) + moments / projection (~4 m^2) in the Beyn precision,
    and ~9 n^2 complex128 for g, the fixed-point inverses and the injection on the lead cell (both variants)."""
    lead = 16 * 9 * n * n
    return {"beyn": 16 * (2 * nq + 4) * m * m + lead, "beyn-mixed": 8 * (2 * nq + 4) * m * m + 16 * 2 * m * m + lead}


def _free_memory():
    """Free bytes: GPU (driver free + cupy pool free) in GPU mode, else the host's MemAvailable."""
    if gpu.on:
        import cupy as cp
        free, _ = cp.cuda.Device().mem_info
        pool = cp.get_default_memory_pool()
        return free + pool.total_bytes() - pool.used_bytes()
    try:
        with open("/proc/meminfo") as f:
            return next(int(l.split()[1]) * 1024 for l in f if l.startswith("MemAvailable"))
    except (OSError, StopIteration):
        return float("inf")


def lead_cell_perm(inp, orbs, sections):
    """Orbital order of a lead cell (read_quatrex orbitals orbs, in lead-matrix order) with its `sections` primitive
    cells contiguous along transport and every cell's atoms in the order of the first cell (matched by translation)."""
    if sections <= 1:
        return None
    from scipy.spatial import cKDTree
    norb, ax = inp["norb"], inp["meta"]["axis"]
    off = np.concatenate([[0], np.cumsum(norb)])
    atom_of = np.repeat(np.arange(len(norb)), norb)
    atoms = np.unique(atom_of[orbs])
    x = inp["xyz"][atoms]
    a = inp["meta"]["block_length"] / sections
    cell = np.floor((x[:, ax] - x[:, ax].min() + 0.25 * a) / a).astype(int)
    assert cell.min() == 0 and cell.max() == sections - 1, f"lead cell atoms span cells {cell.min()}..{cell.max()}"
    first = np.where(cell == 0)[0]
    tree = cKDTree(x[first])
    order = []
    for c in range(sections):
        shift = np.zeros(3); shift[ax] = c * a
        mine = np.where(cell == c)[0]
        d, j = tree.query(x[mine] - shift)
        assert d.max() < 1e-2 and len(np.unique(j)) == len(first) == len(mine), f"lead cell {c} does not match cell 0"
        order.extend(mine[np.argsort(j)])
    pos = {o: i for i, o in enumerate(orbs)}
    return np.array([pos[o] for k in order for o in range(off[atoms[k]], off[atoms[k] + 1])])


class Device:
    """Slabs = device blocks; bL, bR = device orbitals (device order) coupled to the left / right lead; dev = the
    read_quatrex orbital of each device orbital."""

    qtt_eta = 1e-5                  # |Im z| up to which Sigma(z) is the qttools OBC at E = Re z
    fallback_eta = 1e-6             # folded lead on the real axis (no qttools): Sigma at E + i fallback_eta

    def __init__(self, inp, sigma_cache=None, obc="auto"):
        self.inp, self.meta = inp, inp["meta"]
        norb, blocks = inp["norb"], inp["blocks"]
        ob = np.repeat(blocks, norb)
        O = {b: np.where(ob == b)[0] for b in np.unique(blocks)}
        tr = self.meta["transport"]
        self.slab_orbs = [O[b] for b in tr["device_blocks"]]           # orbitals of each slab (read_quatrex order)
        self.ns = [len(o) for o in self.slab_orbs]
        self.sb = np.concatenate([[0], np.cumsum(self.ns)])
        dev = np.concatenate(self.slab_orbs)
        self.dev, self.n = dev, len(dev)                               # device orbital -> read_quatrex orbital
        H, S = inp["H"], inp["S"]
        self.H, self.S = H[dev][:, dev].tocsr(), S[dev][:, dev].tocsr()
        self.loc = -np.ones(H.shape[0], int)                           # orbital -> slab-local index
        for o in self.slab_orbs:
            self.loc[o] = np.arange(len(o))
        r, c = (abs(self.H) + abs(self.S)).nonzero()
        so = np.repeat(np.arange(len(self.ns)), self.ns)
        assert np.abs(so[r] - so[c]).max() <= 1, "device blocks are not block tridiagonal"
        ld = inp["lead"]
        self.lead = FoldedLead(ld["H00"], ld["H01"], ld["S00"], ld["S01"])
        lr = inp.get("lead_right")
        self.leadR = self.lead if lr is None else FoldedLead(lr["H00"], lr["H01"], lr["S00"], lr["S01"])
        oL, oR = O[tr["lead_left_block"]][self.lead.F], O[tr["lead_right_block"]][self.leadR.F]
        P = (abs(H) + abs(S)).tocsr()
        self.bL = np.unique(P[O[tr["lead_left_block"]]][:, dev].nonzero()[1])
        self.bR = np.unique(P[O[tr["lead_right_block"]]][:, dev].nonzero()[1])
        assert self.bL.max() < self.sb[1] and self.bR.min() >= self.sb[-2], "leads couple beyond the end slabs"
        for o, blk in ((oL, "lead_left_block"), (oR, "lead_right_block")):
            rows = O[tr[blk]][np.unique(P[O[tr[blk]]][:, dev].nonzero()[0])]
            assert np.isin(rows, o).all(), f"{blk}: device couples to lead orbitals off its face"
        cut = lambda M, r, c: M[r][:, c].toarray()
        self.cL = [cut(M, oL, dev[self.bL]) for M in (H, S)] + [cut(M, dev[self.bL], oL) for M in (H, S)]
        self.cR = [cut(M, oR, dev[self.bR]) for M in (H, S)] + [cut(M, dev[self.bR], oR) for M in (H, S)]
        self.sigma_cache = sigma_cache
        assert obc in ("auto", "qttools", "folded", "beyn", "beyn-cell"), obc
        self.obc, self.qtt, self._inj, self.beyn = obc, None, {}, None
        if obc in ("beyn", "beyn-cell"):
            from .obc_beyn import BeynLead
            sec = int(self.meta.get("cells_per_block", 1)) if obc == "beyn" else 1
            OL, ORr = O[tr["lead_left_block"]], O[tr["lead_right_block"]]
            perm = lead_cell_perm(inp, OL, sec) if sec > 1 else None
            # two leads: one BeynLead per side (own layers, coupling basis, device copies and info); the right lead
            # is lead_right when given, else the same lead matrices
            rl = ld if lr is None else lr
            # memory lean by default (no accuracy cost): primitive layers and the fixed-point operands on the host;
            # QTX_BEYN_LU=mixed: complex64 LU of a_ii + iterative refinement (~1e-11 in Sigma, 2x time, -1.5 GB at 12 nm)
            bo = dict(layers_on="host", fp_store="host", lu_prec=os.environ.get("QTX_BEYN_LU", "double"))
            bl = BeynLead(ld["H00"], ld["H01"], ld["S00"], ld["S01"], sec, perm, sides="L", **bo)
            br = BeynLead(rl["H00"], rl["H01"], rl["S00"], rl["S01"], sec, perm, sides="R", **bo)
            self.beyn = {"L": bl, "R": br}
            scut = lambda M, r, c: M[r][:, c].tocsr()
            self.bcpl = {"L": [scut(M, OL, dev[self.bL]) for M in (H, S)] + [scut(M, dev[self.bL], OL) for M in (H, S)],
                         "R": [scut(M, ORr, dev[self.bR]) for M in (H, S)] + [scut(M, dev[self.bR], ORr) for M in (H, S)]}
            log(f"  lead OBC: spectral OBC (obc_beyn, {'periodized lead, %d primitive cells' % sec if sec > 1 else 'lead cell as given'}"
                f", {bl.method}) at every z")
            return
        if obc != "folded" and qttools_obc() is not None:
            sec = int(self.meta.get("cells_per_block", 1))
            OL, ORr = O[tr["lead_left_block"]], O[tr["lead_right_block"]]
            try:
                self.qtt = QttLead(ld["H00"], ld["H01"], ld["S00"], ld["S01"], sec, lead_cell_perm(inp, OL, sec))
                self.qttR = self.qtt if lr is None else QttLead(lr["H00"], lr["H01"], lr["S00"], lr["S01"], sec,
                                                                 lead_cell_perm(inp, OL, sec))
            except RuntimeError as e:
                if obc == "qttools":
                    raise
                log(f"  qttools OBC not used: {e}")
                self.qtt = None
            self.qL = [cut(M, OL, dev[self.bL]) for M in (H, S)] + [cut(M, dev[self.bL], OL) for M in (H, S)]
            self.qR = [cut(M, ORr, dev[self.bR]) for M in (H, S)] + [cut(M, dev[self.bR], ORr) for M in (H, S)]
        elif obc == "qttools":
            raise RuntimeError(f"obc 'qttools': {QTT_ERROR or 'qttools does not import'} (install it or set QTTOOLS_SRC)")
        elif obc == "auto" and QTT_ERROR and os.environ.get("QTTOOLS_SRC"):
            log(f"  qttools OBC not used: {QTT_ERROR}")
        log(f"  lead OBC on the real axis: {'qttools spectral OBC (%s)' % self.qtt.nevp_kind if self.qtt is not None else 'folded lead (decimation) at E + %g i' % self.fallback_eta}")

    def uses_qttools(self, z):
        return self.qtt is not None and abs(complex(z).imag) <= self.qtt_eta

    def sigma(self, z):
        """Sigma_L on bL, Sigma_R on bR at z. Real axis (|Im z| <= qtt_eta): qttools at E = Re z (the injected modes
        are kept for inject(); cached as qsig_<E>_<Im z>.npz, any such file at the same E is reused), or the folded
        lead at E + i fallback_eta without qttools. Complex z: folded lead (sig_<Re z>_<Im z>.npz)."""
        z = complex(z)
        if self.beyn is not None:
            return self._sigma_beyn(z)
        if self.uses_qttools(z):
            try:
                return self._sigma_qtt_cached(z.real)
            except RuntimeError as e:
                if self.obc == "qttools":
                    raise
                log(f"  {e}; folded-lead Sigma instead")
        if abs(z.imag) < self.fallback_eta:
            z = complex(z.real, self.fallback_eta)
        self._inj = {}
        return self._sigma_folded(z)

    def inject(self, Sg, tol=1e-4, device=False):
        """Injection columns for the Sigma_L Sg of the last sigma(): the qttools incoming modes when they came with
        it, else the Gamma_L eigenchannels above tol (injection()). device (GPU mode): a cupy array."""
        B = self._inj.get(id(Sg))
        if B is None:
            return injection(Sg, tol, device=device)
        return gpu.cp.asarray(B) if device and gpu.on else B

    def _sigma_qtt_cached(self, E):
        d = self.sigma_cache
        hit = sorted(glob.glob(os.path.join(d, f"qsig_{E:+.8f}_*.npz"))) if d else []
        hit = [f for f in hit if abs(float(f.rsplit("_", 1)[1][:-4])) <= self.qtt_eta]
        if hit:
            c = np.load(hit[0])
            out = c["SL"], c["SR"]
            self._inj = {id(out[0]): c["BL"], id(out[1]): c["BR"]}
            return out
        out, inj = [], []
        for side, lead, (Hld, Sld, Hdl, Sdl) in (("L", self.qtt, self.qL), ("R", self.qttR, self.qR)):
            g, u = lead.modes(E, side)
            Adl = E * Sdl - Hdl
            out.append(Adl @ g @ (E * Sld - Hld))
            inj.append(Adl @ u)
        out = tuple(out)
        self._inj = {id(out[0]): inj[0], id(out[1]): inj[1]}
        if d:
            os.makedirs(d, exist_ok=True)
            savez_atomic(os.path.join(d, f"qsig_{E:+.8f}_{0.0:.8e}.npz"), SL=out[0], SR=out[1], BL=inj[0], BR=inj[1])
        return out

    def _sigma_beyn(self, z):
        """obc beyn / beyn-cell: Sigma and (real axis, |Im z| <= qtt_eta, taken at E = Re z) the incoming modes."""
        if abs(z.imag) <= self.qtt_eta:
            z = complex(z.real, 0.0)
        tag = "p" if self.obc == "beyn" else "c"
        f = None
        if self.sigma_cache:
            f = os.path.join(self.sigma_cache, f"bsig{tag}_{z.real:+.8f}_{z.imag:.8e}.npz")
            if os.path.exists(f):
                c = np.load(f)
                out = c["SL"], c["SR"]
                self._inj = {id(out[0]): c["BL"], id(out[1]): c["BR"]}
                return out
        out, inj = [], []
        for side in "LR":
            Hld, Sld, Hdl, Sdl = self.bcpl[side]
            Sg, B = self.beyn[side].sigma(z, side, z * Sdl - Hdl, z * Sld - Hld)
            if gpu.on:
                gpu.free()                       # release the left lead's cached blocks before the right lead
            out.append(Sg)
            inj.append(B)
        out = tuple(out)
        self._inj = {id(out[0]): inj[0], id(out[1]): inj[1]} if z.imag == 0 else {}
        if f:
            os.makedirs(self.sigma_cache, exist_ok=True)
            savez_atomic(f, SL=out[0], SR=out[1], BL=inj[0], BR=inj[1])
        return out

    def _sigma_folded(self, z):
        f = None
        if self.sigma_cache:
            f = os.path.join(self.sigma_cache, f"sig_{z.real:+.8f}_{z.imag:.8e}.npz")
            if os.path.exists(f):
                c = np.load(f)
                return c["SL"], c["SR"]
        gL, gR = self.lead.surface_gfs(z)
        if self.leadR is not self.lead:
            gR = self.leadR.surface_gfs(z)[1]
        if gpu.on:                                           # the two dense products per side on the device too
            a = gpu.asarray
            out = tuple(gpu.asnumpy((z * a(Sdl) - a(Hdl)) @ a(g) @ (z * a(Sld) - a(Hld)))
                        for g, (Hld, Sld, Hdl, Sdl) in ((gL, self.cL), (gR, self.cR)))
        else:
            out = tuple((z * Sdl - Hdl) @ g @ (z * Sld - Hld)
                        for g, (Hld, Sld, Hdl, Sdl) in ((gL, self.cL), (gR, self.cR)))
        if f:
            os.makedirs(self.sigma_cache, exist_ok=True)
            savez_atomic(f, SL=out[0], SR=out[1])
        return out

    def A(self, E, SL, SR):
        """Sparse E S - H - Sigma_L - Sigma_R."""
        rows, cols, vals = [], [], []
        for Sg, b in ((SL, self.bL), (SR, self.bR)):
            ii, jj = np.meshgrid(b, b, indexing="ij")
            rows.append(ii.ravel()); cols.append(jj.ravel()); vals.append(Sg.ravel())
        Sig = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n, self.n))
        return (E * self.S - self.H - Sig).tocsr()


def injection(SL, tol=1e-4, device=False):
    """Columns B with Gamma_L ~ B B^H (eigenvectors of Gamma_L above tol [eV]: propagating channels). device (GPU
    mode): eigensolve and result stay on the device (cupy)."""
    if device and gpu.on:
        cp = gpu.cp
        G = cp.asarray(SL)
        G = 1j * (G - G.conj().T)
        lam, X = cp.linalg.eigh(0.5 * (G + G.conj().T))     # ascending
        k = lam > tol
        return X[:, k] * cp.sqrt(lam[k])
    G = 1j * (SL - SL.conj().T)
    lam, X = eigh(0.5 * (G + G.conj().T))                    # GPU mode: on the device
    lam, X = lam[::-1], X[:, ::-1]                           # ascending, as numpy's eigh
    k = lam > tol
    return X[:, k] * np.sqrt(lam[k])


def slab_blocks(dev, V=None):
    """Blocks HD, SD (s, s), HU, SU (s, s+1) of U^H H U, U^H S U for U = diag(V_s) (V None: U = I, dense blocks)."""
    b, n = dev.sb, len(dev.ns)
    out = dict(HD=[], SD=[], HU=[], SU=[])
    if gpu.on and V is not None:
        return _slab_blocks_gpu(dev, V)
    for M, kD, kU in ((dev.H, "HD", "HU"), (dev.S, "SD", "SU")):
        for s in range(n):
            X = M[b[s]:b[s + 1], b[s]:b[s + 1]]
            X = X.toarray() if V is None else V[s].conj().T @ (X @ V[s])
            out[kD].append(0.5 * (X + X.conj().T))
            if s < n - 1:
                X = M[b[s]:b[s + 1], b[s + 1]:b[s + 2]]
                out[kU].append(X.toarray() if V is None else V[s].conj().T @ (X @ V[s + 1]))
    return out


def _slab_blocks_gpu(dev, V):
    """slab_blocks(dev, V) with the sparse x dense products on the device (the host scipy product runs on one
    thread: 32 s for N 2373 on 22 848 orbitals, minutes for 72 672). numpy blocks, as slab_blocks."""
    cp = gpu.cp
    b, n = dev.sb, len(dev.ns)
    out = dict(HD=[], SD=[], HU=[], SU=[])
    # only V_s and V_{s+1} on the device (all V_s at once: ~60 GB for the 12 nm tile4y level 1, N 121 022; 2026-11-18)
    Vs = cp.asarray(V[0], dtype=complex)
    for s in range(n):
        Vn = cp.asarray(V[s + 1], dtype=complex) if s < n - 1 else None
        for M, kD, kU in ((dev.H, "HD", "HU"), (dev.S, "SD", "SU")):
            X = Vs.conj().T @ (gpu.csr(M[b[s]:b[s + 1], b[s]:b[s + 1]], np.complex128) @ Vs)
            out[kD].append(gpu.asnumpy(0.5 * (X + X.conj().T)))
            if Vn is not None:
                X = gpu.csr(M[b[s]:b[s + 1], b[s + 1]:b[s + 2]], np.complex128) @ Vn
                out[kU].append(gpu.asnumpy(Vs.conj().T @ X))
            del X
        Vs = Vn
        gpu.free()
    return out


class BlockTridiag:
    """LU of the block-tridiagonal A(z) = z S_c - H_c, minus the projected Sigma on the end blocks. The blocks may be
    numpy or cupy arrays (GPU mode: factors and solves on the device; solve() returns an array of the blocks' kind)."""

    def __init__(self, blk, z, SL1, SR1):
        n = len(blk["HD"])
        D = [z * blk["SD"][s] - blk["HD"][s] for s in range(n)]
        D[0] = D[0] - SL1
        D[-1] = D[-1] - SR1
        Up = [z * blk["SU"][s] - blk["HU"][s] for s in range(n - 1)]
        Lo = [z * blk["SU"][s].conj().T - blk["HU"][s].conj().T for s in range(n - 1)]
        self._factor(D, Up, Lo)

    @classmethod
    def from_blocks(cls, D, Up, Lo):
        """From the diagonal blocks D_s and the couplings Up_s = A_{s,s+1}, Lo_s = A_{s+1,s} (D is overwritten)."""
        self = cls.__new__(cls)
        self._factor(D, Up, Lo)
        return self

    def _factor(self, D, Up, Lo):
        n = len(D)
        self.xp = gpu.xp_of(D[0])
        self.Up, self.lu, self.L = Up, [], []
        self.bounds = np.concatenate([[0], np.cumsum([len(d) for d in D])])
        Dt = D[0]
        for s in range(n):
            self.lu.append(gpu.lu_factor(Dt))
            D[s] = Dt = None
            if s < n - 1:
                Lt = gpu.lu_solve(self.lu[s], Lo[s].T, trans=1).T
                self.L.append(Lt)
                Lo[s] = None
                Dt = D[s + 1] - Lt @ Up[s]

    def solve(self, B):
        b, n, xp = self.bounds, len(self.lu), self.xp
        B = xp.asarray(B)
        Y = [B[b[s]:b[s + 1]].astype(complex) for s in range(n)]
        for s in range(1, n):
            Y[s] = Y[s] - self.L[s - 1] @ Y[s - 1]
        X = [None] * n
        X[-1] = gpu.lu_solve(self.lu[-1], Y[-1])
        for s in range(n - 2, -1, -1):
            X[s] = gpu.lu_solve(self.lu[s], Y[s] - self.Up[s] @ X[s + 1])
        return xp.concatenate(X)


def block_solve(blk, z, SL1, SR1, B, host_frac=0.4):
    """X = A_c(z)^-1 B for the block-tridiagonal A_c(z) = z S_c - H_c - SL1 (first block) - SR1 (last block) of
    slab_blocks, in one forward / backward sweep with the right-hand sides carried along (block Thomas; no factors
    kept, so one call per energy):
        forward   C_s = D~_s^-1 U_s,  Z_s = D~_s^-1 Y~_s,  D~_{s+1} = D_{s+1} - L_s C_s,  Y~_{s+1} = B_{s+1} - L_s Z_s
        backward  X_s = Z_s - C_s X_{s+1}
    With gpu.on the blocks are copied to the device one slab at a time (the mode-space solve on the GPU: N 82 557 on
    6 nm in seconds instead of ~2 min with numpy); the C_s stay on the device when they take < host_frac of the free
    memory, else they go to the host. Peak device memory otherwise ~6 dense blocks. Without the GPU: numpy, same
    sweep. B and X are numpy arrays (N x columns)."""
    n = len(blk["HD"])
    xp = gpu.cp if gpu.on else np
    put = (lambda M: gpu.cp.asarray(M)) if gpu.on else (lambda M: M)
    sz = [len(d) for d in blk["HD"]]
    o = np.concatenate([[0], np.cumsum(sz)])
    keep = True
    if gpu.on:
        need = 16 * sum(sz[s] * sz[s + 1] for s in range(n - 1))
        keep = need < host_frac * gpu.cp.cuda.Device().mem_info[0]
    dev_out = gpu.on and gpu.xp_of(B) is not np          # cupy right-hand sides: the solution stays on the device
    B = B if dev_out else np.asarray(B)
    Dt = z * put(blk["SD"][0]) - put(blk["HD"][0]) - put(SL1)
    Yt = put(B[o[0]:o[1]].astype(complex))
    C, Z = [], []
    for s in range(n):
        if s == n - 1:
            Dt = Dt - put(SR1)
        F = gpu.lu_factor(Dt)
        Dt = None
        Z.append(gpu.lu_solve(F, Yt))
        if s < n - 1:
            SU, HU = put(blk["SU"][s]), put(blk["HU"][s])
            Cs = gpu.lu_solve(F, z * SU - HU)
            F = None
            Lo = z * SU.conj().T - HU.conj().T
            SU = HU = None
            Dt = z * put(blk["SD"][s + 1]) - put(blk["HD"][s + 1]) - Lo @ Cs
            Yt = put(B[o[s + 1]:o[s + 2]].astype(complex)) - Lo @ Z[-1]
            Lo = None
            C.append(Cs if keep else gpu.asnumpy(Cs))
            Cs = None
    X = [None] * n
    X[-1] = Z[-1]
    for s in range(n - 2, -1, -1):
        X[s] = Z[s] - put(C[s]) @ X[s + 1]
        C[s] = None
    X = xp.concatenate(X)
    if not dev_out:
        X = gpu.asnumpy(X)
    gpu.free()
    return X


def device_blocks(dev, A, xp=np):
    """Dense diagonal / upper / lower slab blocks of a sparse device matrix A (device order), as numpy or cupy arrays
    (cupy: built on the GPU from the CSR slices)."""
    A, b, n = sp.csr_matrix(A), dev.sb, len(dev.ns)
    dense = (lambda M: gpu.csr_dense(M, np.complex128)) if xp is not np else (lambda M: M.toarray().astype(complex))
    D = [dense(A[b[s]:b[s + 1], b[s]:b[s + 1]]) for s in range(n)]
    Up = [dense(A[b[s]:b[s + 1], b[s + 1]:b[s + 2]]) for s in range(n - 1)]
    Lo = [dense(A[b[s + 1]:b[s + 2], b[s]:b[s + 1]]) for s in range(n - 1)]
    return D, Up, Lo


class StreamBlockTridiag:
    """GPU block-tridiagonal LU of a sparse device matrix for devices whose dense blocks do not fit on the GPU:
    diagonal blocks are densified one at a time, the couplings stay sparse on the device, and the LU factor of every
    Schur-complemented diagonal block is moved to pinned host memory (copied back twice per solve, at bus speed:
    pageable copies made the exact solve copy-bound, ~115 of 137 s per energy on 6 nm / GH200). Peak GPU memory ~5
    dense blocks; host memory one dense block per slab."""

    def __init__(self, dev, A):
        A, b, n = sp.csr_matrix(A), dev.sb, len(dev.ns)
        cp = gpu.cp
        self.bounds = b[:n + 1]
        self.Up = [gpu.csr(A[b[s]:b[s + 1], b[s + 1]:b[s + 2]], np.complex128) for s in range(n - 1)]
        self.Lo = [gpu.csr(A[b[s + 1]:b[s + 2], b[s]:b[s + 1]], np.complex128) for s in range(n - 1)]
        self.lu = []
        Dt = gpu.csr_dense(A[b[0]:b[1], b[0]:b[1]], np.complex128)
        for s in range(n):
            lu = gpu.lu_factor(Dt)
            Dt = None
            if s < n - 1:
                W = gpu.lu_solve(lu, self.Up[s].toarray())          # D~_s^-1 A_{s,s+1}
                Dt = gpu.csr_dense(A[b[s + 1]:b[s + 2], b[s + 1]:b[s + 2]], np.complex128)
                Dt -= self.Lo[s] @ W
                del W
            self.lu.append((gpu.pinned(lu[0]), gpu.pinned(lu[1])))      # pinned: ~80x faster copies back
            del lu
            gpu.free()

    def _lu(self, s):
        return gpu.cp.asarray(self.lu[s][0]), gpu.cp.asarray(self.lu[s][1])

    def solve(self, B):
        cp, b, n = gpu.cp, self.bounds, len(self.lu)
        B = cp.asarray(B)
        Z = [None] * n                                          # Z_s = D~_s^-1 Y~_s
        Y = B[b[0]:b[1]].astype(complex)
        for s in range(n):
            Z[s] = gpu.lu_solve(self._lu(s), Y)
            if s < n - 1:
                Y = B[b[s + 1]:b[s + 2]].astype(complex) - self.Lo[s] @ Z[s]
        X = [None] * n
        X[-1] = Z[-1]
        for s in range(n - 2, -1, -1):
            X[s] = Z[s] - gpu.lu_solve(self._lu(s), self.Up[s] @ X[s + 1])
        gpu.free()
        return cp.concatenate(X)


def exact_last_slab(dev, A, B, chunk=None, info=None):
    """psi on the last slab of A X = B (B nonzero on the first slab only, as the QTBM injection) by the forward block
    elimination alone: Z_0 = D~_0^-1 B_0, Z_s = D~_s^-1 (-A_{s,s-1} Z_{s-1}), X_N = Z_N. No LU factor is kept, so
    nothing grows with the number of slabs (exact T without the back-substitution; 2026-11-18).
    Schur update D~_{s+1} = A_{s+1,s+1} - A_{s+1,s} D~_s^-1 A_{s,s+1} only on the coupled orbitals f of slab s+1 (the
    nonzero rows of A_{s+1,s} / columns of A_{s,s+1}), in column chunks; the |f| x |f| update is staged on the host
    while the LU of slab s is freed, so the GPU holds ~one dense block (+ chunks) at a time. Complex128, no refinement.
    GPU (gpu.on) or numpy. Returns X_N (n_N x columns, numpy)."""
    # column chunk of the Schur update (each chunk: 2 x n_s x chunk complex on the device; 2048 at 33k orbitals = 2 x
    # 1.1 GB): QTX_EXACT_CHUNK for a GPU shared with other jobs
    chunk = chunk or int(os.environ.get("QTX_EXACT_CHUNK", 2048))
    A, b, n = sp.csr_matrix(A), dev.sb, len(dev.ns)
    dev_out = gpu.on and gpu.xp_of(B) is not np          # cupy B: X_N returned on the device
    B = B.reshape(A.shape[0], -1) if dev_out else np.asarray(B).reshape(A.shape[0], -1)
    assert not bool(abs(B[b[1]:]).any()), "exact_last_slab: B must vanish outside the first slab"
    on = gpu.on
    xp = gpu.cp if on else np
    # Fortran order: the (cupyx / scipy) LU then factors in place (C order costs a second dense block)
    # scattered from the COO entries into an F-order zero block: exactly one dense block (cupyx toarray goes through a
    # second dense array)
    def dense(M):
        if not on:
            return np.asfortranarray(M.toarray().astype(complex))
        M = M.tocoo()
        D = xp.zeros(M.shape, complex, order="F")
        step = 1 << 24                                  # entries per scatter (~0.4 GB of indices + values on the device)
        for i in range(0, M.nnz, step):
            j = min(M.nnz, i + step)
            D[xp.asarray(M.row[i:j].astype(np.int32)), xp.asarray(M.col[i:j].astype(np.int32))] = \
                xp.asarray(M.data[i:j].astype(complex))
        gpu.free()
        return D
    tocsr = (lambda M: gpu.csr(M, np.complex128)) if on else (lambda M: M.astype(complex))
    host = (lambda X: X.get()) if on else (lambda X: X)
    pool = gpu.cp.get_default_memory_pool() if on else None
    peak, t0 = 0, time.time()
    Dt = dense(A[b[0]:b[1], b[0]:b[1]])
    Y = xp.asarray(B[b[0]:b[1]]).astype(complex)
    for s in range(n):
        lu = gpu.lu_factor(Dt)
        del Dt
        Z = gpu.lu_solve(lu, Y)
        if s == n - 1:
            break
        Up = A[b[s]:b[s + 1], b[s + 1]:b[s + 2]].tocsc()                  # slab s x slab s+1
        Lo = A[b[s + 1]:b[s + 2], b[s]:b[s + 1]].tocsr()                  # slab s+1 x slab s
        f = np.union1d(np.unique(Up.nonzero()[1]), np.unique(Lo.nonzero()[0]))
        Lof = tocsr(Lo[f])                                                 # |f| x slab s
        Xff = np.empty((len(f), len(f)), complex)                          # A_{s+1,s} D~_s^-1 A_{s,s+1} on f x f
        for c0 in range(0, len(f), chunk):
            c1 = min(len(f), c0 + chunk)
            Wc = gpu.lu_solve(lu, gpu.csr(Up[:, f[c0:c1]], np.complex128).toarray() if on
                              else Up[:, f[c0:c1]].toarray().astype(complex))
            Xff[:, c0:c1] = host(Lof @ Wc)
            if pool is not None:
                peak = max(peak, pool.total_bytes())
            del Wc
        Y = -(tocsr(Lo) @ Z)                                               # right-hand side of slab s+1
        del lu, Z, Lof
        if on:
            gpu.free()
        Dt = dense(A[b[s + 1]:b[s + 2], b[s + 1]:b[s + 2]])
        fx = xp.asarray(f)
        for c0 in range(0, len(f), chunk):
            c1 = min(len(f), c0 + chunk)
            Dt[fx[:, None], fx[None, c0:c1]] -= xp.asarray(Xff[:, c0:c1])
        del Xff
        if pool is not None:
            peak = max(peak, pool.total_bytes())
    X = Z if dev_out else host(Z)
    del lu, Z
    if on:
        gpu.free()
    if info is not None:
        info.update(peak_gb=peak / 2**30, t=time.time() - t0)
    return X


def exact_solve(dev, A, B, n_ref=2, tol=1e-12):
    """Exact solution of the sparse device system A X = B (numpy in, numpy out) and the max relative residual.
    CPU: CSolver (PARDISO / SuperLU + refinement + GMRES). GPU: dense block-tridiagonal LU over the slabs (the device
    is block tridiagonal by construction) + n_ref steps of refinement with the sparse A on the device."""
    if not gpu.on:
        cs = CSolver(A, n_ref=n_ref, tol=tol)
        X = cs.solve(B)
        cs.free()
        return X, cs.res
    cp = gpu.cp
    need = 16 * 5 * sum(int(n) ** 2 for n in dev.ns)      # dense D, Up, Lo + LU and L factors
    if need > 0.7 * cp.cuda.Device().mem_info[0]:
        log(f"  exact_solve: dense block LU needs ~{need / 2**30:.0f} GB, streaming variant (LU factors on the host)")
        bt = StreamBlockTridiag(dev, A)
    else:
        bt = BlockTridiag.from_blocks(*device_blocks(dev, A, cp))
    Ad = gpu.csr(A, np.complex128)
    dev_out = gpu.xp_of(B) is not np                     # cupy B: X returned on the device
    Bd = cp.asarray(B, dtype=complex).reshape(B.shape[0], -1)
    X = bt.solve(Bd)
    for _ in range(n_ref):
        X += bt.solve(Bd - Ad @ X)
    bn = cp.maximum(cp.linalg.norm(Bd, axis=0), 1e-300)
    res = float((cp.linalg.norm(Bd - Ad @ X, axis=0) / bn).max()) if Bd.shape[1] else 0.0
    if res > tol:
        log(f"  exact_solve (GPU block LU): residual {res:.1e} after {n_ref} refinement steps")
    if not dev_out:
        X = gpu.asnumpy(X)
    del bt, Ad, Bd
    gpu.free()
    return (X.reshape(B.shape) if B.ndim == 1 else X), res


class Coarse:
    """Mode-space (Galerkin) system of the per-slab basis V = [V_s] (slab-local orbitals x modes)."""

    def __init__(self, dev, V):
        self.dev, self.V = dev, V
        self.blk = slab_blocks(dev, V)
        if gpu.on:                  # block_solve copies every block to the device per energy: keep them pinned
            self.blk = {k: [gpu.pinned(M) for M in L] for k, L in self.blk.items()}
        self.PL, self.PR = V[0][dev.bL], V[-1][dev.bR - dev.sb[-2]]
        self.off = np.concatenate([[0], np.cumsum([v.shape[1] for v in V])])
        self.N = int(self.off[-1])
        if gpu.on:
            # V in pinned host memory, replaced in place in the caller's list (no second host copy): restrict / prolong
            # of host arrays then stream V_s to the device slab by slab at bus speed and multiply there (12 nm tile4y:
            # V ~70 GB, does not fit next to the solve; the numpy products took most of the time per energy)
            try:
                for s in range(len(V)):
                    V[s] = gpu.pinned(V[s])
            except Exception as e:                          # pinned allocation refused: plain host arrays still work
                log(f"  Coarse: V not pinned ({e}); restrict / prolong stream pageable memory")

    def to_gpu(self):
        """Keep device copies of the blocks and of V: Ac() then factors on the GPU, prolong / restrict take and return
        cupy arrays (the FGMRES loop). Host arrays given to prolong / restrict still work (numpy result)."""
        if gpu.on and not hasattr(self, "_g"):
            cp = gpu.cp
            self._g = dict(blk={k: [cp.asarray(M) for M in L] for k, L in self.blk.items()},
                           V=[cp.asarray(v) for v in self.V])
        return self

    def Ac(self, E, SL, SR, device=None):
        """BlockTridiag of the Galerkin matrix at E (on the GPU after to_gpu(), unless device=False)."""
        SL1, SR1 = self.PL.conj().T @ SL @ self.PL, self.PR.conj().T @ SR @ self.PR
        if hasattr(self, "_g") and device is not False:
            cp = gpu.cp
            return BlockTridiag(self._g["blk"], E, cp.asarray(SL1), cp.asarray(SR1))
        return BlockTridiag(self.blk, E, SL1, SR1)

    def solve(self, E, SL, SR, B):
        """Mode-space solution A_c(E)^-1 B for one set of right-hand sides (numpy in / out): the streaming sweep
        block_solve, on the GPU with gpu.on (blocks copied slab by slab, nothing kept between energies)."""
        SL1, SR1 = self.PL.conj().T @ SL @ self.PL, self.PR.conj().T @ SR @ self.PR
        return block_solve(self.blk, E, SL1, SR1, B)

    def _Vs(self, Y):
        if hasattr(self, "_g") and gpu.xp_of(Y) is not np:
            return self._g["V"]
        return self.V

    def _stream(self):
        """GPU mode without to_gpu(): products on the device with V_s streamed from pinned memory slab by slab."""
        return gpu.on and not hasattr(self, "_g")

    def P(self, side, xp=np):
        """V_0 on the left lead face / V_last on the right lead face (cupy copy cached for xp = cupy)."""
        if xp is np:
            return self.PL if side == "L" else self.PR
        if not hasattr(self, "_Pd"):
            self._Pd = {"L": gpu.cp.asarray(self.PL), "R": gpu.cp.asarray(self.PR)}
        return self._Pd[side]

    def prolong(self, Y):
        """U Y (GPU mode: on the device; cupy Y -> cupy result, numpy Y -> numpy result)."""
        if self._stream():
            cp, xp = gpu.cp, gpu.xp_of(Y)
            dt = np.result_type(Y.dtype, self.V[0].dtype, np.complex64)
            out = xp.empty((self.dev.sb[len(self.V)],) + Y.shape[1:], dt)
            b = self.dev.sb
            for s, v in enumerate(self.V):
                r = cp.asarray(v) @ cp.asarray(Y[self.off[s]:self.off[s + 1]])
                out[b[s]:b[s + 1]] = r if xp is not np else r.get()
            gpu.free()
            return out
        V = self._Vs(Y)
        return gpu.xp_of(Y).vstack([V[s] @ Y[self.off[s]:self.off[s + 1]] for s in range(len(V))])

    def restrict(self, X):
        """U^H X (GPU mode: on the device, as prolong). Slabs where X vanishes are skipped."""
        b = self.dev.sb
        if self._stream():
            cp, xp = gpu.cp, gpu.xp_of(X)
            out = xp.zeros((self.N,) + X.shape[1:], np.result_type(X.dtype, self.V[0].dtype, np.complex64))
            for s, v in enumerate(self.V):
                Xs = X[b[s]:b[s + 1]]
                if bool(Xs.any()):
                    r = cp.asarray(v).conj().T @ cp.asarray(Xs)
                    out[self.off[s]:self.off[s + 1]] = r if xp is not np else r.get()
            gpu.free()
            return out
        V = self._Vs(X)
        return gpu.xp_of(X).vstack([V[s].conj().T @ X[b[s]:b[s + 1]] for s in range(len(V))])


def is_real(blk):
    """True if all slab blocks of H and S are real: A(z) is then complex symmetric (see g_diag)."""
    return all(not np.iscomplexobj(M) or not np.abs(M.imag).any() for k in ("HD", "SD", "HU", "SU") for M in blk[k])


def g_diag(blk, z, SL, SR, iL, iR, sym=False, host=None):
    """Diagonal blocks G_ss of G = A(z)^-1, A = z S - H - Sigma_L - Sigma_R (Sigma_L on rows / columns iL of the first
    slab, Sigma_R on iR of the last), by left-connected block recursion:
        forward   g_s = (D_s - Y_{s-1} U_{s-1})^-1,  Y_s = L_s g_s
        backward  G_ss = g_s + X_s G_{s+1,s+1} Y_s,  X_s = g_s U_s
    (D_s, U_s, L_s = diagonal / upper / lower blocks of A). sym (real H and S: L_s = U_s^T, g_s = g_s^T) uses
    X_s = Y_s^T: 4 block products per slab instead of 5 (6 in the 0.1.0 recursion).
    host (GPU streaming, 2026-11-18): dict(g=[...], Y=[...]) of preallocated pinned host buffers; blk then holds host
    (pinned) blocks that are copied to the device when used, g_s / Y_s go to the host buffers in the forward sweep and
    come back in the backward sweep, G_ss is returned as host arrays. Device peak ~6 blocks instead of all blocks +
    2 per slab (12 nm level 2 on tile4y level 1 cut 1e-8: N1 ~5 500 per slab, ~42 GB of blocks)."""
    ns = len(blk["HD"])
    if host is not None:
        xp = gpu.cp
        put = lambda M: gpu.cp.asarray(M)
        def keep(M, buf):
            M.get(out=buf)
            return buf
        out_ = lambda M: gpu.asnumpy(M)
    else:
        xp = gpu.xp_of(blk["HD"][0])                         # numpy, or cupy for blocks on the GPU
        put = keep = lambda M, *_: M
        out_ = lambda M: M
    Up = lambda s: z * put(blk["SU"][s]) - put(blk["HU"][s])
    Lo = lambda s: z * put(blk["SU"][s]).conj().T - put(blk["HU"][s]).conj().T
    gl, Y = [None] * ns, [None] * (ns - 1)
    Yp = gs = None
    for s in range(ns):
        D = z * put(blk["SD"][s]) - put(blk["HD"][s])
        if s == 0:
            ix = xp.ix_(xp.asarray(iL), xp.asarray(iL))
            D[ix] = D[ix] - xp.asarray(SL)
        if s == ns - 1:
            ix = xp.ix_(xp.asarray(iR), xp.asarray(iR))
            D[ix] = D[ix] - xp.asarray(SR)
        if s > 0:
            D -= Yp @ Up(s - 1)
        gs = xp.linalg.inv(D)
        del D
        gl[s] = keep(gs, host["g"][s] if host else None)
        if s < ns - 1:
            Yp = Lo(s) @ gs
            Y[s] = keep(Yp, host["Y"][s] if host else None)
    Yp = None
    out = [None] * ns
    G = gs
    out[-1] = out_(G)
    for s in range(ns - 2, -1, -1):
        g_, Y_ = put(gl[s]), put(Y[s])
        X = Y_.T if sym else g_ @ Up(s)
        G = g_ + X @ (G @ Y_)
        out[s] = out_(G)
        gl[s] = Y[s] = None
        del g_, Y_, X
    return out


def rho_contour(dev, mu1, mu2, npts, log=log, V=None):
    """Slab density matrices of the open device, rho_ss = -(1/pi) Im int G_ss dz over the upper half circle from mu1
    to mu2 (Gauss-Legendre in the angle), G_ss from g_diag (block recursion; real H, S use the symmetric variant).
    GPU mode: the slab blocks stay on the device, the recursion runs there, G_ss is accumulated on the host.
    V (per-slab level-1 basis, orthonormal columns): the global rho of the open level-1 system (level 2 on level 1),
    i.e. the recursion on the Galerkin blocks V^H (H, S) V with the lead Sigma projected on the end-slab bases;
    rho_ss is then N1_s x N1_s in the coefficients of V_s."""
    blk = slab_blocks(dev, V)
    sym = is_real(blk)
    if sym:
        for k in ("HD", "SD", "HU", "SU"):
            blk[k] = [np.ascontiguousarray(np.real(M)) for M in blk[k]]
    host = None
    if gpu.on:
        nb = sum(M.nbytes for L in blk.values() for M in L)
        free = gpu.cp.cuda.Device().mem_info[0]
        if nb + 16 * 3 * sum(len(h) ** 2 for h in blk["HD"]) < 0.6 * free:
            blk = {k: [gpu.asarray(M) for M in L] for k, L in blk.items()}
        else:                                   # stream: blocks and g_s / Y_s in pinned host memory (g_diag host=)
            blk = {k: [gpu.pinned(M) for M in L] for k, L in blk.items()}
            n_ = [len(h) for h in blk["HD"]]
            host = dict(g=[gpu.pinned(np.empty((m, m), complex)) for m in n_],
                        Y=[gpu.pinned(np.empty((n_[s + 1], n_[s]), complex)) for s in range(len(n_) - 1)])
            log(f"  rho_contour: blocks {nb / 2**30:.1f} GB, GPU free {free / 2**30:.1f} GB -> blocks and recursion "
                f"in pinned host memory, streamed per slab")
    iL, iR = dev.bL, dev.bR - dev.sb[-2]                     # lead faces in the end slabs
    proj = lambda SL, SR: (SL, SR)
    if V is not None:                                        # Sigma on the whole end-slab level-1 blocks
        PL, PR = V[0][iL], V[-1][iR]
        iL, iR = np.arange(V[0].shape[1]), np.arange(V[-1].shape[1])
        proj = lambda SL, SR: (PL.conj().T @ SL @ PL, PR.conj().T @ SR @ PR)
    c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    x, w = np.polynomial.legendre.leggauss(npts)
    th, w = 0.5 * np.pi * (x + 1), 0.5 * np.pi * w
    acc = [np.zeros(h.shape, complex) for h in blk["HD"]]
    t0 = time.time()
    for ip, (t_, w_) in enumerate(zip(th, w)):
        z = c + r * np.exp(1j * t_)
        dz = -1j * r * np.exp(1j * t_) * w_
        t1 = time.time()
        SL, SR = proj(*dev.sigma(z))
        t2 = time.time()
        for a, G in zip(acc, g_diag(blk, z, SL, SR, iL, iR, sym, host=host)):
            a += gpu.asnumpy(G) * dz
        log(f"    contour point {ip + 1}/{npts}, {time.time() - t0:.0f} s (sigma {t2 - t1:.0f} s, "
            f"recursion {time.time() - t2:.0f} s)")
    del blk
    gpu.free()
    return [-((a - a.conj().T) / 2j) / np.pi for a in acc]


def top_modes(rho, n):
    """The n eigenvectors of the Hermitian rho with the largest eigenvalues, largest first."""
    R = 0.5 * (rho + rho.conj().T)
    m = len(R)
    if gpu.on:                                               # full eigh on the GPU (no subset option), top n kept
        lam, X = eigh(R)
        return lam[:n], X[:, :n]
    lam, X = sla.eigh(R, subset_by_index=[m - min(n, m), m - 1], driver="evr")
    return lam[::-1], X[:, ::-1]


def eigh(R):
    """All eigenpairs of the Hermitian R, largest eigenvalue first (numpy results; GPU mode: cupy's eigh)."""
    if gpu.on:
        lam, X = gpu.cp.linalg.eigh(gpu.cp.asarray(R))
        lam, X = gpu.asnumpy(lam[::-1]), gpu.asnumpy(X[:, ::-1])
        gpu.free()
        return lam, X
    lam, X = np.linalg.eigh(R)
    return lam[::-1], X[:, ::-1]
