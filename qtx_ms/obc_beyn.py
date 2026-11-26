"""Lead OBC from the Bloch modes of the periodized lead (QuaTrEx / qttools spectral OBC with Beyn, reimplemented),
for real E and complex z, on numpy or cupy (gpu.on) without qttools.

Lead model (as QuaTrEx): the lead cell of `sections` primitive cells is replaced by the exactly periodic lead built from
its outermost primitive cell (qttools extract_layer / upscale_layer): periodic blocks L_k, k = -s..s (s = sections), only
the nonzero ones stored ({power k: m x m block}; they alone enter the polynomial T(w), degree 2 max|k|), and
lead-cell blocks a_ji, a_ii, a_ij rebuilt from it. sections = 1 keeps the lead cell as given (no periodization).
Everything is linear in z (A = z S - H): the layers of H and S are extracted once per side.

Per z and side:
  1. Beyn on the primitive-cell polynomial T(w) = sum_k L_k(z) w^k (m x m): annulus r_i < |w| < r_o (per primitive
     cell: QuaTrEx r_o, r_i per lead cell to the power 1/s), nq points per circle, probing block Y (m x m0); the
     contour sums need solves T(w_j)^-1 Y only (no inverses). Rank by SVD, small eigenproblem on the host.
  2. Modes upscaled to the lead cell (v, w v, ..., w^(s-1) v; eigenvalue w^s), residual filter, selection:
       real axis (|Im z| <= real_eta): qttools' rule (propagating by |Im k| < min_decay and the group-velocity sign,
       decaying by Im k < -min_decay, with max_decay = 1.5 ln r_o);
       complex z: retarded = decaying into the lead (Im k < 0, i.e. |w| > 1 in qttools' convention), max_decay cut.
  3. g from the modes, x_ii_a_ij = Phi Lambda^-1 Phi^+, g_0 = (a_ii + a_ji x_ii_a_ij)^-1, n_ref fixed-point steps
     g <- (a_ii - a_ji g a_ij)^-1 (qttools num_ref_iterations), injection b from the incoming modes (real axis only).
  4. Sigma = A_dl g A_ld, B = A_dl g a_ji b (A_dl, A_ld sparse device-lead couplings); only Sigma, B leave the device.
"""
import hashlib, os, threading, time
import numpy as np
import scipy.sparse as sp

from . import gpu


def _lead_blocks(a_ji, a_ii, a_ij, q, s):
    """The nonzero primitive blocks {k: m x m} of the periodic lead from the CSR lead-cell blocks (orbital order q): L_k couples cell
    j to cell j + k. L_0 = a_ii[0, 0], L_k = a_ii[0, k], L_-k = a_ii[k, 0] (0 < k < s), L_s = a_ij[0, 0], L_-s = a_ji[0, 0] (qttools
    extract_layer; s = 1: a_ji, a_ii, a_ij). Blocks without a nonzero entry are not stored (L_0 always is)."""
    m = len(q) // s
    c = [q[i * m:(i + 1) * m] for i in range(s)]
    src = {0: (a_ii, 0, 0), s: (a_ij, 0, 0), -s: (a_ji, 0, 0)}
    src.update({k: (a_ii, 0, k) for k in range(1, s)})
    src.update({-k: (a_ii, k, 0) for k in range(1, s)})
    out = {}
    for k in sorted(src):
        M, i, j = src[k]
        B = M[c[i]][:, c[j]]
        if k == 0 or B.count_nonzero():
            out[k] = B.toarray()
    return out


def _shifted(H, S, z, xp):
    """{k: z S_k - H_k} on xp, of the powers stored in either (H, S: {k: block})."""
    out = {}
    for k in sorted(set(H) | set(S)):
        A = 0
        if k in S:
            A = z * xp.asarray(S[k])
        out[k] = A - xp.asarray(H[k]) if k in H else A
    return out


def _blocks_of(L, s, xp, which, dtype=None):
    """One lead-cell block ("ji", "ii" or "ij") of the periodic lead from its nonzero primitive blocks L = {k: block}:
    block (i, j) = L[j - i + o], o = -s, 0, s."""
    m, o = L[0].shape[0], {"ji": -s, "ii": 0, "ij": s}[which]
    out = xp.zeros((s * m, s * m), dtype=dtype or L[0].dtype)
    for i in range(s):
        for j in range(s):
            if j - i + o in L:
                out[i * m:(i + 1) * m, j * m:(j + 1) * m] = L[j - i + o]
    return out


def _aii_mv(L, s, X, trans=False):
    """a_ii X (trans: a_ii^T X) from the primitive blocks (block Toeplitz, block (i, j) = L[j - i]); a_ii is never
    formed. Complex128."""
    m = L[0].shape[0]
    xp = gpu.cp if (gpu.on and not isinstance(X, np.ndarray)) else np
    out = xp.zeros_like(X)
    for i in range(s):
        for j in range(s):
            k = i - j if trans else j - i
            if k in L:
                out[i * m:(i + 1) * m] += (L[k].T if trans else L[k]) @ X[j * m:(j + 1) * m]
    return out


class _AiiSolver:
    """Solves with a_ii (the lead-cell diagonal block). prec "double": complex128 LU. prec "mixed": complex64 LU (half
    the memory of the largest array) + iterative refinement, residual in complex128 from the primitive blocks
    (_aii_mv): X <- X + LU64^-1 (Y - a_ii X) until max_col |R| / |Y| <= tol. a_ii is one lead cell (no 1/eta
    conditioning), so the refinement contracts by ~ kappa(a_ii) eps_32 per step. park() / unpark(): LU (and the blocks)
    to the host and back."""

    def __init__(self, L, s, xp, prec="double", tol=1e-14, maxit=10):
        self.L, self.s, self.xp, self.prec, self.tol, self.maxit = L, s, xp, prec, tol, maxit
        a = _blocks_of(L, s, xp, "ii", dtype=np.complex64 if prec == "mixed" else complex)
        self.lu = gpu.lu_factor(a)
        del a
        self.its, self.res, self.fail = [], 0.0, 0
        self._h = None

    def mv(self, X, trans=False):
        return _aii_mv(self.L, self.s, X, trans)

    def __call__(self, Y, trans=0):
        if self.prec != "mixed":
            return gpu.lu_solve(self.lu, Y, trans=trans)
        xp = self.xp
        c64 = np.complex64
        X = gpu.lu_solve(self.lu, Y.astype(c64), trans=trans).astype(complex)
        nY = xp.linalg.norm(Y, axis=0)
        nY[nY == 0] = 1
        rel, prev = np.inf, np.inf
        for it in range(self.maxit):
            R = Y - self.mv(X, bool(trans))
            rel = float((xp.linalg.norm(R, axis=0) / nY).max())
            # stop at tol or at the rounding floor of the complex128 residual (~ eps |a_ii| |X| / |Y|, 1e-13..1e-12
            # near isolated-cell levels): the step no longer halves the residual
            if rel <= self.tol or rel > 0.5 * prev:
                break
            prev = rel
            X += gpu.lu_solve(self.lu, R.astype(c64), trans=trans).astype(complex)
            del R
        self.its.append(it)
        self.res = max(self.res, rel)
        self.fail += int(rel > 1e-10)
        return X

    def park(self):
        if self.xp is not np:
            self._h = (self.lu[0].get(), self.lu[1].get(), None if self.L is None else {k: x.get() for k, x in self.L.items()})
            self.lu, self.L = None, None
            gpu.free()

    def unpark(self):
        if self._h is not None:
            xp = self.xp
            self.lu = (xp.asarray(self._h[0]), xp.asarray(self._h[1]))
            self.L = None if self._h[2] is None else {k: xp.asarray(x) for k, x in self._h[2].items()}
            self._h = None


class BeynLead:
    """Periodized lead (or the lead cell as given, sections = 1) with the spectral OBC above.
    H00, H01, S00, S01: lead cell (H01: cell n -> n+1 along +x), in lead-matrix order; perm: lead-cell orbital order
    with the primitive cells contiguous along +x (core.lead_cell_perm), None for sections = 1."""

    def __init__(self, H00, H01, S00, S01, sections=1, perm=None, r_o=10.0, r_i=0.8, nq=20, m0=None, n_ref=4,
                 min_decay=1e-3, min_propagation=0.01, residual_tol=1e-3, real_eta=1e-5, seed=42, method="woodbury",
                 basis_tol=1e-14, fp_tol=1e-13, fp_maxit=300, chunk="auto", offload_lu=True, sides="LR",
                 fp_prec="double", fp_store="device", fp_tol_c64=1e-6, lu_prec="double", layers_on="device",
                 ir_tol=1e-14, ir_maxit=10):
        """method "dense": g (n x n) with qttools' n_ref fixed-point inversions (reference); "woodbury": face fixed
        point on S = P_c^H g P_c (r x r) to fp_tol, coupling bases cut at basis_tol (_basis); g, u never formed."""
        n = H00.shape[0]
        s = int(sections)
        assert n % s == 0
        m = n // s
        self.n, self.m, self.s = n, m, s
        p = np.arange(n) if perm is None else np.asarray(perm)
        self.p = {"L": p, "R": p.reshape(s, m)[::-1].ravel()}            # qttools order k <-> lead orbital p[side][k]
        H00, H01, S00, S01 = (sp.csr_matrix(M) for M in (H00, H01, S00, S01))
        self.layer = {}                       # (side, "H" | "S") -> {k: primitive block}, the nonzero blocks (real for a real lead)
        self.sides = sides                    # the lead sides this object serves (Device: one object per lead)
        for side in sides:
            q = self.p[side]
            for nm, M00, M01 in (("H", H00, H01), ("S", S00, S01)):
                M10 = M01.conj().T.tocsr()
                a_ji, a_ij = (M10, M01) if side == "L" else (M01, M10)
                self.layer[side, nm] = _lead_blocks(a_ji, M00, a_ij, q, s)
        self.r_o, self.r_i, self.nq = r_o ** (1 / s), r_i ** (1 / s), int(nq)
        self.max_decay = 1.5 * np.log(r_o)
        self.m0 = int(m0 or max(64, m // 2))
        self.K = 1                                                         # moments (block Hankel) of Beyn, raised in _beyn
        self.n_ref, self.min_decay, self.min_prop, self.res_tol = n_ref, min_decay, min_propagation, residual_tol
        self.real_eta = real_eta
        self.Y = (np.random.default_rng(seed).random((m, m)) + 1j * np.random.default_rng(seed + 1).random((m, m)))
        self.info = {}
        self._dev = None
        self.method, self.basis_tol, self.fp_tol, self.fp_maxit = method, basis_tol, fp_tol, fp_maxit
        self.chunk = max(256, n // 32) if chunk == "auto" else int(chunk)
        self.offload_lu = offload_lu
        self.fp_prec, self.fp_store, self.fp_tol_c64 = fp_prec, fp_store, fp_tol_c64
        self.lu_prec, self.layers_on, self.ir_tol, self.ir_maxit = lu_prec, layers_on, ir_tol, ir_maxit
        self.fp_stall = True                                               # stop the fixed point when it stalls (False: fp_tol / fp_maxit only)
        self._bases = {}
        self.basis_dir = None                                              # folder of the coupling bases (they do not depend on z)

    # ---- device copies -------------------------------------------------------------------------------------
    def _xp(self):
        return gpu.cp if gpu.on else np

    def _arrays(self):
        """Primitive-cell layers of H, S on the device (made once; the lead-cell blocks are rebuilt per z)."""
        if gpu.on and self.layers_on == "device":
            if self._dev is None:
                self._dev = {q: {k: gpu.cp.asarray(x) for k, x in v.items()} for q, v in self.layer.items()}
            return self._dev
        return self.layer                     # host (layers_on "host": uploaded per z in _sigma_woodbury)

    # ---- 1. Beyn ---------------------------------------------------------------------------------------------
    def _beyn(self, L, xp):
        """Eigenpairs (w, v) of sum_k L[k] w^k (L: the nonzero blocks {k: block}) in the annulus. The probing block m0 is doubled
        (up to m) while the rank fills it (all eigenvalues in the annulus need rank < m0); a small cell can hold more eigenvalues
        than its m orbitals (up to the degree times m): then more moments (block Hankel, K = 2, 4, ...)."""
        deg = max(L) - min(L)
        while True:
            w, vs = self._beyn_once(L, xp)
            # near saturation counts as saturated: with more eigenvalues in the annulus than K m0 the numerical rank can stay just
            # below K m0 (test strip at -2.65 eV: rank 15 of 16, 22 eigenvalues, both channels lost)
            if self.info["rank"] < 0.75 * self.K * self.m0 or (self.m0 >= self.m and self.K >= deg):
                return w, vs
            if self.m0 < self.m:
                self.m0 = min(self.m, 2 * self.m0)
                self.info["m0_doubled"] = self.m0
            else:
                self.K = min(deg, 2 * self.K)
                self.info["K"] = self.K

    def _beyn_once(self, L, xp):
        m, K = self.m, self.K
        Y = xp.asarray(self.Y[:, :self.m0])
        P = [xp.zeros((m, self.m0), complex) for _ in range(2 * K)]               # moments
        for rad, sign in ((self.r_o, 1.0), (self.r_i, -1.0)):
            for j in range(self.nq):
                w = rad * np.exp(2j * np.pi * (j + 1.5) / self.nq)            # qttools: (zeta + 0.5) / N, zeta = 1..N
                T = sum(Lk * w ** k for k, Lk in L.items())
                X = xp.linalg.solve(T, Y)
                for q in range(2 * K):
                    P[q] += (sign * w ** (q + 1) / self.nq) * X
        hankel = lambda o: P[o] if K == 1 else xp.concatenate([xp.concatenate([P[a + b + o] for b in range(K)], axis=1) for a in range(K)], axis=0)
        P0, P1 = hankel(0), hankel(1)
        U, sv, Vh = xp.linalg.svd(P0, full_matrices=False)
        sv_h = gpu.asnumpy(sv) if xp is not np else sv
        keep = int((sv_h > sv_h.max() * m * np.finfo(float).eps).sum())
        U, sv, Vh = U[:, :keep], sv[:keep], Vh[:keep]
        A = (U.conj().T @ P1 @ Vh.conj().T) / sv[None, :]
        A = gpu.asnumpy(A) if xp is not np else A
        w, v = np.linalg.eig(A)                                              # small (rank x rank), host
        vs = U[:m] @ xp.asarray(v)
        self.info["rank"] = keep
        self.info["saturated"] = keep >= 0.75 * K * self.m0
        return w, vs

    # ---- 2. upscale, filter, select ---------------------------------------------------------------------------
    def _select(self, w, vs, a_xx, xp, real):
        s = self.s
        if s > 1:
            wsx = xp.asarray(w)
            vs = xp.concatenate([vs * (wsx ** k)[None, :] for k in range(s)], axis=0)
            w = w ** s
        vs = vs / xp.linalg.norm(vs, axis=0)[None, :]
        wx = xp.asarray(w)
        a_ji, a_ii, a_ij = a_xx
        with np.errstate(divide="ignore", invalid="ignore"):
            prod = (a_ji @ vs) / wx[None, :] + a_ii @ vs + (a_ij @ vs) * wx[None, :]
            res = xp.linalg.norm(prod, axis=0) / xp.abs(wx)
            dEk = -sum((1j * nn) * xp.sum(vs.conj() * (ax @ vs), axis=0) * wx ** nn
                       for ax, nn in ((a_ji, -1), (a_ii, 0), (a_ij, 1)))
            ks = -1j * xp.log(wx)
        dEk = xp.nan_to_num(dEk, nan=0, posinf=0, neginf=0)
        ks = xp.nan_to_num(ks, nan=0, posinf=0, neginf=0)
        ok = res < self.res_tol
        fast_enough = ks.imag > -self.max_decay
        if real:
            eps = np.finfo(float).eps
            prop_ratio = abs(dEk.real) / (abs(dEk.imag) + eps)
            prop = (xp.abs(ks.imag) < self.min_decay) & (prop_ratio > self.min_prop) & (dEk.real < 0)
            dec = (ks.imag < -self.min_decay) | ((prop_ratio <= self.min_prop) & (ks.imag < -1e-14))
            refl = (prop | (dec & fast_enough)) & ok
            inj = (dEk.real > 0) & (xp.abs(ks.imag) < self.min_decay) & (prop_ratio > self.min_prop)
        else:
            refl = (ks.imag < 0) & fast_enough & ok
            inj = xp.zeros(len(w), bool)
        self.info.update(modes=int(refl.sum()), channels=int(inj.sum()))
        return wx, vs, refl, inj, dEk

    # ---- coupling bases (Woodbury) ------------------------------------------------------------------------------
    def _basis(self, side):
        """Bases of the column space (P_c) and row space (P_r) of a_ij(z) = z S_ij - H_ij for every z: SVD of
        [H_ij / |H_ij|, S_ij / |S_ij|] on the nonzero rows / columns ("support"), cut at basis_tol relative.
        Stored on the host on the support only: Uc (|rows| x rc), Ur (|cols| x rr), P_c = Uc on rows, P_r = Ur on cols;
        cores K_H = P_c^H H_ij P_r, K_S = P_c^H S_ij P_r, so a_ij = P_c (z K_S - K_H) P_r^H and
        a_ji = z S_ij^H - H_ij^H = P_r (z K_S^H - K_H^H) P_c^H. Once per side (SVD on the GPU when gpu.on)."""
        if side in self._bases:
            return self._bases[side]
        t0 = time.time()
        f = self._basis_file(side)
        if f and os.path.exists(f):
            with np.load(f) as c:
                b = {k: c[k] for k in c.files}
            b.update(Uc=b["Uc"].astype(complex), Ur=b["Ur"].astype(complex), rc=int(b["rc"]), rr=int(b["rr"]), err=tuple(b["err"]),
                     support=tuple(int(v) for v in b["support"]), t=time.time() - t0, loaded=True)
            self._bases[side] = b
            return b
        H, S, n_s, m = self.layer[side, "H"], self.layer[side, "S"], self.s, self.m
        pw = [k for k in sorted(set(H) | set(S)) if k > 0]                 # block (i, j) of a_ij = L[s + j - i] (0 < s + j - i <= s)
        nz = {k: sum((abs(M[k]) for M in (H, S) if k in M), np.zeros((m, m))) for k in pw}
        nzr, nzc = {k: nz[k].any(axis=1) for k in pw}, {k: nz[k].any(axis=0) for k in pw}
        del nz
        bl = [(i, j, n_s + j - i) for i in range(n_s) for j in range(n_s) if n_s + j - i in nzr]
        ri = [np.where(np.any([nzr[k] for a, _, k in bl if a == i] or [np.zeros(m, bool)], axis=0))[0] for i in range(n_s)]
        cj = [np.where(np.any([nzc[k] for _, b, k in bl if b == j] or [np.zeros(m, bool)], axis=0))[0] for j in range(n_s)]
        rows, cols = (np.concatenate([a * m + x for a, x in enumerate(v)]).astype(int) for v in (ri, cj))
        ro, co = (np.concatenate([[0], np.cumsum([len(x) for x in v])]) for v in (ri, cj))

        def sub(M):                                                        # a_ij on its support, from the nonzero blocks only
            out = np.zeros((len(rows), len(cols)), M[0].dtype)
            for i, j, k in bl:
                if k in M:
                    out[ro[i]:ro[i + 1], co[j]:co[j + 1]] = M[k][np.ix_(ri[i], cj[j])]
            return out
        Hs, Ss = sub(H), sub(S)
        nH, nS = np.linalg.norm(Hs), max(np.linalg.norm(Ss), 1e-300)
        xp = self._xp()

        def span(X):
            U, sv, _ = xp.linalg.svd(xp.asarray(X), full_matrices=False)
            svh = gpu.asnumpy(sv) if xp is not np else sv
            return U[:, :int((svh > self.basis_tol * svh[0]).sum())].copy()     # xp array; the rest of U is freed

        Uc = span(np.hstack([Hs / nH, Ss / nS]))
        Ur = span(np.hstack([Hs.conj().T / nH, Ss.conj().T / nS]))
        K, err = {}, []                                              # cores and reconstruction errors: products on xp (12 nm: 131 s on 8 cores)
        for key, M in (("H", Hs), ("S", Ss)):
            Md = xp.asarray(M)
            Kd = Uc.conj().T @ Md @ Ur
            err.append(float(xp.abs(Uc @ Kd @ Ur.conj().T - Md).max()) / max(float(xp.abs(Md).max()), 1e-300))
            K[key] = gpu.asnumpy(Kd) if xp is not np else Kd
            del Md, Kd
        Uc, Ur = ((gpu.asnumpy(U) if xp is not np else U).astype(complex) for U in (Uc, Ur))     # real lead: real SVD and products
        if xp is not np:
            gpu.free()
        b = dict(Uc=Uc, Ur=Ur, rows=rows, cols=cols, KH=K["H"], KS=K["S"], rc=Uc.shape[1], rr=Ur.shape[1], err=tuple(err),
                 t=time.time() - t0, support=(len(rows), len(cols)))
        self._bases[side] = b
        if f:
            os.makedirs(self.basis_dir, exist_ok=True)
            tmp = f"{f[:-4]}.{os.getpid()}.{threading.get_ident()}.tmp.npz"   # several ranks may build the same side: last rename wins
            real = lambda U: U.real if not U.imag.any() else U
            np.savez(tmp, **{k: (real(v) if k in ("Uc", "Ur") else v) for k, v in b.items() if k != "t"})
            os.replace(tmp, f)
        return b

    def _basis_file(self, side):
        """File of the coupling basis of a side in basis_dir: named by a hash of the lead layers and basis_tol (None: no basis_dir)."""
        if not self.basis_dir:
            return None
        h = hashlib.blake2b(digest_size=12)
        h.update(f"{side} {self.basis_tol:.3e} {self.s} {self.m}".encode())
        for nm in ("H", "S"):
            for k, M in sorted(self.layer[side, nm].items()):
                h.update(str(k).encode())
                h.update(np.ascontiguousarray(M).tobytes())
        return os.path.join(self.basis_dir, f"beyn_basis_{side}_{h.hexdigest()}.npz")

    def _sigma_woodbury(self, z, side, Adl, Ald, xp, D, real):
        """Face fixed point, memory lean: g = G0 (I + P_r W P_r^H G0), G0 = a_ii^-1, W = M (I - C M)^-1, iterated on
        M = a_ji-core S a_ij-core (rr x rr) with S = P_c^H g P_c:  M <- Ah + Bh W Dh,
          Ah = Kji (P_c^H G0 P_c) Kij, Bh = Kji P_c^H G0 P_r, C = P_r^H G0 P_r, Dh = P_r^H G0 P_c Kij;
        start from the Beyn modes (g0 = (a_ii + a_ji X)^-1 by Woodbury, rank = #modes); stop at fp_tol or stagnation.
        Sigma = A_dl G0 A_ld + (A_dl G0 P_r) W (P_r^H G0 A_ld). G0 applied by LU solves in column chunks (self.chunk);
        the face-sized products go to the host; the LU is parked on the host during the fixed point (offload_lu)."""
        bs = self._basis(side)
        gpu_on = xp is not np
        pool = gpu.cp.get_default_memory_pool() if gpu_on else None
        peak = [0]
        phase_peak, phase_t, t_mark = {}, {}, [time.time()]

        def mark(name="other"):                                            # end of a phase (or of a part of it): time and GPU memory
            gpu.sync()
            phase_t[name] = phase_t.get(name, 0.0) + time.time() - t_mark[0]
            t_mark[0] = time.time()
            if pool is not None:
                b = pool.total_bytes()
                peak[0] = max(peak[0], b)
                phase_peak[name] = max(phase_peak.get(name, 0), b)

        to = (lambda a: a.get()) if gpu_on else (lambda a: a)
        n = self.n
        rows_c, rows_r = xp.asarray(bs["rows"]), xp.asarray(bs["cols"])
        Uc, Ur = xp.asarray(bs["Uc"]), xp.asarray(bs["Ur"])
        rc, rr = bs["rc"], bs["rr"]
        Kij = xp.asarray(z * bs["KS"] - bs["KH"])
        Kji = xp.asarray(z * bs["KS"].conj().T - bs["KH"].conj().T)

        def a_ij_mv(V):
            out = xp.zeros((n, V.shape[1]), complex)
            out[rows_c] = Uc @ (Kij @ (Ur.conj().T @ V[rows_r]))
            return out

        def a_ji_mv(V):
            out = xp.zeros((n, V.shape[1]), complex)
            out[rows_r] = Ur @ (Kji @ (Uc.conj().T @ V[rows_c]))
            return out

        L = _shifted(D[side, "H"], D[side, "S"], z, xp)
        w, vs = self._beyn(L, xp)
        a_ii_op = type("_AiiOp", (), {"__matmul__": lambda _, V: _aii_mv(L, self.s, V)})()
        wx, vs, refl, inj, dEk = self._select_mv(w, vs, a_ji_mv, a_ii_op, a_ij_mv, xp, real)
        mark("beyn")
        if self.lu_prec == "mixed":                    # complex64 LU; the complex128 blocks stay for the residual
            sol = _AiiSolver(L, self.s, xp, "mixed", self.ir_tol, self.ir_maxit)
        else:                                          # complex128 LU only, the blocks are not needed any more
            sol = _AiiSolver(L, self.s, xp, "double")
            sol.L = L = None
        mark("lu")
        solve = sol
        vr, wr = vs[:, refl], wx[refl]
        vinv = xp.linalg.solve(vr.conj().T @ vr, vr.conj().T)                 # Phi^+ (k x n)
        Um = a_ji_mv(vr / wr[None, :])                                        # a_ji X = Um vinv
        G0U = solve(Um)
        VG0 = solve(xp.ascontiguousarray(vinv.T), trans=1).T                  # vinv G0
        k = Um.shape[1]
        ps = self.p[side]
        Ald_q = sp.csc_matrix(sp.csr_matrix(Ald)[ps].astype(complex))
        Adl_q = sp.csr_matrix(Adl)[:, ps].astype(complex).tocsr()
        if gpu_on:
            import cupyx.scipy.sparse as csp
            Adl_d = csp.csr_matrix(Adl_q)
        else:
            Adl_d = Adl_q
        nd, nf, cb = Adl_q.shape[0], Ald_q.shape[1], self.chunk
        # lean (fp_store "host" on the GPU): the rr x rr accumulators and operands live on the host, filled chunk by chunk
        lean = gpu_on and self.fp_store == "host"
        acc = (lambda r_, c_: np.empty((r_, c_), complex)) if lean else (lambda r_, c_: xp.empty((r_, c_), complex))
        put = to if lean else (lambda v: v)
        # P_c columns: A = P_c^H G0 P_c, Dm = P_r^H G0 P_c
        A = acc(rc, rc)
        Dm = acc(rr, rc)
        for c0 in range(0, rc, cb):
            c1 = min(rc, c0 + cb)
            Y = xp.zeros((n, c1 - c0), complex)
            Y[rows_c] = Uc[:, c0:c1]
            Y = solve(Y)
            A[:, c0:c1] = put(Uc.conj().T @ Y[rows_c])
            Dm[:, c0:c1] = put(Ur.conj().T @ Y[rows_r])
            mark("chunks")
            del Y
        Ad = xp.asarray(A)
        S0 = Ad - (Uc.conj().T @ G0U[rows_c]) @ xp.linalg.solve(xp.eye(k, dtype=complex) + VG0 @ Um, VG0[:, rows_c] @ Uc)
        del G0U, VG0, Um, vinv
        M = put(Kji @ (S0 @ Kij))
        del S0
        Ah, Dh = acc(rr, rr), acc(rr, rr)
        for c0 in range(0, rr, cb):                       # column chunks: no rr x rr temporaries
            c1 = min(rr, c0 + cb)
            Ah[:, c0:c1] = put(Kji @ (Ad @ Kij[:, c0:c1]))
        del Ad, A
        Dmd = xp.asarray(Dm)
        for c0 in range(0, rr, cb):
            c1 = min(rr, c0 + cb)
            Dh[:, c0:c1] = put(Dmd @ Kij[:, c0:c1])
        del Dmd, Dm
        mark("chunks")
        # P_r columns: Bh = Kji P_c^H G0 P_r, C = P_r^H G0 P_r, F = A_dl G0 P_r (host)
        Bh = acc(rr, rr)
        C = acc(rr, rr)
        F = np.empty((nd, rr), complex)
        for c0 in range(0, rr, cb):
            c1 = min(rr, c0 + cb)
            Y = xp.zeros((n, c1 - c0), complex)
            Y[rows_r] = Ur[:, c0:c1]
            Y = solve(Y)
            Bh[:, c0:c1] = put(Kji @ (Uc.conj().T @ Y[rows_c]))
            C[:, c0:c1] = put(Ur.conj().T @ Y[rows_r])
            F[:, c0:c1] = to(Adl_d @ Y)
            mark("chunks")
            del Y
        # A_ld columns: Sigma0 = A_dl G0 A_ld, E2 = P_r^H G0 A_ld (host)
        Sig = np.empty((nd, nf), complex)
        E2 = np.empty((rr, nf), complex)
        for c0 in range(0, nf, cb):
            c1 = min(nf, c0 + cb)
            Y = solve(xp.asarray(Ald_q[:, c0:c1].toarray()))
            Sig[:, c0:c1] = to(Adl_d @ Y)
            E2[:, c0:c1] = to(Ur.conj().T @ Y[rows_r])
            mark("chunks")
            del Y
        if gpu_on and self.offload_lu:
            sol.park()
            del L
            gpu.free()
        if lean:                                          # bases and cores to the host during the fixed point
            park = {"Uc": to(Uc), "Ur": to(Ur), "Kij": to(Kij), "Kji": to(Kji)}
            Uc = Ur = Kij = Kji = None
            M = xp.asarray(M)
            gpu.free()
        idx = xp.arange(rr)
        # fixed-point operands: fp_store "device" keeps the four rr x rr operands on the GPU; "host" parks the
        # complex128 operands on the host and streams them in per step (GH200: C2C link); fp_prec "mixed" iterates with
        # complex64 copies on the device to fp_tol_c64 (or stagnation), then refines in complex128 (operands from the host)
        mixed = self.fp_prec == "mixed"
        host = gpu_on and (self.fp_store == "host" or mixed)
        ops = {"Ah": Ah, "Bh": Bh, "C": C, "Dh": Dh}
        del Ah, Bh, C, Dh
        ops64 = {k: xp.asarray(v).astype(np.complex64) for k, v in ops.items()} if mixed else None
        if host:
            ops = {k: (v if isinstance(v, np.ndarray) else to(v)) for k, v in ops.items()}
            gpu.free()
        mark("fp_setup")

        def op(k, dt):
            if dt == np.complex64:
                return ops64[k]
            return xp.asarray(ops[k]) if host else ops[k]

        def W_of(M, dt):
            X = op("C", dt) @ M
            X *= -1
            X[idx, idx] += 1
            W = xp.linalg.solve(X.T, M.T).T                                   # M (I - C M)^-1
            del X
            return W

        def step(M, dt):
            W = W_of(M, dt)
            t = W @ op("Dh", dt)
            del W
            Mn = op("Bh", dt) @ t
            del t
            Mn += op("Ah", dt)
            return Mn

        dt = np.complex64 if mixed else complex
        M = M.astype(dt)
        hist, ph, n64 = [], [], 0
        for it in range(self.fp_maxit):
            Mn = step(M, dt)
            d = float(xp.abs(Mn - M).max() / xp.abs(Mn).max())
            M = Mn
            hist.append(d)
            ph.append(d)
            mark("fp")
            done = d <= (self.fp_tol_c64 if dt == np.complex64 else self.fp_tol)
            # stagnation: on the real axis the propagating directions are neutral, the change settles at the error of
            # the start (2808a: 300 steps -> 5e-9, Sigma already 1e-9 vs dense); stop when 3 steps gave < 2x
            done = done or (self.fp_stall and len(ph) > 4 and ph[-1] > 0.5 * ph[-4])
            # plateau already reached (12 nm: steps 5-7, Sigma flat from step 3-4): a small change that one step improved by < 10x
            done = done or (self.fp_stall and len(ph) > 1 and ph[-1] < 1e-8 and ph[-1] > 0.1 * ph[-2])
            if done and dt == np.complex64:                                 # switch to complex128 refinement
                n64, dt, ph = len(hist), complex, []
                M = M.astype(complex)
                del ops64
                ops64 = None
                continue
            if done:
                break
        W = W_of(M, complex)
        del M, ops
        if gpu_on:
            gpu.free()
        if lean:
            Uc, Ur, Kij, Kji = (xp.asarray(park[k]) for k in ("Uc", "Ur", "Kij", "Kji"))
            del park
        Fd = xp.asarray(F)
        Sig += to(Fd @ (W @ xp.asarray(E2)))
        mark("sigma")
        del Fd, F, E2
        if real and int(inj.sum()):
            sol.unpark()

            def gmv(Y):                                                       # g Y = G0 (Y + P_r W P_r^H G0 Y)
                G = solve(Y)
                T = Y.copy()
                T[rows_r] += Ur @ (W @ (Ur.conj().T @ G[rows_r]))
                return solve(T)

            vi = vs[:, inj] / xp.sqrt(dEk[inj].real)[None, :]
            b = vi / wx[inj][None, :] + gmv(a_ij_mv(vi))                       # x_ii_a_ij = -g a_ij
            Bv = to(Adl_d @ gmv(a_ji_mv(b)))
        else:
            Bv = np.zeros((nd, 0), complex)
        mark()
        self.info.update(ir_its=(max(sol.its) if sol.its else 0), ir_res=sol.res, ir_fail=sol.fail,
                         ir_mean=(float(np.mean(sol.its)) if sol.its else 0.0))
        del sol
        self.info.update(fp_it=len(hist), fp_c64=n64, fp_hist=hist, fp_last=hist[-1] if hist else 0.0, rc=rc, rr=rr,
                         basis_err=bs["err"], basis_t=bs["t"], support=bs["support"], peak_gb=peak[0] / 2**30,
                         phase_gb={k: v / 2**30 for k, v in phase_peak.items()},
                         phase_t=phase_t)
        return Sig, Bv

    def _select_mv(self, w, vs, a_ji_mv, a_ii, a_ij_mv, xp, real):
        """_select with the couplings as products (Woodbury path)."""
        class _Op:
            def __init__(self, f):
                self.f = f
            def __matmul__(self, V):
                return self.f(V)
        return self._select(w, vs, (_Op(a_ji_mv), a_ii, _Op(a_ij_mv)), xp, real)

    # ---- 3. + 4. -------------------------------------------------------------------------------------------------
    def sigma(self, z, side, Adl, Ald):
        """(Sigma, B) on the host for one side at z (method woodbury or dense, see __init__)."""
        if self.method == "woodbury":
            t0 = time.time()
            z = complex(z)
            xp = self._xp()
            real = abs(z.imag) <= self.real_eta
            out = self._sigma_woodbury(z, side, Adl, Ald, xp, self._arrays(), real)
            self.info.update(t=time.time() - t0, real=real, res=float("nan"))
            return out
        return self._sigma_dense(z, side, Adl, Ald)

    def _sigma_dense(self, z, side, Adl, Ald):
        """(Sigma, B) on the host for one side at z. Adl: device face x lead cell, Ald: lead cell x device face
        (scipy sparse, lead-matrix order). B has one column per incoming channel (none off the real axis)."""
        t0 = time.time()
        z = complex(z)
        xp = self._xp()
        D = self._arrays()
        real = abs(z.imag) <= self.real_eta
        L = _shifted(D[side, "H"], D[side, "S"], z, xp)
        a_xx = tuple(_blocks_of(L, self.s, xp, q) for q in ("ji", "ii", "ij"))
        w, vs = self._beyn(L, xp)
        del L
        wx, vs, refl, inj, dEk = self._select(w, vs, a_xx, xp, real)
        a_ji, a_ii, a_ij = a_xx
        vr, wr = vs[:, refl], wx[refl]
        vinv = xp.linalg.solve(vr.conj().T @ vr, vr.conj().T)
        g = xp.linalg.inv(a_ii + a_ji @ ((vr / wr[None, :]) @ vinv))
        for _ in range(self.n_ref - 1):
            g = xp.linalg.inv(a_ii - a_ji @ g @ a_ij)
        g = xp.linalg.inv(a_ii - a_ji @ g @ a_ij)                          # qttools x_ii_ref
        if real and int(inj.sum()):
            vi = vs[:, inj] / xp.sqrt(dEk[inj].real)[None, :]
            b = vi / wx[inj][None, :] + g @ (a_ij @ vi)                        # x_ii_a_ij = -g a_ij
            gu = g @ (a_ji @ b)
        else:
            gu = xp.zeros((self.n, 0), complex)
        ps = self.p[side]
        Ald_q, Adl_q = sp.csr_matrix(Ald)[ps].astype(complex), sp.csr_matrix(Adl)[:, ps].astype(complex)
        if xp is np:
            S = np.asarray(Adl_q @ (Ald_q.T @ g.T).T)
            B = np.asarray(Adl_q @ gu)
        else:
            import cupyx.scipy.sparse as csp
            Ald_d, Adl_d = csp.csr_matrix(Ald_q), csp.csr_matrix(Adl_q)
            S = (Adl_d @ (Ald_d.T @ g.T).T).get()
            B = (Adl_d @ gu).get()
        res = float(xp.abs(xp.eye(self.n) - (a_ii - a_ji @ g @ a_ij) @ g).max())
        self.info.update(res=res, t=time.time() - t0, real=real)
        return S, B
