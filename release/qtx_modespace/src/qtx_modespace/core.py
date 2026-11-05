"""Device, lead self-energies, sparse / block-tridiagonal solvers and the contour slab density matrix.

Lead OBC on the real axis (|Im z| <= Device.qtt_eta; the transmission, FGMRES and exact solves at real E): qttools'
spectral OBC (QuaTrEx) by default, Sigma at E = Re z and the flux-normalized incoming Bloch modes as injection
(one column per propagating channel, as QuaTrEx's QTBM). qttools is found on sys.path or in $QTTOOLS_SRC (a
QuaTrEx src folder; needs mpi4py, numba). When it does not import (or fails at an energy) the folded lead below is
used at E + i fallback_eta with the Gamma eigenchannels as injection. Complex z (contours): always the folded lead.

Device: the device blocks of a read_quatrex dict are the slabs; the lead blocks enter through Sigma_L, Sigma_R.
Lead Sigma(z): the lead-cell interior (orbitals not coupled to the neighbouring cells) is folded onto the coupled
faces F, A_eff(z) = A_FF - A_FI A_II^-1 A_IF with A = z S - H, then Lopez-Sancho decimation of the folded chain
(complex Hermitian H, any S).
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


def log(*a):
    print(*a, flush=True)


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


def sancho_both(A00, A01, A10, tol=1e-11, maxit=300):
    """Lopez-Sancho decimation with on-site A00 = z S00 - H00, couplings A01 (cell n -> n+1), A10 (n+1 -> n).
    Returns the surface Green's functions gL (last cell of the left-extending chain) and gR (first cell of the
    right-extending chain)."""
    e = A00.astype(complex)
    esR, esL = e.copy(), e.copy()
    a, b = A01.astype(complex), A10.astype(complex)
    n = len(e)
    for _ in range(maxit):
        X = sla.solve(e, np.hstack([a, b]), check_finite=False)
        ga, gb = X[:, :n], X[:, n:]
        agb, bga = a @ gb, b @ ga
        esR -= agb
        esL -= bga
        e -= agb + bga
        a, b = -a @ ga, -b @ gb
        if max(np.abs(a).max(), np.abs(b).max()) < tol:
            break
    else:
        log(f"  sancho_both: no convergence after {maxit} steps")
    return sla.inv(esL, check_finite=False), sla.inv(esR, check_finite=False)


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

    def surface_gfs(self, z):
        Aeff = z * self.SFF - self.HFF
        if len(self.I):
            cs = CSolver(z * self.SII - self.HII)
            X = cs.solve(z * self.SIF - self.HIF)
            cs.free()
            Aeff = Aeff - (z * self.SFI - self.HFI) @ X
        return sancho_both(Aeff, z * self.S01 - self.H01, z * self.S01.conj().T - self.H01.conj().T)


_QTT = None


def qttools_obc():
    """(Spectral, Full) from qttools (QuaTrEx), or None when it does not import or runs on cupy."""
    global _QTT
    if _QTT is None:
        _QTT = False
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
            except ImportError:
                continue
            _QTT = (Spectral, Full) if xp is np else False
            break
    return _QTT or None


class QttLead:
    """Lead cell (H00, S00, coupling H01, S01 to the next cell along +x) through qttools' spectral OBC at real E.
    sections = primitive cells per lead cell (the NEVP runs on one primitive cell). Per side: g = surface Green's
    function of the semi-infinite lead (cell next to the device included), u = lead-cell amplitudes of the
    flux-normalized incoming modes (device source A_dl u, B B^H = Gamma on the device face). perm: lead-cell orbital
    order with the primitive cells contiguous along +x and in the same internal order; qttools reads the periodic
    layer from the outermost primitive cell, cells numbered inwards (+x for the left lead, reversed for the right)."""

    def __init__(self, H00, H01, S00, S01, sections=1, perm=None, **opts):
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
        kw.update(opts)
        self.sections = sections
        self.solver = Spectral(Full(), block_sections=sections, **kw)

    def modes(self, E, side):
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
            g, b = self.solver(a_xx, "", return_injected=True)
        bad = [str(x.message) for x in w if "recursion error" in str(x.message)]
        if bad:
            raise RuntimeError(f"qttools OBC {side} at E = {E}: {bad[0]}")
        g = g[0]
        u = g @ (a_ji @ b[0])
        ip = np.argsort(self.p[side])
        return g[np.ix_(ip, ip)], u[ip]


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
        assert obc in ("auto", "qttools", "folded"), obc
        self.obc, self.qtt, self._inj = obc, None, {}
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
            raise RuntimeError("obc 'qttools': qttools does not import (install it or set QTTOOLS_SRC)")
        log(f"  lead OBC on the real axis: {'qttools spectral OBC' if self.qtt is not None else 'folded lead (decimation) at E + %g i' % self.fallback_eta}")

    def uses_qttools(self, z):
        return self.qtt is not None and abs(complex(z).imag) <= self.qtt_eta

    def sigma(self, z):
        """Sigma_L on bL, Sigma_R on bR at z. Real axis (|Im z| <= qtt_eta): qttools at E = Re z (the injected modes
        are kept for inject(); cached as qsig_<E>_<Im z>.npz, any such file at the same E is reused), or the folded
        lead at E + i fallback_eta without qttools. Complex z: folded lead (sig_<Re z>_<Im z>.npz)."""
        z = complex(z)
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

    def inject(self, Sg, tol=1e-4):
        """Injection columns for the Sigma_L Sg of the last sigma(): the qttools incoming modes when they came with
        it, else the Gamma_L eigenchannels above tol (injection())."""
        B = self._inj.get(id(Sg))
        return injection(Sg, tol) if B is None else B

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
            np.savez(os.path.join(d, f"qsig_{E:+.8f}_{0.0:.8e}.npz"), SL=out[0], SR=out[1], BL=inj[0], BR=inj[1])
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
        out = tuple((z * Sdl - Hdl) @ g @ (z * Sld - Hld) for g, (Hld, Sld, Hdl, Sdl) in ((gL, self.cL), (gR, self.cR)))
        if f:
            os.makedirs(self.sigma_cache, exist_ok=True)
            np.savez(f, SL=out[0], SR=out[1])
        return out

    def A(self, E, SL, SR):
        """Sparse E S - H - Sigma_L - Sigma_R."""
        rows, cols, vals = [], [], []
        for Sg, b in ((SL, self.bL), (SR, self.bR)):
            ii, jj = np.meshgrid(b, b, indexing="ij")
            rows.append(ii.ravel()); cols.append(jj.ravel()); vals.append(Sg.ravel())
        Sig = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n, self.n))
        return (E * self.S - self.H - Sig).tocsr()


def injection(SL, tol=1e-4):
    """Columns B with Gamma_L ~ B B^H (eigenvectors of Gamma_L above tol [eV]: propagating channels)."""
    G = 1j * (SL - SL.conj().T)
    lam, X = np.linalg.eigh(0.5 * (G + G.conj().T))
    k = lam > tol
    return X[:, k] * np.sqrt(lam[k])


def slab_blocks(dev, V=None):
    """Blocks HD, SD (s, s), HU, SU (s, s+1) of U^H H U, U^H S U for U = diag(V_s) (V None: U = I, dense blocks)."""
    b, n = dev.sb, len(dev.ns)
    out = dict(HD=[], SD=[], HU=[], SU=[])
    for M, kD, kU in ((dev.H, "HD", "HU"), (dev.S, "SD", "SU")):
        for s in range(n):
            X = M[b[s]:b[s + 1], b[s]:b[s + 1]]
            X = X.toarray() if V is None else V[s].conj().T @ (X @ V[s])
            out[kD].append(0.5 * (X + X.conj().T))
            if s < n - 1:
                X = M[b[s]:b[s + 1], b[s + 1]:b[s + 2]]
                out[kU].append(X.toarray() if V is None else V[s].conj().T @ (X @ V[s + 1]))
    return out


class BlockTridiag:
    """LU of the block-tridiagonal A(z) = z S_c - H_c, minus the projected Sigma on the end blocks."""

    def __init__(self, blk, z, SL1, SR1):
        n = len(blk["HD"])
        D = [z * blk["SD"][s] - blk["HD"][s] for s in range(n)]
        D[0] = D[0] - SL1
        D[-1] = D[-1] - SR1
        self.Up = [z * blk["SU"][s] - blk["HU"][s] for s in range(n - 1)]
        Lo = [z * blk["SU"][s].conj().T - blk["HU"][s].conj().T for s in range(n - 1)]
        self.lu, self.L = [], []
        Dt = D[0]
        for s in range(n):
            self.lu.append(sla.lu_factor(Dt, check_finite=False))
            if s < n - 1:
                Lt = sla.lu_solve(self.lu[s], Lo[s].T, trans=1, check_finite=False).T
                self.L.append(Lt)
                Dt = D[s + 1] - Lt @ self.Up[s]
        self.bounds = np.concatenate([[0], np.cumsum([len(d) for d in D])])

    def solve(self, B):
        b, n = self.bounds, len(self.lu)
        Y = [B[b[s]:b[s + 1]].astype(complex) for s in range(n)]
        for s in range(1, n):
            Y[s] = Y[s] - self.L[s - 1] @ Y[s - 1]
        X = [None] * n
        X[-1] = sla.lu_solve(self.lu[-1], Y[-1], check_finite=False)
        for s in range(n - 2, -1, -1):
            X[s] = sla.lu_solve(self.lu[s], Y[s] - self.Up[s] @ X[s + 1], check_finite=False)
        return np.concatenate(X)


class Coarse:
    """Mode-space (Galerkin) system of the per-slab basis V = [V_s] (slab-local orbitals x modes)."""

    def __init__(self, dev, V):
        self.dev, self.V = dev, V
        self.blk = slab_blocks(dev, V)
        self.PL, self.PR = V[0][dev.bL], V[-1][dev.bR - dev.sb[-2]]
        self.off = np.concatenate([[0], np.cumsum([v.shape[1] for v in V])])
        self.N = int(self.off[-1])

    def Ac(self, E, SL, SR):
        return BlockTridiag(self.blk, E, self.PL.conj().T @ SL @ self.PL, self.PR.conj().T @ SR @ self.PR)

    def prolong(self, Y):
        return np.vstack([self.V[s] @ Y[self.off[s]:self.off[s + 1]] for s in range(len(self.V))])

    def restrict(self, X):
        b = self.dev.sb
        return np.vstack([self.V[s].conj().T @ X[b[s]:b[s + 1]] for s in range(len(self.V))])


def is_real(blk):
    """True if all slab blocks of H and S are real: A(z) is then complex symmetric (see g_diag)."""
    return all(not np.iscomplexobj(M) or not np.abs(M.imag).any() for k in ("HD", "SD", "HU", "SU") for M in blk[k])


def g_diag(blk, z, SL, SR, iL, iR, sym=False):
    """Diagonal blocks G_ss of G = A(z)^-1, A = z S - H - Sigma_L - Sigma_R (Sigma_L on rows / columns iL of the first
    slab, Sigma_R on iR of the last), by left-connected block recursion:
        forward   g_s = (D_s - Y_{s-1} U_{s-1})^-1,  Y_s = L_s g_s
        backward  G_ss = g_s + X_s G_{s+1,s+1} Y_s,  X_s = g_s U_s
    (D_s, U_s, L_s = diagonal / upper / lower blocks of A). sym (real H and S: L_s = U_s^T, g_s = g_s^T) uses
    X_s = Y_s^T: 4 block products per slab instead of 5 (6 in the 0.1.0 recursion)."""
    ns = len(blk["HD"])
    Up = lambda s: z * blk["SU"][s] - blk["HU"][s]
    Lo = lambda s: z * blk["SU"][s].conj().T - blk["HU"][s].conj().T
    gl, Y = [None] * ns, [None] * (ns - 1)
    for s in range(ns):
        D = z * blk["SD"][s] - blk["HD"][s]
        if s == 0:
            D[np.ix_(iL, iL)] -= SL
        if s == ns - 1:
            D[np.ix_(iR, iR)] -= SR
        if s > 0:
            D -= Y[s - 1] @ Up(s - 1)
        gl[s] = np.linalg.inv(D)
        del D
        if s < ns - 1:
            Y[s] = Lo(s) @ gl[s]
    out = [None] * ns
    out[-1] = G = gl[-1]
    for s in range(ns - 2, -1, -1):
        X = Y[s].T if sym else gl[s] @ Up(s)
        out[s] = G = gl[s] + X @ (G @ Y[s])
        gl[s] = Y[s] = None
    return out


def rho_contour(dev, mu1, mu2, npts, log=log):
    """Slab density matrices of the open device, rho_ss = -(1/pi) Im int G_ss dz over the upper half circle from mu1
    to mu2 (Gauss-Legendre in the angle), G_ss from g_diag (block recursion; real H, S use the symmetric variant)."""
    blk = slab_blocks(dev)
    sym = is_real(blk)
    if sym:
        for k in ("HD", "SD", "HU", "SU"):
            blk[k] = [np.ascontiguousarray(np.real(M)) for M in blk[k]]
    iL, iR = dev.bL, dev.bR - dev.sb[-2]                     # lead faces in the end slabs
    c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    x, w = np.polynomial.legendre.leggauss(npts)
    th, w = 0.5 * np.pi * (x + 1), 0.5 * np.pi * w
    acc = [np.zeros(h.shape, complex) for h in blk["HD"]]
    t0 = time.time()
    for ip, (t_, w_) in enumerate(zip(th, w)):
        z = c + r * np.exp(1j * t_)
        dz = -1j * r * np.exp(1j * t_) * w_
        t1 = time.time()
        SL, SR = dev.sigma(z)
        t2 = time.time()
        for a, G in zip(acc, g_diag(blk, z, SL, SR, iL, iR, sym)):
            a += G * dz
        log(f"    contour point {ip + 1}/{npts}, {time.time() - t0:.0f} s (sigma {t2 - t1:.0f} s, "
            f"recursion {time.time() - t2:.0f} s)")
    return [-((a - a.conj().T) / 2j) / np.pi for a in acc]


def top_modes(rho, n):
    """The n eigenvectors of the Hermitian rho with the largest eigenvalues, largest first."""
    R = 0.5 * (rho + rho.conj().T)
    m = len(R)
    lam, X = sla.eigh(R, subset_by_index=[m - min(n, m), m - 1], driver="evr")
    return lam[::-1], X[:, ::-1]
