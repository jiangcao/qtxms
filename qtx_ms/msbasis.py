"""Two-level mode-space basis (the qtmode method) built from the generic input set (qtx_ms.prep.inputs).

Level 1 (orbitals -> block modes), per transport slab:
  cores : crystalline atoms, found from device.xyz: label core_label and at least d_core from any atom
          with another label. Basis = bulk Bloch states in an energy window, restricted to the block
          (real Gram matrix, Re B B^H). The bulk H(k) is read off the device H at the deepest core atom
          (diamond lattice from its neighbour vectors), so no TB model is called.
          Only for empirical TB of a crystalline channel; ab initio inputs use cores=False.
  tiles : all other atoms (every atom with cores=False), cut into tiles of ~tile_orbs orbitals by
          recursive bisection in the transverse plane. Basis = dominant eigenvectors of the local
          spectral function A_PP(E) = -Im G_PP / pi, G = (E S - H_emb + iW)^-1 in a d_emb embedding
          with a CAP W on its outer d_cap, at a few energies (randomized range finder); the sets of all
          energies are merged by one SVD, keep s^2 > tile_thr * a_ref.
  U1 = diag(blocks) per slab (orthonormal), H1 = U1^T H U1, S1 = U1^T S U1 in slab blocks.
  enrichment (optional, enrich_round): residuals of window Ritz pairs of local clusters of H1 (slabs s +- w,
          or a tile and the tiles coupled to it) appended to the blocks (qtmode.enrich with generalized S,
          complex H and local instead of global Ritz problems).
Level 2 (block modes -> slab modes): contour-integrated slab density matrix of the open level-1 system
  rho_ss = -(1/pi) Im int_arc G1_ss(z) dz,  G1 = (z S1 - H1 - Sigma1(z))^-1,  arc over [mu1, mu2],
  eigenvectors with lambda > thr * lambda_max.  U = U1 U2 (block diagonal over slabs).
Leads: exact Sigma(z) at complex z, lead interior folded onto its faces, then decimation of the folded chain
(general: complex Hermitian H, any S; fast path for real H, S = I: qtmode face decimation on the smaller face).
On the real axis (|Im z| <= Device.qtt_eta) Sigma and the injected modes come from qttools' spectral OBC (QuaTrEx)
when it imports (on sys.path or in $QTTOOLS_SRC, a QuaTrEx src folder; needs the `qtx` extra: mpi4py, numba):
flux-normalized incoming Bloch modes, one column per propagating channel, like QuaTrEx's QTBM.
"""
import os, sys, time, warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
from scipy.spatial import cKDTree
try:
    import pypardiso
except ImportError:          # pragma: no cover
    pypardiso = None

from qtx_ms import gpu


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ exact lead surface GFs (from qtmode.lead_ms)
def _sancho_both(h00, h01, z, tol=1e-11, maxit=300):
    """Lopez-Sancho returning both bulk-side surface on-site matrices:
    esR (cells n+1.. folded onto n) and esL (cells ..n-1 folded onto n), chain n -> n+1 via h01.
    Complex symmetric chain: n+1 -> n coupling is h01^T."""
    n = len(h00)
    I = np.eye(n)
    e = h00.astype(complex)
    esR, esL = e.copy(), e.copy()
    a, b = h01.astype(complex), h01.T.astype(complex)
    for it in range(maxit):
        X = sla.solve(z * I - e, np.hstack([a, b]), check_finite=False)
        ga, gb = X[:, :n], X[:, n:]
        agb, bga = a @ gb, b @ ga
        esR += agb
        esL += bga
        e += agb + bga
        a, b = a @ ga, b @ gb
        if max(np.abs(a).max(), np.abs(b).max()) < tol:
            break
    return esR, esL, it + 1


def face_surface_gfs_small(Heff, iL, iR, V, z):
    """Exact surface GFs using only the chain on the smaller face Lf (eliminate every Rf):
      right lead, first cell, on Lf : (z - esR + V^T gRR V)^-1          (no R_0 on its left)
      left lead, last cell, on Rf   : R_0 couples only to L_0; L_0 without R_0 has on-site
                                      esL - HLR gRR HRL, then one |Rf| solve."""
    HLL, HLR = Heff[np.ix_(iL, iL)], Heff[np.ix_(iL, iR)]
    HRR, HRL = Heff[np.ix_(iR, iR)], Heff[np.ix_(iR, iL)]
    IR, IL = np.eye(len(iR)), np.eye(len(iL))
    gRR = sla.solve(z * IR - HRR, np.hstack([HRL, V]), check_finite=False)
    t_in, t_v = HLR @ gRR[:, :len(iL)], V.T @ gRR[:, len(iL):]
    h00 = HLL + t_in + t_v
    h01 = HLR @ gRR[:, len(iL):]                                   # L_n -> L_n+1
    esR, esL, nit = _sancho_both(h00, h01, z)
    gR = sla.inv(z * IL - (esR - t_v), check_finite=False)
    GL0 = sla.inv(z * IL - (esL - t_in), check_finite=False)
    gL = sla.inv(z * IR - HRR - HRL @ GL0 @ HLR, check_finite=False)
    return gL, gR, nit


# ------------------------------------------------------------------ sparse complex solves
class CSolver:
    """Factorization of a complex sparse matrix A. PARDISO on the real-equivalent
    [[Ar, -Ai], [Ai, Ar]] when pypardiso is available, else SuperLU. Each solve does n_ref steps of
    iterative refinement, then GMRES preconditioned by the factorization on the columns whose residual
    is still above tol (PARDISO perturbs small pivots; plain refinement can then stall or diverge)."""

    def __init__(self, A, n_ref=2, tol=1e-12, maxit=60):
        A = sp.csr_matrix(A)
        self.A, self.n, self.n_ref, self.tol, self.maxit = A, A.shape[0], n_ref, tol, maxit
        self.res, self.its = 0.0, 0                       # max relative residual, max GMRES its (last solve)
        if pypardiso is None:
            self.lu = spla.splu(A.astype(complex).tocsc())
            self.npert = 0
            return
        Ar, Ai = sp.csr_matrix(A.real), sp.csr_matrix(A.imag)
        self.M = sp.bmat([[Ar, -Ai], [Ai, Ar]], format="csr")
        self.ps = pypardiso.PyPardisoSolver(mtype=11)
        self.ps.factorize(self.M)
        self.npert = int(self.ps.get_iparm(14))           # number of perturbed pivots

    def factor_solve(self, B):
        if pypardiso is None:
            return self.lu.solve(B.astype(complex))
        Z = self.ps.solve(self.M, np.vstack([B.real, B.imag]))
        Z = Z.reshape(2 * self.n, -1)
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
        self.its = 0
        for c in np.where(res > self.tol)[0]:
            X[:, c], its = gmres_right(lambda v: self.A @ v, lambda v: self.factor_solve(v[:, None])[:, 0],
                                       B[:, c], X[:, c], self.tol, self.maxit)
            res[c] = np.linalg.norm(B[:, c] - self.A @ X[:, c]) / bn[c]
            self.its = max(self.its, its)
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
        return x0, 0
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
    return x0 + Z[:, :j + 1] @ y, j + 1


# ------------------------------------------------------------------ device and leads
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
    sections = primitive cells per lead cell (the NEVP runs on one primitive cell). Per side:
      g  surface Green's function of the semi-infinite lead (cell next to the device included),
      u  lead-cell amplitudes of the flux-normalized incoming modes: the device source is A_dl u
         (QuaTrEx's contact source -A_ji b, with the contact cell eliminated), so B B^H = Gamma on the device face.
    perm: lead-cell orbital order with the primitive cells contiguous along +x and in the same internal order. qttools
    reads the periodic layer from the first (outermost) primitive row, with the cells numbered towards a_ij (inwards):
    +x for the left lead, reversed for the right lead. g and u are returned in the original order."""

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
        self.toeplitz_dev = []                       # Toeplitz in the primitive cells? (the NEVP uses cell 0's row)
        for nm, M in (("H", self.H00), ("S", self.S00)):
            d = max([abs(blk(M, i, i) - blk(M, 0, 0)).max() for i in range(sections)] +
                    [abs(blk(M, i, i + 1) - blk(M, 0, 1)).max() for i in range(sections - 1)])
            self.toeplitz_dev.append(float(d))
            if d > 1e-2 * max(abs(M).max(), 1):
                raise RuntimeError(f"lead cell is not {sections} identical primitive cells (max |d{nm}| {d:.2e})")
            if d > 1e-8:
                log(f"  qttools lead: primitive cells of the lead cell differ by up to {d:.1e} in {nm}; "
                    f"the OBC uses the first cell's couplings (as QuaTrEx)")
        kw = dict(num_ref_iterations=4)
        kw.update(opts)
        self.sections = sections
        self.solver = Spectral(Full(), block_sections=sections, **kw)

    def modes(self, E, side):
        a00 = E * self.S00 - self.H00                     # in perm order (cells along +x)
        a01 = E * self.S01 - self.H01                     # cell n -> n+1
        a10 = E * self.S01.conj().T - self.H01.conj().T  # cell n+1 -> n
        a_ji, a_ij = (a10, a01) if side == "L" else (a01, a10)   # a_ji: towards the outer cell
        r = np.argsort(self.p["L"])[self.p[side]]         # perm order -> this side's cell order
        a_xx = tuple(M[np.ix_(r, r)][None] for M in (a_ji, a00, a_ij))
        if self.sections > 1:
            # exactly periodic lead from the outermost primitive cell (as QuaTrEx's contact): refining g with the
            # slightly non-periodic cell blocks breaks the recursion (MoS2/HfO2: 8e-3 eV between cells, error 0.72)
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
        ip = np.argsort(self.p[side])                      # back to the lead-matrix order
        return g[np.ix_(ip, ip)], u[ip]


def lead_cell_perm(inp, orbs, sections):
    """Orbital order of a lead cell (global orbitals orbs, in lead-matrix order) with its `sections` primitive cells
    contiguous along the transport axis and every cell's atoms in the order of the first cell (matched by
    translation by the primitive period)."""
    if sections <= 1:
        return None
    norb, ax = inp["norb"], inp["meta"]["axis"]
    off = np.concatenate([[0], np.cumsum(norb)])
    atom_of = np.repeat(np.arange(len(norb)), norb)
    atoms = np.unique(atom_of[orbs])
    x = inp["xyz"][atoms]
    a = inp["meta"].get("lead_cell_length", np.ptp(x[:, ax]) * sections / (sections - 1)) / sections
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
    pos = {o: i for i, o in enumerate(orbs)}                      # global orbital -> lead-matrix index
    return np.array([pos[o] for k in order for o in range(off[atoms[k]], off[atoms[k] + 1])])


class Device:
    """Device part of a generic input set: slabs = device blocks, lead coupling, exact Sigma(z).
    obc: "auto" (qttools on the real axis, else the folded-lead decimation), "qttools" (fail instead of falling back
    on the real axis) or "folded" (folded lead only; the old name "mgms" is accepted)."""

    qtt_eta = 1e-5                  # |Im z| up to which Sigma(z) is taken from qttools at E = Re z

    def __init__(self, inp, fast_lead=None, obc="auto"):
        self.inp, m = inp, inp["meta"]
        self.meta = m
        norb, blocks = inp["norb"], inp["blocks"]
        self.off = np.concatenate([[0], np.cumsum(norb)])
        ob = np.repeat(blocks, norb)
        O = {b: np.where(ob == b)[0] for b in np.unique(blocks)}
        tr = m["transport"]
        self.dev_blocks = tr["device_blocks"]
        self.slab_orbs = [O[b] for b in self.dev_blocks]              # global orbitals of each slab
        self.ns = [len(o) for o in self.slab_orbs]
        self.sb = np.concatenate([[0], np.cumsum(self.ns)])           # slab bounds in device order
        dev = np.concatenate(self.slab_orbs)
        self.dev = dev
        self.n = len(dev)
        H, S = inp["H"], inp["S"]
        self.H = H[dev][:, dev].tocsr()
        self.S = S[dev][:, dev].tocsr()
        self.loc = -np.ones(H.shape[0], int)                         # global orbital -> slab-local
        for o in self.slab_orbs:
            self.loc[o] = np.arange(len(o))
        r, c = (abs(self.H) + abs(self.S)).nonzero()
        so = np.repeat(np.arange(len(self.ns)), self.ns)
        assert np.abs(so[r] - so[c]).max() <= 1, "device blocks are not block tridiagonal"
        self.lead = FoldedLead(inp["lead"]["H00"], inp["lead"]["H01"], inp["lead"]["S00"], inp["lead"]["S01"],
                               fast=fast_lead)
        lr = inp.get("lead_right")                                  # different right lead (e.g. under bias)
        self.leadR = self.lead if lr is None else FoldedLead(lr["H00"], lr["H01"], lr["S00"], lr["S01"],
                                                             fast=fast_lead)
        oL = O[tr["lead_left_block"]][self.lead.oL]                # lead-cell orbitals carrying g_L, g_R
        oR = O[tr["lead_right_block"]][self.leadR.oR]
        P = (abs(H) + abs(S)).tocsr()
        self.bL = np.unique(P[O[tr["lead_left_block"]]][:, dev].nonzero()[1])
        self.bR = np.unique(P[O[tr["lead_right_block"]]][:, dev].nonzero()[1])
        assert self.bL.max() < self.sb[1] and self.bR.min() >= self.sb[-2], "leads couple beyond end slabs"
        for o, b, blk in ((oL, self.bL, "lead_left_block"), (oR, self.bR, "lead_right_block")):
            rows = O[tr[blk]][np.unique(P[O[tr[blk]]][:, dev].nonzero()[0])]
            assert np.isin(rows, o).all(), f"{blk}: device couples to lead orbitals off its face"
        cut = lambda M, r, c: M[r][:, c].toarray()
        # A = z S - H blocks between device boundary and lead face: (lead, dev) and (dev, lead)
        self.cL = [cut(M, oL, dev[self.bL]) for M in (H, S)] + [cut(M, dev[self.bL], oL) for M in (H, S)]
        self.cR = [cut(M, oR, dev[self.bR]) for M in (H, S)] + [cut(M, dev[self.bR], oR) for M in (H, S)]
        # qttools path: whole lead cell (not only its faces) against the device boundary orbitals
        obc = {"mgms": "folded"}.get(obc, obc)                 # old name of the folded-lead OBC
        assert obc in ("auto", "qttools", "folded"), obc
        self.obc, self.qtt, self._inj = obc, None, {}
        if obc != "folded" and qttools_obc() is not None:
            sec = int(m.get("source", {}).get("cells_per_block", 1))
            perm = lead_cell_perm(inp, O[tr["lead_left_block"]], sec)
            lead = inp["lead"]
            try:
                self.qtt = QttLead(lead["H00"], lead["H01"], lead["S00"], lead["S01"], sec, perm)
                self.qttR = self.qtt if lr is None else QttLead(lr["H00"], lr["H01"], lr["S00"], lr["S01"], sec, perm)
            except RuntimeError as e:
                if obc == "qttools":
                    raise
                log(f"  qttools OBC not used: {e}")
                self.qtt = None
            self.qL = [cut(M, O[tr["lead_left_block"]], dev[self.bL]) for M in (H, S)] + \
                      [cut(M, dev[self.bL], O[tr["lead_left_block"]]) for M in (H, S)]
            self.qR = [cut(M, O[tr["lead_right_block"]], dev[self.bR]) for M in (H, S)] + \
                      [cut(M, dev[self.bR], O[tr["lead_right_block"]]) for M in (H, S)]
        elif obc == "qttools":
            raise RuntimeError(f"obc='qttools': {QTT_ERROR or 'qttools does not import'} (install the qtx extra, set QTTOOLS_SRC)")

    sigma_cache = None              # directory: Sigma(z) is read from / written to sig_<Re z>_<Im z>.npz there
                                    # (qsig_... with the injected modes BL, BR for the qttools path)

    def uses_qttools(self, z):
        return self.qtt is not None and abs(complex(z).imag) <= self.qtt_eta

    def sigma(self, z):
        """Sigma_L on bL, Sigma_R on bR (device indices) at complex z: A_dl g_L A_ld, A_dr g_R A_rd.
        qttools path (real axis): at E = Re z; the injected modes are kept for inject()."""
        z = complex(z)
        q = self.uses_qttools(z)
        if q:
            try:
                return self._sigma_cached(z, "qsig", self._sigma_qtt)
            except RuntimeError as e:
                if self.obc == "qttools":
                    raise
                log(f"  {e}; folded-lead Sigma instead")
        return self._sigma_cached(z, "sig", self._sigma)

    def _sigma_cached(self, z, pre, fun):
        f = self.sigma_cache and os.path.join(self.sigma_cache, f"{pre}_{z.real:+.8f}_{z.imag:.8e}.npz")
        if f and os.path.exists(f):
            c = np.load(f)
            out = c["SL"], c["SR"]
            if "BL" in c:
                self._inj = {id(out[0]): c["BL"], id(out[1]): c["BR"]}
            return out
        out, inj = fun(z)
        self._inj = {id(out[0]): inj[0], id(out[1]): inj[1]} if inj else {}
        if f:
            os.makedirs(self.sigma_cache, exist_ok=True)
            np.savez(f, SL=out[0], SR=out[1], **(dict(BL=inj[0], BR=inj[1]) if inj else {}))
        return out

    def _sigma_qtt(self, z):
        E = z.real
        out, inj = [], []
        for side, lead, (Hld, Sld, Hdl, Sdl) in (("L", self.qtt, self.qL), ("R", self.qttR, self.qR)):
            g, u = lead.modes(E, side)
            Adl = E * Sdl - Hdl
            out.append(Adl @ g @ (E * Sld - Hld))
            inj.append(Adl @ u)
        return tuple(out), tuple(inj)

    def inject(self, Sg, tol=1e-4):
        """Injection columns for the self-energy Sg returned by the last sigma(): the qttools incoming modes when
        they came with it, else the Gamma eigenchannels above tol (injection())."""
        B = self._inj.get(id(Sg))
        return injection(Sg, tol)[0] if B is None else B

    def _sigma(self, z):
        gL, gR = self.lead.surface_gfs(z)
        if self.leadR is not self.lead:
            gR = self.leadR.surface_gfs(z)[1]
        out = []
        a = gpu.asarray                                       # GPU mode: the dense products on the device too
        for g, (Hld, Sld, Hdl, Sdl) in ((gL, self.cL), (gR, self.cR)):
            out.append(gpu.asnumpy((z * a(Sdl) - a(Hdl)) @ a(g) @ (z * a(Sld) - a(Hld))))
        return tuple(out), None

    def A(self, E, SL, SR):
        """Sparse fine operator E S - H - Sigma_L - Sigma_R."""
        rows, cols, vals = [], [], []
        for Sg, b in ((SL, self.bL), (SR, self.bR)):
            ii, jj = np.meshgrid(b, b, indexing="ij")
            rows.append(ii.ravel()); cols.append(jj.ravel()); vals.append(Sg.ravel())
        Sig = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n, self.n))
        return (E * self.S - self.H - Sig).tocsr()


def sancho_both(A00, A01, A10, tol=1e-11, maxit=300):
    """Lopez-Sancho decimation of the chain with on-site A00 = z S00 - H00 and couplings A01 (cell n ->
    n+1), A10 (n+1 -> n), general (complex, non-Hermitian at complex z). Returns the surface Green's
    functions gR of the right-extending chain (its first cell) and gL of the left-extending chain (its
    last cell), both from one decimation. numpy or cupy arrays (the result lives where the input does)."""
    xp = gpu.xp_of(A00)
    e = A00.astype(complex)
    esR, esL = e.copy(), e.copy()
    a, b = A01.astype(complex), A10.astype(complex)
    n = len(e)
    for it in range(maxit):
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


class FoldedLead:
    """Periodic lead cell: H00, S00 and the coupling H01, S01 (cell n -> n+1 along +x).
    Faces F = orbitals of H01 / S01 (rows Rf, columns Lf); the interior I couples only inside its cell
    and is folded onto F at complex z:  A_eff(z) = A_FF - A_FI A_II^-1 A_IF,  A = z S - H.
      general (complex Hermitian H, any S): Sancho decimation of the folded chain on F with
          A01 = z S01 - H01, A10 = z S01^H - H01^H;  g_L, g_R on F (oL = oR = F).
      real symmetric H, S = I (fast): face_surface_gfs_small, chain on the smaller face;
          g_L on Rf (oL), g_R on Lf (oR).
    """

    def __init__(self, H00, H01, S00, S01, fast=None):
        H00, H01, S00, S01 = (sp.csr_matrix(M) for M in (H00, H01, S00, S01))
        self.n = H00.shape[0]
        P01 = abs(H01) + abs(S01)
        self.Rf = np.unique(P01.nonzero()[0])
        self.Lf = np.unique(P01.nonzero()[1])
        F = np.union1d(self.Lf, self.Rf)
        self.F, self.I = F, np.setdiff1d(np.arange(self.n), F)
        orth = abs(S00 - sp.identity(self.n)).max() < 1e-14 and (S01.nnz == 0 or abs(S01).max() < 1e-14)
        real = not (np.iscomplexobj(H00.data) or np.iscomplexobj(H01.data))
        self.fast = (orth and real) if fast is None else fast
        assert not self.fast or (orth and real), "fast lead path needs real H and S = I"
        pos = {v: i for i, v in enumerate(F)}
        self.iL = np.array([pos[c] for c in self.Lf]); self.iR = np.array([pos[r] for r in self.Rf])
        self.oL, self.oR = (self.Rf, self.Lf) if self.fast else (F, F)
        I = self.I
        self.HFF, self.SFF = H00[F][:, F].toarray(), S00[F][:, F].toarray()
        self.HFI, self.SFI = H00[F][:, I].toarray(), S00[F][:, I].toarray()
        self.HIF, self.SIF = H00[I][:, F].toarray(), S00[I][:, F].toarray()
        self.HII, self.SII = H00[I][:, I].tocsr(), S00[I][:, I].tocsr()
        self.H01, self.S01 = H01[F][:, F].toarray(), S01[F][:, F].toarray()

    def surface_gfs(self, z):
        if gpu.on and not self.fast:
            return self._surface_gfs_gpu(z)
        Aeff = z * self.SFF - self.HFF
        if len(self.I):                             # no interior when H01 couples the whole cell
            cs = CSolver(z * self.SII - self.HII)
            X = cs.solve(z * self.SIF - self.HIF)
            cs.free()
            Aeff = Aeff - (z * self.SFI - self.HFI) @ X
        if self.fast:                               # Aeff = z - Heff, Heff complex symmetric
            Heff = z * np.eye(len(Aeff)) - Aeff
            V = self.H01[np.ix_(self.iR, self.iL)]
            gL, gR, _ = face_surface_gfs_small(0.5 * (Heff + Heff.T), self.iL, self.iR, V, z)
            return gL, gR
        A01 = z * self.S01 - self.H01
        A10 = z * self.S01.conj().T - self.H01.conj().T
        return sancho_both(Aeff, A01, A10)

    def _surface_gfs_gpu(self, z):
        """General path on the GPU: dense LU of the interior, decimation on the device; numpy results."""
        d = getattr(self, "_dev", None)
        if d is None:
            d = self._dev = {k: gpu.asarray(getattr(self, k)) for k in
                             ("HFF", "SFF", "HFI", "SFI", "HIF", "SIF", "H01", "S01")}
            d["HII"], d["SII"] = gpu.csr(self.HII, np.complex128), gpu.csr(self.SII, np.complex128)
        Aeff = z * d["SFF"] - d["HFF"]
        if len(self.I):
            lu = gpu.lu_factor((z * d["SII"] - d["HII"]).toarray())
            X = gpu.lu_solve(lu, (z * d["SIF"] - d["HIF"]).astype(complex))
            del lu
            Aeff = Aeff - (z * d["SFI"] - d["HFI"]) @ X
        gL, gR = sancho_both(Aeff, z * d["S01"] - d["H01"], z * d["S01"].conj().T - d["H01"].conj().T)
        return gpu.asnumpy(gL), gpu.asnumpy(gR)


def injection(SL, tol=1e-4):
    """Columns B with Gamma_L ~ B B^H: eigenvectors of Gamma_L above tol [eV] (propagating channels)."""
    G = 1j * (SL - SL.conj().T)
    lam, X = np.linalg.eigh(0.5 * (G + G.conj().T))
    k = lam > tol
    return X[:, k] * np.sqrt(lam[k]), lam, X


# ------------------------------------------------------------------ level 1: partition
def find_cores(labels, geo, core_label="Si", d_core=6.0):
    """Core flag per atom: label core_label and >= d_core from every atom with another label."""
    other = labels != core_label
    if not other.any():
        return labels == core_label, np.full(len(labels), np.inf)
    d, _ = geo.tree(np.where(other)[0]).query(geo.points(np.arange(len(labels))), k=1)
    return (~other) & (d >= d_core), d


def bisect(atoms, pos2d, w, nt):
    """Split atoms into nt groups of ~equal weight w by recursive bisection along the longer axis."""
    if nt <= 1 or len(atoms) <= 1:
        return [atoms]
    ax = int(np.argmax(np.ptp(pos2d[atoms], axis=0)))
    o = atoms[np.argsort(pos2d[atoms, ax], kind="stable")]
    cum = np.cumsum(w[o])
    k1 = nt // 2
    cut = int(np.searchsorted(cum, cum[-1] * k1 / nt)) + 1
    return bisect(o[:cut], pos2d, w, k1) + bisect(o[cut:], pos2d, w, nt - k1)


def parse_cut(spec):
    """Tile cuts, ';'-separated, applied in order: '<axis>:<c1>,<c2>,...@<s1>,<s2>,...' (cut planes) or
    '<axis>:n<k>@<slabs>' (k pieces of equal orbital weight) -> list of dict(axis, n, at, slabs) (see partition);
    None / '' -> None."""
    if not spec:
        return None
    out = []
    for item in spec.split(";"):
        ax, rest = item.split(":")
        at, sl = rest.split("@") if "@" in rest else (rest, "")
        out.append(dict(axis=int(ax), n=int(at[1:]) if at.startswith("n") else None,
                        at=None if at.startswith("n") else sorted(float(x) for x in at.split(",")),
                        slabs=[int(x) for x in sl.split(",")] if sl else None))
    return out


def cut_pieces(atoms, coord, at, period=None):
    """Atoms grouped by the intervals between the cut positions `at` along one coordinate; with a period the
    coordinate is taken modulo it and the pieces below at[0] and above at[-1] are one piece (wrap-around)."""
    c = coord[atoms] % period if period else coord[atoms]
    k = np.searchsorted(np.asarray(at), c)
    if period:
        k[k == len(at)] = 0
    return [np.sort(atoms[k == j]) for j in np.unique(k)]


def partition(dev, cores=True, core_label="Si", d_core=6.0, tile_orbs=2500, axis=0, xsplit=1, cut=None):
    """Blocks per slab: dict(kind, slab, atoms (global atom ids, sorted), orbs (slab-local orbitals)).
    Tiles: recursive transverse bisection to ~tile_orbs orbitals, then each tile cut into xsplit pieces of equal orbital
    weight along the transport axis (xsplit = 1: tiles span the whole slab in x).
    cut = [dict(axis, n, at, slabs), ...] (parse_cut): instead of the bisection, the slabs in `slabs` (all if None) are cut at the
    planes coord[axis] = at (periodic axis: modulo the period, the outermost pieces joined), every other slab is one
    tile; '0:n2@1,2' = slabs 1, 2 cut into two halves of equal orbital weight along x, slabs 0, 3 one tile each;
    '2:8.5,15.5@1,2' = the MoS2 layer and the (wrapped) oxide of slabs 1, 2 as two tiles each; '0:n2@1,2;2:9,15.5@1,2'
    both (4 tiles in slabs 1, 2)."""
    inp = dev.inp
    labels, xyz, norb = inp["labels"], inp["xyz"], inp["norb"]
    core, dmin = (find_cores(labels, Geometry(xyz, dev.meta.get("periodic")), core_label, d_core) if cores
                  else (np.zeros(len(labels), bool), None))
    tr = [i for i in range(3) if i != axis]
    blocks = []
    for s, b in enumerate(dev.dev_blocks):
        a = np.where(inp["blocks"] == b)[0]
        if core[a].any():
            blocks.append(dict(kind="core", slab=s, atoms=a[core[a]]))
        rest = a[~core[a]]
        if cut is not None:
            pieces = [rest]
            for c in cut:
                if c["slabs"] is not None and s not in c["slabs"]:
                    continue
                per = dict(periodic_axes(dev.meta.get("periodic"))).get(c["axis"])
                pieces = [q for t in pieces for q in (
                    [np.sort(u) for u in bisect(t, xyz[:, [c["axis"]]], norb, c["n"])] if c["n"]
                    else cut_pieces(t, xyz[:, c["axis"]], c["at"], per))]
        else:
            pieces = bisect(rest, xyz[:, tr], norb, max(1, int(round(norb[rest].sum() / tile_orbs))))
        for j, t in enumerate(pieces):
            for tx in bisect(t, xyz[:, [axis]], norb, xsplit):
                blocks.append(dict(kind="tile", slab=s, atoms=np.sort(tx), tpar=j))   # tpar: transverse piece
    for blk in blocks:
        g = np.concatenate([np.arange(dev.off[i], dev.off[i + 1]) for i in blk["atoms"]])
        blk["orbs"] = dev.loc[g]
    return blocks, core, dmin


# ------------------------------------------------------------------ level 1: cores
class BulkFromDevice:
    """Bulk diamond Bloch Hamiltonian read off the device H at a deep core atom.
    H(k)[sa, sb] = sum_j H[a_sa, j] exp(i k . (r_j - r_a)), j in sublattice sb, a_0 = deepest core atom,
    a_1 = its deepest nearest neighbour. Basis amplitude on a core atom at r: c_sub(k) exp(i k . r)."""

    def __init__(self, inp, core, dmin):
        H, xyz, norb = inp["H"].tocsr(), inp["xyz"], inp["norb"]
        off = np.concatenate([[0], np.cumsum(norb)])
        atom_of = np.repeat(np.arange(len(norb)), norb)
        ca = np.where(core)[0]
        a0 = ca[np.argmax(dmin[ca])]
        nb0 = np.unique(atom_of[H[off[a0]:off[a0 + 1]].indices])
        d = xyz[nb0] - xyz[a0]
        r = np.linalg.norm(d, axis=1)
        nn = nb0[(r > 1e-6) & (r < 1.05 * r[r > 1e-6].min())]
        assert len(nn) == 4, f"core atom {a0}: {len(nn)} nearest neighbours, expected 4 (diamond)"
        v = xyz[nn] - xyz[a0]
        self.v0 = v[0]
        self.Afcc = np.array([v[0] - v[1], v[0] - v[2], v[0] - v[3]])
        self.a_lat = float(np.linalg.norm(self.Afcc[0]) * np.sqrt(2))
        self.Ainv = np.linalg.inv(self.Afcc)
        self.r0 = xyz[a0]
        a1 = nn[np.argmax(dmin[nn])]
        self.ns = int(norb[a0])
        self.terms = []                        # (sa, sb, M, d)
        for a in (a0, a1):
            sa = self.sublattice(xyz[a][None])[0]
            for j in np.unique(atom_of[H[off[a]:off[a + 1]].indices]):
                assert norb[j] == self.ns, f"neighbour {j} of core atom {a} has {norb[j]} orbitals"
                sb = self.sublattice(xyz[j][None])[0]
                M = H[off[a]:off[a + 1], off[j]:off[j + 1]].toarray()
                self.terms.append((sa, sb, M, xyz[j] - xyz[a]))
        self.atoms = (a0, a1)
        Hk = self.hk(np.random.default_rng(1).normal(size=(4, 3)))
        assert np.abs(Hk - np.conj(np.transpose(Hk, (0, 2, 1)))).max() < 1e-10, "extracted H(k) not Hermitian"

    def sublattice(self, r, tol=1e-3):
        """0 / 1 for positions on the A / B sublattice of the diamond lattice through atom a_0."""
        out = -np.ones(len(r), int)
        for s, sh in ((0, 0.0), (1, 1.0)):
            c = (r - self.r0 - sh * self.v0) @ self.Ainv
            out[np.abs(c - np.round(c)).max(1) < tol] = s
        assert (out >= 0).all(), "atom off the diamond lattice"
        return out

    def hk(self, K):
        ns = self.ns
        Hk = np.zeros((len(K), 2 * ns, 2 * ns), complex)
        for sa, sb, M, d in self.terms:
            Hk[:, sa * ns:(sa + 1) * ns, sb * ns:(sb + 1) * ns] += M[None] * np.exp(1j * K @ d)[:, None, None]
        return Hk

    def kgrid(self, Lx, Wy, Wz, oversample=2.0):
        """As qtmode.cms.core_kgrid: k_x commensurate with Lx, k_y, k_z oversampled for the core size."""
        a = self.a_lat
        nkx = int(round(Lx / a)) * 2
        kx = (np.arange(nkx) - nkx // 2) * 2 * np.pi / Lx
        ky = np.arange(-2 * np.pi / a, 2 * np.pi / a, np.pi / (oversample * Wy))
        kz = np.arange(-2 * np.pi / a, 2 * np.pi / a, np.pi / (oversample * Wz))
        return np.array(np.meshgrid(kx, ky, kz, indexing="ij")).reshape(3, -1).T

    def states(self, K, ewin, chunk=20000):
        ks, cs = [], []
        for c0 in range(0, len(K), chunk):
            kk = K[c0:c0 + chunk]
            E, V = np.linalg.eigh(self.hk(kk))
            sel = np.argwhere((E > ewin[0]) & (E < ewin[1]))
            ks.append(kk[sel[:, 0]]); cs.append(V[sel[:, 0], :, sel[:, 1]])
        return np.vstack(ks), np.vstack(cs)


def core_basis(bulk, pos, kst, cst, thr=1e-2, chunk=4000):
    """Real orthonormal basis of the Bloch states (kst, cst) restricted to atoms at pos (qtmode.cms.core_basis)."""
    ns = bulk.ns
    sub = bulk.sublattice(pos)
    n = len(pos) * ns
    C = np.zeros((n, n))
    for c0 in range(0, len(kst), chunk):
        k, c = kst[c0:c0 + chunk], cst[c0:c0 + chunk]
        ph = np.exp(1j * (k @ pos.T))
        amp = np.where(sub[None, :, None] == 0, c[:, None, :ns], c[:, None, ns:]) * ph[:, :, None]
        B = amp.reshape(len(k), -1).T
        C += (B @ B.conj().T).real
    lam, Q = np.linalg.eigh(C)
    lam, Q = lam[::-1], Q[:, ::-1]
    S = np.sqrt(np.clip(lam, 0, None))
    return Q[:, S > thr * S[0]], S


# ------------------------------------------------------------------ level 1: tiles
def periodic_axes(periodic):
    """[(axis, length)] from meta["periodic"]: None, {"axis": k, "length": L} or a list of such dicts."""
    if not periodic:
        return []
    return [(int(p["axis"]), float(p["length"])) for p in ([periodic] if isinstance(periodic, dict) else periodic)]


class Geometry:
    """Distance queries on atom positions, minimum image along the periodic transverse axes
    (meta["periodic"]: {"axis": k, "length": L} or a list of them, e.g. a 2D sheet folded at k_y = 0, or a
    QuaTrEx Gamma-point supercell periodic in y and z)."""

    def __init__(self, xyz, periodic=None):
        self.xyz = np.asarray(xyz, float)
        self.box = None
        self.per = periodic_axes(periodic)
        if self.per:
            self.lo = self.xyz.min(0) - 1.0
            self.box = np.full(3, 2 * np.ptp(self.xyz, axis=0).max() + 1e3)
            for ax, L in self.per:
                self.box[ax] = L

    def _map(self, r):
        if self.box is None:
            return r
        q = r - self.lo
        for ax, L in self.per:
            q[:, ax] %= L
        return q

    def tree(self, idx=None):
        r = self.xyz if idx is None else self.xyz[idx]
        return cKDTree(self._map(r), boxsize=self.box)

    def points(self, idx):
        return self._map(self.xyz[idx])


def _tile_embedding(inp, geo, tree, atoms, d_emb, d_cap, eta_cap, obc=None):
    """CAP-terminated embedding of a tile: device atoms within d_emb (lead-block atoms dropped when obc is a Device),
    CAP eta_cap ((d - (d_emb - d_cap)) / d_cap)^2 on the outer d_cap shell. Tile orbitals pidx in tile atom order
    (the same row order for every d_emb)."""
    norb = inp["norb"]
    off = np.concatenate([[0], np.cumsum(norb)])
    near = tree.query_ball_point(geo.points(atoms), d_emb)
    emb = np.union1d(np.unique(np.concatenate([np.asarray(q, int) for q in near])), atoms)
    if obc is not None:
        emb = emb[np.isin(emb, obc_atoms(obc))]
    d, _ = geo.tree(atoms).query(geo.points(emb), k=1)
    w = eta_cap * np.clip((d - (d_emb - d_cap)) / d_cap, 0, None) ** 2
    oe = np.concatenate([np.arange(off[i], off[i + 1]) for i in emb])
    ne = norb[emb]
    eoff = np.concatenate([[0], np.cumsum(ne)])
    pe = np.searchsorted(emb, atoms)
    pidx = np.concatenate([np.arange(eoff[j], eoff[j + 1]) for j in pe])
    return dict(emb=emb, oe=oe, pidx=pidx, He=inp["H"][oe][:, oe].tocsr(), Se=inp["S"][oe][:, oe].tocsr(),
                W=sp.diags(np.repeat(w, ne)), faces=[] if obc is None else obc_faces(obc, oe), d_emb=d_emb)


def _sig_emb(obc, e, z):
    """-Sigma_L/R(z) on the embedded lead-face orbitals (sparse, embedding size)."""
    n = e["He"].shape[0]
    if not e["faces"] or not any(len(p) for p, _ in e["faces"]):
        return sp.csr_matrix((n, n), dtype=complex)
    SL, SR = obc.sigma(z)
    M = sp.csr_matrix((n, n), dtype=complex)
    for (pos, fi), Sg in zip(e["faces"], (SL, SR)):
        if len(pos):
            ii, jj = np.meshgrid(pos, pos, indexing="ij")
            M = M - sp.csr_matrix((Sg[np.ix_(fi, fi)].ravel(), (ii.ravel(), jj.ravel())), shape=(n, n))
    return M


ENV_MIN_COMP = 2000        # environment components smaller than this are factorized together


class GpuEmbedding:
    """Device copies of one CAP(+OBC) embedding for the dense Schur-complement G_PP on the GPU, made once per block
    (blocks of z S - K, K = H - i W, as CSR; per point only z S - K is formed on the device and Sigma(z) scattered into
    the lead-face rows). Memory-lean (2026-11-17, for the 12 nm slabs):
      1. the environment E is split into its connected components (graph of H, S and the lead faces restricted to E;
         a slab embedding has a left and a right piece): one LU per component, A_EE is block diagonal, exact;
      2. A_EP and X = A_EE^-1 A_EP in column chunks of ~chunk_bytes, folded into S_P = A_PP - A_PE X at once;
      3. S_P^-1 by its LU and solves against identity column chunks added straight into the accumulator (no inverse).
    Peak ~ (max_c n_Ec^2 + 2 m^2) x 16 B + chunks instead of (n_E^2 + 2 n_E m + 3 m^2) x 16 B.
    Same class as the release (qtx_modespace.level1.GpuEmbedding), 2026-11-17."""

    def __init__(self, e, chunk_bytes=4e9, min_comp=None, obc=None):
        from scipy.sparse.csgraph import connected_components
        cp = gpu.cp
        He, Se, W, pidx = e["He"], e["Se"], e["W"], np.asarray(e["pidx"])
        n = He.shape[0]
        self.e, self.m, self.chunk_bytes = e, len(pidx), chunk_bytes
        eidx = np.setdiff1d(np.arange(n), pidx)
        self.obc = obc if obc is not None else e.get("obc")
        faces = [(np.asarray(p, int), np.asarray(f, int)) for p, f in
                 ((e.get("faces") or []) if self.obc is not None else [])]
        # components of the environment (lead faces: Sigma couples all face orbitals of one lead)
        G = (abs(He) + abs(Se)).tocsr()[eidx][:, eidx]
        loc = np.full(n, -1)
        loc[eidx] = np.arange(len(eidx))
        rows, cols = [], []
        for p, _ in faces:
            q = loc[p][loc[p] >= 0]
            if len(q) > 1:
                rows.append(q[:-1]); cols.append(q[1:])
        if rows:
            r_, c_ = np.concatenate(rows), np.concatenate(cols)
            G = G + sp.csr_matrix((np.ones(len(r_)), (r_, c_)), shape=G.shape)
        ncomp, lab = connected_components(G, directed=False)
        sizes = np.bincount(lab, minlength=ncomp)
        min_comp = ENV_MIN_COMP if min_comp is None else min_comp
        big = [c for c in np.argsort(-sizes) if sizes[c] >= min_comp]
        groups = [eidx[lab == c] for c in big]
        small = np.isin(lab, big, invert=True)
        if small.any():
            groups.append(eidx[small])                          # small pieces together (still block diagonal)
        self.env_sizes = [len(g) for g in groups]
        K, Sc = (He - 1j * W).tocsr(), sp.csr_matrix(Se)
        d = lambda M: gpu.csr(M, np.complex128)
        self.PP = (d(Sc[pidx][:, pidx]), d(K[pidx][:, pidx]))
        self.env = []
        for g in groups:
            self.env.append(dict(
                EE=(d(Sc[g][:, g]), d(K[g][:, g])),
                EPT=(d(Sc[g][:, pidx].T.tocsr()), d(K[g][:, pidx].T.tocsr())),     # rows = P: column chunks by rows
                PE=(d(Sc[pidx][:, g]), d(K[pidx][:, g])),
                idx=g))
        # where the lead-face orbitals sit in P and in each environment group: (rows in the group, rows in Sigma)
        def where(idx):
            l = np.full(n, -1)
            l[idx] = np.arange(len(idx))
            out = []
            for p, f in faces:
                ok = l[p] >= 0
                out.append((cp.asarray(l[p][ok]), cp.asarray(f[ok])))
            return out
        self.fP = where(pidx)
        for g in self.env:
            g["f"] = where(g["idx"])
        self.has_sigma = any(len(r) for r, _ in self.fP) or any(len(r) for g in self.env for r, _ in g["f"])

    @staticmethod
    def _sub(M, z, Sg, fr, fc):
        """M -= Sigma[fr-rows, fc-cols] for every lead (dense M, in place)."""
        cp = gpu.cp
        for (ro, sr), (co, sc), G in zip(fr, fc, Sg):
            if len(ro) and len(co):
                ix = cp.ix_(ro, co)
                M[ix] = M[ix] - G[cp.ix_(sr, sc)]
        return M

    def accumulate(self, z, dz, acc, env_prec="double", refine=1, R=None):
        """acc += G_PP(z) dz (cupy m x m), or G_PP(z) R dz for probe columns R (cupy m x r)."""
        cp, m = gpu.cp, self.m
        Sg = [cp.asarray(M) for M in self.obc.sigma(z)] if self.has_sigma else []
        Sp = self._sub((z * self.PP[0] - self.PP[1]).toarray(), z, Sg, self.fP, self.fP)
        for g in self.env:
            nE = len(g["idx"])
            AEE = self._sub((z * g["EE"][0] - g["EE"][1]).toarray(), z, Sg, g["f"], g["f"])
            if env_prec == "single":
                lu = gpu.lu_factor(AEE.astype(np.complex64))
            else:
                lu, AEE = gpu.lu_factor(AEE), None                  # overwrites A_EE
            APE = z * g["PE"][0] - g["PE"][1]
            k = max(1, min(m, int(self.chunk_bytes // (16 * 3 * max(nE, 1)))))
            for j0 in range(0, m, k):
                j1 = min(m, j0 + k)
                AEP = cp.ascontiguousarray((z * g["EPT"][0][j0:j1] - g["EPT"][1][j0:j1]).toarray().T)
                for (ro, sr), (co, sc), G in zip(g["f"], self.fP, Sg):       # Sigma on (E rows, P columns in chunk)
                    sel = (co >= j0) & (co < j1)
                    if len(ro) and bool(sel.any()):
                        ix = cp.ix_(ro, co[sel] - j0)
                        AEP[ix] = AEP[ix] - G[cp.ix_(sr, sc[sel])]
                if env_prec == "single":
                    X = gpu.lu_solve(lu, AEP.astype(np.complex64)).astype(complex)
                    for _ in range(refine):
                        X += gpu.lu_solve(lu, (AEP - AEE @ X).astype(np.complex64))
                else:
                    X = gpu.lu_solve(lu, AEP)
                del AEP
                Sp[:, j0:j1] -= APE @ X                             # A_PE X: sparse part, then its Sigma part
                for (ro, sr), (co, sc), G in zip(self.fP, g["f"], Sg):
                    if len(ro) and len(co):
                        Sp[ro, j0:j1] += G[cp.ix_(sr, sc)] @ X[co]
                del X
            del lu, AEE, APE
        lu = gpu.lu_factor(Sp)                                      # overwrites S_P
        del Sp
        if R is not None:
            acc += gpu.lu_solve(lu, R) * dz
            return
        k = max(1, min(m, int(self.chunk_bytes // (16 * 2 * m))))
        for j0 in range(0, m, k):
            j1 = min(m, j0 + k)
            I = cp.zeros((m, j1 - j0), complex)
            I[cp.arange(j0, j1), cp.arange(j1 - j0)] = 1.0
            acc[:, j0:j1] += gpu.lu_solve(lu, I) * dz
            del I
        del lu


def tile_rho_contour(inp, geo, tree, atoms, contour, d_emb=12.0, d_cap=3.0, eta_cap=0.3, obc=None, gpp="schur",
                     env_prec="single", refine=1, emb_far=None, workers=1, threads=None, rhs=None,
                     dense_fill=0.02, dense_max=30000, chunk_bytes=4e9):
    """Accelerated contour integral of the tile block of the CAP(+OBC) embedding Green's function,
    acc = int_arc G_PP(z) R dz on the upper arc (Gauss-Legendre in theta, as tile_modes(contour=)); returns
    (acc, info). R = rhs (m x r probes; None: identity, the full G_PP).
    gpp = "schur": G_PP = (A_PP - A_PE A_EE^-1 A_EP)^-1 R, A = z S - H + iW - Sigma_emb(z): the environment E is
      factorized alone and only m columns are solved; env_prec = "single": A_EE LU in complex64, `refine` steps of
      iterative refinement of X = A_EE^-1 A_EP in complex128 (residual with the sparse A_EE), Schur complement and
      its inverse in complex128. "full": one LU of the whole embedding (tile_modes' original path, double).
    emb_far = [(imz, d), ...]: contour points with Im z >= imz use the smaller embedding d (largest matching imz);
      G decays faster away from the real axis, the full d_emb is needed near the window edges only.
    workers > 1: contour points split over forked processes with `threads` BLAS threads each (threadpoolctl)."""
    mu1, mu2, npts = contour
    c, rr = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    x, wq = np.polynomial.legendre.leggauss(int(npts))
    th, wq = 0.5 * np.pi * (x + 1), 0.5 * np.pi * wq
    zs = c + rr * np.exp(1j * th)
    rules = sorted(emb_far or [])
    dz = []
    for z in zs:
        d = d_emb
        for imz, dd in rules:
            if z.imag >= imz:
                d = dd
        dz.append(min(d, d_emb))
    embs = {d: _tile_embedding(inp, geo, tree, atoms, d, d_cap, eta_cap, obc) for d in sorted(set(dz))}
    m = len(embs[d_emb if d_emb in embs else max(embs)]["pidx"])
    r = m if rhs is None else rhs.shape[1]
    cdt = np.complex64 if env_prec == "single" else np.complex128

    def gpu_blocks(e):
        """Device copies of one embedding, made once: E / P blocks of S and K = H - i W as CSR, and where the lead-face
        orbitals sit in E and P (per point only z S - K is formed on the device and Sigma(z) scattered into the
        blocks; assembling and slicing the sparse A on the host cost more than the factorization)."""
        if "_gpu" not in e:
            n, pidx = e["He"].shape[0], e["pidx"]
            eidx = np.setdiff1d(np.arange(n), pidx)
            K, Sc = (e["He"] - 1j * e["W"]).tocsr(), sp.csr_matrix(e["Se"])
            sub = (("E", eidx), ("P", pidx))
            gS = {a + b: gpu.csr(Sc[i][:, j], np.complex128) for a, i in sub for b, j in sub}
            gK = {a + b: gpu.csr(K[i][:, j], np.complex128) for a, i in sub for b, j in sub}
            faces = []
            for pos, fi in (e.get("faces") or [] if obc is not None else []):
                pos, fi = np.asarray(pos, int), np.asarray(fi, int)
                d = {}
                for a, i in sub:
                    loc = np.full(n, -1)
                    loc[i] = np.arange(len(i))
                    d[a] = (gpu.cp.asarray(loc[pos][loc[pos] >= 0]), gpu.cp.asarray(fi[loc[pos] >= 0]))
                faces.append(d)
            e["_gpu"] = gS, gK, faces, any(len(f[a][0]) for f in faces for a in "EP")
        return e["_gpu"]

    def point_gpu(k, e, RP):
        cp, pidx, z = gpu.cp, e["pidx"], zs[k]
        if gpp == "full":
            A = (z * e["Se"] - e["He"] + 1j * e["W"] + _sig_emb(obc, e, z)).tocsr()
            R = cp.zeros((A.shape[0], r), complex)
            R[cp.asarray(pidx)] = cp.asarray(RP)
            return gpu.lu_solve(gpu.lu_factor(gpu.csr_dense(A, np.complex128)), R)[cp.asarray(pidx)]
        gS, gK, faces, has_sigma = gpu_blocks(e)
        Sg = [cp.asarray(M) for M in obc.sigma(z)] if has_sigma else []

        def block(ab):                                              # dense A_ab = z S - K - Sigma on the device
            M = (z * gS[ab] - gK[ab]).toarray()
            for f, G in zip(faces, Sg):
                (ro, fr), (co, fc) = f[ab[0]], f[ab[1]]
                if len(ro) and len(co):
                    ix = cp.ix_(ro, co)
                    M[ix] = M[ix] - G[cp.ix_(fr, fc)]
            return M

        AEE, AEP = block("EE"), block("EP")
        if env_prec == "single":
            lu = gpu.lu_factor(AEE.astype(cdt))
            X = gpu.lu_solve(lu, AEP.astype(cdt)).astype(complex)
            for _ in range(refine):
                X += gpu.lu_solve(lu, (AEP - AEE @ X).astype(cdt))
        else:
            lu = gpu.lu_factor(AEE)
            X = gpu.lu_solve(lu, AEP)
        del lu, AEE, AEP
        Sp = block("PP") - (z * gS["PE"] - gK["PE"]) @ X            # A_PE X: sparse part, then its Sigma part
        for f, G in zip(faces, Sg):
            (ro, fr), (co, fc) = f["P"], f["E"]
            if len(ro) and len(co):
                Sp[ro] = Sp[ro] + G[cp.ix_(fr, fc)] @ X[co]
        del X
        return cp.linalg.inv(Sp) if rhs is None else gpu.solve(Sp, cp.asarray(RP))

    def point(k):
        z, e = zs[k], embs[dz[k]]
        He, Se, W, pidx = e["He"], e["Se"], e["W"], e["pidx"]
        n = He.shape[0]
        dense = (He + Se).nnz > dense_fill * n * n and n <= dense_max
        RP = np.eye(m, dtype=complex) if rhs is None else rhs.astype(complex)
        if gpu.on and dense:
            return point_gpu(k, e, RP)
        A = (z * Se - He + 1j * W + _sig_emb(obc, e, z)).tocsr()
        if gpp == "full":
            R = np.zeros((n, r), complex)
            R[pidx] = RP
            if dense:
                lu = sla.lu_factor(A.toarray(), check_finite=False, overwrite_a=True)
                return sla.lu_solve(lu, R, check_finite=False)[pidx]
            cs = CSolver(A)
            G = cs.solve(R)[pidx]
            cs.free()
            return G
        eidx = np.setdiff1d(np.arange(n), pidx)
        Ar = A[eidx]
        AEE, AEP = Ar[:, eidx].tocsr(), Ar[:, pidx].toarray()
        Ap = A[pidx]
        APP, APE = Ap[:, pidx].toarray(), Ap[:, eidx].tocsr()
        if dense:
            AEEd = AEE.toarray()                                          # dense: threaded GEMM residual
            lu = sla.lu_factor(AEEd.astype(cdt), check_finite=False, overwrite_a=True)
            X = sla.lu_solve(lu, AEP.astype(cdt), check_finite=False).astype(complex)
            if env_prec == "single":
                for _ in range(refine):
                    X += sla.lu_solve(lu, (AEP - AEEd @ X).astype(cdt), check_finite=False)
            del lu, AEEd
        else:
            cs = CSolver(AEE)
            X = cs.solve(AEP)
            cs.free()
        Sp = APP - APE @ X
        del X
        return np.linalg.inv(Sp) if rhs is None else sla.solve(Sp, RP, check_finite=False)

    def lean(k):
        """GpuEmbedding of point k's embedding (GPU + dense + Schur), made once per embedding; else None."""
        e = embs[dz[k]]
        n = e["He"].shape[0]
        if not (gpu.on and gpp != "full" and (e["He"] + e["Se"]).nnz > dense_fill * n * n and n <= dense_max):
            return None
        if "_lean" not in e:
            e["_lean"] = GpuEmbedding(e, chunk_bytes=chunk_bytes, obc=obc)
        return e["_lean"]

    def run(ks):
        acc = (gpu.cp if gpu.on else np).zeros((m, r), complex)
        Rd = None if rhs is None or not gpu.on else gpu.cp.asarray(rhs.astype(complex))
        for k in ks:
            w = -1j * rr * np.exp(1j * th[k]) * wq[k]
            ge = lean(k)
            if ge is not None:
                ge.accumulate(zs[k], w, acc, env_prec=env_prec, refine=refine, R=Rd)
            else:
                acc += point(k) * w
        return gpu.asnumpy(acc)

    t0 = time.time()
    if gpu.on:                                    # no fork after CUDA initialization: points one after another
        acc = run(range(len(zs)))
        for e_ in embs.values():
            e_.pop("_gpu", None)
            e_.pop("_lean", None)
        gpu.free()
    elif workers <= 1:
        if threads:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(threads):
                acc = run(range(len(zs)))
        else:
            acc = run(range(len(zs)))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        cost = np.array([embs[d]["He"].shape[0] ** 3 for d in dz], float)
        load, parts = np.zeros(workers), [[] for _ in range(workers)]
        for k in np.argsort(-cost):                                       # greedy balance by n^3
            j = int(np.argmin(load))
            parts[j].append(int(k))
            load[j] += cost[k]

        def child(j):
            from threadpoolctl import threadpool_limits
            try:
                with threadpool_limits(threads or 1):
                    q.put((j, run(parts[j]), None))
            except Exception as ex:                                       # noqa: BLE001
                q.put((j, None, repr(ex)))
        ps = [ctx.Process(target=child, args=(j,)) for j in range(workers) if parts[j]]
        for p_ in ps:
            p_.start()
        acc = np.zeros((m, r), complex)
        for _ in ps:
            j, a_, err = q.get()
            if err:
                raise RuntimeError(f"tile_rho_contour worker {j}: {err}")
            acc += a_
        for p_ in ps:
            p_.join()
    info = dict(t=time.time() - t0, n_emb={d: embs[d]["He"].shape[0] for d in embs}, d_per_point=dz,
                imz=zs.imag.tolist(), nemb_atoms=len(embs[max(embs)]["emb"]))
    return acc, info


def tile_modes(inp, geo, tree, atoms, energies, d_emb=12.0, d_cap=3.0, eta_cap=0.3, rank=None, n_power=1,
               tile_thr=1e-4, a_ref=0.335, rng=None, dense_fill=0.02, dense_max=30000, eta=0.0, contour=None,
               rho_thr=1e-3, sketch=0, probes="gauss", boundary="cap", obc=None, fast=None):
    """Tile basis from the local spectral function A_PP(E) = i (G - G^H)_PP / 2 pi, G = (E S - H + iW)^-1,
    in a CAP-terminated embedding (qtmode.cms.tile_spectral_modes); randomized range finder, all energies
    merged: Y = [X_E sqrt(lambda_E)], SVD, keep s^2 > tile_thr * a_ref. Real H, S: A X = -Im(G X) / pi
    with real probes (one factorization). Complex Hermitian H, S: G^H = (E S - H - iW)^-1 is factorized
    as well and the basis is complex. Embeddings with nnz > dense_fill n^2 (long-range ab initio H, e.g.
    MoS2/HfO2: ~1300 nonzeros per row) and n <= dense_max use one dense LU per energy instead (G^H solves by
    the transposed factorization). Rows: tile orbitals in atom order.
    eta > 0: spectral function at E + i eta (Lorentzian width eta for every state, also for states that never
    reach the CAP, e.g. traps deep in an oxide, whose peaks are otherwise too narrow to be sampled).
    contour = (mu1, mu2, npts) (real H, S): instead of the sampled A(E), the tile density matrix of the window,
    rho_PP = -(1/pi) Im int_arc G_PP dz on the upper arc from mu1 to mu2 (Gauss-Legendre in theta, as
    rho_contour), built exactly (one solve with the m tile columns per point); keeps the eigenvectors with
    occupation > rho_thr, so every state in the window counts in full, whatever its width. Returns
    (U, occupations, embedding atoms) in that case. sketch = r > 0: one-pass Nystrom sketch instead,
    Y = rho Omega with r Gaussian probes accumulated over the points (r solves per point instead of m),
    rho ~ Y (Omega^T Y)^+ Y^T; r must exceed the number of kept occupations (warned if not, by 20%).
    probes: "gauss" or "sign" (random +-1, the real random-phase probe: every orbital, also of a state
    localized on a few atoms, overlaps each probe with weight exactly 1).
    boundary = "schur": no CAP; the tile P sees the environment E (embedding atoms outside the tile, a closed
    cluster cut at d_emb) through the Schur-complement self-energy Sigma_P(z) = A_PE A_EE(z)^-1 A_EP,
    A = z S - H, and G_PP = (A_PP - Sigma_P)^-1 is formed explicitly (m solves with A_EE per point, dense m x m
    inverse). Sigma_P is Hermitian with poles at the closed-environment levels on the real axis, so it needs
    complex z: the contour, or eta > 0 at the sampled energies (then A_PP is formed exactly, no range finder;
    complex H: G(z)^H = G(conj z), one solve per energy).
    obc = Device (contour, CAP boundary): the embedding keeps device atoms only; lead-block atoms are replaced by
    the exact lead self-energy, -Sigma_L/R(z) added on the lead-face orbitals inside the embedding (the principal
    submatrix when the embedding holds only part of the face). The CAP stays on the other embedding cuts.
    fast = dict (contour, CAP boundary): tile_rho_contour options (gpp, env_prec, refine, emb_far, workers, threads):
    Schur-complement G_PP with a low-precision environment, smaller embeddings away from the axis, contour points
    in parallel."""
    assert boundary in ("cap", "schur")
    assert obc is None or (contour is not None and boundary == "cap"), "obc: contour with CAP boundary only"
    assert boundary == "cap" or contour is not None or eta > 0, "boundary schur: needs the contour or eta > 0"
    rng = np.random.default_rng(0) if rng is None else rng
    norb = inp["norb"]
    off = np.concatenate([[0], np.cumsum(norb)])
    near = tree.query_ball_point(geo.points(atoms), d_emb)
    emb = np.union1d(np.unique(np.concatenate([np.asarray(q, int) for q in near])), atoms)
    if obc is not None:
        emb = emb[np.isin(emb, obc_atoms(obc))]
    d, _ = geo.tree(atoms).query(geo.points(emb), k=1)
    w = eta_cap * np.clip((d - (d_emb - d_cap)) / d_cap, 0, None) ** 2 * (boundary == "cap")
    oe = np.concatenate([np.arange(off[i], off[i + 1]) for i in emb])
    ne = norb[emb]
    eoff = np.concatenate([[0], np.cumsum(ne)])
    pe = np.searchsorted(emb, atoms)
    pidx = np.concatenate([np.arange(eoff[j], eoff[j + 1]) for j in pe])
    He = inp["H"][oe][:, oe].tocsr()
    Se = inp["S"][oe][:, oe].tocsr()
    real = not (np.iscomplexobj(He.data) or np.iscomplexobj(Se.data))
    W = sp.diags(np.repeat(w, ne))
    n, m = He.shape[0], len(pidx)
    r = min(rank or int(np.clip(0.1 * m, 100, 600)), m)
    dense = (He + Se).nnz > dense_fill * n * n and n <= dense_max
    faces = [] if obc is None else obc_faces(obc, oe)

    def sig_emb(z):                                                      # sparse -Sigma on the embedded face orbitals
        if not faces:
            return 0
        SL, SR = obc.sigma(z)
        M = 0
        for (pos, fi), Sg in zip(faces, (SL, SR)):
            if len(pos):
                ii, jj = np.meshgrid(pos, pos, indexing="ij")
                M = M - sp.csr_matrix((Sg[np.ix_(fi, fi)].ravel(), (ii.ravel(), jj.ravel())), shape=(n, n))
        return M
    if boundary == "schur":
        eidx = np.setdiff1d(np.arange(n), pidx)
        HPP, SPP = He[pidx][:, pidx].toarray(), Se[pidx][:, pidx].toarray()
        HPE, SPE = He[pidx][:, eidx], Se[pidx][:, eidx]
        HEP, SEP = He[eidx][:, pidx], Se[eidx][:, pidx]
        HEE, SEE = He[eidx][:, eidx], Se[eidx][:, eidx]

        def gpp(z):                                                      # (A_PP - A_PE A_EE^-1 A_EP)^-1
            B = (z * SEP - HEP).toarray()
            if dense:
                lu = sla.lu_factor((z * SEE - HEE).toarray(), check_finite=False, overwrite_a=True)
                X = sla.lu_solve(lu, B, check_finite=False)
                del lu
            else:
                cs = CSolver(z * SEE - HEE)
                X = cs.solve(B)
                cs.free()
            return np.linalg.inv(z * SPP - HPP - (z * SPE - HPE) @ X)
    if contour is not None and boundary == "cap" and fast is not None:
        assert real, "tile contour: real H, S only (G(z) complex symmetric, rho = -Im int G dz / pi)"
        Om = None
        if sketch:
            Om = (rng.standard_normal((m, min(sketch, m))) if probes == "gauss" else
                  rng.choice([-1.0, 1.0], size=(m, min(sketch, m))))
        acc, info = tile_rho_contour(inp, geo, tree, atoms, contour, d_emb, d_cap, eta_cap, obc, rhs=Om,
                                     dense_fill=dense_fill, dense_max=dense_max, **fast)
        log(f"    tile rho ({fast}): embeddings {info['n_emb']} orb, {info['t']:.0f} s")
        if sketch:
            Y = -acc.imag / np.pi
            d, V = np.linalg.eigh(0.5 * (Om.T @ Y + Y.T @ Om))
            ok = d > 1e-10 * d.max()
            F = Y @ (V[:, ok] / np.sqrt(d[ok]))
            X, sv, _ = np.linalg.svd(F, full_matrices=False)
            lam = sv ** 2
            return X[:, lam > rho_thr], lam, len(emb)
        rho = -acc.imag / np.pi
        lam, X = np.linalg.eigh(0.5 * (rho + rho.T))
        keep = lam > rho_thr
        return X[:, keep][:, ::-1], lam[::-1], len(emb)
    if contour is not None:
        assert real, "tile contour: real H, S only (G(z) complex symmetric, rho = -Im int G dz / pi)"
        mu1, mu2, npts = contour
        c, rr = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
        x, wq = np.polynomial.legendre.leggauss(int(npts))
        th, wq = 0.5 * np.pi * (x + 1), 0.5 * np.pi * wq
        if sketch:
            Om = (rng.standard_normal((m, min(sketch, m))) if probes == "gauss" else
                  rng.choice([-1.0, 1.0], size=(m, min(sketch, m))))
            R = np.zeros((n, Om.shape[1]))
            R[pidx] = Om
        else:
            R = np.zeros((n, m))
            R[pidx, np.arange(m)] = 1.0
        acc = np.zeros((m, R.shape[1]), complex)
        for t_, w_ in zip(th, wq):
            z = c + rr * np.exp(1j * t_)
            if boundary == "schur":
                G = gpp(z) @ R[pidx]
            elif dense:
                lu = sla.lu_factor((z * Se - He + 1j * W + sig_emb(z)).toarray(), check_finite=False, overwrite_a=True)
                G = sla.lu_solve(lu, R, check_finite=False)[pidx]
                del lu
            else:
                cs = CSolver(z * Se - He + 1j * W + sig_emb(z))
                G = cs.solve(R)[pidx]
                cs.free()
            acc += G * (-1j * rr * np.exp(1j * t_) * w_)
        if sketch:
            Y = -acc.imag / np.pi                                         # rho Omega
            d, V = np.linalg.eigh(0.5 * (Om.T @ Y + Y.T @ Om))
            ok = d > 1e-10 * d.max()
            F = Y @ (V[:, ok] / np.sqrt(d[ok]))                             # rho ~ F F^T
            X, sv, _ = np.linalg.svd(F, full_matrices=False)
            lam = sv ** 2
            keep = lam > rho_thr
            if keep.sum() > 0.8 * Om.shape[1]:
                log(f"    tile sketch: {keep.sum()} occupations > {rho_thr:g} with {Om.shape[1]} probes; increase sketch")
            return X[:, keep], lam, len(emb)
        rho = -acc.imag / np.pi
        lam, X = np.linalg.eigh(0.5 * (rho + rho.T))
        keep = lam > rho_thr
        return X[:, keep][:, ::-1], lam[::-1], len(emb)
    Ys = []
    for E in energies:
        E = E + 1j * eta
        if boundary == "schur":
            G = gpp(E)
            lam, X = np.linalg.eigh((1j / (2 * np.pi)) * (G - G.conj().T))
            Ys.append(X * np.sqrt(np.clip(lam, 0, None))[None])
            continue
        if dense:
            lu = sla.lu_factor((E * Se - He + 1j * W).toarray(), check_finite=False, overwrite_a=True)
            cs = csH = None
        else:
            cs = CSolver(E * Se - He + 1j * W)
            csH = None if real else CSolver(np.conj(E) * Se - He - 1j * W)

        def apply_A(X):
            R = np.zeros((n, X.shape[1]), X.dtype)
            R[pidx] = X
            if dense:
                G = sla.lu_solve(lu, R, check_finite=False)
                if real:
                    return -G[pidx].imag / np.pi
                return (1j / (2 * np.pi)) * (G - sla.lu_solve(lu, R, trans=2, check_finite=False))[pidx]
            if real:
                return -cs.solve(R)[pidx].imag / np.pi
            return (1j / (2 * np.pi)) * (cs.solve(R) - csH.solve(R))[pidx]

        X0 = rng.standard_normal((m, r))
        if not real:
            X0 = X0 + 1j * rng.standard_normal((m, r))
        Y = apply_A(X0)
        for _ in range(n_power):
            Y = apply_A(np.linalg.qr(Y)[0])
        Q = np.linalg.qr(Y)[0]
        Bm = Q.conj().T @ apply_A(Q)
        lam, X = np.linalg.eigh(0.5 * (Bm + Bm.conj().T))
        if dense:
            del lu
        else:
            cs.free()
        if csH is not None:
            csH.free()
        Ys.append((Q @ X) * np.sqrt(np.clip(lam, 0, None))[None])
    Uu, s, _ = np.linalg.svd(np.hstack(Ys), full_matrices=False)
    keep = s ** 2 > tile_thr * a_ref
    return Uu[:, keep], s, len(emb)


def obc_atoms(dev):
    """Atoms of the device slabs (cached on dev)."""
    if getattr(dev, "_atoms", None) is None:
        atom_of = np.repeat(np.arange(len(dev.inp["norb"])), dev.inp["norb"])
        dev._atoms = np.unique(atom_of[dev.dev])
    return dev._atoms


def obc_faces(dev, oe):
    """For global orbitals oe: [(positions in oe on the left lead face, their rows in Sigma_L), (same, right)]."""
    if getattr(dev, "_gdev", None) is None:
        dev._gdev = -np.ones(dev.inp["H"].shape[0], int)
        dev._gdev[dev.dev] = np.arange(dev.n)
    dl = dev._gdev[oe]
    out = []
    for b in (dev.bL, dev.bR):
        pos = np.where(np.isin(dl, b))[0]
        out.append((pos, np.searchsorted(b, dl[pos])))
    return out


# ------------------------------------------------------------------ level 1: tile embedding by message passing
class TileBP:
    """Tile Green's functions of the open device by iterated tile self-energies (block Gaussian belief
    propagation / cavity method). A = z S - H - Sigma_L - Sigma_R (exact lead Sigma on the end slabs), tiles
    i with orbital sets t_i (device indices). Round n:
        G_i^(n) = (A_ii - sum_j M_{j->i}^(n-1))^-1,   M^(0) = 0 (closed tiles + leads),
        M_{j->i}^(n) = A_ij g_j^(n, -i) A_ji,
    mode "cavity": g_j^(-i) = Green's function of tile j with tile i removed, on the orbitals b of j coupled to i:
        g_bb = ((G_j)_bb^-1 + (M_{i->j})_bb)^-1  (M_{i->j} lives on b only, so this undoes i's own contribution);
        exact on a tree of tiles (one tile per slab: RGF after ~ns rounds), approximate with loops.
    mode "full": g_j = G_j (the tile's full G, which already contains i's self-energy: double counting).
    Everything is dense per tile; couplings are kept on the coupled orbitals only."""

    def __init__(self, dev, tiles, mode="cavity"):
        assert mode in ("cavity", "full")
        self.dev, self.t, self.mode = dev, [np.asarray(t) for t in tiles], mode
        own = -np.ones(dev.n, int)
        for i, t in enumerate(self.t):
            own[t] = i
        assert (own >= 0).all(), "tiles must cover the device"
        loc = np.zeros(dev.n, int)
        for t in self.t:
            loc[t] = np.arange(len(t))
        self.loc = loc
        P = (abs(dev.H) + abs(dev.S)).tocoo()
        r, c = P.row, P.col
        for b in (dev.bL, dev.bR):                                    # lead Sigma couples all of bL (bR)
            ii, jj = np.meshgrid(b, b, indexing="ij")
            r, c = np.concatenate([r, ii.ravel()]), np.concatenate([c, jj.ravel()])
        oi, oj = own[r], own[c]
        x = oi != oj
        self.b = {}                                                   # (i, j): local orbitals of i coupled to j
        for i, j in set(zip(oi[x].tolist(), oj[x].tolist())):
            sel = (oi == i) & (oj == j)
            self.b[i, j] = np.unique(loc[r[sel]])
        self.nbr = [sorted(j for (i_, j) in self.b if i_ == i) for i in range(len(self.t))]
        H, S = dev.H.tocsr(), dev.S.tocsr()
        self.HD = [H[t][:, t].toarray() for t in self.t]
        self.SD = [S[t][:, t].toarray() for t in self.t]
        self.HC, self.SC = {}, {}                                      # (i, j): on b[i, j] x b[j, i]
        for (i, j), bi in self.b.items():
            bj = self.b[j, i]
            self.HC[i, j] = H[self.t[i][bi]][:, self.t[j][bj]].toarray()
            self.SC[i, j] = S[self.t[i][bi]][:, self.t[j][bj]].toarray()
        self.edges = sorted(self.b)

    def _blocks(self, z, SL, SR):
        dev = self.dev
        AD = [z * S_ - H_ for H_, S_ in zip(self.HD, self.SD)]
        AC = {k: z * self.SC[k] - self.HC[k] for k in self.edges}
        for Sg, bb in ((SL, dev.bL), (SR, dev.bR)):
            tl = [np.isin(self.t[i], bb) for i in range(len(self.t))]
            for i in range(len(self.t)):
                if not tl[i].any():
                    continue
                li = np.where(tl[i])[0]
                pi = np.searchsorted(bb, self.t[i][li])
                AD[i][np.ix_(li, li)] -= Sg[np.ix_(pi, pi)]
                for j in self.nbr[i]:
                    bi, bj = self.b[i, j], self.b[j, i]
                    mi, mj = np.isin(self.t[i][bi], bb), np.isin(self.t[j][bj], bb)
                    if mi.any() and mj.any():
                        AC[i, j][np.ix_(mi, mj)] -= Sg[np.ix_(np.searchsorted(bb, self.t[i][bi][mi]),
                                                               np.searchsorted(bb, self.t[j][bj][mj]))]
        return AD, AC

    def _G(self, AD, M, i):
        D = AD[i].astype(complex)
        for j in self.nbr[i]:
            bi = self.b[i, j]
            D[np.ix_(bi, bi)] -= M[i, j]
        return np.linalg.inv(D)

    def _msg(self, AC, G, M, j, i):
        """M_{j->i} = A_ij g A_ji from G_j; cavity g = G_bb (1 + M_{i->j} G_bb)^-1 on b = b[j, i]."""
        bj = self.b[j, i]
        g = G[j][np.ix_(bj, bj)]
        if self.mode == "cavity":
            g = np.linalg.solve((np.eye(len(bj)) + M[j, i] @ g).T, g.T).T
        return AC[i, j] @ g @ AC[j, i]

    def solve(self, z, SL, SR, maxit=40, tol=1e-8, damp=0.0, keep=(), order="gs", Gref=None):
        """Iterate to max_i |G_i^(n) - G_i^(n-1)| / |G_i^(n)| < tol (or maxit rounds). order "jacobi": all
        messages from the previous round; "gs": sweeps over the tiles (alternating forward / backward), each
        tile's outgoing messages updated right after its G. Returns (G, hist, conv[, err]): final tile G's,
        {n: [G_i^(n) for i in keep]}, the change per round, and with Gref (exact tile blocks) the max relative
        error per round."""
        AD, AC = self._blocks(z, SL, SR)
        nt = len(self.t)
        M = {k: np.zeros((len(self.b[k]),) * 2, complex) for k in self.edges}   # M[i, j] = M_{j->i} on b[i, j]
        G = [self._G(AD, M, i) for i in range(nt)]                  # round 0: closed tiles + leads
        hist, conv, err = {0: [G[i] for i in keep]}, [], []
        rel = lambda G: max(np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(G, Gref))
        if Gref is not None:
            err.append(rel(G))
        for n in range(1, maxit + 1):
            G0 = list(G)
            if order == "jacobi":
                Mn = {(i, j): self._msg(AC, G, M, j, i) for (i, j) in self.edges}
                M = Mn if damp == 0 else {k: (1 - damp) * Mn[k] + damp * M[k] for k in self.edges}
                G = [self._G(AD, M, i) for i in range(nt)]
            else:
                for j in (range(nt) if n % 2 else range(nt - 1, -1, -1)):
                    G[j] = self._G(AD, M, j)
                    for i in self.nbr[j]:
                        Mn = self._msg(AC, G, M, j, i)
                        M[i, j] = Mn if damp == 0 else (1 - damp) * Mn + damp * M[i, j]
            hist[n] = [G[i] for i in keep]
            conv.append(max(np.linalg.norm(a - b) / np.linalg.norm(a) for a, b in zip(G, G0)))
            if Gref is not None:
                err.append(rel(G))
            if conv[-1] < tol:
                break
        return (G, hist, conv) if Gref is None else (G, hist, conv, err)


# ------------------------------------------------------------------ level 1: assembly
def slab_U(dev, blocks, bases):
    """Per slab: sparse U1_s (slab-local orbitals x block modes), and the owning block of each column."""
    Us, own = [], []
    for s in range(len(dev.ns)):
        bl = [i for i, b in enumerate(blocks) if b["slab"] == s]
        rows, cols, vals, c0 = [], [], [], 0
        for i in bl:
            Q = bases[i]
            R, C = np.meshgrid(blocks[i]["orbs"], np.arange(c0, c0 + Q.shape[1]), indexing="ij")
            rows.append(R.ravel()); cols.append(C.ravel()); vals.append(Q.ravel())
            own.append(np.full(Q.shape[1], i)); c0 += Q.shape[1]
        Us.append(sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(dev.ns[s], c0)))
    return Us, np.concatenate(own)


def slab_blocks(dev, Us):
    """Level-1 slab blocks: HD, SD (s, s), HU, SU (s, s+1) of U^T H U, U^T S U."""
    b = dev.sb
    nsl = len(Us)
    Us = [u.toarray() if sp.issparse(u) else np.asarray(u) for u in Us]    # sparse H times dense U: fast
    out = dict(HD=[], SD=[], HU=[], SU=[])
    for M, kD, kU in ((dev.H, "HD", "HU"), (dev.S, "SD", "SU")):
        for s in range(nsl):
            X = Us[s].conj().T @ (M[b[s]:b[s + 1], b[s]:b[s + 1]] @ Us[s])
            out[kD].append(0.5 * (X + X.conj().T))
            if s < nsl - 1:
                out[kU].append(Us[s].conj().T @ (M[b[s]:b[s + 1], b[s + 1]:b[s + 2]] @ Us[s + 1]))
    return out


# ------------------------------------------------------------------ slab bases: transport residual enrichment
def _dense_or_sparse_inv(A, dense_fill=0.02):
    """Solver for a sub-block of the fine operator: dense LU if > dense_fill filled, else CSolver."""
    n = A.shape[0]
    if A.nnz > dense_fill * n * n:
        lu = sla.lu_factor(A.toarray(), check_finite=False, overwrite_a=True)
        return lambda R: sla.lu_solve(lu, R, check_finite=False), lambda: None
    cs = CSolver(A)
    return cs.solve, cs.free


def residual_enrich(dev, Us, energies, sigma, corr="cluster", width=1, tol=1e-2, nadd=40, inj_tol=1e-4, log=log,
                    rhs="inject", nprobe=20, rng=None, weights=None):
    """One round of enrichment of the slab bases Us (slab-local orbitals x modes, orthonormal columns) from the
    residual of the original open system. Per training energy E (real, or complex contour points):
        A = E S - H - Sigma_L - Sigma_R,  psi = U A_c^-1 U^H b  (mode space, A_c = U^H A U),
        rhs "inject": b = QTBM injection columns (the states transport sees);
        rhs "random": b = nprobe Gaussian probes over all device orbitals, so e below samples the range of the
          mode-space error operator (I - pi) A^-1 (randomized range finder of what U misses at E, e.g. oxide
          traps that the injection hardly excites),
        weights = contour weights dz (rhs "random", real H and S): the same probes at every point and only the
          integrated correction -Im sum_z e(z) dz / pi = (rho - rho_MS) omega is used, i.e. the part of the window
          density matrix that U misses (states outside the window, which dominate (I - pi) A^-1 omega at a single
          point, are filtered out),
        r = b - A psi  (U^H r = 0),  e ~ A^-1 r = psi_exact - psi  by
          corr "raw"     : e = r
          corr "slab"    : e_s = A_ss^-1 r_s                       (block Jacobi)
          corr "cluster" : e_s = [A_KK^-1 r_K]_s, K = slabs s-w..s+w  (overlapping Schwarz, closed at the cluster ends)
          corr "exact"   : e = A^-1 r                               (psi_exact enters the basis at E); dense
                           block-tridiagonal LU over the fine slabs when every slab has <= 12000 orbitals,
                           else a sparse direct solve
    The slab blocks e_s / |psi| (all channels, all energies) are orthogonalized against U_s; the left singular
    vectors with singular value > tol (relative error in psi) are appended, at most nadd per slab.
    Returns (added per slab, stats); stats per energy: |r| / |b|, |e| / |psi|."""
    nsl, sb = len(Us), dev.sb
    U = sp.block_diag(Us, format="csr")
    blk = slab_blocks(dev, Us)
    PL, PR = Us[0][dev.bL].toarray(), Us[-1][dev.bR - sb[-2]].toarray()
    cols = [[] for _ in range(nsl)]
    st = dict(E=[], res=[], err=[], t=0.0)
    t0 = time.time()
    rng = np.random.default_rng(0) if rng is None else rng
    if weights is not None:
        assert rhs == "random"
        b0 = rng.standard_normal((dev.n, nprobe)).astype(complex)
        eacc = np.zeros((dev.n, nprobe), complex)
        pacc = np.zeros((dev.n, nprobe), complex)
    fine = None
    if corr == "exact" and max(dev.ns) <= 12000:
        fine = slab_blocks(dev, [sp.identity(n, format="csr") for n in dev.ns])
    for ip, E in enumerate(energies):
        SL, SR = sigma(E)
        if weights is not None:
            b = b0
        elif rhs == "inject":
            B = dev.inject(SL, inj_tol)
            if B.shape[1] == 0:
                continue
            b = np.zeros((dev.n, B.shape[1]), complex)
            b[dev.bL] = B
        else:
            b = rng.standard_normal((dev.n, nprobe)).astype(complex)
        A = dev.A(E, SL, SR)
        Ac = BlockTridiag(blk, E, PL.conj().T @ SL @ PL, PR.conj().T @ SR @ PR)
        psi = U @ Ac.solve(U.conj().T @ b)
        r = b - A @ psi
        if corr == "raw":
            e = r
        elif corr == "exact" and fine is not None:
            SL1 = np.zeros((dev.ns[0],) * 2, complex)
            SL1[np.ix_(dev.bL, dev.bL)] = SL
            SR1 = np.zeros((dev.ns[-1],) * 2, complex)
            SR1[np.ix_(dev.bR - sb[-2], dev.bR - sb[-2])] = SR
            e = BlockTridiag(fine, E, SL1, SR1).solve(r)
        elif corr == "exact":
            cs = CSolver(A)
            e = cs.solve(r)
            cs.free()
        else:
            w = 0 if corr == "slab" else width
            e = np.zeros_like(r)
            for s_ in range(nsl):
                lo, hi = max(0, s_ - w), min(nsl, s_ + w + 1)
                K = slice(sb[lo], sb[hi])
                solve, free = _dense_or_sparse_inv(A[K, K].tocsr())
                eK = solve(r[K])
                free()
                e[sb[s_]:sb[s_ + 1]] = eK[sb[s_] - sb[lo]:sb[s_ + 1] - sb[lo]]
        npsi = np.linalg.norm(psi)
        if weights is not None:
            eacc += e * weights[ip]
            pacc += psi * weights[ip]
        st["E"].append([float(np.real(E)), float(np.imag(E))]); st["res"].append(float(np.linalg.norm(r) / np.linalg.norm(b)))
        st["err"].append(float(np.linalg.norm(e) / npsi))
        for s_ in range(nsl):
            if weights is None:
                cols[s_].append(e[sb[s_]:sb[s_ + 1]] / npsi)
        log(f"    residual E {E:.4f}: {b.shape[1]} {'channels' if rhs == 'inject' else 'probes'}, |r|/|b| {st['res'][-1]:.2e}, |e|/|psi| {st['err'][-1]:.2e}, "
            f"{time.time() - t0:.0f} s")
    if weights is not None:
        de, rms = -eacc.imag / np.pi, -pacc.imag / np.pi               # (rho - rho_MS) omega, rho_MS omega
        st["rho_err"] = float(np.linalg.norm(de) / np.linalg.norm(de + rms))
        log(f"    integrated: |(rho - rho_MS) w| / |rho w| {st['rho_err']:.2e}")
        nr = np.linalg.norm(de + rms)
        for s_ in range(nsl):
            cols[s_].append(de[sb[s_]:sb[s_ + 1]] / nr)
    added = []
    for s_ in range(nsl):
        Y = np.hstack(cols[s_])
        Q = Us[s_]
        for _ in range(2):
            Y = Y - Q @ (Q.conj().T @ Y)
        u, sv, _ = np.linalg.svd(Y, full_matrices=False)
        k = min(int(np.sum(sv > tol)), nadd)
        added.append(u[:, :k])
        st.setdefault("sv", []).append(sv[:max(k, 5)].tolist())
    st["t"] = time.time() - t0
    return added, st


# ------------------------------------------------------------------ level 1: residual enrichment
def cluster_dense(blk, sl):
    """Dense H1, S1 of the consecutive slabs sl from the slab blocks."""
    o = np.concatenate([[0], np.cumsum([blk["HD"][s].shape[0] for s in sl])])
    dt = np.result_type(*blk["HD"], *blk["SD"], *(blk["HU"] or [0.0]), *(blk["SU"] or [0.0]))
    H, S = np.zeros((o[-1], o[-1]), dt), np.zeros((o[-1], o[-1]), dt)
    for i, s in enumerate(sl):
        a, b = slice(o[i], o[i + 1]), slice(o[i + 1], o[i + 2]) if i + 1 < len(sl) else None
        H[a, a], S[a, a] = blk["HD"][s], blk["SD"][s]
        if b is not None:
            H[a, b], S[a, b] = blk["HU"][s], blk["SU"][s]
            H[b, a], S[b, a] = blk["HU"][s].conj().T, blk["SU"][s].conj().T
    return H, S, o


def block_pairs(dev, blocks, bases, tol=0.0):
    """Fine and level-1 couplings of every coupled block pair i <= j (same or adjacent slab, max |H_ij| + |S_ij|
    > tol): {(i, j): (H_ij, S_ij, Q_i^H H_ij Q_j, Q_i^H S_ij Q_j)}, H_ij, S_ij sparse in device orbitals."""
    g = [dev.sb[b["slab"]] + b["orbs"] for b in blocks]
    P = (abs(dev.H) + abs(dev.S)).tocsr()
    out = {}
    for i in range(len(blocks)):
        Pi, Hi, Si = P[g[i]].tocsc(), dev.H[g[i]].tocsc(), dev.S[g[i]].tocsc()
        for j in range(i, len(blocks)):
            if abs(blocks[i]["slab"] - blocks[j]["slab"]) > 1:
                continue
            Pij = Pi[:, g[j]]
            if Pij.nnz == 0 or Pij.max() <= tol:
                continue
            Hij, Sij = Hi[:, g[j]].tocsr(), Si[:, g[j]].tocsr()
            Qi, Qj = bases[i], bases[j]
            out[i, j] = (Hij, Sij, Qi.conj().T @ (Hij @ Qj), Qi.conj().T @ (Sij @ Qj))
    return out


def enrich_clusters(dev, blocks, pairs, mode="slab", width=1):
    """[(cluster blocks, centre blocks)]. slab: slabs s-width..s+width around the blocks of slab s
    (width < 0: one cluster = the device, every block centre); tile: block i and every block coupled to it."""
    nsl, nb = len(dev.ns), len(blocks)
    if mode == "tile":
        nbr = [{i} for i in range(nb)]
        for i, j in pairs:
            nbr[i].add(j); nbr[j].add(i)
        return [(sorted(nbr[i]), [i]) for i in range(nb)]
    if width < 0:
        return [(list(range(nb)), list(range(nb)))]
    sl = [b["slab"] for b in blocks]
    return [([i for i in range(nb) if abs(sl[i] - s) <= width], [i for i in range(nb) if sl[i] == s])
            for s in range(nsl)]


def enrich_round(dev, blocks, bases, ec, lwin=(-0.45, 0.40), rthr=1.0, smin=0.1, width=1, mode="slab", tol=0.0,
                 slabs=None):
    """One round of residual enrichment of the level-1 block bases (qtmode.enrich_round, generalized).
    Per cluster C (enrich_clusters: mode "slab" = slabs s-width..s+width around slab s, "tile" = a block and
    the blocks coupled to it, couplings <= tol ignored), closed in the level-1 basis: Ritz pairs
    H1_C c = lam S1_C c with lam - ec in lwin, x = U1_C c (|x| = |c| = 1). Only the residual on the centre
    blocks is used,
        r_i = sum_{j in C} (H_ij - lam S_ij) Q_j c_j,
    because the outer cluster blocks carry the artificial cluster cut (closed-end states, the missing coupling
    to blocks outside C), while the cuts around the centre are interior to C (all its couplings in C for
    "tile" with tol = 0, and for "slab" with width >= 1). Pairs with |r_centre| > rthr [eV] are flagged; per
    centre block the part of r_i (flagged columns) orthogonal to the block basis is compressed by SVD
    (keep s > smin [eV]) and appended. width < 0: one global cluster (H1 Ritz pairs of the device, as qtmode).
    slabs: enrich only centre blocks in these slabs (e.g. the interior slabs, when the end slabs are cleaned
    against the lead bands instead). Returns ({block index: added orthonormal columns}, stats); `bases` is not
    modified."""
    t0 = time.time()
    pairs = block_pairs(dev, blocks, bases, tol)
    clusters = enrich_clusters(dev, blocks, pairs, mode, width)
    if slabs is not None:
        clusters = [(cb, [i for i in c if blocks[i]["slab"] in slabs]) for cb, c in clusters]
        clusters = [(cb, c) for cb, c in clusters if c]
    R = {}                                          # centre block -> flagged residual columns (block orbitals)
    n_win, n_spur, rn_all, sizes = 0, 0, [], []
    for cb, centre in clusters:
        o = np.concatenate([[0], np.cumsum([bases[i].shape[1] for i in cb])])
        pos = {i: k for k, i in enumerate(cb)}
        dt = np.result_type(*(bases[i] for i in cb), *(p[2] for p in pairs.values()))
        H1, S1 = np.zeros((o[-1], o[-1]), dt), np.zeros((o[-1], o[-1]), dt)
        for (i, j), (_, _, h, s_) in pairs.items():
            if i in pos and j in pos:
                a, b = slice(o[pos[i]], o[pos[i] + 1]), slice(o[pos[j]], o[pos[j] + 1])
                H1[a, b], S1[a, b] = h, s_
                if i != j:
                    H1[b, a], S1[b, a] = h.conj().T, s_.conj().T
        H1, S1 = 0.5 * (H1 + H1.conj().T), 0.5 * (S1 + S1.conj().T)
        sizes.append(len(H1))
        lam, C = sla.eigh(H1, S1, subset_by_value=(ec + lwin[0], ec + lwin[1]), check_finite=False)
        if not len(lam):
            continue
        C = C / np.linalg.norm(C, axis=0)
        X = {j: bases[j] @ C[o[k]:o[k + 1]] for k, j in enumerate(cb)}
        Rc = {}
        for i in centre:
            r = 0
            for j in cb:
                key, tr = ((i, j), False) if (i, j) in pairs else ((j, i), True) if (j, i) in pairs else (None, 0)
                if key is None:
                    continue
                Hij, Sij = pairs[key][:2]
                if tr:
                    Hij, Sij = Hij.conj().T, Sij.conj().T
                r = r + Hij @ X[j] - (Sij @ X[j]) * lam
            Rc[i] = r
        rn = np.sqrt(sum(np.linalg.norm(r, axis=0) ** 2 for r in Rc.values()))
        spur = rn > rthr
        n_win += len(lam); n_spur += int(spur.sum()); rn_all.append(rn)
        for i in centre:
            R[i] = Rc[i][:, spur]
    rn_all = np.concatenate(rn_all) if rn_all else np.zeros(0)
    added = {}
    for i, Rb in R.items():
        if not Rb.shape[1]:
            continue
        Q = bases[i]
        for _ in range(2):
            Rb = Rb - Q @ (Q.conj().T @ Rb)
        Ub, sv, _ = np.linalg.svd(Rb, full_matrices=False)
        keep = sv > smin
        if keep.any():
            Z = Ub[:, keep]
            Z = Z - Q @ (Q.conj().T @ Z)
            added[i] = np.linalg.qr(Z)[0]
    stats = dict(N1=int(sum(q.shape[1] for q in bases)), n_clusters=len(clusters),
                 n_cluster_max=int(max(sizes)), n_cluster_mean=float(np.mean(sizes)),
                 cluster_blocks=[len(c) for c, _ in clusters], n_window=int(n_win), n_spurious=n_spur,
                 added=int(sum(z.shape[1] for z in added.values())),
                 res_hist=np.histogram(rn_all, bins=[0, 0.1, 0.3, 0.5, 1, 2, 3, 5, 100])[0].tolist(),
                 t=time.time() - t0)
    return added, stats


def project(blk, V):
    """Blocks of V^H (blk) V for block-diagonal V = diag(V_s)."""
    n = len(V)
    return dict(HD=[V[s].conj().T @ blk["HD"][s] @ V[s] for s in range(n)],
                SD=[V[s].conj().T @ blk["SD"][s] @ V[s] for s in range(n)],
                HU=[V[s].conj().T @ blk["HU"][s] @ V[s + 1] for s in range(n - 1)],
                SU=[V[s].conj().T @ blk["SU"][s] @ V[s + 1] for s in range(n - 1)])


def rgf_blocks(blk, z, SL1, SR1, width=1):
    """RGF of A_c(z) = z S_c - H_c - P^H Sigma P (end slabs): blocks G[(i, j)] of A_c^-1 for |i - j| <= width.
    Left-connected g_s = (A_ss - A_s,s-1 g_s-1 A_s-1,s)^-1, then G_ss = g_s + g_s A_s,s+1 G_s+1,s+1 A_s+1,s g_s,
    G_ij = -g_i A_i,i+1 G_i+1,j and G_ji = -G_j,i+1 A_i+1,i g_i (i < j)."""
    n = len(blk["HD"])
    D = [z * blk["SD"][s] - blk["HD"][s] for s in range(n)]
    D[0] = D[0] - SL1
    D[-1] = D[-1] - SR1
    Up = [z * blk["SU"][s] - blk["HU"][s] for s in range(n - 1)]
    Lo = [z * blk["SU"][s].conj().T - blk["HU"][s].conj().T for s in range(n - 1)]
    g = [None] * n
    for s in range(n):
        g[s] = np.linalg.inv(D[s] if s == 0 else D[s] - Lo[s - 1] @ g[s - 1] @ Up[s - 1])
    G = {(n - 1, n - 1): g[-1]}
    for s in range(n - 2, -1, -1):
        gU, Lg = g[s] @ Up[s], Lo[s] @ g[s]
        for j in range(s + 1, min(s + width, n - 1) + 1):
            G[(s, j)] = -gU @ G[(s + 1, j)]
            G[(j, s)] = -G[(j, s + 1)] @ Lg
        G[(s, s)] = g[s] - G[(s, s + 1)] @ Lg
    return G


class BlockTridiag:
    """LU of the block-tridiagonal coarse operator A_c(z) = z S_c - H_c - P^H Sigma P (end slabs)."""

    def __init__(self, blk, z, SL1, SR1):
        n = len(blk["HD"])
        D = [z * blk["SD"][s] - blk["HD"][s] for s in range(n)]
        D[0] = D[0] - SL1
        D[-1] = D[-1] - SR1
        self.Up = [z * blk["SU"][s] - blk["HU"][s] for s in range(n - 1)]
        self.Lo = [z * blk["SU"][s].conj().T - blk["HU"][s].conj().T for s in range(n - 1)]
        self.lu, self.L = [], []
        Dt = D[0]
        for s in range(n):
            self.lu.append(sla.lu_factor(Dt, check_finite=False))
            if s < n - 1:
                Lt = sla.lu_solve(self.lu[s], self.Lo[s].T, trans=1, check_finite=False).T   # Lo_s Dt_s^-1
                self.L.append(Lt)
                Dt = D[s + 1] - Lt @ self.Up[s]
        self.sizes = [len(d) for d in D]
        self.bounds = np.concatenate([[0], np.cumsum(self.sizes)])

    def solve(self, B):
        b = self.bounds
        n = len(self.sizes)
        Y = [B[b[s]:b[s + 1]].astype(complex) for s in range(n)]
        for s in range(1, n):
            Y[s] = Y[s] - self.L[s - 1] @ Y[s - 1]
        X = [None] * n
        X[-1] = sla.lu_solve(self.lu[-1], Y[-1], check_finite=False)
        for s in range(n - 2, -1, -1):
            X[s] = sla.lu_solve(self.lu[s], Y[s] - self.Up[s] @ X[s + 1], check_finite=False)
        return np.concatenate(X)


def block_solve(blk, z, SL1, SR1, B, host_frac=0.4):
    """X = A_c(z)^-1 B for the block-tridiagonal A_c(z) = z S_c - H_c - SL1 (first block) - SR1 (last block) of
    slab_blocks, in one forward / backward sweep with the right-hand sides carried along (block Thomas; no factors
    kept, so one call per energy):
        forward   C_s = D~_s^-1 U_s,  Z_s = D~_s^-1 Y~_s,  D~_{s+1} = D_{s+1} - L_s C_s,  Y~_{s+1} = B_{s+1} - L_s Z_s
        backward  X_s = Z_s - C_s X_{s+1}
    With gpu.on the blocks are copied to the device one slab at a time (the mode-space solve on the GPU: N 82 557 on
    6 nm in seconds instead of ~2 min with numpy); the C_s stay on the device when they take < host_frac of the free
    memory, else they go to the host. Peak device memory otherwise ~6 dense blocks. Without the GPU: numpy, same
    sweep. B and X are numpy arrays (N x columns). Ported from the release (qtx_modespace.core.block_solve,
    --ms-solve gpu), 2026-11-17."""
    n = len(blk["HD"])
    xp = gpu.cp if gpu.on else np
    put = (lambda M: gpu.cp.asarray(M)) if gpu.on else (lambda M: M)
    sz = [len(d) for d in blk["HD"]]
    o = np.concatenate([[0], np.cumsum(sz)])
    keep = True
    if gpu.on:
        need = 16 * sum(sz[s] * sz[s + 1] for s in range(n - 1))
        keep = need < host_frac * gpu.cp.cuda.Device().mem_info[0]
    B = np.asarray(B)
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
    X = gpu.asnumpy(xp.concatenate(X))
    gpu.free()
    return X


class MRCoarse:
    """Cholesky of W^H W, W = A U with U = diag(U_s) over slabs (minimal-residual coarse correction, used with
    twolevel.TwoLevel(restrict='AU')). W_s = A[:, slab s] U_s lives on slabs s-1..s+1, so W^H W is block
    pentadiagonal; it is assembled densely from the dense W_s."""

    def __init__(self, dev, A, Us):
        b, nsl = dev.sb, len(Us)
        A = sp.csc_matrix(A)
        W, rng = [], []
        for s in range(nsl):
            lo, hi = b[max(s - 1, 0)], b[min(s + 2, nsl)]
            W.append(np.asarray((A[lo:hi, b[s]:b[s + 1]] @ Us[s]).todense()))
            rng.append((lo, hi))
        off = np.concatenate([[0], np.cumsum([u.shape[1] for u in Us])])
        G = np.zeros((off[-1], off[-1]), complex)
        for s in range(nsl):
            for t in range(s, min(s + 3, nsl)):
                lo, hi = max(rng[s][0], rng[t][0]), min(rng[s][1], rng[t][1])
                if lo >= hi:
                    continue
                X = W[s][lo - rng[s][0]:hi - rng[s][0]].conj().T @ W[t][lo - rng[t][0]:hi - rng[t][0]]
                G[off[s]:off[s + 1], off[t]:off[t + 1]] = X
                G[off[t]:off[t + 1], off[s]:off[s + 1]] = X.conj().T
        del W
        self.cho = sla.cho_factor(G, check_finite=False)

    def solve(self, B):
        return sla.cho_solve(self.cho, B, check_finite=False)


# ------------------------------------------------------------------ level 2: contour slab density matrix
def rho_contour(dev, blk, PL, PR, mu1, mu2, npts, log=log):
    """rho_ss = -(1/pi) Im int_arc G1_ss dz over the upper arc from mu1 to mu2 (Gauss-Legendre in theta).
    G_NN = g^L_N, G_ss = g^L_s + g^L_s A_{s,s+1} G_{s+1,s+1} A_{s+1,s} g^L_s (left-connected g^L)."""
    ns = len(blk["HD"])
    c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    x, w = np.polynomial.legendre.leggauss(npts)
    th, w = 0.5 * np.pi * (x + 1), 0.5 * np.pi * w
    acc = [np.zeros(h.shape, complex) for h in blk["HD"]]
    xp = gpu.cp if gpu.on else np                 # GPU: blocks and recursion on the device, acc on the host
    if gpu.on:
        blk = {k: [xp.asarray(M) for M in L] for k, L in blk.items()}
        PL, PR = xp.asarray(PL), xp.asarray(PR)
    t0 = time.time()
    for ip, (t_, w_) in enumerate(zip(th, w)):
        z = c + r * np.exp(1j * t_)
        dz = -1j * r * np.exp(1j * t_) * w_
        SL, SR = (xp.asarray(x_) for x_ in dev.sigma(z))
        D = [z * blk["SD"][s] - blk["HD"][s] for s in range(ns)]
        D[0] = D[0] - PL.conj().T @ SL @ PL
        D[-1] = D[-1] - PR.conj().T @ SR @ PR
        Up = [z * blk["SU"][s] - blk["HU"][s] for s in range(ns - 1)]
        Lo = [z * blk["SU"][s].conj().T - blk["HU"][s].conj().T for s in range(ns - 1)]
        gl = [None] * ns
        for s in range(ns):
            gl[s] = xp.linalg.inv(D[s] if s == 0 else D[s] - Lo[s - 1] @ gl[s - 1] @ Up[s - 1])
            D[s] = None
        G = gl[-1]
        acc[-1] += gpu.asnumpy(G) * dz
        for s in range(ns - 2, -1, -1):
            G = gl[s] + gl[s] @ Up[s] @ G @ Lo[s] @ gl[s]
            gl[s + 1] = None
            acc[s] += gpu.asnumpy(G) * dz
        if ip % 6 == 5:
            log(f"    contour point {ip + 1}/{npts}, {time.time() - t0:.0f} s")
    del blk
    gpu.free()
    return [-((a - a.conj().T) / 2j) / np.pi for a in acc]


def level2_basis(rho, thr=1e-5, S=None):
    """Per slab: eigenvectors of rho_ss with lambda > thr * lambda_max (largest first).
    S (list of slab overlaps S_ss): eigenproblem of L^H rho L (S = L L^H, eigenvalues = occupations in the
    S metric), vectors mapped back by L^-H and orthonormalized."""
    V = []
    for s, R in enumerate(rho):
        R = 0.5 * (R + R.conj().T)
        if S is None:
            lam, X = np.linalg.eigh(R)
            V.append(X[:, lam > thr * lam[-1]][:, ::-1])
            continue
        Lc = np.linalg.cholesky(0.5 * (S[s] + S[s].conj().T))
        lam, Y = np.linalg.eigh(Lc.conj().T @ R @ Lc)
        X = sla.solve_triangular(Lc.conj().T, Y[:, lam > thr * lam[-1]][:, ::-1], lower=False)
        V.append(np.linalg.qr(X)[0])
    return V


# ------------------------------------------------------------------ exact solve on the GPU
class StreamBlockTridiag:
    """GPU block-tridiagonal LU of a sparse device matrix for devices whose dense blocks do not fit on the GPU
    (ported from the release, qtx_modespace.core.StreamBlockTridiag, 2026-11-17): the diagonal blocks are densified
    one at a time, the couplings stay sparse on the device, and the LU factor of every Schur-complemented diagonal
    block is moved to pinned host memory (copied back per solve at bus speed; pageable copies ran at 4.4 GB/s on
    GH200 and made the 6 nm exact solve copy-bound). Peak GPU memory ~5 dense blocks; host memory one dense
    block per slab. solve(B) takes and returns cupy arrays."""

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


def exact_solve(dev, A, B, n_ref=2, tol=1e-12, stream=None):
    """A X = B for the sparse device matrix A (numpy in / out), returns (X, max relative residual).
    CPU: CSolver. GPU (gpu.on): dense block-tridiagonal LU over the slabs on the device (left-to-right elimination,
    Dt_{s+1} = D_{s+1} - A_{s+1,s} Dt_s^-1 A_{s,s+1}) + n_ref refinement steps with the sparse A on the device.
    stream: None = StreamBlockTridiag (LU factors on the host) when the dense blocks need more than 70% of the free
    GPU memory (5 sum n_s^2 x 16 B, as the release), True / False = force it / the dense block LU."""
    if not gpu.on:
        cs = CSolver(A, n_ref=n_ref, tol=tol)
        X = cs.solve(B)
        cs.free()
        return X, cs.res
    cp, b, ns = gpu.cp, dev.sb, len(dev.ns)
    A = sp.csr_matrix(A)
    if stream is None:
        need = 16 * 5 * sum(int(n) ** 2 for n in dev.ns)      # dense D, Up, Lo + LU and L factors
        stream = need > 0.7 * cp.cuda.Device().mem_info[0]
        if stream:
            log(f"  exact_solve: dense block LU needs ~{need / 2**30:.0f} GB, streaming variant (LU factors on the host)")
    lu, L, Up, Dt = [], [], [], None
    if stream:
        solve = StreamBlockTridiag(dev, A).solve
    else:
        blk = lambda i, j: gpu.csr_dense(A[b[i]:b[i + 1], b[j]:b[j + 1]], np.complex128)
        Up = [blk(s, s + 1) for s in range(ns - 1)]
        Dt = blk(0, 0)
        for s in range(ns):
            lu.append(gpu.lu_factor(Dt))
            if s < ns - 1:
                Lt = gpu.lu_solve(lu[s], blk(s + 1, s).T, trans=1).T          # A_{s+1,s} Dt_s^-1
                L.append(Lt)
                Dt = blk(s + 1, s + 1) - Lt @ Up[s]

        def solve(R):
            Y = [R[b[s]:b[s + 1]] for s in range(ns)]
            for s in range(1, ns):
                Y[s] = Y[s] - L[s - 1] @ Y[s - 1]
            X = [None] * ns
            X[-1] = gpu.lu_solve(lu[-1], Y[-1])
            for s in range(ns - 2, -1, -1):
                X[s] = gpu.lu_solve(lu[s], Y[s] - Up[s] @ X[s + 1])
            return cp.concatenate(X)
    Ad = gpu.csr(A, np.complex128)
    Bd = cp.asarray(np.asarray(B).reshape(len(B), -1), dtype=complex)
    X = solve(Bd)
    for _ in range(n_ref):
        X += solve(Bd - Ad @ X)
    bn = cp.maximum(cp.linalg.norm(Bd, axis=0), 1e-300)
    res = float((cp.linalg.norm(Bd - Ad @ X, axis=0) / bn).max()) if Bd.shape[1] else 0.0
    X = gpu.asnumpy(X)
    del lu, L, Up, Ad, Bd, Dt
    gpu.free()
    return (X.reshape(np.shape(B)) if np.ndim(B) == 1 else X), res
