"""Two-level mode-space basis (the qtmode method) built from the generic input set (mgms.prep.inputs).

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
Level 2 (block modes -> slab modes): contour-integrated slab density matrix of the open level-1 system
  rho_ss = -(1/pi) Im int_arc G1_ss(z) dz,  G1 = (z S1 - H1 - Sigma1(z))^-1,  arc over [mu1, mu2],
  eigenvectors with lambda > thr * lambda_max.  U = U1 U2 (block diagonal over slabs).
Leads: exact Sigma(z) at complex z, lead interior folded onto its faces, then decimation of the folded chain
(general: complex Hermitian H, any S; fast path for real H, S = I: qtmode face decimation on the smaller face).
"""
import os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
from scipy.spatial import cKDTree
try:
    import pypardiso
except ImportError:          # pragma: no cover
    pypardiso = None

from qtmode.lead_ms import face_surface_gfs_small


def log(*a):
    print(*a, flush=True)


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
class Device:
    """Device part of a generic input set: slabs = device blocks, lead coupling, exact Sigma(z)."""

    def __init__(self, inp, fast_lead=None):
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
        oL = O[tr["lead_left_block"]][self.lead.oL]                # lead-cell orbitals carrying g_L, g_R
        oR = O[tr["lead_right_block"]][self.lead.oR]
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

    sigma_cache = None              # directory: Sigma(z) is read from / written to sig_<Re z>_<Im z>.npz there

    def sigma(self, z):
        """Sigma_L on bL, Sigma_R on bR (device indices) at complex z: A_dl g_L A_ld, A_dr g_R A_rd."""
        z = complex(z)
        if self.sigma_cache:
            f = os.path.join(self.sigma_cache, f"sig_{z.real:+.8f}_{z.imag:.8e}.npz")
            if os.path.exists(f):
                c = np.load(f)
                return c["SL"], c["SR"]
        out = self._sigma(z)
        if self.sigma_cache:
            os.makedirs(self.sigma_cache, exist_ok=True)
            np.savez(f, SL=out[0], SR=out[1])
        return out

    def _sigma(self, z):
        gL, gR = self.lead.surface_gfs(z)
        out = []
        for g, (Hld, Sld, Hdl, Sdl) in ((gL, self.cL), (gR, self.cR)):
            out.append((z * Sdl - Hdl) @ g @ (z * Sld - Hld))
        return tuple(out)

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
    last cell), both from one decimation."""
    e = A00.astype(complex)
    esR, esL = e.copy(), e.copy()
    a, b = A01.astype(complex), A10.astype(complex)
    n = len(e)
    for it in range(maxit):
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
    """Periodic lead cell: H00, S00 and the coupling H01, S01 (cell n -> n+1 along +x).
    Faces F = orbitals of H01 / S01 (rows Rf, columns Lf); the interior I couples only inside its cell
    and is folded onto F at complex z:  A_eff(z) = A_FF - A_FI A_II^-1 A_IF,  A = z S - H.
      general (complex Hermitian H, any S): Sancho decimation of the folded chain on F with
          A01 = z S01 - H01, A10 = z S01^H - H01^H;  g_L, g_R on F (oL = oR = F).
      real symmetric H, S = I (fast): qtmode.lead_ms.face_surface_gfs_small, chain on the smaller face;
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


def partition(dev, cores=True, core_label="Si", d_core=6.0, tile_orbs=2500, axis=0):
    """Blocks per slab: dict(kind, slab, atoms (global atom ids, sorted), orbs (slab-local orbitals))."""
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
        nt = max(1, int(round(norb[rest].sum() / tile_orbs)))
        for t in bisect(rest, xyz[:, tr], norb, nt):
            blocks.append(dict(kind="tile", slab=s, atoms=np.sort(t)))
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
class Geometry:
    """Distance queries on atom positions, minimum image along a periodic transverse axis
    (meta["periodic"] = {"axis": k, "length": L}, e.g. a 2D sheet folded at k_y = 0)."""

    def __init__(self, xyz, periodic=None):
        self.xyz = np.asarray(xyz, float)
        self.box = None
        if periodic:
            lo = self.xyz.min(0) - 1.0
            self.lo = lo
            self.box = np.full(3, 2 * np.ptp(self.xyz, axis=0).max() + 1e3)
            self.box[periodic["axis"]] = periodic["length"]
            self.ax, self.L = periodic["axis"], periodic["length"]

    def _map(self, r):
        if self.box is None:
            return r
        q = r - self.lo
        q[:, self.ax] %= self.L
        return q

    def tree(self, idx=None):
        r = self.xyz if idx is None else self.xyz[idx]
        return cKDTree(self._map(r), boxsize=self.box)

    def points(self, idx):
        return self._map(self.xyz[idx])


def tile_modes(inp, geo, tree, atoms, energies, d_emb=12.0, d_cap=3.0, eta_cap=0.3, rank=None, n_power=1,
               tile_thr=1e-4, a_ref=0.335, rng=None):
    """Tile basis from the local spectral function A_PP(E) = i (G - G^H)_PP / 2 pi, G = (E S - H + iW)^-1,
    in a CAP-terminated embedding (qtmode.cms.tile_spectral_modes); randomized range finder, all energies
    merged: Y = [X_E sqrt(lambda_E)], SVD, keep s^2 > tile_thr * a_ref. Real H, S: A X = -Im(G X) / pi
    with real probes (one factorization). Complex Hermitian H, S: G^H = (E S - H - iW)^-1 is factorized
    as well and the basis is complex. Rows: tile orbitals in atom order."""
    rng = np.random.default_rng(0) if rng is None else rng
    norb = inp["norb"]
    off = np.concatenate([[0], np.cumsum(norb)])
    near = tree.query_ball_point(geo.points(atoms), d_emb)
    emb = np.union1d(np.unique(np.concatenate([np.asarray(q, int) for q in near])), atoms)
    d, _ = geo.tree(atoms).query(geo.points(emb), k=1)
    w = eta_cap * np.clip((d - (d_emb - d_cap)) / d_cap, 0, None) ** 2
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
    Ys = []
    for E in energies:
        cs = CSolver(E * Se - He + 1j * W)
        csH = None if real else CSolver(E * Se - He - 1j * W)

        def apply_A(X):
            R = np.zeros((n, X.shape[1]), X.dtype)
            R[pidx] = X
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
        cs.free()
        if csH is not None:
            csH.free()
        Ys.append((Q @ X) * np.sqrt(np.clip(lam, 0, None))[None])
    Uu, s, _ = np.linalg.svd(np.hstack(Ys), full_matrices=False)
    keep = s ** 2 > tile_thr * a_ref
    return Uu[:, keep], s, len(emb)


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
    out = dict(HD=[], SD=[], HU=[], SU=[])
    for M, kD, kU in ((dev.H, "HD", "HU"), (dev.S, "SD", "SU")):
        for s in range(nsl):
            X = (Us[s].conj().T @ (M[b[s]:b[s + 1], b[s]:b[s + 1]] @ Us[s])).toarray()
            out[kD].append(0.5 * (X + X.conj().T))
            if s < nsl - 1:
                out[kU].append((Us[s].conj().T @ (M[b[s]:b[s + 1], b[s + 1]:b[s + 2]] @ Us[s + 1])).toarray())
    return out


def project(blk, V):
    """Blocks of V^H (blk) V for block-diagonal V = diag(V_s)."""
    n = len(V)
    return dict(HD=[V[s].conj().T @ blk["HD"][s] @ V[s] for s in range(n)],
                SD=[V[s].conj().T @ blk["SD"][s] @ V[s] for s in range(n)],
                HU=[V[s].conj().T @ blk["HU"][s] @ V[s + 1] for s in range(n - 1)],
                SU=[V[s].conj().T @ blk["SU"][s] @ V[s + 1] for s in range(n - 1)])


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
    t0 = time.time()
    for ip, (t_, w_) in enumerate(zip(th, w)):
        z = c + r * np.exp(1j * t_)
        dz = -1j * r * np.exp(1j * t_) * w_
        SL, SR = dev.sigma(z)
        D = [z * blk["SD"][s] - blk["HD"][s] for s in range(ns)]
        D[0] = D[0] - PL.conj().T @ SL @ PL
        D[-1] = D[-1] - PR.conj().T @ SR @ PR
        Up = [z * blk["SU"][s] - blk["HU"][s] for s in range(ns - 1)]
        Lo = [z * blk["SU"][s].conj().T - blk["HU"][s].conj().T for s in range(ns - 1)]
        gl = [None] * ns
        for s in range(ns):
            gl[s] = np.linalg.inv(D[s] if s == 0 else D[s] - Lo[s - 1] @ gl[s - 1] @ Up[s - 1])
        G = gl[-1]
        acc[-1] += G * dz
        for s in range(ns - 2, -1, -1):
            G = gl[s] + gl[s] @ Up[s] @ G @ Lo[s] @ gl[s]
            acc[s] += G * dz
        if ip % 6 == 5:
            log(f"    contour point {ip + 1}/{npts}, {time.time() - t0:.0f} s")
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
