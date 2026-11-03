"""Ab initio tight-binding input: sparse H, S, xyz structure, orbitals per atom.

Reads the device, sorts atoms along the transport direction, partitions them into slabs so that
A = E S - H is block tridiagonal, and builds the lead self-energies (Sancho-Rubio, non-orthogonal).

Assumptions (check against your files):
  * H and S are the full device matrices (no k-points), orbitals of each atom consecutive and atoms
    in the same order as in the xyz file.
  * The first and last slabs are one lead unit cell each, and the device starts and ends with
    at least two identical copies of that cell (standard for OMEN/NEGF device files).
"""
import os
import numpy as np
import scipy.sparse as sp
import scipy.io
import scipy.linalg as sla

UNITS = {"eV": 1.0, "Ry": 13.605693, "Ha": 27.211386}


# ------------------------------------------------------------------ readers
def read_xyz(path):
    with open(path) as f:
        lines = f.read().splitlines()
    nat = int(lines[0].split()[0])
    sp_, xyz = [], []
    for ln in lines[2:2 + nat]:
        w = ln.split()
        sp_.append(w[0])
        xyz.append([float(v) for v in w[1:4]])
    return np.array(sp_), np.array(xyz)


def parse_norb(spec, species):
    """spec: int ('10'), per species ('Si:10,H:1'), or a file with one integer per atom."""
    spec = str(spec)
    if os.path.isfile(spec):
        norb = np.loadtxt(spec, dtype=int).ravel()
    elif ":" in spec:
        d = {k.strip(): int(v) for k, v in (p.split(":") for p in spec.split(","))}
        missing = set(species) - set(d)
        if missing:
            raise ValueError(f"no orbital count for species {sorted(missing)}")
        norb = np.array([d[s] for s in species])
    else:
        norb = np.full(len(species), int(spec))
    if len(norb) != len(species):
        raise ValueError(f"{len(norb)} orbital counts for {len(species)} atoms")
    return norb


def read_sparse(path, n, one_based="auto"):
    """.npz (scipy.sparse.save_npz), .mtx (Matrix Market), or text triplets 'i j re [im]'.
    Upper-triangle-only input is completed to the full Hermitian matrix."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        M = sp.load_npz(path)
    elif ext == ".mtx":
        M = scipy.io.mmread(path)
    else:
        d = np.loadtxt(path, comments=("#", "%"))
        i, j = d[:, 0].astype(int), d[:, 1].astype(int)
        v = d[:, 2] + (1j * d[:, 3] if d.shape[1] > 3 else 0)
        if one_based == "auto":
            one_based = i.min() >= 1 and j.min() >= 1 and max(i.max(), j.max()) == n
        if one_based:
            i, j = i - 1, j - 1
        M = sp.coo_matrix((v, (i, j)), shape=(n, n))
    M = sp.csr_matrix(M)
    if M.shape != (n, n):
        raise ValueError(f"{path}: shape {M.shape}, expected ({n},{n}) from xyz + orbitals")
    if sp.tril(M, -1).nnz == 0 and sp.triu(M, 1).nnz > 0:
        print(f"  {os.path.basename(path)}: upper triangle only -> completing Hermitian")
        M = M + sp.triu(M, 1).conj().T
    M.eliminate_zeros()
    return M


# ------------------------------------------------------------------ device
class TBDevice:
    def __init__(self, H, S, species, xyz, norb, axis=0, slab_width=None, tol_zero=1e-8):
        self.species, self.norb = species, norb
        n = int(norb.sum())
        herr = abs(H - H.conj().T).max() / abs(H).max()
        serr = abs(S - S.conj().T).max()
        print(f"  n = {n} orbitals, {len(species)} atoms, nnz(H) = {H.nnz}, nnz(S) = {S.nnz}")
        print(f"  Hermiticity: |H-H^+|/|H| = {herr:.1e}, |S-S^+| = {serr:.1e}")
        off = np.concatenate([[0], np.cumsum(norb)])
        atom_of = np.repeat(np.arange(len(norb)), norb)

        # coupling range along transport, from the nonzero pattern of H and S
        P = (abs(H) > tol_zero) + (abs(S) > tol_zero)
        r, c = P.nonzero()
        ar, ac = atom_of[r], atom_of[c]
        x = xyz[:, axis]
        reach = np.abs(x[ar] - x[ac]).max()
        w = slab_width if slab_width else reach + 1e-3
        if w < reach:
            raise ValueError(f"slab width {w:.3f} < coupling range {reach:.3f} along transport")
        # put bin edges halfway between atomic planes, so planes never sit on an edge
        planes = np.unique(np.round(x, 3))
        gap = np.diff(planes).min() if len(planes) > 1 else w
        x0 = x.min() - 0.5 * gap
        slab_of_atom = np.floor((x - x0) / w).astype(int)
        order = np.lexsort((xyz[:, (axis + 2) % 3], xyz[:, (axis + 1) % 3], x, slab_of_atom))
        perm = np.concatenate([np.arange(off[a], off[a + 1]) for a in order])
        self.H = H[perm][:, perm].tocsr()
        self.S = S[perm][:, perm].tocsr()
        self.xyz, self.atom_order, self.perm = xyz[order], order, perm
        so = slab_of_atom[order]
        nslab = so.max() + 1
        sizes = np.array([norb[order][so == k].sum() for k in range(nslab)])
        if (sizes == 0).any():
            raise ValueError("empty slab: slab width too small relative to atom spacing")
        self.bounds = np.concatenate([[0], np.cumsum(sizes)])
        self.S_n, self.n = nslab, n
        # check block tridiagonality
        slab_of_orb = np.repeat(np.arange(nslab), sizes)
        r, c = (abs(self.H) + abs(self.S)).nonzero()
        assert np.abs(slab_of_orb[r] - slab_of_orb[c]).max() <= 1
        print(f"  coupling range {reach:.3f} (xyz units), slab width {w:.3f} -> {nslab} slabs, "
              f"sizes {sizes.min()}..{sizes.max()} orbitals")
        self._check_leads()

    def blk(self, M, i, j):
        b = self.bounds
        return M[b[i]:b[i + 1], b[j]:b[j + 1]].toarray()

    def _check_leads(self):
        for name, (a, b, c) in {"left": (0, 1, 2), "right": (self.S_n - 1, self.S_n - 2, self.S_n - 3)}.items():
            sz = np.diff(self.bounds)
            if not (sz[a] == sz[b] == sz[c]):
                print(f"  WARNING {name} lead: end slabs differ in size ({sz[a]},{sz[b]},{sz[c]}); "
                      "set --slab-width to the lead unit-cell length (or a multiple)")
                continue
            d0 = abs(self.blk(self.H, a, a) - self.blk(self.H, b, b)).max()
            d1 = abs(self.blk(self.H, a, b) - self.blk(self.H, b, c)).max() if name == "left" else \
                abs(self.blk(self.H, b, a) - self.blk(self.H, c, b)).max()
            ok = "ok" if max(d0, d1) < 1e-4 else "WARNING: device end is not periodic"
            print(f"  {name} lead cell check: |dH00| = {d0:.1e}, |dH01| = {d1:.1e}  {ok}")

    # ---------------- lead blocks: A = (E+i eta) S - H
    def lead_blocks(self, side, E, eta):
        z = E + 1j * eta
        if side == "L":
            i, j = 0, 1
        else:
            i, j = self.S_n - 2, self.S_n - 1
        A00 = z * self.blk(self.S, j if side == "R" else i, j if side == "R" else i) - \
            self.blk(self.H, j if side == "R" else i, j if side == "R" else i)
        A01 = z * self.blk(self.S, i, j) - self.blk(self.H, i, j)       # cell -> next cell (to the right)
        A10 = z * self.blk(self.S, j, i) - self.blk(self.H, j, i)       # cell -> previous cell
        return A00, A01, A10

    def sigma(self, E, eta=1e-6):
        """Retarded Sigma_L (on slab 0) and Sigma_R (on last slab)."""
        A00, A01, A10 = self.lead_blocks("L", E, eta)
        gL = sancho_rubio(A00, a=A10, b=A01)          # lead extends to the left
        SigL = A10 @ gL @ A01
        A00, A01, A10 = self.lead_blocks("R", E, eta)
        gR = sancho_rubio(A00, a=A01, b=A10)          # lead extends to the right
        SigR = A01 @ gR @ A10
        return SigL, SigR

    def lead_bands(self, nk=41, side="L"):
        """Band structure of the lead cell, E(k) for k in [0, pi]."""
        if side == "L":
            i, j = 0, 1
        else:
            i, j = self.S_n - 2, self.S_n - 1
        H0, H1 = self.blk(self.H, i, i), self.blk(self.H, i, j)
        S0, S1 = self.blk(self.S, i, i), self.blk(self.S, i, j)
        ks = np.linspace(0, np.pi, nk)
        return ks, np.array([sla.eigvalsh(H0 + H1 * np.exp(1j * k) + H1.conj().T * np.exp(-1j * k),
                                          S0 + S1 * np.exp(1j * k) + S1.conj().T * np.exp(-1j * k))
                             for k in ks])


def sancho_rubio(A00, a, b, tol=1e-12, maxit=200):
    """Surface GF of a semi-infinite chain with on-site A00, coupling a to the next cell away from the
    surface and b back toward it.  Returns inv(A_surface)."""
    W, Ws = A00.copy(), A00.copy()
    a, b = a.copy(), b.copy()
    for _ in range(maxit):
        g = np.linalg.inv(W)
        agb, bga = a @ g @ b, b @ g @ a
        Ws = Ws - agb
        W = W - agb - bga
        a, b = -a @ g @ a, -b @ g @ b
        if np.abs(a).max() + np.abs(b).max() < tol:
            break
    return np.linalg.inv(Ws)


def injection(SigL, tol=1e-4):
    """Columns B with Gamma_L = B B^dagger (propagating channels only; eigenvalues of Gamma below
    tol [eV] come from the eta broadening of evanescent modes and are dropped)."""
    Gam = 1j * (SigL - SigL.conj().T)
    lam, V = np.linalg.eigh((Gam + Gam.conj().T) / 2)
    k = lam > tol
    return V[:, k] * np.sqrt(lam[k])


def rgf_transmission(dev, E, SigL, SigR, eta=0.0):
    """Exact T(E) by RGF with variable slab sizes."""
    S_n, z = dev.S_n, E + 1j * eta
    A = lambda i, j: z * dev.blk(dev.S, i, j) - dev.blk(dev.H, i, j)
    gR = [None] * S_n
    Ad = A(S_n - 1, S_n - 1) - SigR
    if S_n == 1:
        Ad = Ad - SigL
    gR[-1] = np.linalg.inv(Ad)
    for i in range(S_n - 2, -1, -1):
        Ad = A(i, i) - (SigL if i == 0 else 0)
        gR[i] = np.linalg.inv(Ad - A(i, i + 1) @ gR[i + 1] @ A(i + 1, i))
    G = gR[0]
    for i in range(1, S_n):
        G = -gR[i] @ A(i, i - 1) @ G          # G_{i,0}
    GamL = 1j * (SigL - SigL.conj().T)
    GamR = 1j * (SigR - SigR.conj().T)
    return np.real(np.trace(GamR @ G @ GamL @ G.conj().T))


# ------------------------------------------------------------------ coarse bases
def basis_slab_window(dev, Elo, M):
    """CMS analogue: the M lowest eigenvectors of (H_ii, S_ii) above Elo (mid-gap), per slab.
    (Isolated-slab energies sit above the device band edge, so select by count, not by window.)"""
    Us = []
    for i in range(dev.S_n):
        w, V = sla.eigh(dev.blk(dev.H, i, i), dev.blk(dev.S, i, i))
        V = V[:, w > Elo][:, :M]
        Us.append(np.linalg.qr(V)[0])
    return Us


def basis_ldm(dev, Elo, Ehi, w=2, thr=1e-3):
    """Local density-matrix basis in the window [Elo, Ehi] (excludes valence states when Elo is mid-gap).
    For slab i: generalized eigh of slabs i-w..i+w, keep states in the window, restrict to slab i,
    keep eigenvectors of rho_i = V_i V_i^+ above thr * max."""
    b, Us = dev.bounds, []
    for i in range(dev.S_n):
        lo, hi = max(0, i - w), min(dev.S_n, i + w + 1)
        Hc = dev.H[b[lo]:b[hi], b[lo]:b[hi]].toarray()
        Sc = dev.S[b[lo]:b[hi], b[lo]:b[hi]].toarray()
        _, V = sla.eigh(Hc, Sc, subset_by_value=(Elo, Ehi))
        Vi = V[b[i] - b[lo]:b[i + 1] - b[lo]]
        lam, Q = np.linalg.eigh(Vi @ Vi.conj().T)
        Us.append(Q[:, lam > thr * max(lam.max(), 1e-300)][:, ::-1])
    return Us
