"""sp3d5s* nearest-neighbour tight-binding Hamiltonian for Si (Slater-Koster, two-centre).

Orbitals per atom (10, x2 with spin): s, px, py, pz, dxy, dyz, dzx, dx2-y2, d3z2-r2, s*.
Spin-orbit on p: H_so = (2 Delta/3) L.S, i.e. lambda = Delta/3 (Jancu convention).

Slater-Koster blocks are built by rotating the bond-frame integrals (sigma, pi, delta) with real
spherical-harmonic rotation matrices, so every l = 0, 1, 2 pair is covered by one routine.
For a bond from atom A (orbital alpha) to atom B (orbital beta) along unit vector d:
    <alpha, A | H | beta, B> = sum_m D^la(R)_{alpha m} V(la, lb, |m|) D^lb(R)_{beta m},
with R the rotation that takes z to d. For la > lb the block is <beta,0|H|alpha,-d>^T (homonuclear).

Parameters (eV):
  BOYKIN_SI (default): Boykin, Klimeck, Oyafuso, PRB 69, 115201 (2004), a = 5.431 A;
      gives Eg = 1.131 eV at 0.814 X, m_l = 0.891, m_t = 0.201, Delta_so = 0.047 at Gamma.
  JANCU_SI: Jancu, Scholz, Beltram, Bassani, PRB 57, 6493 (1998), a = 5.430 A;
      Eg = 1.170 eV at 0.846 X, but m_l = 0.70, m_t = 0.23.
"""
import numpy as np
import scipy.sparse as sp

A_SI = 5.431
BOYKIN_SI = dict(
    Es=-2.15168, Ep=4.22925, Ed=13.78950, Esx=19.11650,
    ss_s=-1.95933, sxsx_s=-4.24135, ssx_s=-1.52230,
    sp_s=3.02562, sxp_s=3.15565, sd_s=-2.28485, sxd_s=-0.80993,
    pp_s=4.10364, pp_p=-1.51801, pd_s=-1.35554, pd_p=2.38479,
    dd_s=-1.68136, dd_p=2.58880, dd_d=-1.81400, so=0.01989)
JANCU_SI = dict(
    Es=-2.0196, Ep=4.5448, Ed=14.1836, Esx=19.6748,
    ss_s=-1.9413, sxsx_s=-3.3081, ssx_s=-1.6933,
    sp_s=2.7836, sxp_s=2.8428, sd_s=-2.7998, sxd_s=-0.7003,
    pp_s=4.1068, pp_p=-1.5934, pd_s=-2.1073, pd_p=1.9977,
    dd_s=-1.2327, dd_p=2.5145, dd_d=-2.4734, so=0.0195)     # so = Delta/3

ORB = ["s", "px", "py", "pz", "dxy", "dyz", "dzx", "dx2y2", "dz2", "sx"]
L_OF = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 0])
NORB = 10


# ---------------------------------------------------------------- real harmonics and rotations
def _rfuncs(l, r):
    """Real harmonics (unnormalised consistently) at points r (n, 3); order as in ORB."""
    x, y, z = r.T
    if l == 0:
        return np.ones((len(r), 1))
    if l == 1:
        return np.stack([x, y, z], 1)
    s3 = np.sqrt(3.0)
    return np.stack([s3 * x * y, s3 * y * z, s3 * z * x, 0.5 * s3 * (x * x - y * y), z * z - 0.5 * (x * x + y * y)], 1)


_PTS = np.random.default_rng(0).normal(size=(64, 3))


def _rotmat(l, R):
    """D with f_i(R r) = sum_j D_ij f_j(r): lab orbital i in terms of bond-frame orbitals j."""
    if l == 0:
        return np.ones((1, 1))
    F = _rfuncs(l, _PTS)
    G = _rfuncs(l, _PTS @ R.T)        # f(R r): lab orbitals in terms of bond-frame ones
    return np.linalg.lstsq(F, G, rcond=None)[0].T


def _rot_z_to(d):
    """Rotation matrix R with R @ z = d."""
    d = d / np.linalg.norm(d); z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d); c = d @ z
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


# bond-frame index of m = 0 / |m| = 1 / |m| = 2 within each l block (order of _rfuncs)
_MABS = {0: np.array([0]), 1: np.array([1, 1, 0]), 2: np.array([2, 1, 1, 2, 0])}


def _vtab(P):
    """V[(a, b)] -> array over |m| for orbital groups a, b in {'s','p','d','sx'}, a <= b in l."""
    return {("s", "s"): [P["ss_s"]], ("sx", "sx"): [P["sxsx_s"]], ("s", "sx"): [P["ssx_s"]], ("sx", "s"): [P["ssx_s"]],
            ("s", "p"): [P["sp_s"]], ("sx", "p"): [P["sxp_s"]], ("s", "d"): [P["sd_s"]], ("sx", "d"): [P["sxd_s"]],
            ("p", "p"): [P["pp_s"], P["pp_p"]], ("p", "d"): [P["pd_s"], P["pd_p"]],
            ("d", "d"): [P["dd_s"], P["dd_p"], P["dd_d"]]}


_GROUPS = [("s", 0, slice(0, 1)), ("p", 1, slice(1, 4)), ("d", 2, slice(4, 9)), ("sx", 0, slice(9, 10))]


def sk_block(d, P=BOYKIN_SI):
    """10 x 10 hopping block <alpha, 0 | H | beta, d> for a bond vector d (any length)."""
    d = np.asarray(d, float)
    B = _blk_lower(d, P); Bm = _blk_lower(-d, P)
    for ga, la, sa in _GROUPS:              # la > lb: <a,0|H|b,d> = <b,0|H|a,-d> (real, homonuclear)
        for gb, lb, sb in _GROUPS:
            if la > lb:
                B[sa, sb] = Bm[sb, sa].T
    return B


def _same_comp(la, i, lb, j):
    """Do bond-frame components i (of l = la) and j (of l = lb) share the azimuthal function?"""
    if la == 0:
        return j == (4 if lb == 2 else 2)                   # s couples to z (p) / z2 (d)
    # la = 1, lb = 2: pz-dz2 (sigma), px-dzx and py-dyz (pi)
    return (i, j) in ((2, 4), (0, 2), (1, 1))


def _blk_lower(d, P):
    """Same as sk_block but only the la <= lb part (used for the transposed blocks)."""
    R = _rot_z_to(np.asarray(d, float)); D = {l: _rotmat(l, R) for l in (0, 1, 2)}
    V = _vtab(P); B = np.zeros((NORB, NORB))
    for ga, la, sa in _GROUPS:
        for gb, lb, sb in _GROUPS:
            if la <= lb:
                v = np.asarray(V[(ga, gb)]); Mb = np.zeros((2 * la + 1, 2 * lb + 1))
                if la == lb:
                    Mb = np.diag(v[_MABS[lb]])
                else:
                    for i, ma in enumerate(_MABS[la]):
                        for j, mb in enumerate(_MABS[lb]):
                            if ma == mb and _same_comp(la, i, lb, j):
                                Mb[i, j] = v[ma]
                B[sa, sb] = D[la] @ Mb @ D[lb].T
    return B


def onsite(P=BOYKIN_SI):
    return np.diag([P["Es"], P["Ep"], P["Ep"], P["Ep"]] + [P["Ed"]] * 5 + [P["Esx"]])


def so_block(P=BOYKIN_SI):
    """20 x 20 spin-orbit term on one atom, basis (10 orbitals up, 10 down); H = 2 lambda L.S."""
    lam = P["so"]
    Lx = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]); Ly = np.array([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]])
    Lz = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    sx = np.array([[0, 1], [1, 0]]) / 2; sy = np.array([[0, -1j], [1j, 0]]) / 2; sz = np.array([[1, 0], [0, -1]]) / 2
    Hp = 2 * lam * (np.kron(sx, Lx) + np.kron(sy, Ly) + np.kron(sz, Lz))     # (spin x p) 6 x 6
    H = np.zeros((2 * NORB, 2 * NORB), complex)
    idx = [1, 2, 3, NORB + 1, NORB + 2, NORB + 3]
    H[np.ix_(idx, idx)] = Hp
    return H


# ---------------------------------------------------------------- bulk diamond
NN = A_SI / 4 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])


def bulk_hk(k, P=BOYKIN_SI, a=A_SI, spin=True):
    """H(k) of bulk Si (20 x 20, or 40 x 40 with spin-orbit); k in 1/A, cartesian."""
    nn = NN * a / A_SI
    T = sum(sk_block(d, P) * np.exp(1j * k @ d) for d in nn)
    E0 = onsite(P)
    H = np.block([[E0, T], [T.conj().T, E0]])
    if not spin:
        return H
    Hs = np.kron(np.eye(2), H).astype(complex)
    so = so_block(P)
    for at in range(2):                    # spin-orbit on each atom; reorder to (spin, atom, orb)
        idx = np.r_[at * NORB:(at + 1) * NORB, 2 * NORB + at * NORB:2 * NORB + (at + 1) * NORB]
        Hs[np.ix_(idx, idx)] += so
    return Hs


# ---------------------------------------------------------------- real space
def build_H_si(pos, bonds, P=BOYKIN_SI, spin=False, cell=None):
    """Sparse real-space Hamiltonian for Si atoms at pos (n, 3) with bond list (nb, 2).
    No passivation / surface terms: dangling bonds are left as they are. cell: optional
    periodic vectors (3,) used to unwrap bond vectors (minimum image)."""
    n = len(pos); rows, cols, vals = [], [], []
    E0 = onsite(P)
    ii, jj = np.nonzero(E0)
    for a in range(n):
        rows.append(a * NORB + ii); cols.append(a * NORB + jj); vals.append(E0[ii, jj])
    for a, b in bonds:
        d = pos[b] - pos[a]
        if cell is not None:
            c = np.asarray(cell, float); m = c > 0
            d[m] -= c[m] * np.round(d[m] / c[m])
        T = sk_block(d, P); r, c_ = np.nonzero(np.abs(T) > 0)
        rows += [a * NORB + r, b * NORB + c_]; cols += [b * NORB + c_, a * NORB + r]; vals += [T[r, c_], T[r, c_]]
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * NORB, n * NORB))
    if spin:
        H = sp.kron(sp.identity(2), H, format="csr").astype(complex)
        so = sp.csr_matrix(so_block(P))
        blocks = sp.block_diag([so] * n, format="csr")
        perm = np.concatenate([np.r_[a * NORB:(a + 1) * NORB, n * NORB + a * NORB:n * NORB + (a + 1) * NORB] for a in range(n)])
        Pm = sp.csr_matrix((np.ones(len(perm)), (perm, np.arange(len(perm)))), shape=(len(perm), len(perm)))
        H = H + Pm @ blocks @ Pm.T
    return H
