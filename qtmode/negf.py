"""NEGF core: Au lead self-energies, transmission by sparse LU, oxide self-energy
folding (exact / truncated), Chebyshev interpolation in energy, block RGF."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- leads
def sancho(H00, H01, z, tol=1e-12, maxit=200):
    """Surface GF of a semi-infinite chain extending to the right (cell n -> n+1 via H01)."""
    I = np.eye(len(H00))
    es, e = H00.astype(complex), H00.astype(complex)
    a, b = H01.astype(complex), H01.conj().T.astype(complex)
    for _ in range(maxit):
        g = np.linalg.inv(z * I - e)
        ag, bg = a @ g, b @ g
        es = es + ag @ b
        e = e + ag @ b + bg @ a
        a, b = ag @ a, bg @ b
        if np.abs(a).max() < tol and np.abs(b).max() < tol:
            break
    return np.linalg.inv(z * I - es)


def lead_sigmas(H00, H01, E, eta=1e-6):
    z = E + 1j * eta
    gR = sancho(H00, H01, z)
    gL = sancho(H00, H01.conj().T, z)
    return H01.conj().T @ gL @ H01, H01 @ gR @ H01.conj().T


def gamma(S):
    return 1j * (S - S.conj().T)


def _embed(n, idx, M):
    r, c = np.meshgrid(idx, idx, indexing="ij")
    return sp.csr_matrix((M.ravel(), (r.ravel(), c.ravel())), shape=(n, n))


def transmission_sparse(H, E, SL, SR, iL, iR):
    """T(E) for sparse H (real space, any size) with dense lead self-energies on index sets."""
    n = H.shape[0]
    A = (E * sp.identity(n, format="csr") - H - _embed(n, iL, SL) - _embed(n, iR, SR)).tocsc()
    lu = spla.splu(A)
    B = np.zeros((n, len(iL)), complex)
    B[iL, np.arange(len(iL))] = 1.0
    GN1 = lu.solve(B)[iR]
    return np.real(np.trace(gamma(SR) @ GN1 @ gamma(SL) @ GN1.conj().T))


# ---------------------------------------------------------------- oxide folding
def fold_oxide(H, is_ox, E, orb_pos, rc=None, col_mask=None, chunk=300):
    """Sigma_ox(E) = H_so (E - H_oo)^-1 H_os on the non-oxide orbitals (index space `keep`).

    rc       : truncation radius in A (None = keep everything)
    col_mask : bool over all orbitals; only these interface columns are computed
    Returns (Sigma as csr in keep-space, keep indices, largest dropped |entry|)."""
    H = H.tocsr()
    o, s = np.where(is_ox)[0], np.where(~is_ox)[0]
    Hos = H[o][:, s].tocsc()
    b = np.unique(Hos.nonzero()[1])
    bc = b if col_mask is None else b[col_mask[s[b]]]
    lu = spla.splu((E * sp.identity(len(o), format="csc") - H[o][:, o]).tocsc())
    Hsob = Hos[:, b].T.tocsr()
    tree = cKDTree(orb_pos[s[b]]) if rc is not None else None
    bpos = {v: i for i, v in enumerate(b)}
    rows, cols, vals, tail = [], [], [], 0.0
    for k0 in range(0, len(bc), chunk):
        cj = bc[k0:k0 + chunk]
        S = Hsob @ lu.solve(Hos[:, cj].toarray())          # Sigma[b, cj]
        if rc is None:
            R, C = np.meshgrid(b, cj, indexing="ij")
            rows.append(R.ravel()); cols.append(C.ravel()); vals.append(S.ravel())
        else:
            for jj, nb in enumerate(tree.query_ball_point(orb_pos[s[cj]], rc)):
                nb = np.array(nb, int)
                rows.append(b[nb]); cols.append(np.full(len(nb), cj[jj])); vals.append(S[nb, jj])
                mask = np.ones(len(b), bool); mask[nb] = False
                if mask.any():
                    tail = max(tail, np.abs(S[mask, jj]).max())
    Sig = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(len(s), len(s)))
    Sig.eliminate_zeros()
    return Sig, s, tail


def oxide_levels_near(H, is_ox, E0, k=8):
    """Eigenvalues of the oxide block closest to E0 (trap / defect-level check)."""
    Hoo = H.tocsr()[is_ox][:, is_ox]
    return np.sort(spla.eigsh(Hoo.astype(float), k=k, sigma=E0, return_eigenvectors=False))


class ChebInterp:
    """Polynomial interpolation of an energy-dependent sparse matrix on Chebyshev nodes."""

    def __init__(self, Ea, Eb, n, builder):
        x = np.cos(np.pi * (2 * np.arange(n) + 1) / (2 * n))
        self.nodes = 0.5 * (Ea + Eb) + 0.5 * (Eb - Ea) * x
        self.mats = [builder(E) for E in self.nodes]

    def weights(self, E):
        w = np.ones(len(self.nodes))
        for k, Ek in enumerate(self.nodes):
            for j, Ej in enumerate(self.nodes):
                if j != k:
                    w[k] *= (E - Ej) / (Ek - Ej)
        return w

    def __call__(self, E):
        w = self.weights(E)
        out = w[0] * self.mats[0]
        for wk, M in zip(w[1:], self.mats[1:]):
            out = out + wk * M
        return out


# ---------------------------------------------------------------- block RGF
def split_blocks(M, groups):
    """Dense diagonal and upper (i -> i+1) blocks; checks block-tridiagonality."""
    M = M.tocsr()
    Hd = [M[g][:, g].toarray() for g in groups]
    Hu = [M[g][:, h].toarray() for g, h in zip(groups[:-1], groups[1:])]
    for i in range(len(groups) - 2):
        assert M[groups[i]][:, groups[i + 2]].nnz == 0, f"coupling beyond neighbour block {i}"
    return Hd, Hu


def rgf(Hd, Hu, E, SL, SR, eta=0.0, want_diag=False):
    """Returns T(E) and, optionally, diagonal blocks G_ii and right-connected gR.
    Hd may be a callable i -> block (built on demand to save memory)."""
    z = E + 1j * eta
    get = Hd if callable(Hd) else (lambda i: Hd[i])
    n = len(Hu) + 1
    gL = [None] * n
    P = None
    prev = None
    for i in range(n):
        Di = get(i)
        M = z * np.eye(len(Di)) - Di
        M = M - (SL if i == 0 else Hu[i - 1].conj().T @ prev @ Hu[i - 1])
        if i == n - 1:
            M = M - SR
        cur = np.linalg.inv(M)
        del M
        P = cur if i == 0 else cur @ Hu[i - 1].conj().T @ P    # left-connected G_{i,1}
        if want_diag:
            gL[i] = cur
        prev = cur
    T = np.real(np.trace(gamma(SR) @ P @ gamma(SL) @ P.conj().T))
    if not want_diag:
        return T
    Hd = [get(i) for i in range(n)]
    gR = [None] * n
    for i in reversed(range(n)):
        M = z * np.eye(len(Hd[i])) - Hd[i]
        M = M - (SR if i == n - 1 else Hu[i] @ gR[i + 1] @ Hu[i].conj().T)
        if i == 0:
            M = M - SL
        gR[i] = np.linalg.inv(M)
    G = []
    for i in range(n):
        if i == n - 1:
            G.append(gL[i])
        else:
            G.append(np.linalg.inv(np.linalg.inv(gL[i]) - Hu[i] @ gR[i + 1] @ Hu[i].conj().T))
    return T, G, gR
