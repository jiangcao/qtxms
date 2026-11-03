"""Mode space as the coarse level of a two-grid solver.

P = U (block-diagonal per-slab basis), R = U^dagger, A_c = U^dagger A U (Galerkin).
Coarse-grid projector pi = U A_c^{-1} U^dagger A.  Mode-space solution = pi applied to exact one.
Solver: right-preconditioned GMRES with balancing two-level preconditioner
  M^{-1} = Pc + (I - Pc A) S (I - A Pc),  Pc = U A_c^{-1} U^dagger,
S = Chebyshev polynomial for (-A)^{-1} on [alpha, beta] (complement spectrum).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla


# ---------------------------------------------------------------- bases
def basis_cms(dev, M):
    """Conventional coupled mode space: lowest M eigenvectors of each H_ii."""
    Us = []
    for i in range(dev.S):
        w, V = sla.eigh(dev.slab_H(i), subset_by_index=[0, M - 1])
        Us.append(V)
    return Us


def basis_ldm(dev, Emax, w=3, thr=1e-4, Hfull=None):
    """Local density-matrix basis: diagonalize slabs i-w..i+w (Dirichlet ends),
    take all states below Emax, restrict to slab i, keep eigvecs of rho_i > thr."""
    ns, S = dev.ns, dev.S
    H = dev.full_H() if Hfull is None else Hfull
    Us = []
    for i in range(S):
        lo, hi = max(0, i - w), min(S, i + w + 1)
        Hc = H[lo * ns:hi * ns, lo * ns:hi * ns].toarray()
        # zero-flux (Neumann) ends so longitudinal k~0 states survive in the cluster
        if lo > 0:
            Hc[np.arange(ns), np.arange(ns)] -= dev.t
        if hi < S:
            Hc[np.arange(hi * ns - lo * ns - ns, hi * ns - lo * ns), np.arange(hi * ns - lo * ns - ns, hi * ns - lo * ns)] -= dev.t
        ev, V = sla.eigh(Hc, subset_by_value=(-np.inf, Emax))
        Vi = V[(i - lo) * ns:(i - lo + 1) * ns, :]
        lam, Q = np.linalg.eigh(Vi @ Vi.T)
        Us.append(Q[:, lam > thr * lam.max()][:, ::-1])
    return Us


def assemble_U(Us):
    return sp.block_diag([sp.csr_matrix(u) for u in Us], format="csr")


# ---------------------------------------------------------------- two-level solver
class TwoLevel:
    def __init__(self, dev, H, U, E, Sig, cheb_deg=12, Emax=None, beta=None):
        self.dev, self.U, self.E = dev, U, E
        n, ns = dev.n, dev.ns
        Sbig = sp.lil_matrix((n, n), dtype=complex)
        Sbig[:ns, :ns] = Sig
        Sbig[-ns:, -ns:] = Sig
        self.A = (E * sp.identity(n, dtype=complex, format="csr") - H - Sbig.tocsr()).tocsr()
        Ac = (U.conj().T @ self.A @ U).tocsc()
        self.Ac = Ac
        self.lu = spla.splu(Ac)
        self.N = U.shape[1]
        self.deg = cheb_deg
        self.alpha = max(Emax - E, 1e-3)
        self.beta = beta
        self.nmv = 0

    def mv(self, x):
        self.nmv += 1
        return self.A @ x

    def Pc(self, r):
        return self.U @ self.lu.solve(self.U.conj().T @ r)

    def smooth(self, r):
        """Chebyshev approximation of (-A)^{-1} r on [alpha, beta]; returns z ~ A^{-1} r."""
        if self.deg == 0:
            return np.zeros_like(r)
        a, b = self.alpha, self.beta
        th, de = (b + a) / 2, (b - a) / 2
        sig = th / de
        rho = 1 / sig
        # solve (-A) z = r  -> operator B = -A
        x = np.zeros_like(r)
        res = r.copy()
        d = res / th
        for k in range(self.deg):
            x = x + d
            res = res - (-self.mv(d))
            rho_new = 1 / (2 * sig - rho)
            d = rho_new * rho * d + 2 * rho_new / de * res
            rho = rho_new
        return -x  # A^{-1} = -(-A)^{-1}

    def prec(self, r):
        y = self.Pc(r)
        r2 = r - self.mv(y)
        z = self.smooth(r2)
        return y + z - self.Pc(self.mv(z))

    def solve(self, b, tol=1e-8, maxit=200):
        """Right-preconditioned GMRES, x0 = mode-space solution."""
        x0 = self.Pc(b)
        r0 = b - self.mv(x0)
        bn = np.linalg.norm(b)
        hist = [np.linalg.norm(r0) / bn]
        if hist[0] < tol:
            return x0, hist, x0
        m = maxit
        Vk = np.zeros((len(b), m + 1), complex)
        Zk = np.zeros((len(b), m), complex)
        Hh = np.zeros((m + 1, m), complex)
        beta0 = np.linalg.norm(r0)
        Vk[:, 0] = r0 / beta0
        for j in range(m):
            Zk[:, j] = self.prec(Vk[:, j])
            wv = self.mv(Zk[:, j])
            for i in range(j + 1):  # MGS, 2 passes
                h = np.vdot(Vk[:, i], wv); Hh[i, j] += h; wv -= h * Vk[:, i]
            for i in range(j + 1):
                h = np.vdot(Vk[:, i], wv); Hh[i, j] += h; wv -= h * Vk[:, i]
            Hh[j + 1, j] = np.linalg.norm(wv)
            Vk[:, j + 1] = wv / Hh[j + 1, j]
            e1 = np.zeros(j + 2, complex); e1[0] = beta0
            yk = np.linalg.lstsq(Hh[:j + 2, :j + 1], e1, rcond=None)[0]
            res = np.linalg.norm(Hh[:j + 2, :j + 1] @ yk - e1) / bn
            hist.append(res)
            if res < tol:
                break
        x = x0 + Zk[:, :j + 1] @ yk
        return x, hist, x0


def transmission_from_psi(dev, psi, Sig):
    Gam = 1j * (Sig - Sig.conj().T)
    psiN = psi[-dev.ns:, :]
    return np.real(np.trace(psiN.conj().T @ Gam @ psiN))
