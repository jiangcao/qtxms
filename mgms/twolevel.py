"""Generic two-level (mode space = coarse level) solver for A psi = b, A = E S - H - Sigma.

Coarse: P = U, R = U^+, A_c = U^+ A U (Galerkin), Pc = U A_c^{-1} U^+, pi = Pc A.
Preconditioner (balancing): M^{-1} r = Pc r + (I - Pc A) S (I - A Pc) r.
Smoothers:
  'gmres' : k steps of GMRES on A z = r (z0 = 0).  Works for indefinite complements (TB: valence
            states below E and high conduction states above E).  Nonlinear -> outer is flexible GMRES.
  'cheb'  : Chebyshev for (-A)^{-1} on [alpha, beta]; only valid when the complement is definite
            (effective mass, or a basis window that starts at the bottom of the spectrum).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class TwoLevel:
    def __init__(self, A, U, smoother="gmres", k=10, alpha=None, beta=None, coarse=None, restrict="U"):
        """coarse: optional object with solve(B) for the coarse matrix (e.g. msbasis.BlockTridiag built from
        slab blocks); default: sparse LU of U^H A U.
        restrict: 'U'  -> Galerkin, Pc r = U A_c^-1 U^H r;
                  'AU' -> minimal residual, Pc r = U (W^H W)^-1 W^H r with W = A U (coarse = msbasis.MRCoarse):
                          A Pc is then the orthogonal projector on range(AU), so the coarse correction never
                          increases the residual; Pc A is still a projector (W^H A U = W^H W)."""
        self.A, self.U = A.tocsr(), U.tocsr()
        self.N = U.shape[1]
        self.coarse, self.restrict = coarse, restrict
        if restrict == "AU":
            assert coarse is not None, "restrict='AU' needs a coarse solver for W^H W (msbasis.MRCoarse)"
            self.AH = self.A.conj().T.tocsr()
        if coarse is not None:
            self.Ac = None
        elif self.N:
            Ac = (self.U.conj().T @ self.A @ self.U).tocsc()
            self.lu = spla.splu(Ac)
            # smallest singular value of A_c: near zero = spurious coarse band at this energy
            self.Ac = Ac
        self.smoother, self.k, self.alpha, self.beta = smoother, k, alpha, beta
        self.nmv = 0

    def sigma_min_coarse(self):
        if not self.N or self.Ac is None:
            return np.nan
        if self.N < 2000:
            return np.linalg.svd(self.Ac.toarray(), compute_uv=False).min()
        x = np.random.default_rng(0).standard_normal(self.N) + 0j
        for _ in range(30):  # inverse iteration on A_c^+ A_c
            x = self.lu.solve(self.lu.solve(x, trans="H"))
            x /= np.linalg.norm(x)
        return np.linalg.norm(self.Ac @ x)

    def mv(self, x):
        self.nmv += 1
        return self.A @ x

    def Pc(self, r):
        if not self.N:
            return np.zeros_like(r)
        solve = self.coarse.solve if self.coarse is not None else self.lu.solve
        if self.restrict == "AU":
            return self.U @ solve(self.U.conj().T @ (self.AH @ r))
        return self.U @ solve(self.U.conj().T @ r)

    # ---------------- smoothers
    def smooth(self, r):
        if self.smoother == "gmres":
            return gmres_fixed(self.mv, r, self.k)
        a, b = self.alpha, self.beta
        th, de = (b + a) / 2, (b - a) / 2
        sig = th / de
        rho = 1 / sig
        x = np.zeros_like(r); res = r.copy(); d = res / th
        for _ in range(self.k):
            x = x + d
            res = res + self.mv(d)
            rho_new = 1 / (2 * sig - rho)
            d = rho_new * rho * d + 2 * rho_new / de * res
            rho = rho_new
        return -x

    def prec(self, r):
        y = self.Pc(r)
        r2 = r - self.mv(y) if self.N else r
        z = self.smooth(r2)
        return y + z - self.Pc(self.mv(z)) if self.N else z

    def solve(self, b, tol=1e-8, maxit=300):
        """Flexible GMRES (right preconditioning), x0 = mode-space solution Pc b."""
        x0 = self.Pc(b)
        r0 = b - self.mv(x0) if self.N else b.copy()
        bn = np.linalg.norm(b)
        hist = [np.linalg.norm(r0) / bn]
        if hist[0] < tol:
            return x0, hist, x0
        n, m = len(b), maxit
        V = np.zeros((n, m + 1), complex); Z = np.zeros((n, m), complex)
        Hh = np.zeros((m + 1, m), complex)
        beta0 = np.linalg.norm(r0); V[:, 0] = r0 / beta0
        for j in range(m):
            Z[:, j] = self.prec(V[:, j])
            w = self.mv(Z[:, j])
            for _ in range(2):
                h = V[:, :j + 1].conj().T @ w
                Hh[:j + 1, j] += h
                w -= V[:, :j + 1] @ h
            Hh[j + 1, j] = np.linalg.norm(w)
            V[:, j + 1] = w / Hh[j + 1, j]
            e1 = np.zeros(j + 2, complex); e1[0] = beta0
            y = np.linalg.lstsq(Hh[:j + 2, :j + 1], e1, rcond=None)[0]
            hist.append(np.linalg.norm(Hh[:j + 2, :j + 1] @ y - e1) / bn)
            if hist[-1] < tol:
                break
        x = x0 + Z[:, :j + 1] @ y
        return x, hist, x0


def gmres_fixed(mv, r, k):
    """k steps of GMRES on A z = r from z = 0 (used as smoother)."""
    n = len(r)
    V = np.zeros((n, k + 1), complex); Hh = np.zeros((k + 1, k), complex)
    b0 = np.linalg.norm(r)
    if b0 == 0:
        return np.zeros_like(r)
    V[:, 0] = r / b0
    for j in range(k):
        w = mv(V[:, j])
        h = V[:, :j + 1].conj().T @ w; Hh[:j + 1, j] = h; w = w - V[:, :j + 1] @ h
        Hh[j + 1, j] = np.linalg.norm(w)
        if Hh[j + 1, j] < 1e-14 * b0:
            k = j + 1; break
        V[:, j + 1] = w / Hh[j + 1, j]
    e1 = np.zeros(k + 1, complex); e1[0] = b0
    y = np.linalg.lstsq(Hh[:k + 1, :k], e1, rcond=None)[0]
    return V[:, :k] @ y


def build_A(dev, E, SigL, SigR):
    """Sparse A = E S - H - Sigma_L - Sigma_R on the device (slab ordering of dev)."""
    b = dev.bounds
    n = dev.n
    rows, cols, vals = [], [], []
    for Sig, (lo, hi) in ((SigL, (b[0], b[1])), (SigR, (b[-2], b[-1]))):
        ii, jj = np.meshgrid(np.arange(lo, hi), np.arange(lo, hi), indexing="ij")
        rows.append(ii.ravel()); cols.append(jj.ravel()); vals.append(Sig.ravel())
    Sbig = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return (E * dev.S - dev.H - Sbig).tocsr()
