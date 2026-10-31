"""Device, lead self-energies, sparse / block-tridiagonal solvers and the contour slab density matrix.

Device: the device blocks of a read_quatrex dict are the slabs; the lead blocks enter through Sigma_L, Sigma_R.
Lead Sigma(z): the lead-cell interior (orbitals not coupled to the neighbouring cells) is folded onto the coupled
faces F, A_eff(z) = A_FF - A_FI A_II^-1 A_IF with A = z S - H, then Lopez-Sancho decimation of the folded chain
(complex Hermitian H, any S).
"""
import os, time
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


class Device:
    """Slabs = device blocks; bL, bR = device orbitals (device order) coupled to the left / right lead."""

    def __init__(self, inp, sigma_cache=None):
        self.inp, self.meta = inp, inp["meta"]
        norb, blocks = inp["norb"], inp["blocks"]
        ob = np.repeat(blocks, norb)
        O = {b: np.where(ob == b)[0] for b in np.unique(blocks)}
        tr = self.meta["transport"]
        self.slab_orbs = [O[b] for b in tr["device_blocks"]]           # orbitals of each slab (read_quatrex order)
        self.ns = [len(o) for o in self.slab_orbs]
        self.sb = np.concatenate([[0], np.cumsum(self.ns)])
        dev = np.concatenate(self.slab_orbs)
        self.n = len(dev)
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

    def sigma(self, z):
        """Sigma_L on bL, Sigma_R on bR at complex z (cached as sig_<Re z>_<Im z>.npz in sigma_cache)."""
        z = complex(z)
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


def rho_contour(dev, mu1, mu2, npts, log=log):
    """Slab density matrices of the open device, rho_ss = -(1/pi) Im int G_ss dz over the upper half circle from mu1
    to mu2 (Gauss-Legendre in the angle), G = (z S - H - Sigma(z))^-1 by left-connected block recursion."""
    blk = slab_blocks(dev)
    ns = len(blk["HD"])
    c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    x, w = np.polynomial.legendre.leggauss(npts)
    th, w = 0.5 * np.pi * (x + 1), 0.5 * np.pi * w
    PL, PR = np.eye(dev.ns[0])[dev.bL], np.eye(dev.ns[-1])[dev.bR - dev.sb[-2]]
    acc = [np.zeros(h.shape, complex) for h in blk["HD"]]
    t0 = time.time()
    for ip, (t_, w_) in enumerate(zip(th, w)):
        z = c + r * np.exp(1j * t_)
        dz = -1j * r * np.exp(1j * t_) * w_
        SL, SR = dev.sigma(z)
        D = [z * blk["SD"][s] - blk["HD"][s] for s in range(ns)]
        D[0] = D[0] - PL.T @ SL @ PL
        D[-1] = D[-1] - PR.T @ SR @ PR
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
        log(f"    contour point {ip + 1}/{npts}, {time.time() - t0:.0f} s")
    return [-((a - a.conj().T) / 2j) / np.pi for a in acc]


def top_modes(rho, n):
    """The n eigenvectors of the Hermitian rho with the largest eigenvalues, largest first."""
    R = 0.5 * (rho + rho.conj().T)
    m = len(R)
    lam, X = sla.eigh(R, subset_by_index=[m - min(n, m), m - 1], driver="evr")
    return lam[::-1], X[:, ::-1]
