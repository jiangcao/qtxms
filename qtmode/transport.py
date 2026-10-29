"""Semi-infinite lead self-energies on the coupling surface, and Caroli transmission for a real-space
device and for reduced (coupled-mode-space) models of it.

Lead: periodic cells (H00, H01; H01 couples cell n to n+1). Only the right face Rf of a cell
(rows of H01) and the left face Lf (columns of H01) couple between cells. The cell interior is
folded out at each energy, giving an effective chain on F = Lf u Rf; Lopez-Sancho decimation on
that chain gives both surface Green's functions (left-extending and right-extending lead).

Device: G of the closed device is needed only on its boundary orbitals B (those coupled to the
leads). G_open_BB = [G_closed_BB^-1 - Sigma_BB]^-1, T = Tr[Gamma_L G_LR Gamma_R G_LR^+].
"""
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
try:
    import pypardiso
except ImportError:      # pragma: no cover
    from . import _pardiso_compat as pypardiso   # SciPy SuperLU fallback


def _real_solve(A, B, n_ref=1):
    """Solve A X = B for real sparse A (PARDISO) with iterative refinement."""
    s = pypardiso.PyPardisoSolver(mtype=11)
    A = A.tocsr()
    s.factorize(A)
    X = s.solve(A, B)
    for _ in range(n_ref):
        X = X + s.solve(A, B - A @ X)
    s.free_memory(everything=True)
    return X


def _real_inverse_block(A, rows, cols, chunk=256, n_ref=1):
    """(A^-1)[rows, cols] for real sparse A: one PARDISO factorization, unit right-hand sides
    in chunks, only the requested rows kept (memory ~ n x chunk)."""
    s = pypardiso.PyPardisoSolver(mtype=11)
    A = A.tocsr()
    s.factorize(A)
    n = A.shape[0]
    out = np.zeros((len(rows), len(cols)))
    for c0 in range(0, len(cols), chunk):
        cc = cols[c0:c0 + chunk]
        B = np.zeros((n, len(cc)))
        B[cc, np.arange(len(cc))] = 1.0
        X = s.solve(A, B)
        for _ in range(n_ref):
            X = X + s.solve(A, B - A @ X)
        out[:, c0:c0 + len(cc)] = X[rows]
    s.free_memory(everything=True)
    return out


class Lead:
    """Semi-infinite periodic lead from one cell H00 and the cell-to-cell coupling H01 (sparse)."""

    def __init__(self, H00, H01):
        H00, H01 = H00.tocsr(), H01.tocsr()
        self.n = H00.shape[0]
        self.Rf = np.unique(H01.nonzero()[0])
        self.Lf = np.unique(H01.nonzero()[1])
        self.F = np.union1d(self.Lf, self.Rf)
        self.I = np.setdiff1d(np.arange(self.n), self.F)
        self.H00, self.H01 = H00, H01
        self.posF = {v: i for i, v in enumerate(self.F)}

    def effective_chain(self, E):
        """H00_eff(E) = H_FF + H_FI (E - H_II)^-1 H_IF (real E), and H01 on F x F."""
        H, F, I = self.H00, self.F, self.I
        HIF = H[I][:, F]
        X = _real_solve(E * sp.identity(len(I), format="csr") - H[I][:, I], HIF.toarray())
        Heff = H[F][:, F].toarray() + HIF.T @ X
        V = self.H01[F][:, F].toarray()
        return 0.5 * (Heff + Heff.T), V

    def surface_gfs(self, E, eta=1e-6, tol=1e-10, maxit=200):
        """Surface GFs of the right-extending lead on Lf (its first cell) and of the
        left-extending lead on Rf (its last cell), from one Lopez-Sancho decimation.
        The couplings keep their block structure (a: Rf -> Lf, b: Lf -> Rf) at every step, so only
        those sub-blocks are formed; the solve has |Lf| + |Rf| right-hand sides."""
        Heff, V = self.effective_chain(E)
        z = E + 1j * eta
        iL = np.array([self.posF[c] for c in self.Lf])
        iR = np.array([self.posF[r] for r in self.Rf])
        nF = len(Heff)
        e = Heff.astype(complex)
        esR, esL = e.copy(), e.copy()
        A = V[np.ix_(iR, iL)].astype(complex)          # a = H01 block (Rf rows, Lf cols)
        B = A.T.copy()                                  # b = H01^T block (Lf rows, Rf cols)
        for it in range(maxit):
            M = z * np.eye(nF) - e
            RHS = np.zeros((nF, len(iL) + len(iR)), complex)
            RHS[iR, :len(iL)] = A                       # columns of a (cols Lf)
            RHS[iL, len(iL):] = B                       # columns of b (cols Rf)
            X = sla.solve(M, RHS, check_finite=False)
            ga, gb = X[:, :len(iL)], X[:, len(iL):]     # g a[:, Lf], g b[:, Rf]
            agb = A @ gb[iL]                            # (a g b)[Rf, Rf]
            bga = B @ ga[iR]                            # (b g a)[Lf, Lf]
            esR[np.ix_(iR, iR)] += agb
            esL[np.ix_(iL, iL)] += bga
            e[np.ix_(iR, iR)] += agb
            e[np.ix_(iL, iL)] += bga
            A, B = A @ ga[iL], B @ gb[iR]
            if max(np.abs(A).max(), np.abs(B).max()) < tol:
                break
        I = np.eye(nF)
        gR = sla.solve(z * I - esR, I[:, iL], check_finite=False)[iL]   # right lead, on Lf
        gL = sla.solve(z * I - esL, I[:, iR], check_finite=False)[iR]   # left lead, on Rf
        return gL, gR, it + 1

    def sigmas(self, E, VL, VR, eta=1e-6):
        """Self-energies on device orbitals.
        VL: coupling (rows: lead cell orbitals, cols: device) from the left lead's last cell;
        VR: coupling (rows: device, cols: lead cell orbitals) to the right lead's first cell.
        Returns SigL (on cols of VL), SigR (on rows of VR) as dense matrices on the device
        boundary index sets bL, bR, plus bL, bR and the number of decimation steps."""
        gL, gR, nit = self.surface_gfs(E, eta)
        VL, VR = VL.tocsr(), VR.tocsc()
        bL = np.unique(VL.nonzero()[1])
        bR = np.unique(VR.nonzero()[0])
        A = VL[self.Rf][:, bL].toarray()               # |Rf| x |bL|
        B = VR[bR][:, self.Lf].toarray()               # |bR| x |Lf|
        SigL = A.T @ gL @ A
        SigR = B @ gR @ B.T
        return SigL, SigR, bL, bR, nit


def gamma(S):
    return 1j * (S - S.conj().T)


def caroli(Gbb, SigL, SigR, nL):
    """T from the open-boundary G on [bL, bR] (first nL indices are bL)."""
    GLR = Gbb[:nL, nL:]
    return float(np.real(np.trace(gamma(SigL) @ GLR @ gamma(SigR) @ GLR.conj().T)))


def transmission_open(Ac, SigL, SigR, bL, bR, chunk=128, gam_thr=1e-7):
    """Exact T by one complex SuperLU factorization of the open system A = Ac - Sigma_L - Sigma_R,
    Ac = E - H - U (real sparse). Returns (T, max relative residual of the solves).
    gam_thr: right-hand sides = eigenvectors of Gamma_L with lam > gam_thr*lam_max (Gamma has a long tail,
    no gap; 3 x 5 sp3d5s*: 1e-7 keeps 74-255 of 1 356 columns, T error <= 6e-7; 0 = all columns).
    Replaces the closed-device + Dyson route, which can be O(1) wrong with a tiny residual when a
    closed-device eigenvalue lies near E."""
    import scipy.sparse.linalg as spla
    n = Ac.shape[0]

    def emb(idx, M):
        r_, c_ = np.meshgrid(idx, idx, indexing="ij")
        return sp.csc_matrix((M.ravel(), (r_.ravel(), c_.ravel())), shape=(n, n))

    def gfac(S, thr):   # Gamma = X diag(lam) X^H, eigenpairs with lam > thr*lam_max
        lam, X = np.linalg.eigh(gamma(S))
        m = lam > thr * max(lam.max(), 1e-30)
        return X[:, m], lam[m]

    A = (Ac.astype(complex) - emb(bL, SigL) - emb(bR, SigR)).tocsc()
    # T = ||F_R^H G_RL F_L||_F^2, F = X sqrt(lam): solve only for the dominant unit eigenvectors of Gamma_L,
    # not for every bL orbital (residual stays relative to unit right-hand sides)
    (XL, lamL), (XR, lamR) = gfac(SigL, gam_thr), gfac(SigR, 1e-14)
    FRh = (XR * np.sqrt(lamR)).conj().T
    # symmetric minimum-degree ordering on A^T + A with diagonal pivots: 2.6x less fill, 3.6x faster than
    # COLAMD on the 3 x 5 sp3d5s* device; COLAMD with partial pivoting as fallback if the residual is poor
    for spec, thr, sym in (("MMD_AT_PLUS_A", 0.0, True), ("COLAMD", 1.0, False)):
        lu = spla.splu(A, permc_spec=spec, diag_pivot_thresh=thr, options=dict(SymmetricMode=sym))
        T = 0.0; res = 0.0
        for c0 in range(0, len(lamL), chunk):
            cc = slice(c0, min(c0 + chunk, len(lamL)))
            Bm = np.zeros((n, cc.stop - c0), complex); Bm[bL] = XL[:, cc]
            X = lu.solve(Bm)
            R = Bm - A @ X
            if np.linalg.norm(R) > 1e-11 * np.linalg.norm(Bm):   # one step of iterative refinement
                X += lu.solve(R); R = Bm - A @ X
            res = max(res, float(np.linalg.norm(R) / np.linalg.norm(Bm)))
            T += float(np.sum(np.linalg.norm(FRh @ X[bR], axis=0) ** 2 * lamL[cc]))
        del lu
        if res < 1e-8:
            break
    return T, res


def transmission_real_space(Hdev, E, SigL, SigR, bL, bR):
    """Exact T of the open device (SuperLU, transmission_open)."""
    n = Hdev.shape[0]
    return transmission_open(E * sp.identity(n, format="csc") - Hdev, SigL, SigR, bL, bR)[0]


def transmission_reduced(Hr, E, WL, WR, SigL, SigR):
    """T of a reduced model Hr = W^T H W (real orthonormal W, rows = device orbitals).
    WL, WR: the rows of W on the lead-coupled orbitals bL, bR (sparse or dense). The
    self-energies are projected, Sigma_r = W_b^T Sigma W_b, and G is formed only on the
    reduced columns that touch bL or bR."""
    WL = sp.csr_matrix(WL) if not sp.issparse(WL) else WL.tocsr()
    WR = sp.csr_matrix(WR) if not sp.issparse(WR) else WR.tocsr()
    cL = np.unique(WL.nonzero()[1])
    cR = np.unique(WR.nonzero()[1])
    WLd, WRd = WL[:, cL].toarray(), WR[:, cR].toarray()
    SL = WLd.T @ SigL @ WLd
    SR = WRd.T @ SigR @ WRd
    c = np.concatenate([cL, cR])
    N = Hr.shape[0]
    RHS = np.zeros((N, len(c)), order="F")
    RHS[c, np.arange(len(c))] = 1.0
    M = np.negative(Hr)                                  # one working copy, factorized in place;
    M[np.diag_indices(N)] += E                           # symmetric, so M.T is the same matrix in
    Gc = sla.solve(M.T, RHS, assume_a="sym", check_finite=False, overwrite_a=True,
                   overwrite_b=True)[c]                  # Fortran order (no hidden LAPACK copy)
    del M
    Go = np.linalg.inv(np.linalg.inv(Gc) - sla.block_diag(SL, SR))
    return caroli(Go, SL, SR, len(cL))


# ------------------------------------------------------------------ block-tridiagonal reduced solves
class BlockTridiag:
    """Reduced Hamiltonian Hr (dense, real symmetric) whose columns belong to transport slabs
    (slab_of_col); Hr couples only neighbouring slabs. Stores dense diagonal and upper blocks."""

    def __init__(self, Hr, slab_of_col):
        self.n = int(slab_of_col.max()) + 1
        self.idx = [np.where(slab_of_col == s)[0] for s in range(self.n)]
        self.D = [Hr[np.ix_(i, i)] for i in self.idx]
        self.U = [Hr[np.ix_(i, j)] for i, j in zip(self.idx[:-1], self.idx[1:])]
        for a in range(self.n):
            for b in range(a + 2, self.n):
                assert np.abs(Hr[np.ix_(self.idx[a], self.idx[b])]).max() < 1e-12, "not block tridiagonal"

    def solve(self, E, Sig0, SigN, B):
        """Solve (E - Hr - Sig) X = B with Sig0 on slab 0 and SigN on the last slab (dense, in the
        slab's local column order). B: list of per-slab right-hand sides (None = zero)."""
        n = self.n
        S, y, lus = [], [], []
        for i in range(n):
            Di = (E * np.eye(len(self.idx[i])) - self.D[i]).astype(complex)
            if i == 0:
                Di -= Sig0
            if i == n - 1:
                Di -= SigN
            ncol = next(b.shape[1] for b in B if b is not None)
            bi = B[i] if B[i] is not None else np.zeros((len(self.idx[i]), ncol), complex)
            if i > 0:
                Ui = self.U[i - 1]                                 # (i-1, i) block; lower = Ui^T
                Li = sla.lu_solve(lus[-1], -Ui).T                  # = -Ui^T S_{i-1}^{-1} (S symmetric)
                Di = Di + Li @ Ui
                bi = bi + Li @ y[-1]
            lus.append(sla.lu_factor(Di, check_finite=False))
            y.append(bi)
        X = [None] * n
        X[-1] = sla.lu_solve(lus[-1], y[-1])
        for i in range(n - 2, -1, -1):
            X[i] = sla.lu_solve(lus[i], y[i] - self.U[i] @ X[i + 1])
        return X


def transmission_bt(bt, E, W, bL, bR, SigL, SigR):
    """T for a block-tridiagonal reduced model. W: sparse (n_dev x N) basis; bL/bR lead-coupled
    orbitals, which must lie in the first / last slab's columns."""
    W = W.tocsr()
    WL = W[bL][:, bt.idx[0]].toarray()                 # |bL| x n0
    WR = W[bR][:, bt.idx[-1]].toarray()                # |bR| x nN
    assert abs(np.abs(W[bL]).sum() - np.abs(WL).sum()) < 1e-8 * max(1, np.abs(WL).sum())
    S0 = WL.T @ SigL @ WL
    SN = WR.T @ SigR @ WR
    B = [WL.T.astype(complex)] + [None] * (bt.n - 1)   # columns: lead-coupled orbitals of the left
    X = bt.solve(E, S0, SN, B)
    Y = WR @ X[-1]                                     # G_{bR, bL}
    GL, GR = 1j * (SigL - SigL.conj().T), 1j * (SigR - SigR.conj().T)
    return float(np.real(np.trace(GR @ Y @ GL @ Y.conj().T)))


def transmission_open_pardiso(Hdev, E, SigL, SigR, bL, bR, chunk=128):
    """Exact T by solving the OPEN system (E - H - Sigma) X = B directly (no closed-device Dyson,
    so no loss of accuracy near closed-device eigenvalues). Complex system in real-equivalent form
    [[Ar, -Ai], [Ai, Ar]] factorized once with PARDISO; columns on bL; X restricted to bR."""
    n = Hdev.shape[0]
    def emb(idx, M):
        r, c = np.meshgrid(idx, idx, indexing="ij")
        return sp.csr_matrix((M.ravel(), (r.ravel(), c.ravel())), shape=(n, n))
    A = (E * sp.identity(n, format="csr") - Hdev).astype(complex) - emb(bL, SigL) - emb(bR, SigR)
    Ar, Ai = sp.csr_matrix(A.real), sp.csr_matrix(A.imag)
    M = sp.bmat([[Ar, -Ai], [Ai, Ar]], format="csr")
    s = pypardiso.PyPardisoSolver(mtype=11)
    s.factorize(M)
    Y = np.zeros((len(bR), len(bL)), complex)
    for c0 in range(0, len(bL), chunk):
        cc = np.arange(c0, min(c0 + chunk, len(bL)))
        Bm = np.zeros((2 * n, len(cc)))
        Bm[bL[cc], np.arange(len(cc))] = 1.0
        X = s.solve(M, Bm)
        for _ in range(2):
            X = X + s.solve(M, Bm - M @ X)
        Y[:, cc] = X[bR] + 1j * X[n + bR]
    s.free_memory(everything=True)
    GL, GR = 1j * (SigL - SigL.conj().T), 1j * (SigR - SigR.conj().T)
    return float(np.real(np.trace(GR @ Y @ GL @ Y.conj().T)))


def transmission_bt_deflated(bt, E, W, bL, bR, SigL, SigR, Xs):
    """T for H' = H + Xs * inf * Xs^T: the directions Xs (N x k, orthonormal, in the column order of
    the reduced model) are frozen out. Exact Woodbury limit:
      G' = G - G Xs (Xs^T G Xs)^{-1} Xs^T G,   G = (E - H - Sigma)^{-1}."""
    W = W.tocsr()
    WL = W[bL][:, bt.idx[0]].toarray()
    WR = W[bR][:, bt.idx[-1]].toarray()
    S0, SN = WL.T @ SigL @ WL, WR.T @ SigR @ WR
    k, nl = Xs.shape[1], len(bL)
    B = [np.hstack([Xs[i_].astype(complex), np.zeros((len(i_), nl), complex)]) for i_ in bt.idx]
    B[0][:, k:] = WL.T
    X = bt.solve(E, S0, SN, B)
    GX = [x[:, :k] for x in X]                          # G Xs   (per slab)
    GB = [x[:, k:] for x in X]                          # G WL^T (per slab)
    XtGX = sum(Xs[i_].T @ GX[s] for s, i_ in enumerate(bt.idx))
    XtGB = sum(Xs[i_].T @ GB[s] for s, i_ in enumerate(bt.idx))
    Gbb = GB[-1] - GX[-1] @ np.linalg.solve(XtGX, XtGB)  # rows of the last slab
    Y = WR @ Gbb
    GL, GR = 1j * (SigL - SigL.conj().T), 1j * (SigR - SigR.conj().T)
    return float(np.real(np.trace(GR @ Y @ GL @ Y.conj().T)))
