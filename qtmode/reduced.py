"""Reduced (mode-space) device models with slab-block-tridiagonal structure.

A model is  H_r (N x N, real symmetric), the projected gate profile P_r = W^T diag(p) W, the rows of
W on the lead-coupled orbitals (WL on bL, WR on bR) and the slab of every column (columns ordered
slab by slab). W = U (level 1) or U V (level 2) is never formed for level 2: only its boundary rows
and the projected profile are needed.

T(E; u) for H_r + u P_r with the exact lead self-energies, by a block-Thomas sweep:
  A = E - H_r - u P_r - WL^T Sigma_L WL - WR^T Sigma_R WR   (complex symmetric)
  right-to-left Schur complements  S_s = A_ss - H_s,s+1 S_s+1^-1 H_s+1,s,  M_s+1 = S_s+1^-1 H_s+1,s
  X_0 = S_0^-1 WL^T Gamma_L^1/2,  X_s+1 = M_s+1 X_s,  T = || (WR Gamma_R^1/2)^+ X_last ||_F^2
"""
import numpy as np
import scipy.linalg as sla


class Model:
    def __init__(self, H, P, WL, WR, slab, name=""):
        self.H, self.P, self.name = H, P, name
        self.WL, self.WR = np.asarray(WL), np.asarray(WR)
        self.slab = np.asarray(slab)
        self.ns = int(self.slab.max()) + 1
        assert np.all(np.diff(self.slab) >= 0), "columns must be ordered slab by slab"
        self.idx = [np.where(self.slab == s)[0] for s in range(self.ns)]
        self.sl = [slice(i[0], i[-1] + 1) for i in self.idx]
        self.N = H.shape[0]
        # block-tridiagonal check
        for s in range(self.ns):
            for t in range(s + 2, self.ns):
                assert np.abs(H[self.sl[s], self.sl[t]]).max() < 1e-10
        self.cL = np.where(np.abs(self.WL).sum(0) > 0)[0]
        self.cR = np.where(np.abs(self.WR).sum(0) > 0)[0]
        assert np.all(self.slab[self.cL] == 0) and np.all(self.slab[self.cR] == self.ns - 1)

    def transmission(self, E, u, SL, SR, gL, gR):
        """gL, gR: Gamma^{1/2} of the real-space self-energies (columns = channels)."""
        if gL.shape[1] == 0 or gR.shape[1] == 0:
            return 0.0
        sl, H = self.sl, self.H
        S = self.ns
        WL0 = self.WL[:, sl[0]]
        WRl = self.WR[:, sl[-1]]

        def Ablk(s):
            A = -(H[sl[s], sl[s]] + u * self.P[sl[s], sl[s]]).astype(complex)
            A[np.diag_indices_from(A)] += E
            if s == 0:
                A -= WL0.T @ SL @ WL0
            if s == S - 1:
                A -= WRl.T @ SR @ WRl
            return A

        M = [None] * S
        Sc = Ablk(S - 1)
        for s in range(S - 2, -1, -1):
            Hsn = H[sl[s], sl[s + 1]]
            M[s + 1] = sla.solve(Sc, Hsn.T.astype(complex), check_finite=False, overwrite_a=True)
            Sc = Ablk(s) - Hsn @ M[s + 1]
        X = sla.solve(Sc, WL0.T @ gL, check_finite=False, overwrite_a=True)
        for s in range(1, S):
            X = M[s] @ X
        return float(np.linalg.norm((WRl @ X).T @ gR.conj()) ** 2)


def bt_solve(m, E, u, B, SL=None, SR=None):
    """Full solution X of A X = B on a Model (block Thomas). With SL, SR: open system (complex);
    without: closed system E - H - u P (real if E, B real)."""
    import scipy.linalg as sla
    sl, H, S = m.sl, m.H, m.ns
    cplx = SL is not None or np.iscomplexobj(B)
    dt = complex if cplx else float

    def Ablk(s):
        A = -(H[sl[s], sl[s]] + u * m.P[sl[s], sl[s]]).astype(dt)
        A[np.diag_indices_from(A)] += E
        if SL is not None and s == 0:
            W = m.WL[:, sl[0]]
            A -= W.T @ SL @ W
        if SR is not None and s == S - 1:
            W = m.WR[:, sl[-1]]
            A -= W.T @ SR @ W
        return A

    lus, ys = [None] * S, [None] * S
    Sc = Ablk(S - 1)
    y = B[sl[S - 1]].astype(dt)
    for s in range(S - 1, -1, -1):
        lus[s] = sla.lu_factor(Sc, check_finite=False, overwrite_a=True)
        ys[s] = y
        if s == 0:
            break
        Hps = H[sl[s - 1], sl[s]]                          # A_{s-1,s} = -Hps
        Z = sla.lu_solve(lus[s], np.hstack([Hps.T.astype(dt), y]), check_finite=False)
        k = Hps.shape[0]
        Sc = Ablk(s - 1) - Hps @ Z[:, :k]
        y = B[sl[s - 1]].astype(dt) + Hps @ Z[:, k:]
    X = np.zeros(B.shape, dt)
    X[sl[0]] = sla.lu_solve(lus[0], ys[0], check_finite=False)
    for s in range(1, S):
        X[sl[s]] = sla.lu_solve(lus[s], ys[s] + H[sl[s], sl[s - 1]] @ X[sl[s - 1]], check_finite=False)
    return X


def gap_states(H, G, emin, emax, P=None, gate_slabs=None, slab=None):
    """Eigenpairs of H with eigenvalue in [emin, emax], their Ritz residuals ||(H_full - lam) psi||
    = sqrt(psi^T G psi - lam^2) (G = W^T H_full^2 W), and optionally their weight on gate_slabs."""
    lam, Y = sla.eigh(H, subset_by_value=(emin, emax), driver="evr")
    if len(lam) == 0:
        return lam, lam, lam
    r = np.sqrt(np.clip(np.einsum("ij,ij->j", Y, G @ Y) - lam ** 2, 0, None))
    w = np.zeros_like(lam)
    if gate_slabs is not None:
        m = np.isin(slab, gate_slabs)
        w = (Y[m] ** 2).sum(0)
    return lam, r, w
