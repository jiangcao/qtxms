"""Reduced (mode-space) semi-infinite lead.

The lead cell (H00, H01) is represented in a real orthonormal basis Phi built once from the lead's
own Bloch states: eigenvectors u_nk of H(k) = H00 + H01 e^{ik} + H01^T e^{-ik} with E_nk in an
energy window, on a k grid over [0, pi]; Re u and Im u are stacked and compressed by SVD
(u_{-k} = u_k^*, so [0, pi] spans the whole zone).

  H00r = Phi^T H00 Phi,  H01r = Phi^T H01 Phi    (energy independent)
  g_surf(E) ~ Phi_F g_r(E) Phi_F^T               (Lopez-Sancho on the reduced chain)
  Sigma_L = (Phi_Rf^T A)^T g_L,r (Phi_Rf^T A),   Sigma_R = (B Phi_Lf) g_R,r (B Phi_Lf)^T

with A, B the lead-device couplings on the lead faces, as in transport.Lead.sigmas.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla


def Hk(H00, H01, k):
    return (H00 + H01 * np.exp(1j * k) + H01.T * np.exp(-1j * k)).tocsc()


def lead_bloch_states(H00, H01, ks, ewin, sigma=1.10, k0=60, log=print):
    """Eigenpairs of H(k) with E in ewin for each k; grows the number of requested eigenpairs until
    the largest returned eigenvalue exceeds ewin[1]."""
    out = []
    kk = k0
    for k in ks:
        A = Hk(H00, H01, k)
        while True:
            w, V = spla.eigsh(A, k=kk, sigma=sigma, which="LM", tol=1e-10)
            if w.max() > ewin[1] or kk >= 600:
                break
            kk = int(kk * 1.4)
        m = (w > ewin[0]) & (w < ewin[1])
        o = np.argsort(w[m])
        out.append((k, w[m][o], V[:, m][:, o]))
    return out


def lead_basis(states, thr=1e-6):
    """Real orthonormal basis of span{Re u, Im u}. Returns Phi and the singular values."""
    Y = np.hstack([np.hstack([V.real, V.imag]) for _, _, V in states])
    U, s, _ = np.linalg.svd(Y, full_matrices=False)
    m = s > thr * s[0]
    return U[:, m], s


def sancho_dense(H00, H01, z, tol=1e-12, maxit=300):
    """Surface GF of the chain extending to the right (cell n -> n+1 via H01)."""
    n = len(H00)
    I = np.eye(n)
    es, e = H00.astype(complex), H00.astype(complex)
    a, b = H01.astype(complex), H01.conj().T.astype(complex)
    for it in range(maxit):
        X = sla.solve(z * I - e, np.hstack([a, b]), check_finite=False)
        ga, gb = X[:, :n], X[:, n:]
        agb, bga = a @ gb, b @ ga
        es = es + agb
        e = e + agb + bga
        a, b = a @ ga, b @ gb
        if max(np.abs(a).max(), np.abs(b).max()) < tol:
            break
    return sla.inv(z * I - es, check_finite=False), it + 1


class ReducedLead:
    def __init__(self, lead, Phi):
        """lead: transport.Lead (exact); Phi: (n_cell x m) real orthonormal basis."""
        self.lead = lead
        self.Phi = Phi
        H00, H01 = lead.H00, lead.H01
        self.H00r = Phi.T @ (H00 @ Phi)
        self.H00r = 0.5 * (self.H00r + self.H00r.T)
        self.H01r = Phi.T @ (H01 @ Phi)
        self.PRf = Phi[lead.Rf]          # rows on the right face (couple to the next cell)
        self.PLf = Phi[lead.Lf]          # rows on the left face

    def bands(self, ks):
        return np.array([np.linalg.eigvalsh(self.H00r + self.H01r * np.exp(1j * k)
                                            + self.H01r.T * np.exp(-1j * k)) for k in ks])

    def sigmas(self, E, VL, VR, eta=1e-6):
        z = E + 1j * eta
        gR, n1 = sancho_dense(self.H00r, self.H01r, z)        # right lead, first cell
        gL, n2 = sancho_dense(self.H00r, self.H01r.T, z)      # left lead, last cell
        VL, VR = VL.tocsr(), VR.tocsc()
        bL = np.unique(VL.nonzero()[1])
        bR = np.unique(VR.nonzero()[0])
        A = VL[self.lead.Rf][:, bL].toarray()                 # |Rf| x |bL|
        B = VR[bR][:, self.lead.Lf].toarray()                 # |bR| x |Lf|
        Ar = self.PRf.T @ A                                   # m x |bL|
        Br = B @ self.PLf                                     # |bR| x m
        return Ar.T @ gL @ Ar, Br @ gR @ Br.T, bL, bR, max(n1, n2)


# ======================================================================================
# Face-exact lead with a reduced interior basis
# ======================================================================================
def _sancho_one(h00, h01, z, tol=1e-11, maxit=300, h10=None):
    """Lopez-Sancho for the chain extending to the right (cell n -> n+1 via h01, n+1 -> n via h10).
    Returns the 'bulk-side' surface on-site matrix es (h00 plus the self-energy of cells 2..inf).
    h10 defaults to h01^T: effective chains obtained by eliminating sites at complex z are
    complex symmetric, not Hermitian."""
    n = len(h00)
    I = np.eye(n)
    e = h00.astype(complex)
    es = e.copy()
    a = h01.astype(complex)
    b = (h01.T if h10 is None else h10).astype(complex)
    for it in range(maxit):
        X = sla.solve(z * I - e, np.hstack([a, b]), check_finite=False)
        ga, gb = X[:, :n], X[:, n:]
        agb, bga = a @ gb, b @ ga
        es += agb
        e += agb + bga
        a, b = a @ ga, b @ gb
        if max(np.abs(a).max(), np.abs(b).max()) < tol:
            break
    return es, it + 1


def face_surface_gfs(Heff, iL, iR, V, z):
    """Exact surface GFs from the effective face chain (cell = Lf u Rf, Heff on F, V: Rf_n -> Lf_n+1),
    computed on one face at a time:
      right-extending lead, first cell, on Lf: eliminate every Rf -> chain on Lf
      left-extending lead, last cell, on Rf:   eliminate every Lf -> chain on Rf
    The surface cell lacks the neighbour on its open side; that term is removed from the bulk
    surface on-site matrix returned by Sancho."""
    HLL, HLR = Heff[np.ix_(iL, iL)], Heff[np.ix_(iL, iR)]
    HRR, HRL = Heff[np.ix_(iR, iR)], Heff[np.ix_(iR, iL)]
    IR, IL = np.eye(len(iR)), np.eye(len(iL))
    # chain on Lf (eliminate Rf_n, which couples L_n (HLR) and L_n+1 (V^T))
    gRR = sla.solve(z * IR - HRR, np.hstack([HRL, V]), check_finite=False)
    gRR_HRL, gRR_V = gRR[:, :len(iL)], gRR[:, len(iL):]
    h00 = HLL + HLR @ gRR_HRL + V.T @ gRR_V
    h01 = HLR @ gRR_V                                       # L_n -> L_n+1
    es, n1 = _sancho_one(h00, h01, z)
    gR = sla.inv(z * IL - (es - V.T @ gRR_V), check_finite=False)   # first cell: no R_0
    # chain on Rf (eliminate Lf_n, which couples R_n (HRL) and R_n-1 (V))
    gLL = sla.solve(z * IL - HLL, np.hstack([HLR, V.T]), check_finite=False)
    gLL_HLR, gLL_VT = gLL[:, :len(iR)], gLL[:, len(iR):]
    k00 = HRR + HRL @ gLL_HLR + V @ gLL_VT
    k01 = V @ gLL_HLR                                       # R_n -> R_n+1 (through L_n+1)
    es, n2 = _sancho_one(k00, k01.T, z)                      # chain extending to the left
    gL = sla.inv(z * IR - (es - V @ gLL_VT), check_finite=False)    # last cell: no L_n+1
    return gL, gR, max(n1, n2)


class FaceExactLead:
    """Lead cell in the basis [identity on the coupling faces F] + [reduced basis Q on the interior].
    The interior is folded onto F by a dense Galerkin solve in Q at every energy."""

    def __init__(self, lead, Q):
        self.lead = lead
        H = lead.H00
        F, I = lead.F, lead.I
        self.Q = Q
        self.HFF = H[F][:, F].toarray()
        self.HFIQ = H[F][:, I] @ Q                          # |F| x nQ
        HQQ = Q.T @ (H[I][:, I] @ Q)
        self.lam, self.Z = np.linalg.eigh(0.5 * (HQQ + HQQ.T))
        self.C = self.HFIQ @ self.Z                         # |F| x nQ, in the eigenbasis of HQQ
        self.V = lead.H01[F][:, F].toarray()
        self.iL = np.array([lead.posF[c] for c in lead.Lf])
        self.iR = np.array([lead.posF[r] for r in lead.Rf])

    def heff(self, E):
        return self.HFF + (self.C / (E - self.lam)) @ self.C.T

    def sigmas(self, E, VL, VR, eta=1e-6):
        Heff = self.heff(E)
        Vv = self.V[np.ix_(self.iR, self.iL)]
        fn = face_surface_gfs_small if len(self.iL) <= len(self.iR) else face_surface_gfs
        gL, gR, nit = fn(Heff, self.iL, self.iR, Vv, E + 1j * eta)
        VL, VR = VL.tocsr(), VR.tocsc()
        bL = np.unique(VL.nonzero()[1])
        bR = np.unique(VR.nonzero()[0])
        A = VL[self.lead.Rf][:, bL].toarray()
        B = VR[bR][:, self.lead.Lf].toarray()
        return A.T @ gL @ A, B @ gR @ B.T, bL, bR, nit


def interior_basis(lead, E_train, ewin_eig, n_eig=200, thr=1e-6, solve=None, log=print):
    """Reduced interior basis: interior eigenvectors with lambda in ewin_eig (exact poles) plus the
    compressed response (E_j - H_II)^-1 H_IF at the training energies (smooth background)."""
    H = lead.H00
    F, I = lead.F, lead.I
    HII = H[I][:, I].tocsc()
    HIF = H[I][:, F].toarray()
    sig0 = 0.5 * (ewin_eig[0] + ewin_eig[1])
    w, Vv = spla.eigsh(HII, k=n_eig, sigma=sig0, which="LM", tol=1e-10)
    m = (w > ewin_eig[0]) & (w < ewin_eig[1])
    if not (w.min() < ewin_eig[0] and w.max() > ewin_eig[1]):
        log(f"    warning: {n_eig} interior eigenpairs cover only [{w.min():.3f}, {w.max():.3f}]")
    P = Vv[:, m]
    del Vv
    Q = P.copy()
    for Ej in E_train:                     # incremental: one training energy at a time (memory)
        X = solve(Ej * sp.identity(len(I), format="csr") - HII, HIF)
        for _ in range(2):                 # project out the current basis (twice for stability)
            X -= Q @ (Q.T @ X)
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        del X
        s0 = s[0] if Q.shape[1] == P.shape[1] else s0
        keep = s > thr * s0
        Q = np.hstack([Q, U[:, keep]])
        del U
    Q, _ = np.linalg.qr(Q)
    return Q, int(m.sum()), w


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
