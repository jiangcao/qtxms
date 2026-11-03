"""Effective-mass rough Si nanosheet toy device, uniform leads, RGF reference.

Grid: x (transport) slabs of ny*nz points, finite differences, spacing a.
Roughness: top/bottom surfaces fluctuate (Gaussian-correlated), outside = oxide barrier.
"""
import numpy as np
import scipy.sparse as sp

HB2_2M0 = 0.0380998  # eV nm^2


class Device:
    def __init__(self, L=20.0, W=5.0, T=3.0, a=0.25, mstar=0.26, Vox=3.0,
                 rms=0.3, corr=1.5, n_ideal=6, pad=2, seed=1):
        self.a = a
        self.t = HB2_2M0 / (mstar * a * a)
        self.ny = int(round(W / a))
        self.nz = int(round(T / a)) + 2 * pad
        self.S = int(round(L / a))
        self.ns = self.ny * self.nz
        self.n = self.S * self.ns
        z = (np.arange(self.nz) - pad + 0.5) * a  # cell centres; ideal sheet 0..T
        rng = np.random.default_rng(seed)

        def surface():
            f = rng.standard_normal((self.S, self.ny))
            kx = np.fft.fftfreq(self.S, a) * 2 * np.pi
            ky = np.fft.fftfreq(self.ny, a) * 2 * np.pi
            K2 = kx[:, None] ** 2 + ky[None, :] ** 2
            f = np.real(np.fft.ifft2(np.fft.fft2(f) * np.exp(-K2 * corr ** 2 / 8)))
            return f / f.std() * rms

        top = T + surface()
        bot = 0.0 + surface()
        top[:n_ideal] = T; top[-n_ideal:] = T
        bot[:n_ideal] = 0.0; bot[-n_ideal:] = 0.0
        self.V = np.zeros((self.S, self.ny, self.nz))
        for i in range(self.S):
            for j in range(self.ny):
                out = (z < bot[i, j]) | (z > top[i, j])
                self.V[i, j, out] = Vox
        self.Vlead = self.V[0].copy()
        # transverse 2D Laplacian (Dirichlet at y, z ends)
        t = self.t
        Ty = sp.diags([-t, 2 * t, -t], [-1, 0, 1], shape=(self.ny, self.ny))
        Tz = sp.diags([-t, 2 * t, -t], [-1, 0, 1], shape=(self.nz, self.nz))
        self.Tperp = (sp.kron(Ty, sp.identity(self.nz)) + sp.kron(sp.identity(self.ny), Tz)).tocsr()

    def slab_H(self, i):
        """Diagonal block H_ii (includes 2t from the x-direction)."""
        return (self.Tperp + sp.diags(2 * self.t + self.V[i].ravel())).toarray()

    def lead_H(self):
        return (self.Tperp + sp.diags(2 * self.t + self.Vlead.ravel())).toarray()

    def full_H(self):
        blocks = [sp.csr_matrix(self.slab_H(i)) for i in range(self.S)]
        D = sp.block_diag(blocks, format="csr")
        off = sp.kron(sp.diags([1.0, 1.0], [-1, 1], shape=(self.S, self.S)), sp.identity(self.ns))
        return (D - self.t * off).tocsr()

    # ---------- uniform lead self-energy via transverse modes ----------
    def lead_modes(self):
        if not hasattr(self, "_lm"):
            eps, W = np.linalg.eigh(self.lead_H())
            self._lm = (eps, W)
        return self._lm

    def sigma(self, E, eta=1e-9):
        """Retarded surface self-energy (same for L and R by symmetry)."""
        eps, W = self.lead_modes()
        t = self.t
        z = E + 1j * eta - eps
        s = np.sqrt(z * z - 4 * t * t + 0j)
        g1 = (z - s) / (2 * t * t)
        g2 = (z + s) / (2 * t * t)
        g = np.where(np.abs(t * g1) <= 1.0, g1, g2)
        sig = t * t * g
        return (W * sig) @ W.conj().T, sig

    def gamma_inj(self, E, tol=1e-6):
        """Columns B with Gamma_L = B B^dagger (propagating modes only)."""
        Sig, sig = self.sigma(E)
        eps, W = self.lead_modes()
        gam = -2 * sig.imag
        k = gam > tol * self.t
        B = np.zeros((self.n, k.sum()), complex)
        B[:self.ns] = W[:, k] * np.sqrt(gam[k])
        return B, Sig


def rgf_column(dev, E, Hblocks):
    """Exact first block-column G_{i1} and transmission via RGF."""
    S, t = dev.S, dev.t
    Sig, _ = dev.sigma(E)
    I = np.eye(dev.ns)
    A = [E * I - Hblocks[i] for i in range(S)]
    A[0] = A[0] - Sig
    A[-1] = A[-1] - Sig
    gR = [None] * S
    gR[-1] = np.linalg.inv(A[-1])
    for i in range(S - 2, -1, -1):
        gR[i] = np.linalg.inv(A[i] - t * t * gR[i + 1])
    G = [None] * S
    G[0] = gR[0]
    for i in range(1, S):
        G[i] = -t * (gR[i] @ G[i - 1])
    Gam = 1j * (Sig - Sig.conj().T)
    T = np.real(np.trace(Gam @ G[-1] @ Gam @ G[-1].conj().T))
    return G, T
