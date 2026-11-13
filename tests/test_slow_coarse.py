"""qtx_ms.schwarz.SlowCoarse (copy of the release test) on a synthetic indefinite open system: square-lattice strip at an in-band energy (Helmholtz-like,
many wavelengths), absorbing ends (open leads), restricted additive Schwarz windows along x with absorbing ramps on the
cuts. (A 1D chain is degenerate: the RAS error operator has rank 2 per cut, so the slow modes make the cycle exact.)"""
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

from qtx_ms.schwarz import SlowCoarse, fgmres_block


def strip(nx=240, ny=10, E=0.9, t=1.0, nabs=30, s=1.0):
    """Square-lattice strip (nx x ny sites, index x ny + y) at an in-band energy, absorbing ramps at both x ends."""
    Hx = sp.diags([np.full(nx - 1, -t), np.full(nx - 1, -t)], [-1, 1])
    Hy = sp.diags([np.full(ny - 1, -t), np.full(ny - 1, -t)], [-1, 1])
    H = sp.kron(Hx, sp.eye(ny)) + sp.kron(sp.eye(nx), Hy)
    ramp = np.zeros(nx)
    x = np.arange(nabs)[::-1] / nabs
    ramp[:nabs], ramp[nx - nabs:] = x ** 2, x[::-1] ** 2
    W = sp.diags(np.repeat(ramp, ny))
    return (E * sp.eye(nx * ny) - H + 1j * s * W).tocsr().astype(complex)


class RAS:
    """Windows of w columns (x) every w - 2 ov columns, absorbing ramp of d columns on the inner cuts, LU per window,
    correction kept on the window core."""

    def __init__(self, A, ny=10, w=24, ov=4, d=3, s=1.0):
        n = A.shape[0] // ny
        self.win = []
        starts = list(range(0, n - w, w - 2 * ov)) + [n - w]
        for k, a in enumerate(starts):
            idx = np.arange(a * ny, (a + w) * ny)
            p = np.zeros(w)
            r = (np.arange(d)[::-1] + 1) / d
            if a > 0:
                p[:d] = r ** 2
            if a + w < n:
                p[w - d:] = r[::-1] ** 2
            lu = sla.lu_factor(A[idx][:, idx].toarray() + 1j * s * np.diag(np.repeat(p, ny)))
            lo = 0 if k == 0 else (starts[k - 1] + w + a) // 2 - a
            hi = w if k == len(starts) - 1 else (a + w + starts[k + 1]) // 2 - a
            self.win.append((idx, lu, slice(lo * ny, hi * ny)))

    def __call__(self, r):
        z = np.zeros_like(r)
        for idx, lu, core in self.win:
            z[idx[core]] = sla.lu_solve(lu, r[idx])[core]
        return z


def setup():
    A = strip()
    M = RAS(A)
    rng = np.random.default_rng(1)
    b = np.zeros((A.shape[0], 3), complex)
    b[:30] = rng.standard_normal((30, 3))
    return A, M, b


def test_slow_coarse_space():
    A, M, b = setup()
    sc = SlowCoarse(A, M, m=16, q=3, rounds=3, log=lambda *x: None)
    Q = sc.Q
    assert Q.shape == (A.shape[0], 48) and sc.N == 48
    assert np.allclose(Q.conj().T @ Q, np.eye(48), atol=1e-10)
    r = b[:, :1]
    assert np.linalg.norm(Q.conj().T @ (A @ sc(r) - r)) < 1e-10 * np.linalg.norm(r)      # Galerkin
    assert [d["N"] for d in sc.info["rounds"]] == [16, 32, 48]


def test_slow_coarse_fgmres():
    A, M, b = setup()
    _, its1, res1 = fgmres_block(A, b, np.zeros_like(b), M, 1e-10, 300)
    sc = SlowCoarse(A, M, m=16, q=3, rounds=3, log=lambda *x: None)
    x, its2, res2 = fgmres_block(A, b, sc(b), sc.cycle, 1e-10, 300)
    xe = sp.linalg.spsolve(A.tocsc(), b)
    assert res2.max() < 1e-9 and np.linalg.norm(x - xe) < 1e-8 * np.linalg.norm(xe)
    assert its2.max() < its1.max()
