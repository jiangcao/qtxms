"""Regression of the contour recursion (core.g_diag, core.rho_contour) against the 0.1.0 recursion and a dense
inverse, on small synthetic block-tridiagonal devices (real and complex H, non-orthogonal S).

    PYTHONPATH=src python -m pytest tests -q
"""
import types
import numpy as np
import scipy.sparse as sp
import pytest

from qtx_modespace import core
from ref010 import g_diag_010


def device(cplx, sizes=(7, 9, 8, 6), nL=3, nR=4, seed=0):
    """Random Hermitian block-tridiagonal H (real or complex), S = I + small symmetric blocks, and a Sigma(z) that is
    complex symmetric for real H (as the decimated Sigma of a real lead) with Im < 0."""
    rng = np.random.default_rng(seed)
    sb = np.concatenate([[0], np.cumsum(sizes)])
    n, k = sb[-1], len(sizes)
    H = np.zeros((n, n), complex if cplx else float)
    S = np.eye(n)
    rnd = lambda *sh: rng.standard_normal(sh) + (1j * rng.standard_normal(sh) if cplx else 0)
    for s in range(k):
        a, b = sb[s], sb[s + 1]
        M = rnd(b - a, b - a); H[a:b, a:b] = M + M.conj().T
        Q = 0.1 * rng.standard_normal((b - a, b - a)); S[a:b, a:b] += Q + Q.T
        if s < k - 1:
            c = sb[s + 2]
            C = 0.5 * rnd(b - a, c - b); H[a:b, b:c] = C; H[b:c, a:b] = C.conj().T
            Q = 0.05 * rng.standard_normal((b - a, c - b)); S[a:b, b:c] = Q; S[b:c, a:b] = Q.T
    KL, KR = rng.standard_normal((nL, nL)), rng.standard_normal((nR, nR))
    WL, WR = rng.standard_normal((nL, nL)), rng.standard_normal((nR, nR))
    sig = lambda z: (0.3 * (KL + KL.T) * (1 + 0.1 * z) - 0.5j * (np.eye(nL) + 0.1 * (WL @ WL.T)),
                     0.3 * (KR + KR.T) - 0.4j * (np.eye(nR) + 0.1 * (WR @ WR.T)))
    return types.SimpleNamespace(H=sp.csr_matrix(H), S=sp.csr_matrix(S), sb=sb, ns=list(sizes),
                                 bL=np.arange(nL), bR=np.arange(n - nR, n), sigma=sig)


def dense_diag(d, z):
    """G_ss from the dense inverse of the whole device."""
    SL, SR = d.sigma(z)
    A = z * d.S.toarray() - d.H.toarray()
    A[np.ix_(d.bL, d.bL)] -= SL
    A[np.ix_(d.bR, d.bR)] -= SR
    G = np.linalg.inv(A)
    return [G[d.sb[s]:d.sb[s + 1], d.sb[s]:d.sb[s + 1]] for s in range(len(d.ns))]


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("z", [0.3 + 0.02j, -1.2 + 0.7j, 2.0 + 1e-6j])
def test_g_diag(cplx, z):
    d = device(cplx)
    blk = core.slab_blocks(d)
    sym = core.is_real(blk)
    assert sym == (not cplx)
    SL, SR = d.sigma(z)
    iL, iR = d.bL, d.bR - d.sb[-2]
    new = core.g_diag(blk, z, SL, SR, iL, iR, sym)
    ref = g_diag_010(blk, z, SL, SR, iL, iR)
    den = dense_diag(d, z)
    scale = max(np.abs(g).max() for g in den)
    for gn, gr, gd in zip(new, ref, den):
        assert np.abs(gn - gr).max() < 1e-11 * scale
        assert np.abs(gn - gd).max() < 1e-11 * scale
    if sym:                                    # complex symmetric A: the general branch gives the same blocks
        gen = core.g_diag(blk, z, SL, SR, iL, iR, False)
        assert max(np.abs(a - b).max() for a, b in zip(new, gen)) < 1e-11 * scale


@pytest.mark.parametrize("cplx", [False, True])
def test_rho_contour(cplx):
    """rho_contour = -(1/pi) Im of the quadrature of the dense G_ss on the same contour; Hermitian."""
    d = device(cplx, seed=3)
    mu1, mu2, npts = -1.0, 1.5, 12
    rho = core.rho_contour(d, mu1, mu2, npts, log=lambda *a: None)
    c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    x, w = np.polynomial.legendre.leggauss(npts)
    th, w = 0.5 * np.pi * (x + 1), 0.5 * np.pi * w
    acc = [np.zeros((n, n), complex) for n in d.ns]
    for t_, w_ in zip(th, w):
        z = c + r * np.exp(1j * t_)
        for a, G in zip(acc, dense_diag(d, z)):
            a += G * (-1j * r * np.exp(1j * t_) * w_)
    ref = [-((a - a.conj().T) / 2j) / np.pi for a in acc]
    scale = max(np.abs(a).max() for a in ref)
    for p, q in zip(rho, ref):
        assert np.abs(p - q).max() < 1e-11 * scale
        assert np.abs(p - p.conj().T).max() < 1e-14 * scale
