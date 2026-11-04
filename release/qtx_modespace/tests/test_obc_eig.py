"""OBC eigenmodes (obc_eig) on a 1D chain with a double barrier: the continued lead Sigma against the closed form
Sigma(z) = (z - i sqrt(4 t^2 - z^2)) / 2 (analytic through the band, both half planes), and Beyn's eigenpairs against
the residual and the argument principle (winding number of det T(z) on the contour).

    PYTHONPATH=src python -m pytest tests -q
"""
import numpy as np
import scipy.sparse as sp

from qtx_modespace import obc_eig as oe
from qtx_modespace.core import Device, slab_blocks

T_HOP, CELL, NB = 1.0, 3, 8


def chain(barrier=1.5, sites=(9, 14)):
    n = CELL * NB
    H = sp.diags([-T_HOP * np.ones(n - 1), -T_HOP * np.ones(n - 1)], [-1, 1], shape=(n, n)).tolil()
    for i in sites:
        H[i, i] = barrier
    H, S = H.tocsr(), sp.identity(n, format="csr")
    c0, c1 = np.arange(CELL), np.arange(CELL, 2 * CELL)
    lead = dict(H00=H[c0][:, c0], H01=H[c0][:, c1], S00=S[c0][:, c0], S01=S[c0][:, c1])
    meta = dict(axis=0, transport=dict(device_blocks=list(range(1, NB - 1)), lead_left_block=0, lead_right_block=NB - 1),
                edges=dict(Ec_lead=-2.0, Ev_lead=-2.0))
    return dict(H=H, S=S, norb=np.ones(n, int), blocks=np.repeat(np.arange(NB), CELL), lead=lead, lead_right=None,
                labels=np.array(["X"] * n), xyz=np.c_[np.arange(n, dtype=float), np.zeros(n), np.zeros(n)],
                perm=np.arange(n), meta=meta)


def sigma_exact(z):
    return 0.5 * (z - 1j * np.sqrt(4 * T_HOP ** 2 - z ** 2))


def test_continued_sigma_closed_form():
    dev = Device(chain())
    nprop = oe.channel_counts(dev.lead, 0.4)
    assert nprop == (1, 1)
    sig = oe.ContinuedSigma(dev, nprop)
    for z in (0.4 + 0.05j, 0.4 - 0.05j, 0.3 - 0.2j, -0.7 - 0.1j):
        SL, SR = sig(z)
        assert abs(SL[0, 0] - sigma_exact(z)) < 1e-9 and abs(SR[0, 0] - sigma_exact(z)) < 1e-9, z


def test_beyn_resonances():
    inp = chain()
    dev = Device(inp)
    c, a, b, npts = 0.0, 1.2, 0.4, 32
    sig = oe.ContinuedSigma(dev, oe.channel_counts(dev.lead, c))
    zs, ws = oe.ellipse(c, a, b, npts)
    V = np.random.default_rng(0).standard_normal((dev.n, 8)) + 0j
    blk = slab_blocks(dev)
    Xs = [oe.solve_point(dev, sig, z, V, blk) for z in zs]
    lam, vec, s, k = oe.beyn(Xs, zs, ws, 1, 1e-10, c, a)
    ins = ((lam.real - c) / a) ** 2 + (lam.imag / b) ** 2 < 1
    lam, vec = lam[ins], vec[:, ins]
    # argument principle: number of zeros of det T inside = winding number of det T on a fine contour
    zf, _ = oe.ellipse(c, a, b, 300)
    ld = [np.linalg.slogdet(dev.A(z, *sig(z)).toarray()) for z in zf]
    ph = np.unwrap(np.angle([sgn for sgn, _ in ld]))
    winding = int(round((ph[-1] - ph[0] + (ph[1] - ph[0])) / (2 * np.pi)))
    assert len(lam) == winding > 0
    assert (lam.imag < 0).all()
    assert max(oe.residuals(dev, sig, lam, vec)) < 1e-8
    w = oe.atom_weights(inp, dev, vec[:, 0])
    assert abs(w.sum() - 1) < 1e-12 and w[:CELL].sum() == 0
