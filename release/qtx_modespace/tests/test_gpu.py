"""GPU backend (cupy) against the CPU path on the small synthetic devices of the other tests. Skipped without a
working cupy / CUDA device.

    PYTHONPATH=src python -m pytest tests/test_gpu.py -q
"""
import types
import numpy as np
import scipy.sparse as sp
import pytest

from qtx_modespace import core, gpu, level1, schwarz
from test_level1 import wire, CONTOUR
from test_rho_contour import device

try:
    gpu.enable()
    gpu.enable(False)
    HAVE_GPU = True
except RuntimeError:
    HAVE_GPU = False
pytestmark = pytest.mark.skipif(not HAVE_GPU, reason="cupy / CUDA not available")


@pytest.fixture
def on_gpu():
    gpu.enable()
    yield
    gpu.enable(False)


def rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-300)


@pytest.mark.parametrize("cplx", [False, True])
def test_sigma_folded(cplx):
    inp = wire(cplx=cplx)
    dev = core.Device(inp, obc="folded")
    z = -0.3 + 0.05j
    ref = dev.sigma(z)
    gpu.enable()
    try:
        got = dev.sigma(z)
    finally:
        gpu.enable(False)
    assert all(isinstance(g, np.ndarray) for g in got)
    assert max(rel(g, r) for g, r in zip(got, ref)) < 1e-11


@pytest.mark.parametrize("cplx", [False, True])
def test_rho_contour(cplx):
    d = device(cplx, seed=3)
    ref = core.rho_contour(d, -1.0, 1.5, 12, log=lambda *a: None)
    gpu.enable()
    try:
        got = core.rho_contour(d, -1.0, 1.5, 12, log=lambda *a: None)
    finally:
        gpu.enable(False)
    scale = max(np.abs(r).max() for r in ref)
    assert max(np.abs(g - r).max() for g, r in zip(got, ref)) < 1e-11 * scale


def test_local_rho(on_gpu):
    inp = wire()
    dev = core.Device(inp)
    geo = level1.Geometry(inp["xyz"])
    b = level1.partition(inp, dev, "tile", tile_orbs=12)[3]
    e = level1.embedding(inp, geo, geo.tree(), b["atoms"], 3.0, 1.5, 0.5, obc=dev)
    for kw, tol in ((dict(gpp="full"), 1e-11), (dict(gpp="schur"), 1e-11),
                    (dict(gpp="schur", env_prec="single", refine=2), 1e-7)):
        got, _ = level1.local_rho(e, CONTOUR, **kw)
        gpu.enable(False)
        ref, _ = level1.local_rho(e, CONTOUR, **kw)
        gpu.enable()
        assert isinstance(got, np.ndarray) and rel(got, ref) < tol, kw


def test_eigh_top_modes(on_gpu):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 40)) + 1j * rng.standard_normal((40, 40))
    R = X @ X.conj().T
    lam, U = core.top_modes(R, 5)
    gpu.enable(False)
    lr, Ur = core.top_modes(R, 5)
    assert rel(lam, lr) < 1e-12
    assert np.abs(np.abs(np.sum(U.conj() * Ur, axis=0)) - 1).max() < 1e-10       # same vectors up to a phase


@pytest.mark.parametrize("cplx", [False, True])
def test_exact_solve_block_lu(cplx, on_gpu):
    inp = wire(nb=8, cplx=cplx)
    dev = core.Device(inp)
    E = -0.4
    SL, SR = dev.sigma(E + 1e-6j)
    B = core.injection(SL)
    b = np.zeros((dev.n, B.shape[1]), complex)
    b[dev.bL] = B
    A = dev.A(E, SL, SR)
    X, res = core.exact_solve(dev, A, b)
    assert isinstance(X, np.ndarray) and res < 1e-12
    Xr = sp.linalg.spsolve(A.tocsc(), b)
    assert rel(X, Xr) < 1e-10
    # streaming variant (large devices: sparse couplings on the device, LU factors on the host)
    st = core.StreamBlockTridiag(dev, A)
    Xs = gpu.asnumpy(st.solve(gpu.cp.asarray(b)))
    assert all(isinstance(f[0], np.ndarray) for f in st.lu)
    assert rel(Xs, Xr) < 1e-10


@pytest.mark.parametrize("coarse", [True, False])
def test_fgmres_cap_schwarz_gpu(coarse):
    """Same CAP-Schwarz + FGMRES as test_level1 on the GPU: same solution, same iteration counts as the CPU run."""
    inp = wire(nb=8)
    dev = core.Device(inp)
    E = -0.4
    SL, SR = dev.sigma(E + 1e-6j)
    B = core.injection(SL)
    b = np.zeros((dev.n, B.shape[1]), complex)
    b[dev.bL] = B
    A = dev.A(E, SL, SR)
    x = core.CSolver(A).solve(b)
    V = None
    if coarse:
        V, _, _ = level1.build_level1(inp, dev, CONTOUR, kind="slab", d_emb=3.0, d_cap=1.5, eta_cap=0.5, cut=1e-4,
                                      log=lambda *a: None)
    out = {}
    for mode in ("cpu", "gpu"):
        gpu.enable(mode == "gpu")
        try:
            sw = schwarz.CapSchwarz(dev, inp, width=4.0, s=2.0, d=1.0, solver="dense", log=lambda *a: None)
            sw.set_energy(A, log=lambda *a: None)
            cg, x0 = None, gpu.asarray(np.zeros_like(b))
            if coarse:
                c = core.Coarse(dev, V).to_gpu()
                Ac = c.Ac(E, SL, SR)
                cg = lambda r: c.prolong(Ac.solve(c.restrict(r)))
                x0 = cg(gpu.asarray(b))
            op = schwarz.operator(A)
            xs, its, res = schwarz.fgmres_block(op, gpu.asarray(b), x0, schwarz.two_level(op, sw, cg), tol=1e-11,
                                                m=200)
            out[mode] = (gpu.asnumpy(xs), its, res)
        finally:
            gpu.enable(False)
    xs, its, res = out["gpu"]
    assert res.max() < 1e-10
    assert np.linalg.norm(xs - x) < 1e-8 * np.linalg.norm(x)
    assert np.abs(its - out["cpu"][1]).max() <= 1


def test_cli_gpu_transmission(tmp_path):
    """cli pieces in GPU mode: closed solves (exact by the GPU block LU) agree with the direct solve."""
    import json
    from qtx_modespace import cli
    inp = wire(nb=8)
    dev = core.Device(inp)
    V, _, _ = level1.build_level1(inp, dev, CONTOUR, kind="slab", d_emb=3.0, d_cap=1.5, eta_cap=0.5, cut=1e-4,
                                  log=lambda *a: None)
    gpu.enable()
    try:
        c = core.Coarse(dev, V).to_gpu()
        sw = schwarz.CapSchwarz(dev, inp, width=4.0, s=2.0, d=1.0, solver="dense64", log=lambda *a: None)
        a = types.SimpleNamespace(out=str(tmp_path), dE="-0.4", rhs_slabs="0", rhs_face=False, rhs_chunk=5,
                                  rhs_max=0, closed_eta=0.01, cap_dE0=0.0, fgmres_coarse="ms", coarse_eta=0.0, pre=1,
                                  post=1, fgmres_tol=1e-11, fgmres_maxit=200, cap="4,2,1", exact=True, workers=2,
                                  save_x=None)
        cli.closed_solves(a, dev, c, sw, 0.0, dict(N=[v.shape[1] for v in V]))
    finally:
        gpu.enable(False)
    r = json.load(open(tmp_path / "closed.json"))["E"]["-0.4000"]
    assert r["res_max"] < 1e-10 and r["err"] < 1e-8


@pytest.mark.parametrize("cplx", [False, True])
def test_coarse_blocks_and_injection(cplx):
    """U^H H U, U^H S U (slab_blocks with a basis) and the Gamma_L eigenchannels: device vs host."""
    inp = wire(nb=8, cplx=cplx)
    dev = core.Device(inp)
    rng = np.random.default_rng(1)
    V = [np.linalg.qr(rng.standard_normal((n, n // 2)) + 1j * rng.standard_normal((n, n // 2)))[0] for n in dev.ns]
    SL, _ = dev.sigma(-0.4 + 1e-6j)
    ref, Bref = core.slab_blocks(dev, V), core.injection(SL)
    gpu.enable()
    try:
        got, B = core.slab_blocks(dev, V), core.injection(SL)
    finally:
        gpu.enable(False)
    for k in ref:
        assert len(got[k]) == len(ref[k])
        for g, r in zip(got[k], ref[k]):
            assert isinstance(g, np.ndarray) and rel(g, r) < 1e-12, k
    assert B.shape == Bref.shape and rel(B @ B.conj().T, Bref @ Bref.conj().T) < 1e-10
