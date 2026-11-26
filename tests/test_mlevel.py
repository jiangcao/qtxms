"""qtx_ms.mlevel (density-matrix hierarchy with repeated merging) on the synthetic strip: the level operators are
Galerkin, the multilevel cycle with an exact bottom level is a preconditioner under which block FGMRES reaches the
direct solution, the window halo only changes the windows (not the level operators), and the row-restricted window
inverses of CapSchwarz give the same smoother as the LUs."""
import numpy as np
import scipy.sparse as sp
import pytest

from conftest import strip
from qtx_ms import msbasis as mb
from qtx_ms.mlevel import Hierarchy, Timed, gcycle
from qtx_ms.schwarz import CapSchwarz, fgmres_block, operator

quiet = lambda *x: None
E, MU, CUT = 0.35, (0.1, 0.6), 0.1          # a narrow window and a high cut: the small strip then compresses
SIG = {}


def setup(solver="dense", **kw):
    inp = strip(nd=8, ny=4, nz=2, no=2)
    dev = mb.Device(inp, obc="folded")
    dev.sigma_cache = SIG["dir"]                # the contour Sigma is shared by the tests (same z points)
    sw = CapSchwarz(dev, inp, 12.0, 1.0, 2.0, spacing=4.0, solver=solver)
    SL, SR = dev.sigma(E + 1e-6j)
    A = dev.A(E, SL, SR).tocsr()
    B = dev.inject(SL)
    b = np.zeros((dev.n, B.shape[1]), complex)
    b[dev.bL] = B
    sw.set_energy(E, A, log=quiet)
    h = Hierarchy(dev, inp, sw, l1_lu="c128", log=quiet, **kw)
    return dev, sw, h, A, b, SL, SR


@pytest.fixture(scope="module", autouse=True)
def sigma_dir(tmp_path_factory):
    SIG["dir"] = str(tmp_path_factory.mktemp("sigma"))


def build(h, SL, SR, halos=(None, None), cut=CUT):
    DM = h.window_dm(None, "", MU[0], MU[1], 16)
    h.level1(h.basis1(DM, "dm", cut), E, SL, SR)
    for hl in halos:
        h.merge("x", cut, "dm", MU[0], MU[1], 16, hl)
    return h


def prolongation(h, m):
    """Dense fine <- level-m prolongation."""
    P = np.eye(h.LV[m].N, dtype=complex)
    for k in range(m, 0, -1):
        P = h.LV[k].P(P)
    return P


def test_rows_smoother_equals_lu():
    dev, sw, _, A, b, _, _ = setup("dense")
    sr = CapSchwarz(dev, sw.dev.inp, 12.0, 1.0, 2.0, spacing=4.0, solver="rows")
    sr.set_energy(E, A, log=quiet)
    r = np.random.default_rng(0).standard_normal((dev.n, 3)) + 0j
    assert np.allclose(sr(r), sw(r), atol=1e-10)


def test_levels_are_galerkin_and_cycle_converges():
    dev, sw, h, A, b, SL, SR = setup()
    assert len(sw.idx) >= 8 and b.shape[1] >= 2
    build(h, SL, SR)
    assert len(h.LV) == 3 and h.LV[3].N < h.LV[2].N < h.LV[1].N < dev.n
    for m in (1, 2, 3):
        P = prolongation(h, m)
        assert np.allclose(P.conj().T @ P, np.eye(P.shape[1]), atol=1e-10)             # orthonormal block bases
        assert np.allclose(h.LV[m].A.dense(), P.conj().T @ (A @ P), atol=1e-9)         # Galerkin, Sigma included
    xe = sp.linalg.spsolve(A.tocsc(), b)
    Aop = operator(A, 1)
    _, its0, _ = fgmres_block(Aop, b, np.zeros_like(b), sw, 1e-10, 200)
    for m in (1, 2, 3):
        prec, tm = h.cycle(m, h.exact(m), Aop, sw)
        x, its, res = fgmres_block(Aop, b, np.zeros_like(b), prec, 1e-10, 200)
        assert res.max() < 1e-9 and np.linalg.norm(x - xe) < 1e-8 * np.linalg.norm(xe)
        assert its.max() < its0.max() and tm.n > 0


@pytest.mark.parametrize("smooth", ["same", "full"])
def test_halo(smooth):
    dev, sw, h, A, b, SL, SR = setup(halo_smooth=smooth)
    build(h, SL, SR, halos=(0, None))
    i2, i3 = h.LV[2].info, h.LV[3].info
    assert i2["halo_max"] >= 1 and i2["halo_rho"] == 0 and i3["halo_rho"] == i3["halo_max"]
    assert i2["halo_smoother"] == (0 if smooth == "same" else i2["halo_max"])
    assert i2["halo_cost"]["0"] < 1 and i2["rho_windows"] == [int(h.LV[1].r[list(ks)].sum()) for ks in h.LV[2].kids]
    P = prolongation(h, 3)
    assert np.allclose(h.LV[3].A.dense(), P.conj().T @ (A @ P), atol=1e-9)             # the operators keep all couplings
    xe = sp.linalg.spsolve(A.tocsc(), b)
    Aop = operator(A, 1)
    prec, _ = h.cycle(3, h.exact(3), Aop, sw)
    x, its, res = fgmres_block(Aop, b, np.zeros_like(b), prec, 1e-10, 200)
    assert res.max() < 1e-9 and np.linalg.norm(x - xe) < 1e-8 * np.linalg.norm(xe)


def test_dmc_selection_and_top_smoother():
    dev, sw, h, A, b, SL, SR = setup()
    DM = h.window_dm(None, "", MU[0], MU[1], 16)
    h.level1(h.basis1(DM, "dmc", CUT), E, SL, SR)
    h.merge("x", CUT, "dmc", MU[0], MU[1], 16)
    own = h.own_smoother()
    L = h.LV[2]
    Aop = operator(A, 1)
    prec, _ = h.cycle(2, gcycle(L.A, own, h.exact(2)), Aop, sw)      # smoothed deepest level around its exact solve
    x, its, res = fgmres_block(Aop, b, np.zeros_like(b), prec, 1e-10, 200)
    xe = sp.linalg.spsolve(A.tocsc(), b)
    assert res.max() < 1e-9 and np.linalg.norm(x - xe) < 1e-8 * np.linalg.norm(xe)


def test_lead_sigma_factor_equals_dense_projection():
    """Sigma of a lead window from the lead-core factor = the dense level-1 window matrix projected down."""
    dev, sw, h, A, b, SL, SR = setup()
    build(h, SL, SR)
    sg = h.sigma1(h.LV[1].U, SL, SR)
    for m in (2, 3):
        L = h.LV[m]
        for w in ([0, 1], [len(L.r) - 2, len(L.r) - 1], list(range(len(L.r)))):
            cw = h.cores_of(L, w)
            assert set(cw) & h.lead_cores
            o = np.concatenate([[0], np.cumsum([h.LV[1].U[k].shape[1] for k in cw])]).astype(int)
            loc = {k: q for q, k in enumerate(cw)}
            M1 = np.zeros((o[-1], o[-1]), complex)
            for (i, j), Bk in sg.items():
                if i in loc and j in loc:
                    M1[o[loc[i]]:o[loc[i] + 1], o[loc[j]]:o[loc[j] + 1]] = Bk
            ref = h.projop(L, w, M1)
            assert np.abs(ref).max() > 1e-6
            assert np.allclose(h.sig_of(h.sig_factor(L, w), sg), ref, atol=1e-12)


def test_mode_space_solution_is_galerkin():
    """P A_m^-1 P^H b through the level operators = the Galerkin mode-space solution with the dense prolongation."""
    dev, sw, h, A, b, SL, SR = setup()
    build(h, SL, SR)
    for m in (1, 2, 3):
        c = b
        for k in range(1, m + 1):
            c = h.LV[k].PH(c)
        c = h.exact(m)(c)
        for k in range(m, 0, -1):
            c = h.LV[k].P(c)
        P = prolongation(h, m)
        assert np.allclose(c, P @ np.linalg.solve(P.conj().T @ (A @ P), P.conj().T @ b), atol=1e-9)


def test_slab_lu_equals_dense_solve_and_block_list_is_not_a_copy():
    """SlabLU (block-tridiagonal LU over x slabs) = dense solve on every level; the A block list of a level points
    into the operator's block rows (A_k is stored once)."""
    dev, sw, h, A, b, SL, SR = setup()
    build(h, SL, SR)
    rng = np.random.default_rng(1)
    for m in (1, 2, 3):
        L = h.LV[m]
        ex = h.exact(m)
        assert len(ex.sizes) > 1 and sum(ex.sizes) == L.N
        r = rng.standard_normal((L.N, 3)) + 1j * rng.standard_normal((L.N, 3))
        assert np.allclose(ex(r), np.linalg.solve(L.A.dense(), r), atol=1e-9)
    L = h.LV[3]
    assert all(np.shares_memory(L.Ab[i][q], L.A.R[i]) for i in range(len(L.r)) for q in range(len(L.Ab[i])))


@pytest.mark.parametrize("solver", ["rows", "lu"])
def test_identical_windows_are_one(solver):
    """Windows of the same blocks that write different blocks (y pairs whose windows cover the full width): one window
    (one LU) and the same correction as with the windows apart."""
    dev, sw, h, A, b, SL, SR = setup(l1_solver=solver)
    build(h, SL, SR, halos=())
    L = h.LV[1]
    nb = len(L.r)
    assert nb >= 6
    wins = [np.arange(0, 4), np.arange(0, 4), np.arange(2, nb), np.arange(2, nb)]
    writes = [(0, 2), (1,), tuple(range(4, nb)), (3,)]               # every block once, in no particular order
    Sm = h.ras(L, wins, writes)
    one = [h.ras(L, [w if c == q else None for c, w in enumerate(wins)], writes) for q in range(len(wins))]
    assert len(Sm.sizes) == 2 and all(len(s.sizes) == 1 for s in one)
    rng = np.random.default_rng(1)
    r = rng.standard_normal((L.N, 3)) + 1j * rng.standard_normal((L.N, 3))
    y = Sm(r)
    assert np.allclose(y, sum(s(r) for s in one), atol=1e-10 * np.abs(y).max()) and np.abs(y).min() > 0
