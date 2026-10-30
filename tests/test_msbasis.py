"""Unit tests of the generic two-level mode-space code (mgms.msbasis, mgms.twolevel, mgms.prep.inputs)."""
import numpy as np
import scipy.sparse as sp
import pytest

from mgms import msbasis as mb
from mgms.abtb import sancho_rubio
from mgms.prep.inputs import write_inputs, read_inputs
from mgms.twolevel import TwoLevel
from conftest import strip, dense_T, gauge_phase

E_TEST = [0.3, 1.1]


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def herm(M):
    return 0.5 * (M + M.conj().T)


# ------------------------------------------------------------------ inputs, solvers
def test_inputs_roundtrip(tmp_path, real_strip):
    i = real_strip
    write_inputs(tmp_path, i["labels"], i["xyz"], i["norb"], i["blocks"], i["H"], i["S"], lead=i["lead"], meta=i["meta"])
    r = read_inputs(tmp_path)
    assert abs(r["H"] - i["H"]).max() == 0 and abs(r["S"] - i["S"]).max() == 0
    for k in ("H00", "H01", "S00", "S01"):
        assert abs(r["lead"][k] - i["lead"][k]).max() == 0
    np.testing.assert_array_equal(r["norb"], i["norb"])
    np.testing.assert_array_equal(r["blocks"], i["blocks"])
    np.testing.assert_allclose(r["xyz"], i["xyz"], atol=1e-7)
    assert r["meta"]["transport"] == i["meta"]["transport"]


def test_csolver_complex():
    rng = np.random.default_rng(1)
    n = 300
    A = sp.random(n, n, 0.02, random_state=1) + 1j * sp.random(n, n, 0.02, random_state=2) + 4 * sp.identity(n)
    B = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
    X = mb.CSolver(A).solve(B)
    assert np.linalg.norm(A @ X - B) < 1e-10 * np.linalg.norm(B)


def test_sancho_both_matches_abtb():
    rng = np.random.default_rng(2)
    n = 6
    H00 = herm(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    S00 = np.eye(n) + 0.05 * herm(rng.normal(size=(n, n)))
    H01 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    S01 = 0.05 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    z = 0.4 + 1e-3j
    A00, A01, A10 = z * S00 - H00, z * S01 - H01, z * S01.conj().T - H01.conj().T
    gL, gR = mb.sancho_both(A00, A01, A10)
    assert rel(gR, sancho_rubio(A00, a=A01, b=A10)) < 1e-9
    assert rel(gL, sancho_rubio(A00, a=A10, b=A01)) < 1e-9


def test_block_tridiag():
    rng = np.random.default_rng(0)
    n = [5, 7, 6]
    cplx = lambda *s: rng.normal(size=s) + 1j * rng.normal(size=s)
    blk = dict(HD=[herm(cplx(k, k)) for k in n], SD=[np.eye(k) + 0.1 * herm(cplx(k, k)) for k in n],
               HU=[cplx(n[i], n[i + 1]) for i in range(2)], SU=[0.1 * cplx(n[i], n[i + 1]) for i in range(2)])
    z = 0.3 + 0.01j
    SL, SR = cplx(5, 5), cplx(6, 6)
    o = np.cumsum([0] + n)
    A = np.zeros((18, 18), complex)
    for s in range(3):
        A[o[s]:o[s + 1], o[s]:o[s + 1]] = z * blk["SD"][s] - blk["HD"][s]
    for s in range(2):
        A[o[s]:o[s + 1], o[s + 1]:o[s + 2]] = z * blk["SU"][s] - blk["HU"][s]
        A[o[s + 1]:o[s + 2], o[s]:o[s + 1]] = (z * blk["SU"][s].conj().T - blk["HU"][s].conj().T)
    A[:5, :5] -= SL
    A[-6:, -6:] -= SR
    B = cplx(18, 2)
    assert rel(mb.BlockTridiag(blk, z, SL, SR).solve(B), np.linalg.solve(A, B)) < 1e-12


def test_geometry_periodic():
    xyz = np.array([[0.0, 0.5, 0.0], [0.0, 9.5, 0.0], [5.0, 5.0, 3.0]])
    geo = mb.Geometry(xyz, dict(axis=1, length=10.0))
    d, j = geo.tree([1]).query(geo.points([0]), k=1)
    assert abs(d[0] - 1.0) < 1e-12
    d0, _ = mb.Geometry(xyz).tree([1]).query(xyz[[0]], k=1)
    assert abs(d0[0] - 9.0) < 1e-12


# ------------------------------------------------------------------ leads and exact transmission
@pytest.mark.parametrize("nonorth", [False, True])
@pytest.mark.parametrize("gauge", [False, True])
def test_folded_lead_vs_unfolded(nonorth, gauge):
    """Sigma from the folded lead (interior plane eliminated) = Sigma from the unfolded cell."""
    inp = strip(nonorth=nonorth, complex_gauge=gauge)
    dev = mb.Device(inp)
    assert dev.lead.fast == (not nonorth and not gauge)
    assert len(dev.lead.I) > 0
    for E in E_TEST:
        _, SL, SR, _ = dense_T(inp, E)
        SLn, SRn = dev.sigma(E + 1e-6j)
        assert rel(SLn, SL[np.ix_(dev.bL, dev.bL)]) < 1e-8
        assert rel(SRn, SR[np.ix_(dev.bR, dev.bR)]) < 1e-8
        assert np.abs(SL).sum() - np.abs(SL[np.ix_(dev.bL, dev.bL)]).sum() < 1e-12


def test_fast_lead_equals_general():
    inp = strip(nonorth=False)
    df, dg = mb.Device(inp), mb.Device(inp, fast_lead=False)
    assert df.lead.fast and not dg.lead.fast
    for E in E_TEST:
        a, b = df.sigma(E + 1e-6j), dg.sigma(E + 1e-6j)
        assert rel(a[0], b[0]) < 1e-9 and rel(a[1], b[1]) < 1e-9


def exact_T(inp, E, tol=1e-10):
    dev = mb.Device(inp)
    SL, SR = dev.sigma(E + 1e-6j)
    B, lam, _ = mb.injection(SL, tol)
    b = np.zeros((dev.n, B.shape[1]), complex)
    b[dev.bL] = B
    psi = mb.CSolver(dev.A(E, SL, SR)).solve(b)
    GR = 1j * (SR - SR.conj().T)
    pr = psi[dev.bR]
    return float(np.real(np.einsum("ic,ij,jc->", pr.conj(), GR, pr))), B.shape[1]


@pytest.mark.parametrize("nonorth", [False, True])
@pytest.mark.parametrize("gauge", [False, True])
def test_qtbm_T_vs_caroli(nonorth, gauge):
    inp = strip(nonorth=nonorth, complex_gauge=gauge)
    for E in E_TEST:
        T, nc = exact_T(inp, E)
        Tc = dense_T(inp, E)[0]
        assert nc > 0 and Tc > 0.1
        assert abs(T - Tc) < 1e-6 * Tc


@pytest.mark.parametrize("nonorth", [False, True])
def test_gauge_invariance_T(nonorth):
    for E in E_TEST:
        assert abs(exact_T(strip(nonorth=nonorth), E)[0] - exact_T(strip(nonorth=nonorth, complex_gauge=True), E)[0]) < 1e-8


# ------------------------------------------------------------------ level 1
def test_tile_modes_gauge_covariant():
    """Complex-H tile basis spans D x (real-H tile basis); same singular values."""
    r, c = strip(nonorth=True), strip(nonorth=True, complex_gauge=True)
    atoms = np.where(r["blocks"] == 2)[0][:20]                  # 40 orbitals < probe rank: exact range finder
    geo = mb.Geometry(r["xyz"])
    Qr, sr, _ = mb.tile_modes(r, geo, geo.tree(), atoms, E_TEST, tile_thr=1e-3)
    Qc, sc, _ = mb.tile_modes(c, geo, geo.tree(), atoms, E_TEST, tile_thr=1e-3)
    assert np.isrealobj(Qr) and np.iscomplexobj(Qc)
    assert Qr.shape == Qc.shape and Qr.shape[1] > 0
    np.testing.assert_allclose(sr[:Qr.shape[1]], sc[:Qc.shape[1]], rtol=1e-8)
    D = np.diag(gauge_phase(r["xyz"][atoms], 2))
    Pr, Pc = D @ Qr @ Qr.T @ D.conj().T, Qc @ Qc.conj().T
    assert np.abs(Pr - Pc).max() < 1e-8


def test_partition_covers_device(real_strip):
    dev = mb.Device(real_strip)
    blocks, core, _ = mb.partition(dev, cores=False, tile_orbs=40)
    for s in range(len(dev.ns)):
        orbs = np.sort(np.concatenate([b["orbs"] for b in blocks if b["slab"] == s]))
        np.testing.assert_array_equal(orbs, np.arange(dev.ns[s]))
    assert not core.any()


def test_bulk_from_device_sp3s():
    """Bulk H(k) read off the device H at a core atom = the sp3s* bulk H(k) of the model."""
    from mgms.prep import model
    from mgms.prep.structure import build_open_device
    from mgms.prep.hamiltonian import TBParams, build_H
    from mgms.prep.si_sheet import LABELS
    from qtmode.cms import _bulk_hk_batch
    model.select("sp3s")
    st = build_open_device(n_dev_blocks=1, n_lead_blocks=1, nt=(4, 4), half_width=(9.0, 9.0), rough_rms=0.0)
    P = TBParams()
    inp = dict(labels=np.array(LABELS)[st.species], xyz=st.pos, norb=st.norb, H=build_H(st, P))
    core, dmin = mb.find_cores(inp["labels"], mb.Geometry(st.pos), "Si", 3.0)
    assert core.sum() > 4
    bulk = mb.BulkFromDevice(inp, core, dmin)
    K = np.random.default_rng(0).normal(size=(20, 3))
    e1 = np.linalg.eigvalsh(bulk.hk(K))
    e2 = np.linalg.eigvalsh(_bulk_hk_batch(P, K))
    assert np.abs(e1 - e2).max() < 1e-10
    pos = st.pos[np.where(core)[0]]
    kst, cst = bulk.states(bulk.kgrid(8 * bulk.a_lat, np.ptp(pos[:, 1]) + 1, np.ptp(pos[:, 2]) + 1), (1.1, 2.3))
    Q, _ = mb.core_basis(bulk, pos, kst, cst)
    assert Q.shape[1] > 0 and np.abs(Q.T @ Q - np.eye(Q.shape[1])).max() < 1e-10


# ------------------------------------------------------------------ full two-level pipeline
@pytest.mark.parametrize("gauge", [False, True])
def test_two_level_pipeline(gauge):
    """Tiles -> U1 -> contour rho -> U = U1 U2; two-level FGMRES with coarse U converges to exact T."""
    inp = strip(nd=4, nonorth=True, complex_gauge=gauge)
    dev = mb.Device(inp)
    blocks, _, _ = mb.partition(dev, cores=False, tile_orbs=40)
    geo = mb.Geometry(inp["xyz"])
    tree = geo.tree()
    bases = [mb.tile_modes(inp, geo, tree, b["atoms"], E_TEST, tile_thr=1e-3)[0] for b in blocks]
    Us, _ = mb.slab_U(dev, blocks, bases)
    for U in Us:
        assert np.abs((U.conj().T @ U).toarray() - np.eye(U.shape[1])).max() < 1e-10
    blk1 = mb.slab_blocks(dev, Us)
    PL, PR = Us[0][dev.bL].toarray(), Us[-1][dev.bR - dev.sb[-2]].toarray()
    rho = mb.rho_contour(dev, blk1, PL, PR, 0.1, 1.3, 8, log=lambda *a: None)
    for R in rho:
        assert np.abs(R - R.conj().T).max() < 1e-10 and np.linalg.eigvalsh(R).min() > -1e-8
    V = mb.level2_basis(rho, 1e-6)
    blk2 = mb.project(blk1, V)
    U2 = sp.block_diag([sp.csr_matrix(Us[s] @ V[s]) for s in range(len(Us))], format="csr")
    for E in E_TEST:
        SL, SR = dev.sigma(E + 1e-6j)
        B, _, _ = mb.injection(SL, 1e-10)
        A = dev.A(E, SL, SR)
        Ac = mb.BlockTridiag(blk2, E, (PL @ V[0]).conj().T @ SL @ (PL @ V[0]),
                             (PR @ V[-1]).conj().T @ SR @ (PR @ V[-1]))
        tl = TwoLevel(A, U2, smoother="gmres", k=5, coarse=Ac)
        GR = 1j * (SR - SR.conj().T)
        T = 0.0
        for c in range(B.shape[1]):
            b = np.zeros(dev.n, complex)
            b[dev.bL] = B[:, c]
            x, hist, _ = tl.solve(b, tol=1e-11)
            assert hist[-1] < 1e-11 and len(hist) < 60
            T += float(np.real(x[dev.bR].conj() @ GR @ x[dev.bR]))
        Te = exact_T(inp, E)[0]
        assert abs(T - Te) < 1e-7 * Te
