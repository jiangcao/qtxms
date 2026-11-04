"""Unit tests of the generic two-level mode-space code (qtx_ms.msbasis, qtx_ms.twolevel, qtx_ms.prep.inputs)."""
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import pytest

from qtx_ms import msbasis as mb
from qtx_ms.abtb import sancho_rubio
from qtx_ms.prep.inputs import write_inputs, read_inputs
from qtx_ms.twolevel import TwoLevel
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


def test_geometry_two_periodic_axes():
    """Minimum image along y and z at once (QuaTrEx Gamma supercell); the transport axis x stays open."""
    xyz = np.array([[0.0, 0.5, 0.3], [0.0, 9.5, 7.7], [7.5, 0.5, 0.3]])
    per = [dict(axis=1, length=10.0), dict(axis=2, length=8.0)]
    geo = mb.Geometry(xyz, per)
    d, _ = geo.tree([1]).query(geo.points([0]), k=1)
    assert abs(d[0] - np.hypot(1.0, 0.6)) < 1e-12
    d, _ = geo.tree([2]).query(geo.points([0]), k=1)
    assert abs(d[0] - 7.5) < 1e-12
    assert mb.periodic_axes(per[0]) == [(1, 10.0)] and mb.periodic_axes(None) == []


# ------------------------------------------------------------------ leads and exact transmission
@pytest.mark.parametrize("nonorth", [False, True])
@pytest.mark.parametrize("gauge", [False, True])
def test_folded_lead_vs_unfolded(nonorth, gauge):
    """Sigma from the folded lead (interior plane eliminated) = Sigma from the unfolded cell."""
    inp = strip(nonorth=nonorth, complex_gauge=gauge)
    dev = mb.Device(inp, obc="folded")
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
    df, dg = mb.Device(inp, obc="folded"), mb.Device(inp, fast_lead=False, obc="folded")
    assert df.lead.fast and not dg.lead.fast
    for E in E_TEST:
        a, b = df.sigma(E + 1e-6j), dg.sigma(E + 1e-6j)
        assert rel(a[0], b[0]) < 1e-9 and rel(a[1], b[1]) < 1e-9


def exact_T(inp, E, tol=1e-10, obc="folded"):
    dev = mb.Device(inp, obc=obc)
    SL, SR = dev.sigma(E + 1e-6j)
    B = dev.inject(SL, tol)
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


@pytest.mark.skipif(mb.qttools_obc() is None, reason="qttools not importable (qtx extra, QTTOOLS_SRC)")
@pytest.mark.parametrize("nonorth", [False, True])
@pytest.mark.parametrize("gauge", [False, True])
def test_qttools_obc_vs_folded(nonorth, gauge):
    """qttools Sigma = folded-lead Sigma; its incoming modes B: B B^H = Gamma_L, one column per channel; same T."""
    inp = strip(nonorth=nonorth, complex_gauge=gauge)
    dq, dm = mb.Device(inp, obc="qttools"), mb.Device(inp, obc="folded")
    for E in E_TEST:
        SLq, SRq = dq.sigma(E + 1e-6j)
        SLm, SRm = dm.sigma(E + 1e-6j)
        assert rel(SLq, SLm) < 1e-5 and rel(SRq, SRm) < 1e-5
        B, Bm = dq.inject(SLq), mb.injection(SLm)[0]
        assert B.shape[1] == Bm.shape[1] > 0
        assert rel(B @ B.conj().T, 1j * (SLq - SLq.conj().T)) < 1e-6
        assert abs(exact_T(inp, E, obc="qttools")[0] - exact_T(inp, E)[0]) < 1e-5


@pytest.mark.skipif(mb.qttools_obc() is None, reason="qttools not importable (qtx extra, QTTOOLS_SRC)")
def test_qttools_supercell_sections():
    """Lead cell of 2 primitive cells, orbitals shuffled: sections=2 with the cell permutation (NEVP on one primitive
    cell, right lead with the cells reversed) = sections=1 (NEVP on the whole cell), both sides."""
    L = strip(nonorth=True)["lead"]
    H00, H01, S00, S01 = (L[k].toarray() for k in ("H00", "H01", "S00", "S01"))
    m = H00.shape[0]
    Z = np.zeros((m, m))
    sup = lambda A0, A1: (np.block([[A0, A1], [A1.conj().T, A0]]), np.block([[Z, Z], [A1, Z]]))
    (H00s, H01s), (S00s, S01s) = sup(H00, H01), sup(S00, S01)
    q = np.random.default_rng(1).permutation(2 * m)                  # stored order: shuffled
    ip = np.argsort(q)                                                # cells contiguous = stored[ip]
    Hs, H1, Ss, S1 = (M[np.ix_(q, q)] for M in (H00s, H01s, S00s, S01s))
    a = mb.QttLead(Hs, H1, Ss, S1, sections=2, perm=ip)
    b = mb.QttLead(Hs, H1, Ss, S1, sections=1)
    for E in E_TEST:
        for side in "LR":
            (ga, ua), (gb, ub) = a.modes(E, side), b.modes(E, side)
            assert rel(ga, gb) < 1e-8
            assert ua.shape[1] == ub.shape[1] > 0
            assert rel(ua @ ua.conj().T, ub @ ub.conj().T) < 1e-8


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
    from qtx_ms.prep import model
    from qtx_ms.prep.structure import build_open_device
    from qtx_ms.prep.hamiltonian import TBParams, build_H
    from qtx_ms.prep.si_sheet import LABELS
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
        Te = exact_T(inp, E, obc=dev.obc)[0]
        assert abs(T - Te) < 1e-7 * Te


def test_gmres_right_with_inexact_preconditioner():
    """The CSolver fallback: GMRES right-preconditioned by an inexact inverse reaches tol."""
    rng = np.random.default_rng(3)
    n = 200
    A = sp.random(n, n, 0.03, random_state=3) + 1j * sp.random(n, n, 0.03, random_state=4) + 3 * sp.identity(n)
    A = A.tocsr()
    b = rng.normal(size=n) + 1j * rng.normal(size=n)
    d = A.diagonal()
    x, its = mb.gmres_right(lambda v: A @ v, lambda v: v / d, b, np.zeros(n, complex), 1e-12, 200)
    assert np.linalg.norm(A @ x - b) < 1e-11 * np.linalg.norm(b) and its < 200


@pytest.mark.parametrize("gauge", [False, True])
def test_mr_coarse(gauge):
    """Minimal-residual coarse correction: W^H W solve, A Pc orthogonal projector (residual never grows),
    Pc A projector, and the two-level FGMRES with it converges."""
    inp = strip(nd=4, nonorth=True, complex_gauge=gauge)
    dev = mb.Device(inp)
    blocks, _, _ = mb.partition(dev, cores=False, tile_orbs=40)
    geo = mb.Geometry(inp["xyz"])
    tree = geo.tree()
    bases = [mb.tile_modes(inp, geo, tree, b["atoms"], E_TEST, tile_thr=1e-3)[0] for b in blocks]
    Us, _ = mb.slab_U(dev, blocks, bases)
    U = sp.block_diag(Us, format="csr")
    E = E_TEST[0]
    SL, SR = dev.sigma(E + 1e-6j)
    A = dev.A(E, SL, SR)
    mr = mb.MRCoarse(dev, A, Us)
    W = (A @ U).toarray()
    v = np.random.default_rng(5).normal(size=U.shape[1]) + 0j
    assert rel(mr.solve(W.conj().T @ W @ v), v) < 1e-9
    tl = TwoLevel(A, U, smoother="gmres", k=5, coarse=mr, restrict="AU")
    r = np.random.default_rng(6).normal(size=dev.n) + 1j * np.random.default_rng(7).normal(size=dev.n)
    r2 = r - A @ tl.Pc(r)
    assert np.linalg.norm(r2) <= np.linalg.norm(r) and np.linalg.norm(r2 - (r2 - A @ tl.Pc(r2))) < 1e-9 * np.linalg.norm(r)
    p1 = tl.Pc(A @ r); assert rel(tl.Pc(A @ p1), p1) < 1e-9
    B, _, _ = mb.injection(SL, 1e-10)
    b = np.zeros(dev.n, complex); b[dev.bL] = B[:, -1]
    x, hist, _ = tl.solve(b, tol=1e-11)
    assert hist[-1] < 1e-11 and rel(x, mb.CSolver(A).solve(b)) < 1e-8


def test_right_lead_shift(tmp_path, real_strip):
    """A right lead written as leadR_* (left lead + dV S) round-trips, leaves Sigma_L unchanged, and its
    surface GF at z equals the unshifted one at z - dV."""
    inp = real_strip
    dV = -0.15
    lr = {k: inp["lead"][k] + (dV * inp["lead"]["S" + k[1:]] if k[0] == "H" else 0) for k in ("H00", "H01", "S00", "S01")}
    write_inputs(tmp_path, inp["labels"], inp["xyz"], inp["norb"], inp["blocks"], inp["H"], inp["S"],
                 lead=inp["lead"], meta=inp["meta"], lead_right=lr)
    r = read_inputs(tmp_path)
    assert r["lead_right"] is not None and abs(r["lead_right"]["H00"] - lr["H00"]).max() == 0
    z = 0.7 + 0.05j
    dev_b, dev_0 = mb.Device(r), mb.Device(inp)
    assert rel(dev_b.sigma(z)[0], dev_0.sigma(z)[0]) < 1e-10
    gR_shift = mb.FoldedLead(lr["H00"], lr["H01"], lr["S00"], lr["S01"]).surface_gfs(z)[1]
    gR_ref = dev_0.lead.surface_gfs(z - dV)[1]
    assert rel(gR_shift, gR_ref) < 1e-9


@pytest.mark.parametrize("mode,width", [("slab", -1), ("slab", 1), ("tile", 0)])
def test_enrich_round(mode, width):
    """Enrichment keeps the blocks orthonormal, adds only directions outside them, and lowers the largest
    window Ritz residual of the global H1 (local clusters, width 1, as well as the global problem)."""
    inp = strip(nd=4, nonorth=True, complex_gauge=True)
    dev = mb.Device(inp)
    blocks, _, _ = mb.partition(dev, cores=False, tile_orbs=20)
    geo = mb.Geometry(inp["xyz"])
    bases = [mb.tile_modes(inp, geo, geo.tree(), b["atoms"], E_TEST, tile_thr=3e-2)[0] for b in blocks]
    lwin = (0.0, 1.4)

    def max_res(bases):
        Us, _ = mb.slab_U(dev, blocks, bases)
        U = sp.block_diag(Us, format="csr")
        H1, S1, _ = mb.cluster_dense(mb.slab_blocks(dev, Us), list(range(len(Us))))
        lam, C = sla.eigh(H1, S1, subset_by_value=lwin)
        X = U @ C
        X /= np.linalg.norm(X, axis=0)
        return np.linalg.norm(dev.H @ X - (dev.S @ X) * lam, axis=0).max()

    added, st = mb.enrich_round(dev, blocks, bases, 0.0, lwin, rthr=0.05, smin=0.01, width=width, mode=mode)
    assert st["n_window"] > 0 and st["added"] > 0
    if mode == "tile":
        assert min(st["cluster_blocks"]) > 1 and max(st["cluster_blocks"]) < len(blocks)
    for i, Z in added.items():
        assert np.abs(Z.conj().T @ Z - np.eye(Z.shape[1])).max() < 1e-10
        assert np.abs(bases[i].conj().T @ Z).max() < 1e-10
    new = [np.hstack([q, added[i]]) if i in added else q for i, q in enumerate(bases)]
    assert max_res(new) < 0.7 * max_res(bases)


@pytest.mark.parametrize("complex_gauge", [False, True])
def test_tile_schur_equals_open_embedding(complex_gauge):
    """boundary='schur' (explicit Sigma_P = A_PE A_EE^-1 A_EP) = the embedding without CAP."""
    r = strip(nonorth=True, complex_gauge=complex_gauge)
    atoms = np.where(r["blocks"] == 2)[0][:20]
    geo = mb.Geometry(r["xyz"])
    Qs, ss, _ = mb.tile_modes(r, geo, geo.tree(), atoms, E_TEST, tile_thr=1e-3, eta=0.05, boundary="schur")
    Qc, sc, _ = mb.tile_modes(r, geo, geo.tree(), atoms, E_TEST, tile_thr=1e-3, eta=0.05, eta_cap=0.0)
    assert Qs.shape == Qc.shape and Qs.shape[1] > 0
    assert np.abs(Qs @ Qs.conj().T - Qc @ Qc.conj().T).max() < 1e-8
    if not complex_gauge:
        _, os_, _ = mb.tile_modes(r, geo, geo.tree(), atoms, None, contour=(-1, 1, 8), boundary="schur", rho_thr=1e-6)
        _, oc, _ = mb.tile_modes(r, geo, geo.tree(), atoms, None, contour=(-1, 1, 8), eta_cap=0.0, rho_thr=1e-6)
        np.testing.assert_allclose(os_, oc, atol=1e-10)


def _ms_T(dev, Us, E, tol=1e-10):
    blk = mb.slab_blocks(dev, Us)
    PL, PR = Us[0][dev.bL].toarray(), Us[-1][dev.bR - dev.sb[-2]].toarray()
    SL, SR = dev.sigma(E + 1e-6j)
    B = mb.injection(SL, tol)[0]
    b = np.zeros((dev.n, B.shape[1]), complex)
    b[dev.bL] = B
    U = sp.block_diag(Us, format="csr")
    psi = U @ mb.BlockTridiag(blk, E, PL.conj().T @ SL @ PL, PR.conj().T @ SR @ PR).solve(U.conj().T @ b)
    pr = psi[dev.bR]
    return float(np.real(np.einsum("ic,ij,jc->", pr.conj(), 1j * (SR - SR.conj().T), pr)))


@pytest.mark.parametrize("corr", ["exact", "cluster"])
def test_residual_enrich(corr):
    """Residual enrichment from the original open system: exact correction makes MS exact at the training
    energy; the cluster correction reduces the MS error there."""
    inp = strip(nd=4, nonorth=True)
    dev = mb.Device(inp)
    blocks, _, _ = mb.partition(dev, cores=False, tile_orbs=40)
    geo = mb.Geometry(inp["xyz"])
    bases = [mb.tile_modes(inp, geo, geo.tree(), b["atoms"], E_TEST, tile_thr=3e-2)[0] for b in blocks]
    Us, _ = mb.slab_U(dev, blocks, bases)
    E = 0.7
    Te = exact_T(inp, E, obc=dev.obc)[0]
    e0 = abs(_ms_T(dev, Us, E) - Te) / Te
    added, st = mb.residual_enrich(dev, Us, [E], lambda z: dev.sigma(z + 1e-6j), corr, tol=1e-8, nadd=100,
                                   inj_tol=1e-10, log=lambda *a: None)
    U2 = [sp.csr_matrix(sp.hstack([u, sp.csr_matrix(q)])) for u, q in zip(Us, added)]
    for u in U2:
        assert np.abs((u.conj().T @ u).toarray() - np.eye(u.shape[1])).max() < 1e-10
    e1 = abs(_ms_T(dev, U2, E) - Te) / Te
    assert e0 > 1e-4
    assert e1 < (1e-8 if corr == "exact" else 0.5 * e0), (e0, e1)


@pytest.mark.parametrize("gauge", [False, True])
def test_rgf_blocks_vs_dense(gauge):
    """rgf_blocks = blocks |s - s'| <= 2 of the dense inverse of the level-1 operator (complex z, random U1)."""
    inp = strip(nd=4, ny=3, nz=2, no=2, complex_gauge=gauge)
    dev = mb.Device(inp, obc="folded")
    rng = np.random.default_rng(3)
    Us = [sp.csr_matrix(np.linalg.qr(rng.normal(size=(n, n - 3)) + 1j * gauge * rng.normal(size=(n, n - 3)))[0])
          for n in dev.ns]
    blk = mb.slab_blocks(dev, Us)
    z = 0.7 + 0.05j
    SL, SR = dev.sigma(z)
    PL, PR = Us[0][dev.bL].toarray(), Us[-1][dev.bR - dev.sb[-2]].toarray()
    U = sp.block_diag(Us, format="csr")
    G = np.linalg.inv((U.conj().T @ dev.A(z, SL, SR) @ U).toarray())
    cb = np.concatenate([[0], np.cumsum([u.shape[1] for u in Us])])
    Gr = mb.rgf_blocks(blk, z, PL.conj().T @ SL @ PL, PR.conj().T @ SR @ PR, width=2)
    assert len(Gr) == 4 + 2 * 3 + 2 * 2
    for (i, j), X in Gr.items():
        assert rel(X, G[cb[i]:cb[i + 1], cb[j]:cb[j + 1]]) < 1e-11


@pytest.mark.parametrize("dense_fill", [0.0, 1.0])            # dense LU path / sparse (CSolver) path
def test_tile_rho_fast_equals_original(dense_fill):
    """tile_modes(fast=...): Schur G_PP (double / single + refinement), whole-embedding LU, contour points over
    forked workers, all equal the original serial contour path; lead Sigma in the embedding (obc=dev)."""
    r = strip(nonorth=True)
    dev = mb.Device(r, obc="folded")
    geo = mb.Geometry(r["xyz"])
    atoms = np.where(r["blocks"] == 1)[0][:20]                 # next to the left lead face
    kw = dict(contour=(-0.5, 1.5, 8), rho_thr=1e-8, obc=dev, dense_fill=dense_fill, d_emb=5.0, d_cap=2.0)
    U0, s0, _ = mb.tile_modes(r, geo, geo.tree(), atoms, None, **kw)
    rho0 = (U0 * s0[:U0.shape[1]]) @ U0.T
    for fast, tol in ((dict(gpp="schur", env_prec="double"), 1e-11), (dict(gpp="schur", env_prec="single", refine=2), 1e-9),
                      (dict(gpp="full"), 1e-11), (dict(gpp="schur", env_prec="double", workers=2, threads=1), 1e-11)):
        U, s, _ = mb.tile_modes(r, geo, geo.tree(), atoms, None, fast=fast, **kw)
        assert np.abs((U * s[:U.shape[1]]) @ U.T - rho0).max() < tol, fast
    # smaller embedding far from the axis: still a close approximation (G decays at large Im z)
    U, s, _ = mb.tile_modes(r, geo, geo.tree(), atoms, None, fast=dict(gpp="schur", emb_far=[(0.5, 3.0)]), **kw)
    assert np.abs((U * s[:U.shape[1]]) @ U.T - rho0).max() < 5e-2
