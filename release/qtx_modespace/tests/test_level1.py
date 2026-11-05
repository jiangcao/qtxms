"""Level 1 (local density matrix per block) and the FGMRES + CAP-Schwarz solver on a small synthetic device.

Device: a 2 x 3 atom wire along x (1 A spacing, 2 orbitals per atom, nearest-neighbour H, S = I + small NN overlap),
leads = 2-cell blocks of the ideal wire, random on-site disorder in the device blocks.

    PYTHONPATH=src python -m pytest tests -q
"""
import numpy as np
import scipy.sparse as sp
import pytest

from qtx_modespace import core, level1, schwarz


def wire(nb=6, cells=2, ny=2, nz=3, norb=2, seed=0, cplx=False):
    rng = np.random.default_rng(seed)
    nx = nb * cells
    xyz = np.array([[x, y, z] for x in range(nx) for y in range(ny) for z in range(nz)], float)
    na = len(xyz)
    blocks = (xyz[:, 0] // cells).astype(int)
    eps = np.zeros((na, norb))
    eps[:, 1] = 1.5
    dev = (blocks > 0) & (blocks < nb - 1)
    eps[dev] += 0.3 * rng.standard_normal((dev.sum(), norb))
    t = np.array([[-1.0, 0.3], [0.3, -0.6]])
    if cplx:
        t = t + 0.2j * np.array([[0, 1], [-1, 0]])
    n = na * norb
    H = np.zeros((n, n), complex if cplx else float)
    S = np.eye(n)
    for i in range(na):
        H[i * norb:(i + 1) * norb, i * norb:(i + 1) * norb] = np.diag(eps[i])
        for j in range(i + 1, na):
            if abs(np.linalg.norm(xyz[i] - xyz[j]) - 1) < 1e-9:
                f = 1.0 if xyz[i, 0] != xyz[j, 0] else 0.5
                H[i * norb:(i + 1) * norb, j * norb:(j + 1) * norb] = f * t
                H[j * norb:(j + 1) * norb, i * norb:(i + 1) * norb] = f * t.conj().T
                S[i * norb:(i + 1) * norb, j * norb:(j + 1) * norb] = 0.05 * np.eye(norb)
                S[j * norb:(j + 1) * norb, i * norb:(i + 1) * norb] = 0.05 * np.eye(norb)
    H, S = sp.csr_matrix(H), sp.csr_matrix(S)
    ob = np.repeat(blocks, norb)
    o0, o1 = np.where(ob == 0)[0], np.where(ob == 1)[0]
    lead = dict(H00=H[o0][:, o0], H01=H[o0][:, o1], S00=S[o0][:, o0], S01=S[o0][:, o1])
    meta = dict(name="wire", axis=0, periodic=None,
                transport=dict(device_blocks=list(range(1, nb - 1)), lead_left_block=0, lead_right_block=nb - 1),
                edges=dict(Ec_lead=0.0, Ev_lead=-1.0))
    return dict(labels=np.array(["A"] * na), xyz=xyz, norb=np.full(na, norb), blocks=blocks, H=H, S=S, lead=lead,
                lead_right=None, perm=np.arange(n), meta=meta)


CONTOUR = (-1.5, 0.5, 16)


@pytest.mark.parametrize("cplx", [False, True])
def test_slab_obc_whole_device_equals_global_rho(cplx):
    """CAP + OBC embedding that holds the whole device and no CAP shell: the local rho of a slab is the global rho_ss."""
    inp = wire(cplx=cplx)
    dev = core.Device(inp)
    ref = core.rho_contour(dev, *CONTOUR, log=lambda *a: None)
    geo = level1.Geometry(inp["xyz"])
    tree = geo.tree()
    for b in level1.partition(inp, dev, "slab"):
        e = level1.embedding(inp, geo, tree, b["atoms"], 100.0, 1.0, 0.3, obc=dev)
        assert e["ncap"] == 0 and sum(e["nface"]) > 0
        for gpp in ("schur", "full"):
            rho, _ = level1.local_rho(e, CONTOUR, gpp=gpp)
            R = ref[b["slab"]][np.ix_(b["orbs"], b["orbs"])]
            assert np.abs(rho - R).max() < 1e-10 * np.abs(R).max()


def test_schur_single_and_full_agree_with_cap():
    inp = wire()
    dev = core.Device(inp)
    geo = level1.Geometry(inp["xyz"])
    b = level1.partition(inp, dev, "tile", tile_orbs=12)[3]
    e = level1.embedding(inp, geo, geo.tree(), b["atoms"], 3.0, 1.5, 0.5)
    assert e["ncap"] > 0
    r1, _ = level1.local_rho(e, CONTOUR, gpp="full")
    r2, _ = level1.local_rho(e, CONTOUR, gpp="schur")
    r3, _ = level1.local_rho(e, CONTOUR, gpp="schur", env_prec="single", refine=2)
    assert np.abs(r1 - r2).max() < 1e-11 * np.abs(r1).max()
    assert np.abs(r1 - r3).max() < 1e-7 * np.abs(r1).max()


@pytest.mark.parametrize("kind,tile_orbs,xsplit", [("slab", 0, 1), ("slab", 0, 2), ("tile", 12, 1), ("tile", 6, 2)])
def test_partition_covers_slabs(kind, tile_orbs, xsplit):
    inp = wire()
    dev = core.Device(inp)
    blocks = level1.partition(inp, dev, kind, tile_orbs or 2500, xsplit)
    for s, n in enumerate(dev.ns):
        o = np.concatenate([b["orbs"] for b in blocks if b["slab"] == s])
        assert np.array_equal(np.sort(o), np.arange(n))
    assert len(blocks) == {("slab", 1): 4, ("slab", 2): 8, ("tile", 1): 8, ("tile", 2): 32}[(kind, xsplit)]


@pytest.mark.parametrize("boundary", ["obc", "cap"])
def test_build_level1_cache_and_ms(tmp_path, boundary):
    """Level-1 basis: cached blocks reused with a higher cut; mode-space T close to exact for a low cut."""
    inp = wire()
    dev = core.Device(inp)
    kw = dict(kind="tile", tile_orbs=12, boundary=boundary, d_emb=4.0, d_cap=1.5, eta_cap=0.5, cache=str(tmp_path),
              log=lambda *a: None)
    V, occ, info = level1.build_level1(inp, dev, CONTOUR, cut=1e-10, **kw)
    V2, _, _ = level1.build_level1(inp, dev, CONTOUR, cut=0.3, **kw)
    assert all(v2.shape[1] <= v.shape[1] for v, v2 in zip(V, V2)) and sum(v.shape[1] for v in V2) < sum(
        v.shape[1] for v in V)
    assert len(list(tmp_path.glob("*/block*.npz"))) == len(level1.partition(inp, dev, "tile", 12))
    for v in V:
        assert np.abs(v.conj().T @ v - np.eye(v.shape[1])).max() < 1e-10
    E = -0.4
    SL, SR = dev.sigma(E + 1e-6j)
    b = np.zeros((dev.n, core.injection(SL).shape[1]), complex)
    b[dev.bL] = core.injection(SL)
    GR = 1j * (SR - SR.conj().T)
    T = lambda x: float(np.real(np.einsum("ic,ij,jc->", x[dev.bR].conj(), GR, x[dev.bR])))
    x = core.CSolver(dev.A(E, SL, SR)).solve(b)
    c = core.Coarse(dev, V)
    xm = c.prolong(c.Ac(E, SL, SR).solve(c.restrict(b)))
    assert abs(T(xm) - T(x)) < 0.05 * T(x)


@pytest.mark.parametrize("coarse", [True, False])
def test_fgmres_cap_schwarz_converges(coarse):
    inp = wire(nb=8)
    dev = core.Device(inp)
    E = -0.4
    SL, SR = dev.sigma(E + 1e-6j)
    B = core.injection(SL)
    b = np.zeros((dev.n, B.shape[1]), complex)
    b[dev.bL] = B
    A = dev.A(E, SL, SR)
    x = core.CSolver(A).solve(b)
    sw = schwarz.CapSchwarz(dev, inp, width=4.0, s=2.0, d=1.0, solver="dense", log=lambda *a: None)
    sw.set_energy(A, log=lambda *a: None)
    cg, x0 = None, np.zeros_like(b)
    if coarse:
        V, _, _ = level1.build_level1(inp, dev, CONTOUR, kind="slab", d_emb=3.0, d_cap=1.5, eta_cap=0.5, cut=1e-4,
                                      log=lambda *a: None)
        c = core.Coarse(dev, V)
        Ac = c.Ac(E, SL, SR)
        cg = lambda r: c.prolong(Ac.solve(c.restrict(r)))
        x0 = cg(b)
    xs, its, res = schwarz.fgmres_block(A, b, x0, schwarz.two_level(A, sw, cg), tol=1e-11, m=200)
    assert res.max() < 1e-10
    assert np.linalg.norm(xs - x) < 1e-8 * np.linalg.norm(x)


@pytest.mark.parametrize("coarse,eta,face", [("ms", 0.0, False), ("none", 0.0, False), ("ms", 0.01, True)])
def test_closed_system_unit_rhs(tmp_path, coarse, eta, face):
    """--system closed: A0 = z S - H (no Sigma), unit right-hand sides on the left contact slab, FGMRES + CAP-Schwarz
    (two- and one-level) agrees with the direct solve."""
    import json, types
    from qtx_modespace import cli
    inp = wire(nb=8)
    dev = core.Device(inp)
    V, _, _ = level1.build_level1(inp, dev, CONTOUR, kind="slab", d_emb=3.0, d_cap=1.5, eta_cap=0.5, cut=1e-4,
                                  log=lambda *a: None)
    c = core.Coarse(dev, V)
    sw = schwarz.CapSchwarz(dev, inp, width=4.0, s=2.0, d=1.0, solver="dense", log=lambda *a: None)
    a = types.SimpleNamespace(out=str(tmp_path), dE="-0.4", rhs_slabs="0", rhs_face=face, rhs_chunk=5, rhs_max=0,
                              closed_eta=eta, cap_dE0=0.0, fgmres_coarse=coarse, coarse_eta=0.0, pre=1, post=1,
                              fgmres_tol=1e-11, fgmres_maxit=200, cap="4,2,1", exact=True, workers=2, save_x=str(tmp_path / "x"))
    cli.closed_solves(a, dev, c, sw, 0.0, dict(N=[v.shape[1] for v in V]))
    r = json.load(open(tmp_path / "closed.json"))["E"]["-0.4000"]
    assert r["columns"] == (len(dev.bL) if face else dev.ns[0])
    assert r["res_max"] < 1e-10 and r["err"] < 1e-8
    assert len(list((tmp_path / "x").glob("closed_*.npz"))) == 1


@pytest.mark.parametrize("cplx", [False, True])
def test_fastop_matches_csr(cplx):
    rng = np.random.default_rng(1)
    A = sp.random(300, 300, 0.05, random_state=2) + (1j * sp.random(300, 300, 0.01, random_state=3) if cplx else 0)
    X = rng.standard_normal((300, 7)) + 1j * rng.standard_normal((300, 7))
    for th in (1, 3):
        op = schwarz.FastOp(A, th)
        assert np.abs(op @ X - A @ X).max() < 1e-12
        assert np.abs(op @ X[:, 0] - A @ X[:, 0]).max() < 1e-12
        assert np.abs(op @ X.real - A @ X.real).max() < 1e-12
