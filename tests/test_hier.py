"""qtx_ms.hier: against the dense solution and qtx_ms.mlsolve; thread ranks; every block pair and every Sigma point once; GPU."""
import collections
import numpy as np
import pytest

from conftest import strip
from qtx_ms import devstore, dist, gpu, mlsolve
from qtx_ms import msbasis as mb
from qtx_ms.hier import obc, plan, run
from qtx_ms.prep.inputs import write_inputs

ARGS = ["--cap", "12,1,2", "--cap-spacing", "4", "--cap-split", "2", "--cap-tov", "1.5", "--up", "y:1e-3,x:1e-3,x:1e-3",
        "--mu=-0.25,0.5", "--npts", "8", "--Ec", "0.35", "--dE=-0.02,0.05"]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("hier")
    inp = strip(nd=8, ny=4, nz=2, no=2)
    inp["meta"]["periodic"] = [dict(axis=1, length=8.0)]
    inp["meta"].setdefault("edges", {})["Ec_lead"] = 0.35
    write_inputs(d / "dev", inp["labels"], inp["xyz"], inp["norb"], inp["blocks"], inp["H"], inp["S"], inp["lead"], inp["meta"])
    devstore.main(["--inputs", str(d / "dev"), "--out", str(d / "store")])
    ref = mlsolve.solve(mlsolve.parse(["--inputs", str(d / "dev"), "--sigma-cache", str(d / "sigma"), "--l1", "dmc:1e-3",
                                       "--halo-smooth", "same"] + ARGS))
    return d, inp, ref


def go(d, comm=None, store=False, *more):
    a = ["--inputs", str(d / "dev"), "--cut", "1e-3"] + ARGS + (["--dev-store", str(d / "store")] if store else []) + list(more)
    return run.run(run.parse(a), comm or dist.Comm())


def same(r, ref):
    for k, v in ref["E"].items():
        w = r["E"][k]
        assert w["converged"] and abs(w["T"] - v["T"]) < 1e-8 and abs(max(w["its"]) - max(v["its"])) <= 1
        assert r["N"] == v["N"]


def test_serial(env):
    d, inp, ref = env
    r = go(d)
    same(r, ref)
    h = r["h"]
    for v in r["E"].values():                                         # dense solve, the same injection channels
        (SL, SR), B = h.dev._sigma(v["E"] + 1e-6j)[0], h.obc.inject(v["E"] + 1e-6j)
        b = np.zeros((h.dev.n, B.shape[1]), complex)
        b[h.dev.bL] = B
        y = np.linalg.solve(h.dev.A(v["E"], SL, SR).toarray(), b)[h.dev.bR]
        assert abs(np.real(np.einsum("ic,ij,jc->", y.conj(), 1j * (SR - SR.conj().T), y)) - v["T"]) < 1e-9
    # Galerkin pairs against the dense product
    P = np.eye(h.dev.n)
    for k in range(1, h.K + 1):
        Uk = np.zeros((P.shape[1], h.N[k]))
        for b in range(h.lv[k].n):
            i = h.fine.cores[b] if k == 1 else h.entries(k - 1, h.lv[k].kids[b])
            Uk[np.ix_(i, h.entries(k, [b]))] = h.U[k][b]
        P = P @ Uk
        for q, M in enumerate((h.dev.S, h.dev.H)):
            D = P.T @ (M @ P)
            for i in range(h.lv[k].n):
                assert np.abs(h.dense(k, [i], h.lv[k].nbr[i])[q] - D[np.ix_(h.entries(k, [i]), h.entries(k, h.lv[k].nbr[i]))]).max() < 1e-10


@pytest.mark.parametrize("size", [2, 3])
def test_ranks(env, tmp_path, size, monkeypatch):
    d, _, ref = env
    calls = collections.Counter()
    sig = mb.Device._sigma

    def counted(self, z):
        calls[complex(z)] += 1
        return sig(self, z)
    monkeypatch.setattr(mb.Device, "_sigma", counted)
    more = ["--sigma-cache", str(tmp_path / "sig"), "--cache", str(tmp_path / "b")]
    rs = dist.ThreadComm.run(size, lambda c: go(d, c, True, *more))
    assert len(calls) == 10 and set(calls.values()) == {1}            # 8 contour + 2 real points, each on one rank
    for r in rs:
        same(r, ref)
    h = rs[0]["h"]
    for k in range(1, h.K + 1):                                        # a pair is formed once, by the owner of its first block
        own = [{p for p in r["h"].SH[k] if r["h"].lv[k].owner[p[0]] == q} for q, r in enumerate(rs)]
        assert sum(map(len, own)) == len(set().union(*own)) == len(h.lv[k].pairs)
        for q, r in enumerate(rs):
            assert set(r["h"].SH[k]) == set(h.plan.need_pairs[k][q])
    calls.clear()
    same(go(d, None, False, *more), ref)                               # Sigma and bases of the ranks, one process
    assert not calls


def test_no_cache_ranks(env):
    d, _, ref = env
    for r in dist.ThreadComm.run(2, lambda c: go(d, c)):
        same(r, ref)


def test_partial_cache_is_not_mixed(env, tmp_path):
    """A lost level is rebuilt together with the levels above it."""
    d, _, ref = env
    go(d, None, False, "--cache", str(tmp_path))
    f = next(tmp_path.glob("hier_*"))
    (f / "U2_0.npy").unlink()
    np.save(f / "U3_0.npy", np.zeros_like(np.load(f / "U3_0.npy")))    # must not be read
    same(go(d, None, False, "--cache", str(tmp_path)), ref)


def test_injection(env):
    d, inp, _ = env
    dev = mb.Device(devstore.read_any(str(d / "dev")), obc="folded")
    o = obc.OBC(dev, dist.Comm(), log=lambda *x: None)
    z = 0.4 + 1e-6j
    o.prepare([], [z])
    B, B0 = o.inject(z), mb.injection(o.sigma(z, 0))[0]
    assert B.shape == B0.shape and np.abs(B @ B.conj().T - B0 @ B0.conj().T).max() < 1e-10


def test_identical_windows_shared():
    """y pairs over the full width have the same window: one window, two parents."""
    nx, ny = 6, 4
    ix, iy = np.arange(nx * ny) // ny, np.arange(nx * ny) % ny
    dy = np.abs(iy[:, None] - iy[None]); dy = np.minimum(dy, ny - dy)
    G = (np.abs(ix[:, None] - ix[None]) <= 2) & (dy <= 1)
    p = plan.Plan(G, (ix, iy), np.repeat(dist.chunks(np.ones(nx), 2), ny), [("y", 1), ("y", 1), ("x", None)], [[0], [23]], 2)
    assert [len(w.parents) for w in p.levels[1].wins] == [2] * nx and len(p.levels[2].wins) == nx
    assert all((np.abs(ix[w.blocks] - ix[p.levels[2].kids[w.parents[0]][0]]) <= 1).all() for w in p.levels[1].wins)


def test_contour_slices(env):
    """Slices add up to the whole window; a weight scales the step; a weighted basis gives the same T."""
    d = env[0]
    z1, d1 = obc.contours(obc.slices("-0.2,0.6,60", 8, 0.1))
    z2, d2 = obc.contours(obc.slices("-0.2,0.1,40:0.1,0.6,40", 8, 0.1))
    g = lambda z, dz: np.sum(dz / (z - 0.33 + 0.05j))                  # a pole inside the second slice
    assert abs(g(z1, d1) - g(z2, d2)) < 1e-6
    z3, d3 = obc.contours(obc.slices("-0.2,0.1,40:0.1,0.6,40,0.25", 8, 0.1))
    assert np.array_equal(z2, z3) and np.allclose(d3[40:], 0.25 * d2[40:]) and np.array_equal(d3[:40], d2[:40])
    n = [go(d, None, False, "--mu=" + m, "--npts", "16") for m in ("-0.25,0.5", "-0.25,0.1:0.1,0.5", "-0.25,0.1:0.1,0.5,16,0.01")]
    for r in n:
        assert all(abs(v["T"] - env[2]["E"][k]["T"]) < 1e-8 for k, v in r["E"].items())
    assert abs(n[0]["N"][-1] - n[1]["N"][-1]) <= 0.05 * n[0]["N"][-1] and n[2]["N"][-1] < n[1]["N"][-1]


def test_bottom_bt(env):
    """Block-tridiagonal LU of the deepest level = dense LU: the same T and iteration counts, also on ranks and after one merge."""
    d = env[0]
    for up in ("y:1e-3,x:1e-3,x:1e-3", "y:1e-3,x:1e-3", "y:1e-3"):
        r0 = go(d, None, False, "--up", up, "--bottom", "dense")
        assert r0["h"].bottom_res is None
        rs = [go(d, None, False, "--up", up)] + dist.ThreadComm.run(2, lambda c: go(d, c, True, "--up", up))
        for r in rs:
            assert r["h"].bottom_res < 1e-10
            assert r["N"] == r0["N"]
            for k, v in r0["E"].items():
                assert r["E"][k]["converged"] and abs(r["E"][k]["T"] - v["T"]) < 1e-9 and r["E"][k]["its"] == v["its"]


def test_schwarz_only(env):
    """One-level: the same T, more iterations than with the levels; ranks: the same counts."""
    d, _, ref = env
    r = go(d, None, False, "--schwarz", "--maxit", "400")
    rs = dist.ThreadComm.run(2, lambda c: go(d, c, False, "--schwarz", "--maxit", "400"))
    assert not r["h"].U
    for k, v in ref["E"].items():
        for q in [r] + rs:
            w = q["E"][k]
            assert w["converged"] and abs(w["T"] - v["T"]) < 1e-8 and max(w["its"]) > max(v["its"])
            assert abs(max(w["its"]) - max(r["E"][k]["its"])) <= 1
            assert len(w["hist"][0]) == max(w["its"])


def test_slow(env):
    """Two-level with the slow modes of S_0: the same T as the levels, fewer iterations than Schwarz alone; more rounds and ranks work;
    Q is orthonormal and P_c A is a projector (exact coarse solve)."""
    d, _, ref = env
    r0 = go(d, None, False, "--up", "", "--schwarz", "--maxit", "400")
    more = ["--up", "", "--maxit", "400"]
    one, two = go(d, None, False, *more, "--slow", "32,3,1"), go(d, None, False, *more, "--slow", "16,3,2")
    rs = dist.ThreadComm.run(2, lambda c: go(d, c, False, *more, "--slow", "32,3,1"))
    for q in [one, two] + rs:
        assert not q["h"].U and q["h"].slow.N == 32
        for k, v in ref["E"].items():
            w = q["E"][k]
            assert w["converged"] and abs(w["T"] - v["T"]) < 1e-8 and max(w["its"]) < 0.8 * max(r0["E"][k]["its"])
            assert abs(max(w["its"]) - max(one["E"][k]["its"])) <= 4
    s = one["h"].slow
    assert np.abs(s.Q.conj().T @ s.Q - np.eye(s.N)).max() < 1e-12 and np.abs(s.AQ - one["h"].fine @ s.Q).max() < 1e-12
    X = np.random.default_rng(0).standard_normal((s.Q.shape[0], 3)) + 0j
    P = lambda x: s(one["h"].fine @ x)
    assert np.abs(P(P(X)) - P(X)).max() < 1e-9 * np.abs(P(X)).max()
    Q = np.vstack([q["h"].slow.Q for q in rs])
    assert np.abs(Q.conj().T @ Q - np.eye(32)).max() < 1e-12


def test_gpu(env):
    try:
        gpu.enable()
    except RuntimeError:
        pytest.skip("no GPU")
    try:
        same(go(env[0]), env[2])
    finally:
        gpu.enable(False)
