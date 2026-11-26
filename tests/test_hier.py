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


def test_bottom_domains(env):
    """Distributed block-tridiagonal LU (domains + separator system on one rank) = the replicated one: 8 slabs on 3 / 4 ranks
    (2 / 3 separators, holder with and without a domain of its own); smaller factors on the largest rank."""
    d, up = env[0], "y:1e-3,x:1e-3"
    r0 = dist.ThreadComm.run(2, lambda c: go(d, c, True, "--up", up, "--bottom", "btrep"))[0]
    for size, ns in ((3, 2), (4, 3)):
        rs = dist.ThreadComm.run(size, lambda c: go(d, c, True, "--up", up))
        for r in rs:
            h = r["h"]
            assert len(h.bplan["sep"]) == ns and h.bottom_res < 1e-10 and r["N"] == r0["N"]
            assert h.bottom_gb < 0.7 * 3 * sum(n * n for n in h.bottom_n) * 16 / 2 ** 30
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


def test_beyn(env, tmp_path):
    """Beyn Sigma of the subdivided lead (3 cells) = folded lead: contour points to 1e-6, real axis up to the eta = 1e-6 of the folded lead;
    T of both models; the cached files are kept apart; the bases of beyn-real are those of folded."""
    d, inp, ref = env
    dev = mb.Device(devstore.read_any(str(d / "dev")), obc="folded")
    q = {m: obc.OBC(dev, dist.Comm(), log=lambda *x: None, model=m) for m in ("folded", "beyn")}
    zs = [0.35 + 0.1j, 0.5 + 0.05j, 0.42 + 1e-6j]
    for o in q.values():
        o.prepare(zs[:2], zs[2:])
    inp1 = q["beyn"].dev.inp
    sec, pm = obc.smallest_cell(inp1, np.where(np.repeat(inp1["blocks"], inp1["norb"]) == inp1["meta"]["transport"]["lead_left_block"])[0])
    assert sec == 3 and len(pm) == 48 and sorted(q["beyn"]._leads["L"][0].layer["L", "H"]) == [-1, 0, 1]
    for z in zs:
        for side in (0, 1):
            A, B = q["beyn"].sigma(z, side), q["folded"].sigma(z, side)
            assert np.abs(A - B).max() < (1e-5 if z.imag > 1e-3 else 1e-2) * np.abs(B).max()
    z = -2.65 + 1e-6j                                                  # more eigenvalues in the annulus than the cell has orbitals
    for o in q.values():
        o.prepare([], [z])
    A, B = q["beyn"].sigma(z, 0), q["folded"].sigma(z, 0)
    assert q["beyn"].inject(z).shape[1] == q["folded"].inject(z).shape[1] == 2 and np.abs(A - B).max() < 1e-2 * np.abs(B).max()
    for z in zs[2:]:                                                    # the incoming modes: B B^H = Gamma_L
        S, B = q["beyn"].sigma(z, 0), q["beyn"].inject(z)
        G = 1j * (S - S.conj().T)
        assert 0 < B.shape[1] and np.abs(B @ B.conj().T - G).max() < 1e-5 * np.abs(G).max()
    r = {m: go(d, None, False, "--obc", m, "--sigma-cache", str(tmp_path / "s"), "--cache", str(tmp_path / "b")) for m in ("folded", "beyn-real", "beyn")}
    for m in ("beyn-real", "beyn"):
        for k, v in ref["E"].items():
            w = r[m]["E"][k]
            assert w["converged"] and abs(w["T"] - v["T"]) < 1e-2 * abs(v["T"])
    assert r["beyn-real"]["N"] == r["folded"]["N"]
    assert len(list((tmp_path / "s").glob("bmode_*"))) == 2 == len(list((tmp_path / "s").glob("inj_*")))
    assert len(list((tmp_path / "s").glob("bsigL_*"))) == 10 == len(list((tmp_path / "s").glob("bsigR_*")))    # Beyn: one file per side
    assert len(list((tmp_path / "s").glob("sig_*"))) == 10
    assert len(list((tmp_path / "b").glob("hier_*"))) == 2


def test_contour_sigma_only_for_rho(env, tmp_path):
    """Beyn at the contour points only when a density matrix is built: not with all bases cached, again when a basis is lost."""
    d, _, _ = env
    m = lambda s: ["--obc", "beyn", "--sigma-cache", str(tmp_path / s), "--cache", str(tmp_path / "b")]
    eq = lambda r: r["N"] == ref["N"] and all(abs(v["T"] - ref["E"][k]["T"]) < 1e-8 and v["converged"] for k, v in r["E"].items())
    count = lambda s: (len(list((tmp_path / s).glob("bsigL_*"))), len(list((tmp_path / s).glob("sig_*"))))
    ref = go(d, None, False, *m("s0"))
    npts = count("s0")[0] - 2
    assert npts > 0 and count("s0")[1] == 0                            # two energies; every contour point with Beyn
    assert eq(go(d, None, False, *m("s1")))
    assert count("s1") == (2, 0)
    next((tmp_path / "b").glob("hier_*/U2_0.npy")).unlink()
    assert eq(go(d, None, False, *m("s2")))
    assert count("s2") == (npts + 2, 0)


def test_beyn_ranks(env, tmp_path, monkeypatch):
    """Beyn tasks per point and side: every (z, side) once, the left lead on the even ranks, the right lead on the odd ranks; T as on one rank;
    the coupling bases are stored in the basis folder and loaded by a later run."""
    from qtx_ms.obc_beyn import BeynLead
    d, _, _ = env
    calls, where = collections.Counter(), collections.defaultdict(set)
    sig = BeynLead.sigma

    def counted(self, z, side, *x):
        calls[complex(z), side] += 1
        return sig(self, z, side, *x)
    monkeypatch.setattr(BeynLead, "sigma", counted)
    m = lambda s: ["--obc", "beyn", "--sigma-cache", str(tmp_path / s), "--cache", str(tmp_path / "b")]
    ref = go(d, None, False, "--obc", "beyn")                          # one rank, no cache
    calls.clear()
    rs = dist.ThreadComm.run(3, lambda c: go(d, c, True, *m("s")))
    assert len(calls) == 20 and set(calls.values()) == {1}            # 8 contour + 2 real points, two sides, each on one rank
    for q, r in enumerate(rs):
        where[q] = set(r["h"].obc._leads)
        assert r["N"] == ref["N"] and all(abs(v["T"] - ref["E"][k]["T"]) < 1e-9 for k, v in r["E"].items())
    assert where[0] == where[2] == {"L"} and where[1] == {"R"}
    assert len(list((tmp_path / "b").glob("beyn_basis_L_*.npz"))) == 1 == len(list((tmp_path / "b").glob("beyn_basis_R_*.npz")))
    calls.clear()
    r = go(d, None, False, *m("s2"))                                   # bases cached: only the real points, coupling bases loaded
    assert len(calls) == 4 and all(abs(v["T"] - ref["E"][k]["T"]) < 1e-9 for k, v in r["E"].items())
    assert all(r["h"].obc._leads[sd][0]._bases[sd].get("loaded") for sd in "LR")


@pytest.mark.parametrize("kmax", [1, 3, 5])
def test_lead_blocks(kmax):
    """Periodic lead of 5 primitive cells: only the nonzero blocks L_k are stored (real); g of the subdivided lead = decimation of the lead
    cell at complex z and next to the real axis (the Beyn rank exceeds the cell here: higher moments), incoming modes B B^H = Gamma."""
    import scipy.sparse as sp
    from qtx_ms.obc_beyn import BeynLead
    m, s = 12, 5
    n, r = s * m, np.random.default_rng(3)

    def lead(scale):
        L = {k: scale * 0.3 ** k * r.standard_normal((m, m)) for k in range(kmax + 1)}
        L[0] = L[0] + L[0].T
        for k in range(1, kmax + 1):
            L[-k] = L[k].T
        return [sp.csr_matrix(np.block([[L.get(j - i + o, np.zeros((m, m))) for j in range(s)] for i in range(s)])) for o in (0, s)]
    (H0, H1), (S0, S1) = lead(1.0), lead(0.02)
    S0 = S0 + sp.identity(n)
    fl = mb.FoldedLead(H0, H1, S0.tocsr(), S1)
    I = sp.identity(n, format="csr")
    for side, (dev_c, gi) in dict(L=(H1, 0), R=(H1.T.tocsr(), 1)).items():
        for z in (0.3 + 0.1j, 0.3 + 1e-6j):
            for sec in (s, 1):
                bl = BeynLead(H0, H1, S0, S1, sec, np.arange(n) if sec > 1 else None, sides=side, m0=n // sec)
                assert sorted(bl.layer[side, "H"]) == (list(range(-kmax, kmax + 1)) if sec > 1 else [-1, 0, 1])
                assert all(v.dtype == float for q in "HS" for v in bl.layer[side, q].values())
                g = bl.sigma(z if z.imag > 1e-3 else z.real, side, I, I)[0]
                ref = fl.surface_gfs(z)[gi]
                assert np.abs(g[np.ix_(fl.F, fl.F)] - ref).max() < (1e-9 if z.imag > 1e-3 else 1e-3) * np.abs(ref).max()
                if sec > 1:
                    d, u = bl.sigma(z.real, side, I, I)
                    G = 1j * (d - d.conj().T)
                    assert np.abs(u @ u.conj().T - G).max() < 1e-5 * max(np.abs(G).max(), 1)


def test_closed(env, tmp_path):
    """No lead: columns of G_c = (E S - H)^-1 on the contact faces against the dense inverse, all and some columns, ranks; no Sigma is
    computed; the bases of the closed device are not those of the open one."""
    d, _, ref = env
    more = ["--obc", "closed", "--maxit", "200", "--cache", str(tmp_path / "b"), "--chunk", "7", "--out"]
    r = go(d, None, False, *more, str(tmp_path / "r.json"))
    few = go(d, None, False, *more, str(tmp_path / "f.json"), "--rhs-cols", "5")
    go(d, None, False, "--cache", str(tmp_path / "b"))
    assert len(list((tmp_path / "b").glob("hier_*"))) == 2 and not r["h"].obc.mem
    rs = dist.ThreadComm.run(2, lambda c: go(d, c, False, *more, str(tmp_path / "t.json")))
    dev = r["h"].dev
    for k, v in r["E"].items():
        G = np.linalg.inv((v["E"] * dev.S - dev.H).toarray())
        for f, n in (("r", len(dev.bL)), ("f", 5), ("t", len(dev.bL))):
            z = np.load(tmp_path / f"{f}.json.G_{float(k):+.4f}.npz")
            assert len(z["J"]) == n
            for X, Y in ((z["G_LL"], G[np.ix_(dev.bL[z["J"]], dev.bL[z["J"]])]), (z["G_RL"], G[np.ix_(dev.bR, dev.bL[z["J"]])])):
                assert np.abs(X - Y).max() < 1e-6 * np.abs(G).max()
        for q in [v, few["E"][k]] + [x["E"][k] for x in rs]:
            assert q["converged"] and q["T"] is None and q["closed"]["asym"] < 1e-6


def test_gpu(env):
    try:
        gpu.enable()
    except RuntimeError:
        pytest.skip("no GPU")
    try:
        same(go(env[0]), env[2])
    finally:
        gpu.enable(False)


def test_dV_reuses_bases(env, tmp_path):
    """--dV: another gate voltage on the cached bases (and Sigma) of the inputs: H += (dV S + S dV) / 2 on the fine level, Galerkin
    pairs = P^T H' P, T of the dense solve with H'; refused without the cached bases."""
    d, inp, _ = env
    x = inp["xyz"][:, 0]
    dV = 0.08 * np.clip(1.0 - np.abs(x - x.mean()) / (0.3 * np.ptp(x)), 0.0, 1.0)  # gate bump in the middle, 0 near the leads (per atom)
    np.save(tmp_path / "dV.npy", dV)
    with pytest.raises(SystemExit):
        go(d, None, False, "--cache", str(tmp_path / "c"), "--dV", str(tmp_path / "dV.npy"))
    r0 = go(d, None, False, "--cache", str(tmp_path / "c"), "--sigma-cache", str(tmp_path / "s"))
    files = {f: f.stat().st_mtime_ns for f in (tmp_path / "c").rglob("U*.npy")}
    np.save(tmp_path / "zero.npy", np.zeros_like(dV))
    r = go(d, None, True, "--cache", str(tmp_path / "c"), "--sigma-cache", str(tmp_path / "s"),
           "--dV", f"{tmp_path / 'dV.npy'},{tmp_path / 'zero.npy'}")
    assert {f: f.stat().st_mtime_ns for f in (tmp_path / "c").rglob("U*.npy")} == files and r["N"] == r0["N"]
    h = r["h"]
    dev = mb.Device(devstore.read_any(str(d / "dev")), obc="folded")
    v = np.repeat(dV, inp["norb"])[dev.dev]
    Hp = dev.H + 0.5 * (dev.S.multiply(v[:, None]) + dev.S.multiply(v[None, :]))
    P = np.eye(dev.n)
    for k in range(1, h.K + 1):
        Uk = np.zeros((P.shape[1], h.N[k]))
        for b in range(h.lv[k].n):
            i = h.fine.cores[b] if k == 1 else h.entries(k - 1, h.lv[k].kids[b])
            Uk[np.ix_(i, h.entries(k, [b]))] = h.U[k][b]
        P = P @ Uk
    D = P.T @ (Hp @ P)
    for i in range(h.lv[h.K].n):
        assert np.abs(h.dense(h.K, [i], h.lv[h.K].nbr[i])[1] - D[np.ix_(h.entries(h.K, [i]), h.entries(h.K, h.lv[h.K].nbr[i]))]).max() < 1e-10
    for e, w in r["E"].items():
        assert w["converged"]
        z = w["E"] + 1e-6j
        (SL, SR), B = dev._sigma(z)[0], h.obc.inject(z)
        b = np.zeros((dev.n, B.shape[1]), complex)
        b[dev.bL] = B
        y = np.linalg.solve((dev.A(w["E"], SL, SR) - (Hp - dev.H)).toarray(), b)[dev.bR]
        T = np.real(np.einsum("ic,ij,jc->", y.conj(), 1j * (SR - SR.conj().T), y))
        assert abs(T - w["T"]) < 1e-9 and abs(T - r0["E"][e]["T"]) > 1e-4   # the barrier changes T
    with pytest.raises(SystemExit):                                   # build: not into the folder of the inputs' bases
        go(d, None, True, "--cache", str(tmp_path / "c"), "--dV", str(tmp_path / "dV.npy"), "--dV-bases", "build")
    rb = go(d, None, True, "--cache", str(tmp_path / "b"), "--sigma-cache", str(tmp_path / "s"), "--dV", str(tmp_path / "dV.npy"),
            "--dV-bases", "build")
    for e, w in rb["E"].items():
        assert w["converged"] and abs(w["T"] - r["E"][e]["T"]) < 1e-9



def test_dV_source_order():
    """run.to_orbitals: a per-atom / per-orbital potential of the source atom order -> prepared orbitals (atoms reordered)."""
    norb_src = np.array([2, 1, 3, 2])
    order = np.array([2, 0, 3, 1])                                    # prep atom i = source atom order[i]
    off = np.concatenate([[0], np.cumsum(norb_src)])
    perm = np.concatenate([np.arange(off[i], off[i + 1]) for i in order])
    Va = np.array([1.0, 2.0, 3.0, 4.0])
    want = np.repeat(Va[order], norb_src[order])
    assert np.array_equal(run.to_orbitals(Va, norb_src[order], perm), want)
    assert np.array_equal(run.to_orbitals(np.repeat(Va, norb_src), norb_src[order], perm), want)
    assert np.array_equal(run.to_orbitals(Va, norb_src), np.repeat(Va, norb_src))


def test_dV_drain_bias(env, tmp_path):
    """--dV with a drain step (another source-drain bias): dV = -delta on the right part, right lead face included -> Sigma_R(z + delta)
    of the given lead (Beyn); T = the dense solve with H' and an unshifted OBC evaluated at z + delta; bases and left lead reused."""
    d, inp, _ = env
    x = inp["xyz"][:, 0]
    t = np.clip((x - (x.min() + 0.45 * np.ptp(x))) / (0.2 * np.ptp(x)), 0.0, 1.0)
    delta = 0.03
    dV = 0.05 * np.clip(1.0 - np.abs(x - x.mean()) / (0.2 * np.ptp(x)), 0.0, 1.0) - delta * t * t * (3 - 2 * t)
    np.save(tmp_path / "dV.npy", dV)
    c, s = ("--cache", str(tmp_path / "c")), ("--sigma-cache", str(tmp_path / "s"))
    r0 = go(d, None, False, "--obc", "beyn", *c, *s)
    with pytest.raises(AssertionError):                               # the shift needs Beyn
        go(d, None, False, *c, *s, "--dV", str(tmp_path / "dV.npy"))
    r = go(d, None, True, "--obc", "beyn", *c, *s, "--dV", str(tmp_path / "dV.npy"))
    h = r["h"]
    assert h.obc.shift == (0.0, delta) and r["N"] == r0["N"]
    dev = mb.Device(devstore.read_any(str(d / "dev")), obc="folded")
    v = np.repeat(dV, inp["norb"])[dev.dev]
    assert not v[dev.bL].any() and np.all(v[dev.bR] == -delta)
    dH = 0.5 * (dev.S.multiply(v[:, None]) + dev.S.multiply(v[None, :]))
    o = obc.OBC(dev, dist.Comm(), log=lambda *x: None, model="beyn")
    for e, w in r["E"].items():
        z = w["E"] + 1e-6j
        o.prepare([], [z, z + delta])
        SL, SR = o.sigma(z, 0), o.sigma(z + delta, 1)
        assert np.abs(SR - dev._sigma(z + delta)[0][1]).max() < 1e-2 * np.abs(SR).max()     # folded lead at the shifted energy
        B = o.inject(z)
        b = np.zeros((dev.n, B.shape[1]), complex)
        b[dev.bL] = B
        y = np.linalg.solve((dev.A(w["E"], SL, SR) - dH).toarray(), b)[dev.bR]
        T = np.real(np.einsum("ic,ij,jc->", y.conj(), 1j * (SR - SR.conj().T), y))
        assert w["converged"] and abs(T - w["T"]) < 1e-9 and abs(T - r0["E"][e]["T"]) > 1e-4
