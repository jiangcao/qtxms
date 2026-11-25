"""qtx_ms.dist: the domain-decomposed hierarchy (window rho, merges, level operators, smoothers, FGMRES) equals the
serial one. Ranks = threads (ThreadComm, numpy); real MPI / NCCL: tests/dist_mpi.py under mpiexec."""
import numpy as np
import scipy.sparse as sp
import pytest

from conftest import strip
from qtx_ms import dist, gpu, msbasis as mb
from qtx_ms.mlevel import Hierarchy
from qtx_ms.schwarz import CapSchwarz, fgmres_block, operator

quiet = lambda *x: None
E, MU, CUT = 0.35, (0.1, 0.6), 0.1
CASES = {"dm": dict(kind="dm"), "dmc_schur": dict(kind="dmc", core=dict(prec="double", refine=1, probes=0)),
         "free1": dict(kind="dm", l1_free=1), "free2": dict(kind="dm", l1_free=2),
         "fine1": dict(kind="dm", fine=1), "fine2": dict(kind="dmc", fine=2, core=dict(prec="double", refine=1, probes=0)),
         "fine4": dict(kind="dm", fine=4),
         "lu_stream": dict(kind="dm", fine=1, l1_solver="lu", fsolver="dense", stream=0.0)}


def solve(comm, sigma_dir, kind="dm", core=None, l1_free=0, ups="yxx", dirs=None, keep_max=np.inf,
          bottom="dist", share=True, store=None, fine=0, l1_solver="rows", fsolver="rows", stream=None):
    inp = strip(nd=8, ny=4, nz=2, no=2)
    inp["meta"]["periodic"] = [dict(axis=1, length=8.0)]          # y strips (the H of the strip itself is open in y)
    dev = mb.Device(inp, obc="folded")
    if store is not None:                                           # (inp, StoreDevice, {rank: rows held}): test_devstore
        inp, dev, held = store
    dev.sigma_cache = sigma_dir
    sw = CapSchwarz(dev, inp, 12.0, 1.0, 2.0, spacing=4.0, solver=fsolver, split=[2], tov=1.5, stream_gb=stream)
    sw.distribute(comm)
    if store is not None:
        rows = np.unique(np.concatenate([sw.idx[k] for k in sw.own]))
        dev.hold(rows)
        held[comm.rank] = len(rows)
    SL, SR = dev.sigma(E + 1e-6j)
    B = dev.inject(SL)
    b = np.zeros((dev.n, B.shape[1]), complex)
    b[dev.bL] = B
    A = dev.A(E, SL, SR).tocsr()
    fa = dist.FineA(dev, E, SL, SR)
    sw.set_energy(E, fa, log=quiet)
    h = Hierarchy(dev, inp, sw, l1_lu="c128", log=quiet, comm=comm, core=core, l1_free=l1_free, halo_smooth="full",
                  keep_max=keep_max, bottom=bottom, share_points=share, fine_blocks=fine,
                  l1_solver=l1_solver, stream_smoother_gb=stream)
    h.basis_dir = dirs
    part, dv = h.part0, gpu.asarray
    Aop = dist.FineOp(fa, part)
    DM = h.window_dm(None, "", MU[0], MU[1], 8, keep=1e-6, core=core)
    h.level1(h.basis1(DM, kind, CUT), E, SL, SR, A0=Aop)
    for i, ax in enumerate(ups):
        h.merge(ax, CUT, kind, MU[0], MU[1], 8, 1 if i == 0 else None, f"{kind}{i}" if dirs else None)
    if store is not None:
        dev.drop()                                                  # the solves need no held rows (CAP rows: files)
    r = np.random.default_rng(0).standard_normal((dev.n, 3)) + 0j
    out = dict(n=dev.n, N=[h.LV[k].N for k in sorted(h.LV)], own=len(sw.own), Adirect=A @ r,
               Ar=part.gather(Aop @ dv(part.local(r))), S0=part.gather(sw(dv(part.local(r)))), x=[], its=[], A=[], Sm=[], cyc=[])
    xe = sp.linalg.spsolve(A.tocsc(), b)
    bl = dv(part.local(b))
    for m in sorted(h.LV):
        L = h.LV[m]
        if L.N > keep_max:                                          # S, H dropped: no exact solve with this level
            assert L.Sb is None or fine
            continue
        ex = h.exact(m)
        out["A"].append(L.A.dense())
        if m > 1:                                                   # smoother of the level below
            rk = np.random.default_rng(m).standard_normal((L.parent.N, 2)) + 0j
            out["Sm"].append(L.parent.part.gather(L.Sbelow(dv(L.parent.part.local(rk)))))
        prec, _ = h.cycle(m, ex, Aop, sw)
        out["cyc"].append(part.gather(prec(dv(part.local(r)))))     # fine vector: independent of the block bases' gauge
        x, its, res = fgmres_block(Aop, bl, gpu.xp_of(bl).zeros_like(bl), prec, 1e-10, 200, comm=comm)
        assert res.max() < 1e-9
        out["x"].append(part.gather(x))
        out["its"].append(its)
    assert all(np.linalg.norm(gpu.asnumpy(x) - xe) < 1e-8 * np.linalg.norm(xe) for x in out["x"])
    return out


@pytest.fixture(scope="module")
def sigma_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("sigma"))
    REF.update({c: solve(dist.Comm(), d, **kw) for c, kw in CASES.items()})
    return d


REF = {}


@pytest.mark.parametrize("cg", [False, True])
def test_fine_blocks_equal_stored(sigma_dir, cg):
    """Levels formed from the sparse core-pair blocks = the stored Galerkin blocks (every piece of S_k, H_k, A_k), real and
    complex H; nothing of S_k, H_k is stored."""
    inp = strip(nd=8, ny=4, nz=2, no=2, complex_gauge=cg)
    inp["meta"]["periodic"] = [dict(axis=1, length=8.0)]
    dev = mb.Device(inp, obc="folded")
    SL, SR = dev.sigma(E + 1e-6j)
    hs = []
    for fine in (0, 3):
        sw = CapSchwarz(dev, inp, 12.0, 1.0, 2.0, spacing=4.0, solver="rows", split=[2], tov=1.5)
        A = dev.A(E, SL, SR).tocsr()
        sw.set_energy(E, A, log=quiet)
        h = Hierarchy(dev, inp, sw, l1_lu="c128", log=quiet, l1_free=3 - fine, fine_blocks=fine, halo_smooth="full")
        DM = h.window_dm(None, "", MU[0], MU[1], 8, keep=1e-6)
        if hs:                                                      # the same bases: compare entry by entry
            h.level1(hs[0].LV[1].U, E, SL, SR, A0=operator(A))
        else:
            h.level1(h.basis1(DM, "dm", CUT), E, SL, SR, A0=operator(A))
        for ax in "yx":
            h.merge(ax, CUT, "dm", MU[0], MU[1], 8, None, None)
        hs.append(h)
    a, b = hs
    for k in (1, 2, 3):
        La, Lb = a.LV[k], b.LV[k]
        assert type(Lb.Sb).__name__ == "FineMat" and list(La.r) == list(Lb.r)
        if k > 1:
            for u, v in zip(La.U, Lb.U):
                assert np.abs(np.linalg.svd(gpu.asnumpy(u).conj().T @ gpu.asnumpy(v), compute_uv=False) - 1).max() < 1e-8
            Lb.U = La.U                                             # gauge of the merges
        ca = list(range(len(La.r)))
        for nm, Xa, Xb in (("S", La.Sb, Lb.Sb), ("H", La.Hb, Lb.Hb)):
            for r, c in ((ca, ca), (ca[:3], ca[2:]), (ca[1:4], ca[1:4][::-1])):
                assert np.allclose(gpu.asnumpy(a.dense(La, Xa, r, c)), gpu.asnumpy(b.dense(Lb, Xb, r, c)), atol=1e-11), (k, nm)
                b.fb.keep(Lb)                                       # dense pairs kept: the same pieces, one triangle stored
                assert np.allclose(gpu.asnumpy(a.dense(La, Xa, r, c)), gpu.asnumpy(b.dense(Lb, Xb, r, c)), atol=1e-11)
                assert all(i <= j for _, i, j in b.fb.pairs)
                b.fb.release()
        for r, c in ((ca, ca), (ca[:3], ca[2:])):                   # E S - H directly from the sparse A_ij
            assert np.allclose(gpu.asnumpy(a.dense_A(La, E, r, c)), gpu.asnumpy(b.dense_A(Lb, E, r, c)), atol=1e-11), k
    b.fb.set_energy(E + 0.1)                                        # another energy: A_ij rebuilt
    La, Lb = a.LV[1], b.LV[1]
    assert np.allclose(gpu.asnumpy(a.dense_A(La, E + 0.1, ca, ca)), gpu.asnumpy(b.dense_A(Lb, E + 0.1, ca, ca)), atol=1e-11)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("size", [2, 3, 5])
def test_distributed_equals_serial(sigma_dir, case, size):
    ref = REF[case]
    outs = dist.ThreadComm.run(size, lambda c: solve(c, sigma_dir, **CASES[case]))
    assert all(o["own"] > 0 for o in outs)
    for o in outs:
        assert o["N"] == ref["N"]
        assert np.allclose(o["Ar"], o["Adirect"], atol=1e-12) and np.allclose(o["S0"], ref["S0"], atol=1e-11)
        for a, b in zip(o["A"], ref["A"]):                          # shared points change the order of the sums:
            assert np.allclose(*(np.sort_complex(np.linalg.eigvals(v)) for v in (a, b)), atol=1e-9)    # gauge only
        for a, b in zip(o["cyc"], ref["cyc"]):
            assert np.linalg.norm(a - b) < 1e-8 * np.linalg.norm(b)
        for a, b in zip(o["its"], ref["its"]):
            assert np.abs(a - b).max() <= 1
        for a, b in zip(o["x"], ref["x"]):
            assert np.linalg.norm(a - b) < 1e-8 * np.linalg.norm(b)


def test_stored_bases(sigma_dir, tmp_path):
    """Bases written by the first run (rank 0) are read back by every rank."""
    a = dist.ThreadComm.run(3, lambda c: solve(c, sigma_dir, dirs=str(tmp_path)))[0]
    assert len(list(tmp_path.glob("l*.npz"))) == 3
    b = dist.ThreadComm.run(2, lambda c: solve(c, sigma_dir, dirs=str(tmp_path)))[0]
    assert a["N"] == b["N"] and all(
        np.linalg.norm(x - y) < 1e-8 * np.linalg.norm(y) for x, y in zip(a["cyc"], b["cyc"]))


def test_lu_host_view():
    """LU factors through the transposed, C-contiguous host form of a Streamer item (CPU mode: numpy)."""
    rng = np.random.default_rng(0)
    M = rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
    b = rng.standard_normal((9, 2)) + 0j
    lu = gpu.lu_factor(np.asfortranarray(M.copy()))
    it = gpu.lu_host(lu)
    assert it[0].flags.c_contiguous
    assert np.allclose(M @ gpu.lu_solve(gpu.lu_view(it), b), b)
    xs = {}
    gpu.Streamer([it, it]).run(lambda k, v: xs.update({k: gpu.lu_solve(gpu.lu_view(v), b)}))
    assert all(np.allclose(M @ x, b) for x in xs.values())


def test_chunks_and_halo():
    assert dist.chunks([1, 1, 100, 1, 1], 4).tolist() and set(dist.chunks([1, 1, 100, 1, 1], 4)) == {0, 1, 2, 3}
    assert dist.chunks(np.ones(8), 4).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    n = 23

    def f(c):
        part = dist.Part(c, [np.arange(0, 7), np.arange(7, 9), np.arange(9, 23)], [0, 1, 2], n)
        need = np.random.default_rng(c.rank).choice(n, 9, replace=False)
        ex = dist.Exch(part, need)
        g = np.arange(n, dtype=complex)[:, None] * np.ones(2)
        x = ex(part.local(g))
        assert np.array_equal(x[ex.pos[need]], g[need])
        z = part.local(np.zeros((n, 2), complex))
        zx = np.zeros((part.nloc + ex.ng, 2), complex)
        zx[:part.nloc] = z
        zx[part.nloc:] = g[np.array(sorted(range(n), key=lambda i: ex.pos[i]))[-ex.ng:]] if ex.ng else 0
        return part.gather(ex.rev(zx, z))[:, 0], need, part.gown
    outs = dist.ThreadComm.run(3, f)
    sent = {int(i) for r, (_, need, gown) in enumerate(outs) for i in need if gown[i] != r}
    for z, _, _ in outs:
        assert all(z[i] == (i if i in sent else 0) for i in range(n))


def test_dropped_blocks_of_free_levels(sigma_dir):
    """Free levels above keep_max drop S, H after the merge above them: the deeper levels are unchanged."""
    ref = REF["free2"]
    o = dist.ThreadComm.run(3, lambda c: solve(c, sigma_dir, keep_max=ref["N"][2], **CASES["free2"]))[0]
    assert o["N"] == ref["N"] and len(o["A"]) == len(ref["A"]) - 2
    assert all(np.linalg.norm(a - b) < 1e-8 * np.linalg.norm(b) for a, b in zip(o["cyc"], ref["cyc"][2:]))
    assert all(np.abs(a - b).max() <= 1 for a, b in zip(o["its"], ref["its"][2:]))


@pytest.mark.parametrize("case", ["dm", "dmc_schur"])
@pytest.mark.parametrize("size", [3, 5])
def test_options_off(sigma_dir, case, size):
    """Replicated bottom solve, windows integrated by their owners only: the same results, entry by entry."""
    ref = REF[case]
    o = dist.ThreadComm.run(size, lambda c: solve(c, sigma_dir, bottom="repl", share=False, **CASES[case]))[0]
    assert o["N"] == ref["N"] and all(np.allclose(a, b, atol=1e-9) for a, b in zip(o["A"], ref["A"]))
    assert all(np.allclose(a, b, atol=1e-8) for a, b in zip(o["Sm"], ref["Sm"]))


def test_share_points_is_balanced():
    rng = np.random.default_rng(1)
    for size, nc, npts in [(4, 3, 24), (4, 5, 24), (8, 3, 24), (3, 11, 8), (2, 2, 1), (5, 1, 4)]:
        own = dist.chunks(np.ones(nc), min(size, nc)) * (size // min(size, nc))
        cost = rng.uniform(1, 3, nc) ** 3
        pts = dist.share_points(cost, own, npts, size)
        assert pts == dist.share_points(cost, own, npts, size)
        load0, load = np.zeros(size), np.zeros(size)
        for c in range(nc):
            assert sorted(q for d in pts[c].values() for q in d) == list(range(npts)) and own[c] in pts[c]
            load0[own[c]] += cost[c] * npts
            for r, q in pts[c].items():
                load[r] += cost[c] * len(q)
        assert load.max() <= load0.max() + 1e-12
        if npts == 24:
            assert load.max() <= 1.1 * load.mean() + cost.max()      # one point is the unit; helpers pay one more
    # the points really move in the hierarchy tests
    assert any(len(d) > 1 for d in dist.share_points([8.0, 1.0, 1.0], [0, 1, 2], 8, 3))


def test_domains():
    for S in range(1, 30):
        for size in range(1, 10):
            dom, sep, ranks = dist.domains(np.ones(S), size)
            assert len(dom) == min(size, (S + 1) // 2) == len(set(ranks)) and max(ranks) < size
            assert all(b > a for a, b in dom)
            assert sorted([s for a, b in dom for s in range(a, b)] + sep) == list(range(S))
    assert sum(b - a for a, b in dist.domains(np.ones(40), 4)[0]) == 37


class _Tri:
    """Block-tridiagonal operator with DistOp's interface (take_all of global block lists), Sigma on the ends only."""

    def __init__(self, sizes, E, seed=0):
        rng = np.random.default_rng(seed)
        self.off = np.concatenate([[0], np.cumsum(sizes)])
        n = int(self.off[-1])
        H = np.zeros((n, n), complex)
        for s in range(len(sizes)):
            a, b = self.off[s], self.off[s + 1]
            X = rng.standard_normal((b - a, b - a)) + 1j * rng.standard_normal((b - a, b - a))
            H[a:b, a:b] = (X + X.conj().T) / 4
            if s + 1 < len(sizes):
                c = self.off[s + 2]
                T = rng.standard_normal((b - a, c - b)) + 1j * rng.standard_normal((b - a, c - b))
                H[a:b, b:c], H[b:c, a:b] = 0.3 * T, 0.3 * T.conj().T
        A = E * np.eye(n) - H
        for s in (0, len(sizes) - 1):
            a, b = self.off[s], self.off[s + 1]
            A[a:b, a:b] += 0.5j * np.eye(b - a)
        self.A, self.cols = A, [list(range(max(0, s - 1), min(len(sizes), s + 2))) for s in range(len(sizes))]

    def take_all(self, wr, wc):
        r = np.concatenate([np.arange(self.off[k], self.off[k + 1]) for k in wr])
        c = np.concatenate([np.arange(self.off[k], self.off[k + 1]) for k in wc])
        return self.A[np.ix_(r, c)]


@pytest.mark.parametrize("size", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("S", [1, 2, 3, 6, 25])
def test_dist_slab_lu(size, S):
    """The domain solve equals a dense solve; energies inside the spectrum: the interiors are closed (no Sigma)."""
    from qtx_ms.mlevel import DistSlabLU
    sizes = np.random.default_rng(S).integers(3, 9, S)
    if S < size:
        pytest.skip("every rank owns a block")
    worst = 0.0
    for E in (-0.2, 0.0, 0.13, 0.5):
        T = _Tri(sizes, E)
        own = dist.chunks(np.ones(S), size)
        r = np.random.default_rng(3).standard_normal((int(T.off[-1]), 2)) + 0j
        ref = np.linalg.solve(T.A, r)

        def f(c):
            part = dist.Part(c, [np.arange(T.off[k], T.off[k + 1]) for k in range(S)], own, int(T.off[-1]))
            lu = DistSlabLU(T, np.arange(S), part)
            assert lu.sizes == list(sizes)
            one = lu(part.local(r[:, 0]))
            x = lu(part.local(r))
            assert np.allclose(one, x[:, 0])
            return part.gather(x)
        for x in dist.ThreadComm.run(size, f):
            worst = max(worst, np.linalg.norm(x - ref) / np.linalg.norm(ref))
    assert worst < 1e-9, worst
