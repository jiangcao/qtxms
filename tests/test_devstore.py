"""qtx_ms.devstore: sliced H, S on disk = the Device of the full matrices; the distributed hierarchy on the store = serial."""
import json, os
import numpy as np
import pytest

from conftest import strip
import test_dist as td
from qtx_ms import devstore, dist, msbasis as mb


def make(folder, **kw):
    inp = strip(nd=8, ny=4, nz=2, no=2, **kw)
    inp["meta"]["periodic"] = [dict(axis=1, length=8.0)]
    dev = mb.Device(inp, obc="folded")
    devstore.save(dev, inp, str(folder))
    return inp, dev


@pytest.mark.parametrize("cg", [False, True])
def test_store_equals_device(tmp_path, cg):
    inp, dev = make(tmp_path / "st", complex_gauge=cg)
    i2, d2 = devstore.load(str(tmp_path / "st"))
    assert "H" not in i2 and "S" not in i2
    for k in ("dev", "sb", "bL", "bR", "loc"):
        assert np.array_equal(getattr(dev, k), getattr(d2, k)), k
    assert dev.ns == d2.ns and dev.n == d2.n
    rng = np.random.default_rng(1)
    for nm in "HS":
        A, B = getattr(dev, nm), getattr(d2, nm)
        assert abs(A - B.full()).max() == 0
        idx = rng.permutation(dev.n)[:40]
        assert abs(A[idx][:, idx] - B.sub(idx)).max() == 0         # nothing held: from the files
        with pytest.raises(KeyError):
            B[idx]
        B.hold(idx)
        assert abs(A[idx] - B[idx]).max() == 0 and abs(A[idx[:7]] - B[idx[:7]]).max() == 0
        assert abs(A[idx][:, idx] - B.sub(idx)).max() == 0
        far = np.setdiff1d(np.arange(dev.n), idx)[:9]
        with pytest.raises(KeyError):
            B[far]
        mix = np.concatenate([idx[:5], far])
        assert abs(A[mix][:, mix] - B.sub(mix)).max() == 0         # held + not held
        B.drop()
        with pytest.raises(KeyError):
            B[idx]
    z = 0.35 + 1e-3j
    for a, b in zip(dev.sigma(z), d2.sigma(z)):
        assert np.array_equal(a, b)
    S = dev.sigma(z)
    assert abs(dev.A(0.35, *S) - d2.A(0.35, *S)).max() == 0 and isinstance(d2.H, devstore.SlabRows)


def test_sparse_couplings(tmp_path):
    """The lead-device couplings are CSR files; a format 1 store (dense files) loads, and --sparse-couplings adds the CSR files to it."""
    inp, dev = make(tmp_path / "st")
    st = str(tmp_path / "st")
    assert not os.path.exists(st + "/cL0.npy") and os.path.exists(st + "/cL0_data.npy")
    same = lambda d2: all(a.shape == b.shape and abs(a - b).max() == 0 for c, d in ((dev.cL, d2.cL), (dev.cR, d2.cR)) for a, b in zip(c, d))
    assert same(devstore.load(st)[1])
    for s, c in (("L", dev.cL), ("R", dev.cR)):                       # format 1
        for q, M in enumerate(c):
            for k in devstore.CSR:
                os.remove(f"{st}/c{s}{q}_{k}.npy")
            np.save(f"{st}/c{s}{q}.npy", M.toarray())
    info = json.load(open(st + "/store.json"))
    json.dump(dict(info, format=1), open(st + "/store.json", "w"))
    assert same(devstore.load(st)[1])
    devstore.main(["--sparse-couplings", st])
    assert os.path.exists(st + "/cR3_indptr.npy") and same(devstore.load(st)[1])
    assert devstore.coupling(st, "L", 0, len(dev.bL)).nnz < dev.cL[0].shape[0] * dev.cL[0].shape[1]


def test_stale_store_refused(tmp_path):
    src = tmp_path / "inputs"
    from qtx_ms.prep.inputs import write_inputs
    inp = strip()
    w = lambda: write_inputs(src, inp["labels"], inp["xyz"], inp["norb"], inp["blocks"], inp["H"], inp["S"], inp["lead"],
                             inp["meta"])
    w()
    st = str(tmp_path / "st")
    devstore.main(["--inputs", str(src), "--out", st])
    devstore.load(st, source=str(src))
    with pytest.raises(FileExistsError):
        devstore.main(["--inputs", str(src), "--out", st])
    os.utime(src / "H.npz", (1, 1))
    with pytest.raises(ValueError):
        devstore.load(st, source=str(src))
    info = json.load(open(os.path.join(st, "store.json")))
    json.dump(dict(info, format=99), open(os.path.join(st, "store.json"), "w"))
    with pytest.raises(ValueError):
        devstore.load(st)
    assert not [f for f in os.listdir(tmp_path) if ".tmp" in f]


@pytest.mark.parametrize("case", ["dmc_schur", "free2", "fine1", "fine2"])
def test_distributed_on_store(tmp_path, case):
    """Thread ranks with sliced rows: level sizes, operators, its and solutions of the serial run on the full matrices;
    the rows are dropped before the solves (solve(): dev.drop())."""
    inp, _ = make(tmp_path / "st")
    sg = str(tmp_path / "sigma")
    ref = td.solve(dist.Comm(), sg, **td.CASES[case])
    held = {}

    def run(c):
        i2, d2 = devstore.load(str(tmp_path / "st"))
        return td.solve(c, sg, store=(i2, d2, held), **td.CASES[case])
    outs = dist.ThreadComm.run(3, run)
    assert len(held) == 3 and all(0 < v < ref["n"] for v in held.values())
    for o in outs:
        assert o["N"] == ref["N"]
        assert np.allclose(o["Ar"], ref["Ar"], atol=1e-12) and np.allclose(o["S0"], ref["S0"], atol=1e-11)
        for a, b in zip(o["cyc"], ref["cyc"]):
            assert np.linalg.norm(a - b) < 1e-8 * np.linalg.norm(b)
        for a, b in zip(o["its"], ref["its"]):
            assert np.abs(a - b).max() <= 1
        for a, b in zip(o["x"], ref["x"]):
            assert np.linalg.norm(a - b) < 1e-8 * np.linalg.norm(b)


def test_hostmem():
    m = devstore.hostmem()
    assert m["anon"] > 0 and m["cpu"] > 0
