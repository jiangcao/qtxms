"""The single-purpose drivers (mlbasis, mlsolve, mlmode, mlexact) against the study driver and the dense solution; serial and
thread ranks on the device store; the caches of one driver serve the others."""
import json, os
import numpy as np
import pytest

from conftest import strip
from qtx_ms import devstore, dist, mlbasis, mlevel, mlexact, mlmode, mlsolve
from qtx_ms.prep.inputs import write_inputs

BASE = ["--cap", "12,1,2", "--cap-spacing", "4", "--cap-split", "2", "--cap-tov", "1.5", "--l1", "dmc:1e-3",
        "--up", "y:1e-3,x:1e-3,x:1e-3", "--mu=-0.25,0.5", "--npts", "8", "--Ec", "0.35"]
DE = "-0.02,0.05"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("drv")
    inp = strip(nd=8, ny=4, nz=2, no=2)
    inp["meta"]["periodic"] = [dict(axis=1, length=8.0)]
    inp["meta"].setdefault("edges", {})["Ec_lead"] = 0.35
    write_inputs(d / "dev", inp["labels"], inp["xyz"], inp["norb"], inp["blocks"], inp["H"], inp["S"], inp["lead"],
                 inp["meta"])
    devstore.main(["--inputs", str(d / "dev"), "--out", str(d / "store")])
    ex = mlexact.solve(mlexact.parse(args(d, "--save-psi", str(d / "psi"), "--out", str(d / "ex.json"), basis=False)))
    ref = str(d / "ref.json")                                    # study driver: exact level n, mode-space T of every level
    mlevel.main(args(d) + ["--dm-cache", str(d / "dm_ref"), "--dm-keep", "1e-6", "--schur", "--schur-prec", "double", "--halo", "1",
                           "--halo-smooth", "full", "--cap-solver", "rows64", "--sweep", "exact", "--exact-max", "100000",
                           "--out", ref])
    return d, ex, json.load(open(ref))


def args(d, *more, basis=True, store=False):
    a = ["--inputs", str(d / "dev"), "--dE=" + DE, "--sigma-cache", str(d / "sigma")]
    return a + (BASE if basis else ["--Ec", "0.35"]) + (["--dev-store", str(d / "store")] if store else []) + list(more)


def test_exact_driver(env, tmp_path):
    d, ex, ref = env
    from qtx_ms import msbasis as mb
    dev = mb.Device(devstore.read_any(str(d / "dev")), obc="folded")
    for v in ex["E"].values():
        SL, SR = dev.sigma(v["E"] + 1e-6j)
        A = dev.A(v["E"], SL, SR).toarray()
        B = dev.inject(SL)
        b = np.zeros((dev.n, B.shape[1]), complex)
        b[dev.bL] = B
        p = np.linalg.solve(A, b)[dev.bR]
        assert abs(np.real(np.einsum("ic,ij,jc->", p.conj(), 1j * (SR - SR.conj().T), p)) - v["T_exact"]) < 1e-9
    st = mlexact.solve(mlexact.parse(args(d, "--out", str(tmp_path / "e.json"), basis=False, store=True)))   # sliced H, S
    for k in ex["E"]:
        assert abs(st["E"][k]["T_exact"] - ex["E"][k]["T_exact"]) < 1e-10
    n = len(os.listdir(d / "psi"))
    assert n == 2
    with pytest.raises(SystemExit):                              # another reference edge
        mlexact.solve(mlexact.parse(args(d, "--Ec", "0.3", "--out", str(d / "ex.json"), basis=False)))


def basis_then_use(d, tmp, comm=None, store=False, *more):
    dm = ["--dm-cache", str(tmp / "dm")]
    b = mlbasis.build(mlbasis.parse(args(d, *dm, *more, store=store)), comm)
    s = mlsolve.solve(mlsolve.parse(args(d, *dm, *more, store=store)), comm)
    m = mlmode.solve(mlmode.parse(args(d, *dm, *more, "--levels", "all", store=store)), comm)
    return b, s, m


def check(b, s, m, ex, ref):
    for k, v in ex["E"].items():
        r = ref["E"][k]
        x = s["E"][k]
        assert x["converged"] and abs(x["T"] - v["T_exact"]) < 1e-8 * v["T_exact"]
        deep = r["runs"][-1]
        assert np.abs(np.array(x["its"]) - np.array(deep["its"])).max() <= 1
        assert len(x["N"]) == 4
        assert x["N"] == [l["N"] for l in r["levels"]]
        ms = {w["level"]: w["T"] for w in r["mode_space"]}
        got = {w["level"]: w["T"] for w in m["E"][k]["levels"]}
        assert set(got) == {1, 2, 3, 4}
        for lv in got:                                           # same mode-space T as the study driver at every level
            assert abs(got[lv] - ms[lv]) < 1e-7 * max(1.0, abs(ms[lv])), (k, lv)
        assert b["N"] is None or b["N"] == x["N"]


def test_serial(env, tmp_path):
    d, ex, ref = env
    b, s, m = basis_then_use(d, tmp_path)
    check(b, s, m, ex, ref)
    assert b["stored"] == []
    again = mlbasis.build(mlbasis.parse(args(d, "--dm-cache", str(tmp_path / "dm"))))      # nothing to do
    assert again["stored"] == [2, 3, 4] and again["N"] is None


def test_serial_options(env, tmp_path):
    """Free levels and fine blocks; without any cache."""
    d, ex, ref = env
    for more in (["--free-levels", "2"], ["--fine-blocks", "1"], ["--fine-blocks", "4"]):
        s = mlsolve.solve(mlsolve.parse(args(d, *more)))
        check(dict(N=None), s, mlmode.solve(mlmode.parse(args(d, "--levels", "all",
                                                             *(more if more[0] == "--fine-blocks" else [])))), ex, ref)


@pytest.mark.parametrize("more", [(), ("--fine-blocks", "1"), ("--fine-blocks", "4")])
def test_thread_ranks_on_store(env, tmp_path, more):
    d, ex, ref = env
    outs = dist.ThreadComm.run(3, lambda c: basis_then_use(d, tmp_path, c, True, *more))
    for b, s, m in outs:
        check(b, s, m, ex, ref)


def test_json_merge_and_ec_guard(env, tmp_path):
    d, ex, ref = env
    f = str(tmp_path / "o.json")
    mlmode.solve(mlmode.parse(args(d, "--out", f)))
    o = json.load(open(f))
    assert len(o["E"]) == 2 and "T" in o["E"]["0.0500"]
    with pytest.raises(SystemExit):
        mlmode.solve(mlmode.parse(args(d, "--Ec", "0.3", "--out", f)))
