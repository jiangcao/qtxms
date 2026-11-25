"""Distributed hierarchy under real MPI (and NCCL) against the serial one, on the synthetic strip of test_dist.py:
    mpiexec -n 3 python tests/dist_mpi.py [cpu|mpi|nccl]       cpu: numpy; mpi / nccl: cupy with that halo transport
Sigma cache: $QTX_TEST_SIGMA (default /tmp/qtx_dist_sigma_<user>). Prints PASS on rank 0, exit code 1 otherwise."""
import os, sys, tempfile, getpass
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from qtx_ms import dist, gpu

kind = sys.argv[1] if len(sys.argv) > 1 else "cpu"
comm = dist.MPIComm(nccl=kind == "nccl")
assert comm.size > 1, "one rank only: the MPI library under mpi4py does not belong to this launcher"
if kind != "cpu":
    import cupy
    gpu.enable(device=comm.local_rank % cupy.cuda.runtime.getDeviceCount())
    comm.start_nccl()
import test_dist as td


def host(o):
    return {k: ([gpu.asnumpy(v) for v in x] if isinstance(x, list) else gpu.asnumpy(x) if hasattr(x, "shape") else x)
            for k, x in o.items()}


def run(c, **kw):                                                 # solve() works on device vectors in GPU mode
    return td.solve(c, d, **kw)


d = os.environ.get("QTX_TEST_SIGMA", os.path.join(tempfile.gettempdir(), "qtx_dist_sigma_" + getpass.getuser()))
ok = True
for case, kw in td.CASES.items():
    ref = host(run(dist.Comm(), **kw)) if comm.rank == 0 else None
    comm.barrier()
    ref = comm.c.bcast(ref)
    o = host(run(comm, **kw))
    e = dict(N=o["N"] == ref["N"], Ar=abs(o["Ar"] - o["Adirect"]).max(), S0=abs(o["S0"] - ref["S0"]).max(),
             A=max(abs(np.sort_complex(np.linalg.eigvals(a)) - np.sort_complex(np.linalg.eigvals(b))).max()   # gauge free
                   for a, b in zip(o["A"], ref["A"])) if o["N"] == ref["N"] else np.inf,
             Aent=max(abs(a - b).max() for a, b in zip(o["A"], ref["A"])) if o["N"] == ref["N"] else np.inf,
             cyc=max(np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(o["cyc"], ref["cyc"])),
             x=max(np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(o["x"], ref["x"])),
             its=max(int(abs(a - b).max()) for a, b in zip(o["its"], ref["its"])))
    good = e["N"] and e["Ar"] < 1e-11 and e["S0"] < 1e-9 and e["A"] < 1e-8 and e["x"] < 1e-8 and e["cyc"] < 1e-8 and e["its"] <= 1
    good = all(comm.allgather(bool(good)))
    ok &= good
    if comm.rank == 0:
        print(f"{case:10s} {comm.size} ranks {kind}: {'ok' if good else 'FAIL'} {e}", flush=True)
if comm.rank == 0:
    print("PASS" if ok else "FAILED", flush=True)
sys.exit(0 if ok else 1)
