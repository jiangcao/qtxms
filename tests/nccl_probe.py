"""NCCL over the ranks of a launcher (no solver): init, sum, ring send / recv as used by qtx_ms.dist, with bandwidths.
    srun -n 8 ... python tests/nccl_probe.py       prints PASS on rank 0"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qtx_ms import dist, gpu
comm = dist.MPIComm(nccl=True)
assert comm.size > 1, "one rank only"
gpu.enable(device=0)
import cupy as cp
say = lambda *x: print(*x, flush=True) if comm.rank == 0 else None
t = time.time()
comm.start_nccl()
say(f"init {time.time() - t:.1f} s")
x = cp.full(1000, comm.rank + 1.0)
assert float(comm.sum(x)[0]) == comm.size * (comm.size + 1) / 2
say("sum ok")
up, dn = (comm.rank + 1) % comm.size, (comm.rank - 1) % comm.size
half = comm.size // 2
for name, a, b in (("ring", up, dn), ("across", (comm.rank + half) % comm.size, (comm.rank + half) % comm.size)):
    for n in (1, 1 << 16, 1 << 20, 1 << 23):                          # complex entries: 16 B .. 128 MB
        s, r = cp.full(n, comm.rank + 0j), cp.empty(n, complex)
        for rep in range(3):
            cp.cuda.get_current_stream().synchronize(); comm.barrier(); t = time.time()
            comm.exchange({a: s}, {b: r}) if a != b else comm.exchange({a: s}, {a: r})
            cp.cuda.get_current_stream().synchronize(); dt = time.time() - t
        assert bool((r == b).all()), (comm.rank, name, n)
        say(f"{name:7s} {16 * n / 1e6:8.2f} MB  {dt * 1e3:8.2f} ms  {16 * n / dt / 1e9:6.1f} GB/s per rank and direction")
comm.barrier()
say("PASS")
