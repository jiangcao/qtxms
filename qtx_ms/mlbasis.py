"""Basis driver (user, 2026-11-25): the window density matrices (level 1) and the block bases of every further level of the
density-matrix hierarchy into the caches (--dm-cache, --basis-cache), nothing else: no real-axis Sigma, fine operator,
smoothers, LU or solve. The bases are energy independent; qtx_ms.mlsolve, qtx_ms.mlmode and qtx_ms.mlevel with the same basis
options (the cache keys hold them) then skip every contour. An interrupted run is continued by the next one: finished
levels are read from the cache. With --dE the energies are ignored.

    python -m qtx_ms.mlbasis --inputs DIR --gpu --sigma-cache DIR --dm-cache DIR --cap-split 2,1 --up y:1e-5,x:1e-5 \\
        [--dist nccl --dev-store DIR --fine-blocks 1] [--out FILE.json]
"""
import gc, json, os, time
import numpy as np

from qtx_ms import dist, gpu, mldrive as md
from qtx_ms.mlevel import guarded, start

log = md.log


def parse(argv=None):
    p = md.new_parser(__doc__)
    md.device_args(p)
    md.hierarchy_args(p)
    a = md.check(p.parse_args(argv))
    assert a.dm_cache or a.basis_cache not in ("", "none"), "--dm-cache or --basis-cache: where the bases are stored"
    assert a.basis_cache != "none", "--basis-cache none: nothing would be stored"
    return a


def build(a, comm=None):
    """Returns {Ec, N: level sizes, t_rho, t_levels, stored: levels read from the cache}."""
    comm = dist.Comm() if comm is None else comm
    dd, t00 = comm.size > 1, time.time()
    log(f"backend: {gpu.describe()}")
    gpu.track_peak()
    inp, dev, name, Ec = md.read_device(a)
    n = len(a.up.split(",")) + 1
    sw, h, kind, levels, basis = md.build(a, comm, inp, dev, name, Ec, host_blocks=True, l1_free=n, keep_max=0,
                                          smoothers=False)
    t = time.time()
    U1 = basis()
    gc.collect()
    t_rho = time.time() - t
    log(f"  level 1 bases: {t_rho:.0f} s")
    res = dict(Ec=Ec, t_rho=t_rho, args=vars(a))
    t = time.time()
    if all(h.basis_stored(key, k + 2) for k, (*_, key) in enumerate(levels)):
        log("  every level is in the basis store")
        res.update(N=None, stored=list(range(2, n + 1)))
    else:
        zL, zR = np.zeros((len(dev.bL),) * 2, complex), np.zeros((len(dev.bR),) * 2, complex)   # A_k is never formed
        h.level1(U1, Ec, zL, zR, A0=object())
        if a.dev_store and dd and a.fine_blocks:
            md.drop_rows(h, dev)
        for q, (axis, cut, m1, m2, npts, halo, key) in enumerate(levels):
            h.merge(axis, cut, kind, m1, m2, npts, halo, key, basis_only=q == len(levels) - 1)
            h.ubank.clear()                                      # host copies: the bases are in the files
            gc.collect()
            gpu.free()
        res.update(N=[int(h.LV[k].N) for k in sorted(h.LV)],
                   stored=[k for k in range(2, n + 1) if h.LV[k].info.get("stored")])
        log("  N " + " / ".join(str(v) for v in res["N"]))
    res["t_levels"] = time.time() - t
    res["peak_gb_per_rank"] = md.peaks(h, comm)
    log(f"  levels 2..{n}: {res['t_levels']:.0f} s ({len(res['stored'])} of {n - 1} from the store); in {h.basis_dir}")
    if a.out and comm.rank == 0:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    h.free()
    log(f"done, {time.time() - t00:.0f} s")
    return res


def main(argv=None):
    a = parse(argv)
    build(a, start(a.dist, a.gpu))


if __name__ == "__main__":
    guarded(main)
