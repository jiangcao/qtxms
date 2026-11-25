"""Mode-space driver (user, 2026-11-25): T of the Galerkin problem in the bases of the density-matrix hierarchy, psi = P A_m^-1
P^H b, without correction (no fine operator, no smoothers, no FGMRES). Levels: --levels last (default) | all | 2,5,...

The bases are energy independent: the levels are built once (S_k, H_k kept only for the wanted levels), every further energy
redoes the lead Sigma, its projection, and the block-tridiagonal LU of A_m(E) = E S_m - H_m - Sigma_m(E).
T is second-order accurate, psi 1e-3 .. 1e-2; the exact values come from qtx_ms.mlsolve or qtx_ms.mlexact.

    python -m qtx_ms.mlmode --inputs DIR --dE 0.05,0.10,0.15 --gpu --sigma-cache DIR --dm-cache DIR --cap-split 2,1 \\
        --up y:1e-5,x:1e-5 [--levels all] [--dist nccl --dev-store DIR --fine-blocks 1] --out FILE.json
"""
import gc, time

from qtx_ms import dist, gpu, mldrive as md
from qtx_ms.mlevel import FineMat, guarded, start, _xp

log = md.log


def parse(argv=None):
    p = md.new_parser(__doc__)
    md.device_args(p)
    md.hierarchy_args(p)
    p.add_argument("--ncol", type=int, default=0, help="the k strongest injection columns only (0: all; T of these)")
    p.add_argument("--levels", default="last", help="levels solved: last, all (1 .. n) or a comma list")
    a = md.check(p.parse_args(argv))
    n = 1 + len(a.up.split(","))
    a.want = [n] if a.levels == "last" else list(range(1, n + 1)) if a.levels == "all" else sorted(
        {int(v) for v in a.levels.split(",")})
    assert all(1 <= m <= n for m in a.want), f"--levels: 1 .. {n}"
    return a


def solve(a, comm=None):
    """Run the energies of `a` (parse()) on the communicator. Returns {Ec, E: {dE: record}}."""
    comm = dist.Comm() if comm is None else comm
    dd, xp, t00 = comm.size > 1, _xp(), time.time()
    log(f"backend: {gpu.describe()}")
    gpu.track_peak()
    inp, dev, name, Ec = md.read_device(a)
    md.check_out(a, Ec)
    top = max(a.want)
    # every level free: no stored A_k; the operator below level 1 is never applied (placeholder)
    sw, h, kind, levels, basis = md.build(a, comm, inp, dev, name, Ec, host_blocks=True, l1_free=top,
                                          keep_max=0, smoothers=False, keep_levels=a.want)
    part, K = h.part0, a.fine_blocks
    Tof = md.Trans(dev, part, comm)
    res, t_levels = dict(Ec=Ec, E={}), 0.0
    for ie, dE in enumerate(md.energies(a)):
        te, E, tt = time.time(), Ec + dE, {}
        t = time.time()
        SL, SR, b, nch = md.lead(dev, E, a.eta, comm, a.ncol)
        Tof.set(SR)
        bg, nsel = xp.asarray(part.local(b)), b.shape[1]
        del b
        tt["sigma"] = time.time() - t
        t = time.time()
        if ie == 0:
            h.level1(basis(), E, SL, SR, A0=object())
            if a.dev_store and dd and K:
                md.drop_rows(h, dev)
            for axis, cut, m1, m2, npts, halo, key in levels[:top - 1]:
                h.merge(axis, cut, kind, m1, m2, npts, halo, key)
            if a.dev_store and dd and not K:
                md.drop_rows(h, dev)
            if h.fb is not None and not any(isinstance(h.LV[m].Sb, FineMat) for m in a.want):
                h.fb.free()                                      # the wanted levels have stored S, H
                h.mem("sparse core-pair blocks freed")
            h.ubank.clear()                                      # host copies of the bases
            gc.collect()
            gpu.free()
            t_levels = time.time() - t
            log(f"  levels 1..{top}: {t_levels:.0f} s; N " + " / ".join(str(h.LV[k].N) for k in sorted(h.LV)))
        else:
            h.shift(E, SL, SR)
            tt["shift"] = time.time() - t
        rec = dict(E=E, channels=int(nch), ncol=int(nsel), levels=[], t=tt, t_levels=t_levels,
                   N=[int(h.LV[k].N) for k in sorted(h.LV)], args=vars(a))
        for m in a.want:
            t = time.time()
            bot = h.exact(m)
            t_lu = time.time() - t
            c = bg
            for k in range(1, m + 1):
                c = h.LV[k].PH(c)
            c = bot(c)
            for k in range(m, 0, -1):
                c = h.LV[k].P(c)
            T = Tof(c)
            h.mem(f"mode space, level {m} (N {h.LV[m].N})")
            log(f"Ec {dE:+.3f}: mode space level {m} (N {h.LV[m].N}): T {T:.10f}; LU {t_lu:.1f} s, {time.time() - t:.1f} s")
            rec["levels"].append(dict(level=m, N=int(h.LV[m].N), T=T, t_lu=t_lu, t=time.time() - t))
            del bot, c
            gpu.free()
        rec["T"] = rec["levels"][-1]["T"]                        # deepest wanted level
        tt["energy"] = time.time() - te
        rec["peak_gb_per_rank"] = md.peaks(h, comm)
        res["E"][f"{dE:.4f}"] = rec
        md.write(a, comm, Ec, dE, rec)
        del bg
    h.free()
    log(f"done, {time.time() - t00:.0f} s")
    return res


def main(argv=None):
    a = parse(argv)
    solve(a, start(a.dist, a.gpu))


if __name__ == "__main__":
    guarded(main)
