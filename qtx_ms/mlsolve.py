"""Production driver (user, 2026-11-25): per energy one FGMRES solve of A psi = b with the multilevel cycle of the
density-matrix hierarchy and the exact solve of the deepest level, nothing else. (The study driver qtx_ms.mlevel solves with
every level as the bottom, prints the mode-space T of each and compares with an exact psi.)

Freed as soon as the solve does not need it:
  S_k, H_k of every level once the level above it and the smoother of A_k are built (Hierarchy keep_max = 0); those of the
    deepest level and its stored A_n after the LU;
  sparse fine core-pair blocks (--fine-blocks K): after the smoother of A_K, last energy (the next one needs them again);
  device-store rows of H, S: after their last use, last energy; window density matrices: after the level-1 bases.
The solve holds U, the smoothers of A_0 .. A_{n-1}, the fine A, the stored A_k of the levels above --free-levels, the LU.

    python -m qtx_ms.mlsolve --inputs DIR --dE 0.15,0.50 --gpu --sigma-cache DIR --dm-cache DIR --cap-split 2,1 \\
        --up y:1e-5,x:1e-5 [--dist nccl --dev-store DIR --fine-blocks 1 --stream-smoother-gb 0] --out FILE.json
"""
import gc, time
import numpy as np

from qtx_ms import dist, gpu, mldrive as md
from qtx_ms.mlevel import Timed, guarded, start, _xp

log = md.log
quiet = lambda *x: None


def parse(argv=None):
    p = md.new_parser(__doc__)
    md.device_args(p)
    md.hierarchy_args(p)
    g = p.add_argument_group("solve")
    g.add_argument("--ncol", type=int, default=0, help="the k strongest injection columns only (0: all; T of these)")
    g.add_argument("--cap-solver", default="rows64", choices=["dense64", "dense", "rows64", "rows"])
    g.add_argument("--l1-lu", default="c64", choices=["c64", "c128"], help="precision of the level smoothers")
    g.add_argument("--l1-solver", default="rows", choices=["rows", "lu"])
    g.add_argument("--halo-smooth", default="full", choices=["same", "full"])
    g.add_argument("--gamma", type=int, default=1, help="coarse corrections per level: 1 V-cycle, 2 W-cycle")
    g.add_argument("--tol", type=float, default=1e-10)
    g.add_argument("--maxit", type=int, default=100)
    g.add_argument("--chunk", type=int, default=0, help="columns per FGMRES call (0: all)")
    g.add_argument("--fgmres-log", type=int, default=0, help="log the residual estimate every k iterations")
    g.add_argument("--free-levels", type=int, default=0, help="levels 1 .. K without stored A_k (implies --host-blocks)")
    g.add_argument("--keep-sh", default="", help="levels (comma list) whose energy independent S_k, H_k blocks are kept over "
                   "the energies (host memory with --host-blocks / --fine-blocks), e.g. the first level with stored blocks")
    g.add_argument("--no-keep-fine", action="store_true", help="several ranks: do not keep the halo, pattern and values of the fine "
                   "S, H over the energies (real S, H; pinned host memory, 20 bytes per nonzero of the own rows)")
    g.add_argument("--stream-gb", type=float, default=None, help="operators larger than this [GB] streamed from the host")
    g.add_argument("--stream-smoother-gb", type=float, default=None, help="the same for the smoothers alone (0: always)")
    a = md.check(p.parse_args(argv))
    a.free_levels = max(a.free_levels, a.fine_blocks)
    a.host_blocks = a.host_blocks or a.free_levels > 0
    return a


def solve(a, comm=None):
    """Run the energies of `a` (parse()) on the communicator. Returns {Ec, E: {dE: record}}."""
    from qtx_ms.schwarz import fgmres_block, operator
    comm = dist.Comm() if comm is None else comm
    dd, xp, t00 = comm.size > 1, _xp(), time.time()
    log(f"backend: {gpu.describe()}")
    gpu.track_peak()
    inp, dev, name, Ec = md.read_device(a)
    md.check_out(a, Ec)
    sw, h, kind, levels, basis = md.build(
        a, comm, inp, dev, name, Ec, solver=a.cap_solver,
        sw_stream=a.stream_gb if a.stream_smoother_gb is None else a.stream_smoother_gb, l1_lu=a.l1_lu,
        l1_solver=a.l1_solver, halo_smooth=a.halo_smooth, host_blocks=a.host_blocks, stream_gb=a.stream_gb,
        l1_free=a.free_levels, keep_max=0, stream_smoother_gb=a.stream_smoother_gb, coupling_norms=False,
        keep_sh=[int(v) for v in a.keep_sh.split(",")] if a.keep_sh else ())
    part, K = h.part0, a.fine_blocks
    Tof = md.Trans(dev, part, comm)
    dEs, U1, res = md.energies(a), None, dict(Ec=Ec, E={})
    fine = None if a.no_keep_fine or len(dEs) == 1 else {}       # fine S, H data of dist.FineOp over the energies
    t_setup = time.time() - t00
    log(f"  setup {t_setup:.0f} s")
    for ie, dE in enumerate(dEs):
        last, te, E, tt = ie == len(dEs) - 1, time.time(), Ec + dE, {}

        def tick(k, t0):
            tt[k] = time.time() - t0
        t = time.time()
        SL, SR, b, nch = md.lead(dev, E, a.eta, comm, a.ncol)
        Tof.set(SR)
        bg, nsel = xp.asarray(part.local(b)), b.shape[1]
        del b
        tick("sigma", t)
        t = time.time()
        sg = None if a.free_levels else a.stream_gb              # the fine A carries every free-level product: resident
        if dd:
            A = dist.FineA(dev, E, SL, SR)
            Aop = dist.FineOp(A, part, sg, a.threads, fine)
            if last and fine:
                fine.clear()
        else:
            A = dev.A(E, SL, SR).tocsr()
            Aop = operator(A, stream_gb=sg) if gpu.on else operator(A, a.threads)
        sw.set_energy(E, A, log=quiet)
        tick("S0", t)
        log(f"Ec {dE:+.3f}: {nch} columns, {nsel} solved; window factor {tt['S0']:.0f} s; {h.memstr()}")
        h.mem("fine level: operator, right-hand sides, window factors of S0")
        S0, A0 = Timed(sw, "S0"), Timed(Aop, "A")
        # hierarchy
        t = time.time()
        if U1 is None:
            U1 = basis()                                         # energy independent: built once
            gc.collect()
        tick("window rho", t)
        t = time.time()
        store = a.dev_store and dd and last
        h.level1(U1, E, SL, SR, A0=Aop)
        if store and K:                                          # their sparse blocks are on the device
            md.drop_rows(h, dev)
        for axis, cut, m1, m2, npts, halo, key in levels:
            h.merge(axis, cut, kind, m1, m2, npts, halo, key)
            if h.fb is not None and last and len(h.LV) == K + 1:     # A_K smoother built: no S_k, H_k is formed any more
                h.fb.free()
                gpu.free()
                h.mem(f"sparse core-pair blocks freed (levels 1..{K} complete)")
        if store and not K:
            md.drop_rows(h, dev)
        if last:
            h.ubank.clear()                                      # host copies of the bases (kept for the next energy)
        tick("levels", t)
        nl = len(h.LV)
        Ln = h.LV[nl]
        # exact solve of the deepest level
        t = time.time()
        bot = h.exact(nl)
        if h.fb is not None and last:                            # a deepest level formed from the sparse blocks
            h.fb.free()
        Ln.Sb = Ln.Hb = Ln.Ab = None                             # the cycle applies A_1 .. A_{n-1} only
        if isinstance(getattr(Ln.A, "R", None), dict):
            Ln.A.R, Ln.A.st = {}, None
        gc.collect()
        gpu.free()
        tick("LU", t)
        h.mem(f"slab LU of A{nl} (N {Ln.N}): {len(bot.sizes)} slabs of {min(bot.sizes)}..{max(bot.sizes)}, {bot.gb:.2f} GB")
        log(f"  Ec {dE:+.3f}: levels {tt['levels']:.0f} s (" + ", ".join(
            f"level {k} {'stored' if h.LV[k].info.get('stored') else 'rho'}" for k in range(2, nl + 1))
            + f"), LU of level {nl} {tt['LU']:.0f} s")
        h.inventory()
        # FGMRES
        prec, tm = h.cycle(nl, bot, A0, S0, a.gamma)
        t, ch, xs, its, rs = time.time(), a.chunk if a.chunk > 0 else nsel, [], [], []
        for k in range(0, nsel, ch):
            def cb(i, e, k=k, t=t):
                if a.fgmres_log and ((i + 1) % a.fgmres_log == 0 or i == 0):
                    log(f"        chunk {k // ch + 1}/{-(-nsel // ch)} it {i + 1}: residual est max {e.max():.2e}, "
                        f"{time.time() - t:.0f} s")
            bk = bg[:, k:k + ch]
            xk, ik, rk = fgmres_block(Aop, bk, xp.zeros_like(bk), prec, a.tol, a.maxit, cb=cb, comm=comm)
            xs.append(xk), its.append(np.asarray(ik)), rs.append(np.asarray(rk))
            gpu.free()
        x, its, rs = xp.concatenate(xs, axis=1), np.concatenate(its), np.concatenate(rs)
        tick("solve", t)
        T, ok = Tof(x), bool(rs.max() < 10 * a.tol)
        h.mem(f"FGMRES, exact level {nl} (N {Ln.N})")
        tick("energy", te)
        log(f"Ec {dE:+.3f}: T {T:.10f}, its {its.min()}..{its.max()}, res {rs.max():.0e}"
            + ("" if ok else "  ** NOT CONVERGED **") + f"; solve {tt['solve']:.0f} s ("
            + ", ".join(f"{c.name} {c.t:.0f} s" for c in (S0, A0, tm)) + f"), energy {tt['energy']:.0f} s")
        rec = dict(E=E, channels=int(nch), ncol=int(nsel), T=T, its=its.tolist(), res=rs.tolist(), converged=ok,
                   N=[int(h.LV[k].N) for k in range(1, nl + 1)], t=tt, t_setup=t_setup,
                   work={c.name: float(c.t) for c in (S0, A0, tm)}, peak_gb_per_rank=md.peaks(h, comm), mem=h.memlog,
                   args=vars(a))
        res["E"][f"{dE:.4f}"] = rec
        md.write(a, comm, Ec, dE, rec)
        del x, xs, xk, bk, bg, bot, prec, tm, S0, A0, Ln
        h.free()
        sw.free()
        del Aop, A
        gc.collect()
        gpu.free()
    log(f"done, {time.time() - t00:.0f} s")
    return res


def main(argv=None):
    a = parse(argv)
    solve(a, start(a.dist, a.gpu))


if __name__ == "__main__":
    guarded(main)
