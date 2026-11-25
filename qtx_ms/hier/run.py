"""T(E) by FGMRES with the hierarchy:
    python -m qtx_ms.hier.run --inputs DIR --dE 0.15,0.30 --cap-split 4,1 --up y:1e-5,x:1e-5,y:1e-5,x:1e-5,x:1e-5,x:1e-5 \\
        --mu=-0.15,0.8 --sigma-cache DIR --cache DIR --gpu [--dev-store DIR --dist nccl] --out FILE.json"""
import argparse, json, os, time
import numpy as np

from .. import devstore, dist, gpu
from .. import msbasis as mb
from .fine import _xp
from .levels import Hier
from .obc import OBC, contours, slices


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--dev-store", default="", help="sliced device (python -m qtx_ms.devstore)")
    p.add_argument("--dE", required=True, help="energies relative to Ec (comma list)")
    p.add_argument("--Ec", type=float, default=None)
    p.add_argument("--eta", type=float, default=1e-6)
    p.add_argument("--cap", default="16,2,2", help="fine windows: width [A], CAP [eV], ramp [A]")
    p.add_argument("--cap-spacing", type=float, default=8.0)
    p.add_argument("--cap-split", default="", help="strips per periodic transverse axis, e.g. 4,1")
    p.add_argument("--cap-tov", type=float, default=6.0)
    p.add_argument("--cut", type=float, default=1e-5, help="cut of the level 1 rho eigenvalues")
    p.add_argument("--up", default="", help="merges AXIS:CUT,... (not with --schwarz)")
    p.add_argument("--bottom", default="bt", choices=["bt", "dense"], help="exact solve of the deepest level: block-tridiagonal LU over the x "
                   "positions (default), or dense LU")
    p.add_argument("--schwarz", action="store_true", help="no coarse level: the preconditioner is one application of S_0")
    p.add_argument("--slow", default="", help="M,Q,ROUNDS: two-level, no bases: the coarse level = slow modes of the fine smoother S_0 (M vectors, "
                   "Q subspace steps; more rounds: slow modes of the cycle), dense exact solve, built at every energy (hier.slow)")
    p.add_argument("--halo", default="1", help="x halo of the level windows per merge (comma list, the rest: all coupled blocks)")
    p.add_argument("--mu", default="-0.15,0.8", help="contour window relative to Ec, or several: M1,M2[,NPTS[,WEIGHT]]:... "
                   "(rho = weighted sum)")
    p.add_argument("--npts", type=int, default=24)
    p.add_argument("--sigma-cache", default="")
    p.add_argument("--cache", default="", help="folder of the bases")
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--maxit", type=int, default=100)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--fgmres-log", type=int, default=0, help="log the residual estimate every k iterations")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--dist", default="", choices=["", "mpi", "nccl", "auto"])
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    if a.slow:
        if a.up or a.schwarz:
            p.error("--slow is a coarse level of its own: not with --up or --schwarz")
        a.schwarz = True
    if not (a.schwarz or a.up):
        p.error("--up, --schwarz or --slow")
    return a


def run(a, comm):
    from ..schwarz import CapSchwarz, fgmres_block
    log, t00, xp = mb.log, time.time(), _xp()
    if a.dev_store:
        inp, dev = devstore.load(a.dev_store, source=a.inputs)
    else:
        inp = devstore.read_any(a.inputs)
        dev = mb.Device(inp, obc="folded")
    name = os.path.basename(os.path.normpath(a.inputs))
    Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
    dEs, sl = [float(v) for v in a.dE.split(",")], slices(a.mu, a.npts, Ec)
    if not a.schwarz:
        log("contours: " + ", ".join(f"Ec {m1 - Ec:+.2f}..{m2 - Ec:+.2f} ({n} points, weight {w:g})" for m1, m2, n, w in sl))
    log(f"{name}: {dev.n} orbitals, Ec {Ec:.4f} eV, {comm.size} ranks, {gpu.describe()}")
    sw = CapSchwarz(dev, inp, *(float(v) for v in a.cap.split(",")), spacing=a.cap_spacing, tov=a.cap_tov,
                    split=[int(v) for v in a.cap_split.split(",")] if a.cap_split else None)
    obc = OBC(dev, comm, a.sigma_cache or None, log=log)
    reals = [Ec + e + 1j * a.eta for e in dEs]
    obc.prepare([] if a.schwarz else contours(sl)[0], reals)
    ups = [(v.split(":")[0], float(v.split(":")[1])) for v in a.up.split(",") if v and not a.schwarz]
    hs = [None if v == "all" else int(v) for v in a.halo.split(",") if v]
    hs += [None] * (len(ups) - len(hs))
    h = Hier(dev, inp, sw, comm, obc, [(ax, c, hl) for (ax, c), hl in zip(ups, hs)], a.cut, sl, a.cache or None, name, log)
    h.bottom_kind = a.bottom
    h.slow_par = tuple(int(v) for v in a.slow.split(",")) if a.slow else None
    if not a.schwarz:
        h.setup()
    peak = lambda: comm.allgather(round(gpu.cp.get_default_memory_pool().total_bytes() / 2 ** 30, 1) if gpu.on else 0.0)
    log("setup " + ", ".join(f"{k} {v:.1f}" for k, v in h.t.items()) + f" s; total {time.time() - t00:.0f} s; GPU held per rank {peak()} GB")
    f, res = h.fine, dict(Ec=Ec, args=vars(a), N=[h.N[k] for k in sorted(h.N)], t_setup=dict(h.t), E={})
    part, iR = f.part, np.where(f.part.gown[dev.bR] == comm.rank)[0]
    for e, z in zip(dEs, reals):
        t0 = time.time()
        h.t = {}
        Sig = [obc.sigma(z, 0), obc.sigma(z, 1)]
        B = obc.inject(z)
        b = np.zeros((dev.n, B.shape[1]), complex)
        b[dev.bL] = B
        bl = xp.asarray(part.local(b))
        h.tick("load", t0)
        h.set_energy(z.real, Sig, a.schwarz)
        t, xs, its, rs, ch = time.time(), [], [], [], a.chunk or b.shape[1]
        hist = []
        for k in range(0, b.shape[1], ch):
            hist.append([])

            def cb(i, est, k=k):
                hist[-1].append(float(est.max()))
                if a.fgmres_log and (i + 1) % a.fgmres_log == 0:
                    log(f"    columns {k}..: it {i + 1}: residual est max {est.max():.1e}, median {np.median(est):.1e}")
            bk = bl[:, k:k + ch]
            x, i, r = fgmres_block(f, bk, h.slow(bk) if h.slow is not None else xp.zeros_like(bk), h.prec, a.tol, a.maxit, cb=cb, comm=comm)
            xs.append(x), its.append(i), rs.append(r)
        h.tick("solve", t)
        x, its, rs = xp.concatenate(xs, axis=1), np.concatenate(its), np.concatenate(rs)
        y = xp.zeros((len(dev.bR), x.shape[1]), complex)
        y[xp.asarray(iR)] = x[xp.asarray(part.gloc[dev.bR[iR]])]
        y, SR = comm.sum(y), xp.asarray(Sig[1])
        T = float(xp.real(xp.sum(y.conj() * (1j * ((SR - SR.conj().T) @ y)))))
        ok = bool(rs.max() < 10 * a.tol)
        log(f"Ec {e:+.3f}: T {T:.10f}, {b.shape[1]} columns, its {its.min()}..{its.max()}, res {rs.max():.0e}" + ("" if ok else " ** NOT CONVERGED **")
            + f"; energy {time.time() - t0:.1f} s: " + ", ".join(f"{k} {v:.1f}" for k, v in h.t.items())
            + (f"; bottom {len(h.bottom_n)} slabs of {max(h.bottom_n)} at most, residual {h.bottom_res:.0e}" if h.bottom_res is not None else "")
            + f"; smoothers {h.S0.gb:.1f} + {sum(h.gb.values()):.1f} GB, GPU held per rank {peak()} GB")
        res["E"][f"{e:.4f}"] = dict(E=z.real, T=T, its=its.tolist(), slow=h.slow.info if h.slow is not None else None, res=rs.tolist(), converged=ok, hist=hist, t=dict(h.t), t_energy=time.time() - t0)
        if a.out and comm.rank == 0:
            json.dump(res, open(a.out, "w"), indent=1)
    log(f"done, {time.time() - t00:.0f} s")
    res["h"] = h
    return res


def main(argv=None):
    from ..mlevel import start
    a = parse(argv)
    run(a, start(a.dist, a.gpu))


if __name__ == "__main__":
    from ..mlevel import guarded
    guarded(main)
