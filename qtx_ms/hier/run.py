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
    p.add_argument("--obc", default="folded", choices=["folded", "beyn", "beyn-real", "closed"], help="lead Sigma: folded lead at E + i eta, Beyn on the "
                   "periodized lead at every z, Beyn at the real energies only (contour folded: the same bases), or closed: no Sigma at "
                   "any z, right-hand sides = identity on the left contact face (columns of G_c = (E S - H)^-1; no T)")
    p.add_argument("--rhs-cols", type=int, default=0, help="closed: only this many, evenly spaced columns of the identity (0: all)")
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
    p.add_argument("--profile", action="store_true", help="time the parts of the solve (FGMRES matvec / prec / orth, the V-cycle parts, "
                   "a replay of the FGMRES least-squares solves); synchronises the GPU at every timer")
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
    obc = OBC(dev, comm, a.sigma_cache or None, log=log, model=a.obc, basis_dir=a.cache or None)
    reals = [Ec + e + 1j * a.eta for e in dEs]
    ups = [(v.split(":")[0], float(v.split(":")[1])) for v in a.up.split(",") if v and not a.schwarz]
    hs = [None if v == "all" else int(v) for v in a.halo.split(",") if v]
    hs += [None] * (len(ups) - len(hs))
    h = Hier(dev, inp, sw, comm, obc, [(ax, c, hl) for (ax, c), hl in zip(ups, hs)], a.cut, sl, a.cache or None, name, log)
    h.bottom_kind = a.bottom
    h.slow_par = tuple(int(v) for v in a.slow.split(",")) if a.slow else None
    rho = not (a.schwarz or h.have_bases())                            # density matrices to build: only then the contour Sigma
    obc.prepare(contours(sl)[0] if rho else [], reals)
    if not a.schwarz:
        h.setup()
    peak = lambda: comm.allgather(round(gpu.cp.get_default_memory_pool().total_bytes() / 2 ** 30, 1) if gpu.on else 0.0)
    log("setup " + ", ".join(f"{k} {v:.1f}" for k, v in h.t.items()) + f" s; total {time.time() - t00:.0f} s; GPU held per rank {peak()} GB")
    f, res = h.fine, dict(Ec=Ec, args=vars(a), N=[h.N[k] for k in sorted(h.N)], t_setup=dict(h.t), E={})
    part = f.part
    iL, iR = (np.where(part.gown[b] == comm.rank)[0] for b in (dev.bL, dev.bR))
    for e, z in zip(dEs, reals):
        t0 = time.time()
        h.t = {}
        Sig = [obc.sigma(z, 0), obc.sigma(z, 1)]
        closed, nL = a.obc == "closed", len(dev.bL)
        if closed:                                                     # unit vectors of the left contact face: entry J[c] of column c
            J = np.arange(nL) if not 0 < a.rhs_cols < nL else np.linspace(0, nL - 1, a.rhs_cols).round().astype(int)
            nb = len(J)
        else:
            B = obc.inject(z)
            nb = B.shape[1]
            if nb == 0:                                                # no incoming channel (below the lead band edge): T = 0
                log(f"Ec {e:+.3f}: T 0, no incoming channel")
                res["E"][f"{e:.4f}"] = dict(E=z.real, T=0.0, closed={}, its=[], slow=None, res=[], converged=True, hist=[], t={},
                                           t_energy=time.time() - t0)
                continue
        h.tick("load", t0)
        h.set_energy(z.real, Sig, a.schwarz)
        prof, clock = ({}, {}) if a.profile else (None, None)
        h.prof = h.fine.prof = prof
        t, ys, yl, its, rs, ch = time.time(), [], [], [], [], a.chunk or nb
        hist = []
        for k in range(0, nb, ch):
            hist.append([])

            def cb(i, est, k=k):
                hist[-1].append(float(est.max()))
                if a.fgmres_log and (i + 1) % a.fgmres_log == 0:
                    log(f"    columns {k}..: it {i + 1}: residual est max {est.max():.1e}, median {np.median(est):.1e}")
            if closed:
                Bk = np.zeros((nL, len(J[k:k + ch])), complex)
                Bk[J[k:k + ch], np.arange(Bk.shape[1])] = 1
            else:
                Bk = B[:, k:k + ch]
            bk = xp.zeros((part.nloc, Bk.shape[1]), complex)
            bk[xp.asarray(part.gloc[dev.bL[iL]])] = xp.asarray(Bk[iL])
            x, i, r = fgmres_block(_Synced(f) if a.profile else f, bk, h.slow(bk) if h.slow is not None else xp.zeros_like(bk), h.prec, a.tol, a.maxit, cb=cb, comm=comm,
                                   clock=clock)
            its.append(i), rs.append(r)
            for out, face, ii in [(ys, dev.bR, iR)] + ([(yl, dev.bL, iL)] if closed else []):
                y = xp.zeros((len(face), x.shape[1]), complex)
                y[xp.asarray(ii)] = x[xp.asarray(part.gloc[face[ii]])]
                out.append(y)
            del x
        h.tick("solve", t)
        pf = _profile(h, comm, clock, prof, its, log, e) if a.profile else None
        its, rs, T, info = np.concatenate(its), np.concatenate(rs), None, {}
        y = comm.sum(xp.concatenate(ys, axis=1))
        if closed:                                                     # A is real symmetric: so is G_c (the sampled block of it)
            G = comm.sum(xp.concatenate(yl, axis=1))[xp.asarray(J)]
            q = lambda M: float(xp.linalg.norm(M))
            info = dict(G_LL=q(G), G_RL=q(y), asym=q(G - G.T) / q(G), imag=q(G.imag) / q(G))
            if a.out and comm.rank == 0:
                np.savez(f"{a.out}.G_{e:+.4f}.npz", J=J, G_LL=gpu.asnumpy(G), G_RL=gpu.asnumpy(y))
            what = f"closed, |G_LL| {info['G_LL']:.6e}, |G_RL| {info['G_RL']:.6e}, G_LL - G_LL^T {info['asym']:.0e}, Im {info['imag']:.0e}"
        else:
            SR = xp.asarray(Sig[1])
            T = float(xp.real(xp.sum(y.conj() * (1j * ((SR - SR.conj().T) @ y)))))
            what = f"T {T:.10f}"
        ok = bool(rs.max() < 10 * a.tol)
        log(f"Ec {e:+.3f}: {what}, {nb} columns, its {its.min()}..{its.max()}, res {rs.max():.0e}" + ("" if ok else " ** NOT CONVERGED **")
            + f"; energy {time.time() - t0:.1f} s: " + ", ".join(f"{k} {v:.1f}" for k, v in h.t.items())
            + (f"; bottom {len(h.bottom_n)} slabs of {max(h.bottom_n)} at most, residual {h.bottom_res:.0e}" if h.bottom_res is not None else "")
            + f"; smoothers {h.S0.gb:.1f} + {sum(h.gb.values()):.1f} GB, GPU held per rank {peak()} GB")
        res["E"][f"{e:.4f}"] = dict(E=z.real, T=T, closed=info, its=its.tolist(), slow=h.slow.info if h.slow is not None else None, res=rs.tolist(), converged=ok, hist=hist, t=dict(h.t), t_energy=time.time() - t0, profile=pf)
        if a.out and comm.rank == 0:
            json.dump(res, open(a.out, "w"), indent=1)
    log(f"done, {time.time() - t00:.0f} s")
    res["h"] = h
    return res


class _Synced:
    """The fine operator with the GPU synchronised after every product (FGMRES matvec timer when profiling)."""
    def __init__(self, f):
        self.f = f

    def __matmul__(self, x):
        y = self.f @ x
        gpu.sync()
        return y


def _profile(h, comm, clock, prof, its, log, e):
    """Log the parts of the solve: max and mean over the ranks; the least-squares solves of FGMRES replayed with their sizes."""
    h.prof = h.fine.prof = None
    lst = time.time()
    rng = np.random.default_rng(0)
    for n in np.concatenate(its):                                      # one lstsq of (i + 2) x (i + 1) per column and iteration
        for i in range(int(n)):
            M = rng.standard_normal((i + 2, i + 1)) + 1j * rng.standard_normal((i + 2, i + 1))
            np.linalg.lstsq(M, np.eye(i + 2, 1, dtype=complex)[:, 0], rcond=None)
    lst = time.time() - lst
    mine = {f"FGMRES {k}": (v, 0) for k, v in clock.items()}
    mine.update(prof)
    mine["FGMRES least squares (replay, inside orth)"] = (lst, 0)
    allr = comm.allgather(mine)
    keys = list(mine)
    rows = []
    for k in keys:
        ts = [d.get(k, (0.0, 0))[0] for d in allr]
        rows.append((k, max(ts), float(np.mean(ts)), min(ts), mine[k][1]))
    log(f"  profile Ec {e:+.3f} [s] (max / mean / min over {comm.size} ranks, calls on rank 0):")
    for k, mx, mn, mi, n in rows:
        log(f"    {k:42s} {mx:8.2f} {mn:8.2f} {mi:8.2f}" + (f"  {n}" if n else ""))
    return {k: dict(max=mx, mean=mn, min=mi, calls=n) for k, mx, mn, mi, n in rows}


def main(argv=None):
    from ..mlevel import start
    a = parse(argv)
    run(a, start(a.dist, a.gpu))


if __name__ == "__main__":
    from ..mlevel import guarded
    guarded(main)
