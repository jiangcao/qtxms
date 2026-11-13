"""One-level solve: flexible GMRES on the QTBM system A psi = b, right-preconditioned by the cap-Schwarz smoother
alone (schwarz.CapSchwarz: restricted additive Schwarz, windows along x with absorbing ramps on the artificial cuts,
lead faces keep Sigma). No mode-space coarse level, x0 = 0. With --coarse slow: two-level with the coarse level from
the slow error modes of the smoother / cycle, built at every energy (schwarz.SlowCoarse; --slow-m vectors per round,
--slow-q subspace steps, --slow-rounds), x0 = P_c b.

Per energy: lead Sigma (cached), injection b, window factorization, block FGMRES (schwarz.fgmres_block, all channels
at once, residual estimate logged every iteration); T from the iterate. With --exact also a sparse direct solve
(msbasis.CSolver) for the error in psi and T; --psi-dir saves / reuses that exact psi.

    python -m qtx_ms.cap_fgmres --inputs inputs/mos2_hfo2_9072a --dE 0.05,0.2 --workers 4 --exact
"""
import argparse, json, os, time
import numpy as np

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb
from qtx_ms.schwarz import CapSchwarz, SlowCoarse, fgmres_block, fgmres_restart

log = mb.log


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--dE", default="0.05,0.2", help="energies - Ec_lead [eV] (comma list)")
    p.add_argument("--Ec", type=float, help="reference edge for --dE (default Ec_lead of the inputs)")
    p.add_argument("--eta", type=float, default=1e-6, help="Im z of the lead Sigma [eV]")
    p.add_argument("--obc", default="folded", choices=["auto", "qttools", "folded"])
    p.add_argument("--sigma-cache", help="default results_ms/sigma/<inputs name>")
    p.add_argument("--cap", default="16,2,3", help="window width [Å], absorbing strength [eV], ramp [Å]")
    p.add_argument("--cap-spacing", type=float, help="largest distance between window centres [Å] (default: slab middles)")
    p.add_argument("--cap-centres", default="all", choices=["all", "iface", "ends"])
    p.add_argument("--cap-solver", default="dense64", choices=["dense", "dense64", "pardiso"])
    p.add_argument("--workers", type=int, default=4, help="windows factored / solved concurrently")
    p.add_argument("--maxit", type=int, default=300, help="FGMRES iterations in total")
    p.add_argument("--restart", type=int, default=0, help="> 0: restarted FGMRES(m), columns converged at a restart are "
                   "frozen (0: one unrestarted cycle of --maxit)")
    p.add_argument("--tol", type=float, default=1e-8, help="relative residual per column")
    p.add_argument("--coarse", default="none", choices=["none", "slow"], help="none: one-level, x0 = 0; slow: two-level "
                   "with the slow-mode coarse level built at each energy (schwarz.SlowCoarse), x0 = P_c b")
    p.add_argument("--slow-m", type=int, default=64, help="slow: vectors per round")
    p.add_argument("--slow-q", type=int, default=3, help="slow: subspace-iteration steps per round")
    p.add_argument("--slow-rounds", type=int, default=3, help="slow: rounds (coarse size m x rounds)")
    p.add_argument("--pre", type=int, default=1, help="slow: pre-smoothing steps")
    p.add_argument("--post", type=int, default=1, help="slow: post-smoothing steps")
    p.add_argument("--exact", action="store_true", help="also the sparse direct solution: error in psi and T")
    p.add_argument("--psi-dir", help="save / reuse the exact psi here (psi_<E>.npz)")
    p.add_argument("--psi-wait", type=float, default=0, help="with --exact: wait up to this many hours for psi_<E>.npz "
                   "(written by a concurrent exact_T --save-psi) before solving directly")
    p.add_argument("--out", help="json (default results_ms/cap_fgmres/<inputs name>.json, merged across runs)")
    a = p.parse_args()

    name = os.path.basename(os.path.normpath(a.inputs))
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc=a.obc)
    dev.sigma_cache = a.sigma_cache or os.path.join("results_ms", "sigma", name)
    Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
    fjs = a.out or os.path.join("results_ms", "cap_fgmres", name + ".json")
    os.makedirs(os.path.dirname(fjs) or ".", exist_ok=True)
    w, s, d = (float(v) for v in a.cap.split(","))
    log(f"{name}: device {dev.n} orbitals, slabs {dev.ns}; cap-Schwarz {a.cap} {a.cap_centres} {a.cap_solver}, "
        f"{a.workers} workers, " + ("no coarse level, x0 = 0" if a.coarse == "none" else
        f"slow-mode coarse level (m {a.slow_m}, q {a.slow_q}, {a.slow_rounds} rounds), x0 = P_c b") + f"; MKL threads {os.environ.get('MKL_NUM_THREADS')} -> {fjs}")
    S = CapSchwarz(dev, inp, w, s, d, spacing=a.cap_spacing, solver=a.cap_solver, centres=a.cap_centres,
                   workers=a.workers)
    for dE in (float(x) for x in a.dE.split(",")):
        E = Ec + dE
        t = time.time()
        SL, SR = dev.sigma(E + 1j * a.eta)
        tsig = time.time() - t
        B = dev.inject(SL)
        b = np.zeros((dev.n, B.shape[1]), complex)
        b[dev.bL] = B
        A = dev.A(E, SL, SR)
        GR = 1j * (SR - SR.conj().T)
        T = lambda y: float(np.real(np.einsum("ic,ij,jc->", y[dev.bR].conj(), GR, y[dev.bR])))
        log(f"E = Ec {dE:+.3f}: {B.shape[1]} channels; sigma {tsig:.0f} s")
        S.set_energy(E, A)
        hist = []
        def cb(i, est):
            hist.append(est.max())
            if i < 10 or (i + 1) % 10 == 0:
                log(f"  it {i + 1}: residual estimate max {est.max():.2e}, median {np.median(est):.2e}, "
                    f"active {(est > a.tol).sum()}")
        clock = {}
        prec, x0, slow = S, np.zeros_like(b), None
        if a.coarse == "slow":
            slow = SlowCoarse(A, S, a.slow_m, a.slow_q, a.slow_rounds, a.pre, a.post, log=log)
            prec, x0 = slow.cycle, slow(b)
        t = time.time()
        if a.restart > 0:
            x, its, rres, ncyc = fgmres_restart(A, b, x0, prec, a.tol, a.restart, a.maxit, clock, cb, log)
        else:
            x, its, rres = fgmres_block(A, b, x0, prec, a.tol, a.maxit, clock, cb)
            ncyc = 1
        tit = time.time() - t
        Tx = T(x)
        r = dict(E=E, channels=int(B.shape[1]), its=its.tolist(), res=rres.tolist(), res_hist=hist, T=Tx,
                 restart=a.restart, cycles=ncyc, coarse=a.coarse, slow=None if slow is None else slow.info, t_sigma=tsig, t_factor=S.twall, t_iter=tit, clock=clock, n_windows=len(S.idx),
                 window_sizes=[len(i) for i in S.idx], cap=a.cap, centres=a.cap_centres, solver=a.cap_solver)
        log(f"  FGMRES ({a.coarse}" + (f", N {slow.N}, build {slow.info['t_build']:.0f} s" if slow else "") + f"): its {its.min()}..{its.max()}, true residual max {rres.max():.1e}, T {Tx:.6f}; factor "
            f"{S.twall:.0f} s + iterations {tit:.0f} s (" + ", ".join(f"{k} {v:.0f}" for k, v in clock.items()) + ")")
        if a.exact:
            fpsi = os.path.join(a.psi_dir, f"psi_{E:+.8f}.npz") if a.psi_dir else None
            t = time.time()
            psi = None
            tw = time.time()
            while fpsi and a.psi_wait > 0 and not os.path.exists(fpsi) and time.time() - tw < 3600 * a.psi_wait:
                time.sleep(60)
            t = time.time()                                        # direct-solve / reuse time, not the wait
            if fpsi and os.path.exists(fpsi):
                time.sleep(5)                                      # a file being written right now

                # the saved psi may belong to injection columns with other phases / mixing (degenerate or
                # sign-ambiguous Gamma eigenvectors): map it onto this b by the column transform Q, b = b_saved Q
                ps = np.load(fpsi)["psi_m"]
                if ps.shape[0] == dev.n:
                    bs = A @ ps
                    Q = np.linalg.lstsq(bs[dev.bL], b[dev.bL], rcond=None)[0]
                    if np.linalg.norm(b - A @ (ps @ Q)) < 1e-8 * np.linalg.norm(b):
                        psi = ps @ Q
                        log(f"  exact psi reused from {fpsi} (column transform |Q - I| {np.linalg.norm(Q - np.eye(*Q.shape)):.1e})")
            if psi is None:
                cs = mb.CSolver(A)
                psi = cs.solve(b)
                cs.free()
                if fpsi:
                    os.makedirs(a.psi_dir, exist_ok=True)
                    np.savez(fpsi, E=E, psi_m=psi, dev=dev.dev, obc=a.obc, eta=a.eta)
            tex = time.time() - t
            Tex = T(psi)
            r.update(T_exact=Tex, T_err=(Tx - Tex) / Tex, err_psi=float(np.linalg.norm(x - psi) / np.linalg.norm(psi)),
                     t_exact=tex)
            log(f"  exact: T {Tex:.6f}, FGMRES T err {r['T_err']:+.1e}, psi err {r['err_psi']:.1e}; direct {tex:.0f} s")
        S.ops = None                                              # free the window factors
        if slow is not None:
            slow.free()
        res = json.load(open(fjs)) if os.path.exists(fjs) else {"E": {}}
        res["Ec"] = Ec
        res["E"][f"{dE:.4f}"] = r
        json.dump(res, open(fjs, "w"), indent=1)


if __name__ == "__main__":
    main()
