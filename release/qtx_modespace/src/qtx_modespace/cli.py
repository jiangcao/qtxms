"""QuaTrEx device folder -> mode-space basis (saved as .npy) -> optional transmission vs exact.

  1. read the QuaTrEx folder (quatrex.read_quatrex): blocks of --cells contact cells (default: the coupling range),
     optional --trim to block tridiagonal; leads = the first / last block; lead band edges Ev, Ec
  2. level 2 (direct): slab density matrices of the open device, contour integral over [Ec + mu1, Ec + mu2] with the
     exact lead self-energies; per slab the --nmodes eigenvectors with the largest occupation, from an eigensolve of
     the --neig largest (default 2 nmodes; the smallest occupations are near rounding level, and which vectors come
     out there depends on the solve size), optionally only those with occupation > --floor * largest.
     Several windows (--mu 'a1,b1;a2,b2'): each set is taken as above, weighted X sqrt(lambda / lambda_max), stacked
     and merged by one SVD per slab (kept: s^2 > --merge-thr s0^2), so that a higher window (e.g. for a bias that
     lowers the bands) is not swamped by the occupations of the first
     --basis l1 instead: level 1 = local density matrices per block (level1.build_level1): the device slabs as
     blocks (--l1-partition slab, slab size = --cells) or tiles (--l1-partition tile: --tile-orbs per tile, --tile-xsplit
     pieces along transport), each block embedded in the atoms within --l1-emb, closed by a CAP (--l1-boundary cap) or
     by the CAP + the exact lead Sigma on the lead faces (obc), rho of the window, eigenvectors with occupation > --l1-cut
  3. optional cleanup (off by default; not needed for T): the level-2 bases of the slabs listed in --clean-slabs
     (e.g. 0,-1 = next to the leads) are put on the lead cell, and spurious lead bands in [Ec + win1, Ec + win2] are
     removed by appending optimized vectors (clean.RMS). Its convergence is sensitive to rounding-level changes of
     the basis (the number of added vectors varies between runs)
  4. saved: <out>/modes/slab<s>.npy (slab orbitals x modes, orthonormal columns), slab<s>_orbitals.npy (row ->
     orbital index in the QuaTrEx hamiltonian.h5), basis.json (sizes, occupations, cleanup, device info)
     --modes DIR: use a saved basis (DIR/slab<s>.npy, checked against slab<s>_orbitals.npy) instead of 1-4
  5. --transmission: T of the mode-space solution x = U A_c^-1 U^H b (QTBM injection, Galerkin A_c with the exact
     lead Sigma projected) at the --dE energies, with --exact also the exact T (sparse direct solve) and the error,
     and the error of the solution vectors x_MS - x (see psi_errors); --psi-dir keeps the exact x for reuse
  6. --fgmres: iterative solve to --fgmres-tol by flexible GMRES with the CAP-Schwarz smoother (schwarz.CapSchwarz,
     windows --cap along transport, absorbing layers on the artificial cuts) and the mode-space coarse level
     (--fgmres-coarse ms: x0 = mode-space solution, two-level cycle with --pre / --post smoothing) or without it
     (none: one-level, x0 = 0); T_fgmres, iterations and timings per energy
  7. --system closed --fgmres: A0 = z S - H without the lead self-energy, right-hand sides = unit vectors on the
     contact-slab orbitals (--rhs-slabs, --rhs-face), same FGMRES / smoother / coarse level (without Sigma) -> closed.json

Config file: --config FILE.toml sets any option by its long name (dashes or underscores, folder = the device
folder); options on the command line override it. Every run writes the resolved options to <out>/config.toml, so
'python -m qtx_modespace --config <out>/config.toml' repeats it (relative paths: from the launch directory).

    python -m qtx_modespace DEVICE_FOLDER --out ms_out --cells 3 --trim --nmodes 400 \\
        --transmission --dE 0:0.5:0.05 --exact
"""
import argparse, json, os, time, tomllib
import numpy as np

from .quatrex import read_quatrex, lead_bands
from .core import Device, Coarse, CSolver, injection, rho_contour, top_modes, log
from .clean import LeadCell, clean_slab
from .level1 import build_level1, partition
from .schwarz import CapSchwarz, FastOp, fgmres_block, two_level


def energies(spec):
    if ":" in spec:
        lo, hi, st = (float(x) for x in spec.split(":"))
        return list(np.round(np.arange(lo, hi + 0.5 * st, st), 6))
    return [float(x) for x in spec.split(",")]


def plot(fpng, res, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:          # pragma: no cover
        return
    k = sorted(res, key=float)
    d = np.array([float(x) for x in k])
    Tms = np.array([res[x]["T_ms"] for x in k])
    Tex = np.array([res[x].get("T_exact", np.nan) for x in k])
    ex = np.isfinite(Tex).any()
    fig, axs = plt.subplots(2 if ex else 1, 1, figsize=(7, 6.5 if ex else 4), sharex=True, squeeze=False)
    a1 = axs[0, 0]
    if ex:
        a1.plot(d, Tex, "k-", lw=1.5, label="exact")
    a1.plot(d, Tms, "o", ms=3.5, color="#d9622b", label="mode space")
    a1.set_ylabel("T"); a1.legend(frameon=False); a1.set_title(title, fontsize=10); a1.grid(alpha=0.3)
    if ex:
        ok = np.isfinite(Tex) & (Tex > 1e-8)
        a2 = axs[1, 0]
        a2.semilogy(d[ok], np.abs(Tms[ok] - Tex[ok]) / Tex[ok], "o-", ms=3, lw=0.8, color="#2b6cd9")
        a2.set_ylabel("|T_MS - T| / T"); a2.grid(alpha=0.3)
    axs[-1, 0].set_xlabel("E - Ec (eV)")
    fig.tight_layout(); fig.savefig(fpng, dpi=130); plt.close(fig)


def slab_modes(a, dev, mu, npts, neig, nmodes):
    """Top --nmodes (lambda, X) per slab of the contour slab density matrices over [mu1, mu2] (cached eigenpairs)."""
    nsl = len(dev.ns)
    feig = os.path.join(a.out, f"eig_mu{mu[0]:.4f}_{mu[1]:.4f}_n{npts}_top{neig}.npz")
    if os.path.exists(feig):
        z = np.load(feig)
        eig = [(z[f"l{s}"], z[f"X{s}"]) for s in range(nsl)]
        log(f"slab modes from {feig}")
    else:
        frho = os.path.join(a.out, f"rho_mu{mu[0]:.4f}_{mu[1]:.4f}_n{npts}.npz")
        if os.path.exists(frho):
            z = np.load(frho)
            rho = [z[f"r{s}"] for s in range(nsl)]
            log(f"slab density matrices from {frho}")
        else:
            t = time.time()
            rho = rho_contour(dev, mu[0], mu[1], npts, log=log)
            log(f"contour rho [{mu[0]:.4f}, {mu[1]:.4f}] eV, {npts} points: {time.time() - t:.0f} s")
            if a.save_rho:
                np.savez(frho, **{f"r{s}": r for s, r in enumerate(rho)})
        t = time.time()
        eig = []
        for s in range(nsl):
            eig.append(top_modes(rho[s], neig))
            rho[s] = None
        np.savez(feig, **{f"l{s}": e[0] for s, e in enumerate(eig)}, **{f"X{s}": e[1] for s, e in enumerate(eig)})
        log(f"slab eigenpairs (top {neig}): {time.time() - t:.0f} s")
    eig = [(lam[:nmodes], X[:, :nmodes]) for lam, X in eig]
    if a.floor > 0:
        eig = [(lam[lam > a.floor * lam[0]], X[:, lam > a.floor * lam[0]]) for lam, X in eig]
    log(f"window [{mu[0]:.4f}, {mu[1]:.4f}] eV: N {[X.shape[1] for _, X in eig]}; smallest kept occupation / "
        f"largest {[f'{lam[-1] / lam[0]:.1e}' for lam, _ in eig]}")
    return eig


def level1_basis(a, inp, dev, Ec):
    """--basis l1: (V, occupations, info) of level1.build_level1 with the --l1-* options; None after --l1-only."""
    w = [Ec + float(x) for x in (a.l1_mu or a.mu.split(";")[0]).split(",")]
    npts = a.l1_npts or int(str(a.npts).split(",")[0])
    d_cap, eta_cap = (float(x) for x in a.l1_cap.split(","))
    cache = None if a.l1_cache == "none" else (a.l1_cache or os.path.join(a.out, "level1"))
    kw = dict(kind=a.l1_partition, tile_orbs=a.tile_orbs, xsplit=a.tile_xsplit, boundary=a.l1_boundary,
              d_emb=a.l1_emb, d_cap=d_cap, eta_cap=eta_cap, cut=a.l1_cut, store_cut=a.l1_store_cut, cache=cache,
              gpp=a.l1_gpp, env_prec=a.l1_env_prec, workers=a.l1_workers, threads=a.l1_threads or None)
    if a.l1_only:
        k, n = (int(x) for x in a.l1_only.split(","))
        nb = len(partition(inp, dev, a.l1_partition, a.tile_orbs, a.tile_xsplit))
        mine = {i for i in range(nb) if i % n == k}
        build_level1(inp, dev, (w[0], w[1], npts), only=mine, **kw)
        return None
    return build_level1(inp, dev, (w[0], w[1], npts), **kw)


def build_basis(a, inp, dev, Ec, t0):
    nsl, meta = len(dev.ns), inp["meta"]
    if a.basis == "l1":
        out = level1_basis(a, inp, dev, Ec)
        if out is None:
            return None, None
        V, occ, l1info = out
        eig = list(zip(occ, V))
        info = dict(device=dict(meta, n=dev.n, slabs=dev.ns), basis="l1", level1=l1info,
                    occupations={s: dict(first=float(o[0]), last=float(o[-1])) for s, o in enumerate(occ)},
                    clean={}, args=vars(a))
        return finish_basis(a, inp, dev, Ec, t0, V, eig, info)
    nmodes = [int(x) for x in str(a.nmodes).split(",")]
    neig = a.neig or 2 * max(nmodes)
    assert neig >= max(nmodes), "--neig < --nmodes"
    # ------------------------------------------------------------------ level 2
    wins = [[Ec + float(x) for x in w.split(",")] for w in a.mu.split(";")]
    npts = [int(x) for x in str(a.npts).split(",")]
    npts += [npts[-1]] * (len(wins) - len(npts))
    nmodes += [nmodes[-1]] * (len(wins) - len(nmodes))
    sets = [slab_modes(a, dev, mu, n, neig, m) for mu, n, m in zip(wins, npts, nmodes)]
    if len(sets) == 1:
        eig = sets[0]
    else:   # merge the windows: per slab X sqrt(lambda / lambda_max) of every set, one SVD
        eig = []
        for s in range(nsl):
            Y = np.hstack([X * np.sqrt(np.maximum(lam, 0) / lam[0]) for lam, X in (e[s] for e in sets)])
            Q, sv, _ = np.linalg.svd(Y, full_matrices=False)
            m = sv ** 2 > a.merge_thr * sv[0] ** 2
            eig.append((sv[m] ** 2, Q[:, m]))
        log(f"merged {len(sets)} windows: {[sum(e[s][1].shape[1] for e in sets) for s in range(nsl)]} vectors -> "
            f"N {[X.shape[1] for _, X in eig]} (s^2 > {a.merge_thr:g} s0^2)")
    V = [X for _, X in eig]
    log(f"level 2: N {[v.shape[1] for v in V]}; smallest kept occupation / largest "
        f"{[f'{lam[-1] / lam[0]:.1e}' for lam, _ in eig]}")

    # ------------------------------------------------------------------ cleanup of the slabs next to the leads
    info = dict(device=dict(meta, n=dev.n, slabs=dev.ns), basis="l2", mu=wins, npts=npts, nmodes=nmodes,
                neig=neig, floor=a.floor, merge_thr=a.merge_thr if len(wins) > 1 else None,
                occupations={s: dict(first=float(lam[0]), last=float(lam[-1])) for s, (lam, _) in enumerate(eig)},
                clean={}, args=vars(a))
    return finish_basis(a, inp, dev, Ec, t0, V, eig, info)


def finish_basis(a, inp, dev, Ec, t0, V, eig, info):
    """Optional end-slab cleanup, then save modes/ and basis.json."""
    nsl = len(dev.ns)
    slabs = sorted({int(x) % nsl for x in a.clean_slabs.split(",") if x.strip()})
    if slabs:
        cell = LeadCell(inp["lead"])
        ks = np.linspace(0, np.pi, a.nk)
        t = time.time()
        Eex = np.array([np.sort(e) for e in lead_bands(inp["lead"], ks)])
        log(f"exact lead bands on {a.nk} k: {time.time() - t:.0f} s")
        win = tuple(Ec + float(x) for x in a.win.split(","))
        for s in slabs:
            t = time.time()
            V[s], info["clean"][s] = clean_slab(inp, dev, cell, V[s], s, ks, Eex, win, tol=a.tol, kstep=a.kstep,
                                                maxit=a.maxit, map_tol=a.map_tol, log=log)
            info["clean"][s]["t"] = time.time() - t
    info["N"] = [int(v.shape[1]) for v in V]
    for s in range(nsl):
        np.save(os.path.join(a.out, "modes", f"slab{s}.npy"), V[s])
        np.save(os.path.join(a.out, "modes", f"slab{s}_orbitals.npy"), inp["perm"][dev.slab_orbs[s]])
        np.save(os.path.join(a.out, "modes", f"slab{s}_occupations.npy"), eig[s][0])
    json.dump(info, open(os.path.join(a.out, "basis.json"), "w"), indent=1, default=str)
    log(f"basis: N {info['N']}, total {sum(info['N'])} (device {dev.n} orbitals) -> {a.out}/modes; "
        f"{time.time() - t0:.0f} s")
    return V, info


def solve_fgmres(a, dev, c, sw, A, b, E, SL, SR, Xms, Tof):
    """FGMRES + CAP-Schwarz (two-level with the Galerkin coarse level, or one-level) on A x = b at one energy."""
    sw.set_energy(A, a.cap_dE0)
    coarse, x0 = None, np.zeros_like(b)
    if a.fgmres_coarse == "ms":
        Ac = c.Ac(E + 1j * a.coarse_eta, SL, SR) if a.coarse_eta else None
        if Ac is None:
            Ac = c.Ac(E, SL, SR)
        coarse = lambda r: c.prolong(Ac.solve(c.restrict(r)))
        x0 = coarse(b) if a.coarse_eta else Xms
    A = FastOp(A, a.workers)
    prec = two_level(A, sw, coarse, a.pre, a.post)
    hist = []
    t = time.time()
    x, its, res = fgmres_block(A, b, x0, prec, a.fgmres_tol, a.fgmres_maxit, cb=lambda i, e: hist.append(float(e.max())))
    tit = time.time() - t
    sw.free()
    return dict(x=x, T=Tof(x), its=its.tolist(), res=res.tolist(), res_hist=hist, t_factor=sw.twall, t_iter=tit,
                coarse=a.fgmres_coarse, tol=a.fgmres_tol, cap=a.cap, centres=a.cap_centres, split=a.cap_split,
                solver=a.cap_solver, pre=a.pre, post=a.post, coarse_eta=a.coarse_eta, n_windows=len(sw.idx),
                window_sizes=[len(i) for i in sw.idx])


def closed_columns(a, dev):
    """Device orbitals whose unit vectors are the right-hand sides of --system closed (slabs --rhs-slabs, or only
    their lead-face orbitals with --rhs-face)."""
    nsl = len(dev.ns)
    cols = []
    for s in sorted({int(x) % nsl for x in a.rhs_slabs.split(",")}):
        o = np.arange(dev.sb[s], dev.sb[s + 1])
        if a.rhs_face:
            o = o[np.isin(o, np.concatenate([dev.bL, dev.bR]))]
        cols.append(o)
    cols = np.concatenate(cols)
    return cols[:a.rhs_max] if a.rhs_max else cols


def closed_solves(a, dev, c, sw, EcT, info):
    """--system closed: A0 X = I_C with A0 = z S - H (no lead self-energy, z = E + i --closed-eta), I_C = unit vectors
    on the contact-slab orbitals C; fine and coarse levels without Sigma. Solved in chunks of --rhs-chunk columns by
    FGMRES + CAP-Schwarz (two-level with the Galerkin coarse level of the basis, or one-level); with --exact also by a
    sparse direct solve (error of X). Results per energy in <out>/closed.json, X in --save-x (closed_<E>.npy)."""
    cols = closed_columns(a, dev)
    fj = os.path.join(a.out, "closed.json")
    out = json.load(open(fj)) if os.path.exists(fj) else {}
    out.update(Ec=EcT, N=info["N"], columns=int(len(cols)), rhs_slabs=a.rhs_slabs, rhs_face=a.rhs_face)
    res = out.setdefault("E", {})
    Z0L = np.zeros((len(dev.bL), len(dev.bL)), complex)
    Z0R = np.zeros((len(dev.bR), len(dev.bR)), complex)
    for dE in energies(a.dE):
        z = EcT + dE + 1j * a.closed_eta
        A = (z * dev.S - dev.H).tocsr()
        r = dict(E=z.real, eta=a.closed_eta, columns=int(len(cols)))
        log(f"E = Ec {dE:+.3f} (+ {a.closed_eta:g} i), closed system, {len(cols)} unit right-hand sides")
        X = np.zeros((dev.n, len(cols)), complex)
        sw.set_energy(A, a.cap_dE0)
        coarse = None
        if a.fgmres_coarse == "ms":
            Ac = c.Ac(z + 1j * a.coarse_eta, Z0L, Z0R)
            coarse = lambda q: c.prolong(Ac.solve(c.restrict(q)))
        Aop = FastOp(A, a.workers)
        prec = two_level(Aop, sw, coarse, a.pre, a.post)
        its, rr, t = [], [], time.time()
        for k in range(0, len(cols), a.rhs_chunk):
            cc = cols[k:k + a.rhs_chunk]
            b = np.zeros((dev.n, len(cc)), complex)
            b[cc, np.arange(len(cc))] = 1.0
            x0 = coarse(b) if coarse is not None else np.zeros_like(b)
            X[:, k:k + len(cc)], it, rs = fgmres_block(Aop, b, x0, prec, a.fgmres_tol, a.fgmres_maxit)
            its += it.tolist(); rr += rs.tolist()
            log(f"  columns {k}..{k + len(cc) - 1}: its {it.min()}..{it.max()}, residual {rs.max():.1e}; "
                f"{time.time() - t:.0f} s")
        sw.free()
        r.update(its=its, res_max=float(max(rr)), its_max=int(max(its)), its_mean=float(np.mean(its)),
                 t_factor=sw.twall, t_iter=time.time() - t, coarse=a.fgmres_coarse, cap=a.cap, tol=a.fgmres_tol)
        if a.exact:
            t = time.time()
            cs = CSolver(A)
            B = np.zeros((dev.n, len(cols)), complex)
            B[cols, np.arange(len(cols))] = 1.0
            Xe = cs.solve(B)
            cs.free()
            col = np.linalg.norm(X - Xe, axis=0) / np.linalg.norm(Xe, axis=0)
            r.update(err=float(np.linalg.norm(X - Xe) / np.linalg.norm(Xe)), err_col_max=float(col.max()),
                     t_exact=time.time() - t)
        if a.save_x:
            os.makedirs(a.save_x, exist_ok=True)
            np.savez(os.path.join(a.save_x, f"closed_{z.real:+.8f}_{a.closed_eta:.2e}.npz"), X=X, cols=cols,
                     orbitals=np.asarray(dev.inp["perm"])[dev.dev[cols]])
        log(f"  closed: its {min(its)}..{max(its)} (mean {np.mean(its):.1f}), residual {max(rr):.1e}; factor "
            f"{sw.twall:.0f} s + iterations {r['t_iter']:.0f} s"
            + (f"; vs direct: err {r['err']:.1e} (worst column {r['err_col_max']:.1e}), direct {r['t_exact']:.0f} s"
               if "err" in r else ""))
        res[f"{dE:.4f}"] = r
        out["E"] = dict(sorted(res.items(), key=lambda kv: float(kv[0])))
        res = out["E"]
        json.dump(out, open(fj, "w"), indent=1)


def psi_errors(dev, c, A, b, X, Xms):
    """Errors of the mode-space QTBM solutions Xms against the exact X (columns = injected channels), relative to
    the exact solution: Frobenius over all channels (psi), worst channel (psi_max), S norm (psi_S), the right end
    slab (psi_last; T is read off there) and each slab (psi_slab); psi_best = |X - U U^H X| / |X| is the error of
    the best plain-norm approximation in range(U); res = |b - A Xms| / |b|."""
    E, b_ = Xms - X, dev.sb
    fro = lambda M: float(np.linalg.norm(M))
    sn = lambda M: float(np.sqrt(max(np.real(np.vdot(M, dev.S @ M)), 0.0)))
    col = np.linalg.norm(E, axis=0) / np.maximum(np.linalg.norm(X, axis=0), 1e-300)
    return dict(psi=fro(E) / fro(X), psi_max=float(col.max()), psi_S=sn(E) / sn(X),
                psi_last=fro(E[b_[-2]:]) / fro(X[b_[-2]:]),
                psi_slab=[fro(E[b_[s]:b_[s + 1]]) / fro(X[b_[s]:b_[s + 1]]) for s in range(len(dev.ns))],
                psi_best=fro(X - c.prolong(c.restrict(X))) / fro(X), res=fro(b - A @ Xms) / fro(b))


def load_config(p, f):
    """Options of the TOML file f as parser defaults: keys = long option names, lists joined by ',' (lists of lists:
    windows, joined by ';'), values of string options converted to str."""
    acts = {x.dest: x for x in p._actions}
    out = {}
    for k, v in tomllib.loads(open(f).read()).items():
        d = k.replace("-", "_")
        if d not in acts or d in ("help", "config"):
            p.error(f"{f}: unknown option '{k}'")
        if isinstance(v, list):
            v = ";".join(",".join(map(str, x)) for x in v) if v and isinstance(v[0], list) else ",".join(map(str, v))
        elif acts[d].type in (None, str) and not isinstance(v, (str, bool)):
            v = str(v)
        out[d] = v
    return out


def save_config(f, a):
    """The resolved options of the run as TOML (None = unset, left out)."""
    lines = [f"# python -m qtx_modespace --config {f}   (launched from {os.getcwd()})"]
    for k, v in vars(a).items():
        if v is None or k == "config":
            continue
        lines.append(f"{k} = " + (("true" if v else "false") if isinstance(v, bool) else
                                  json.dumps(v) if isinstance(v, str) else repr(v)))
    open(f, "w").write("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="qtx_modespace", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", help="QuaTrEx device folder (quatrex_config.toml, inputs/)")
    p.add_argument("--config", help="TOML file of options (command line overrides it)")
    p.add_argument("--out")
    g = p.add_argument_group("device")
    g.add_argument("--cells", type=int, help="contact cells per block (default: smallest covering the coupling range)")
    g.add_argument("--trim", action="store_true", help="drop H, S entries coupling non-neighbouring blocks")
    g.add_argument("--potential", action="store_true", help="add inputs/potential.npy as (V S + S V) / 2")
    g = p.add_argument_group("basis")
    g.add_argument("--mu", default="-0.15,0.6",
                   help="contour window - Ec_lead [eV]; several windows separated by ';' are merged, e.g. "
                        "'-0.15,0.6;0.5,0.9'")
    g.add_argument("--npts", default="24", help="contour points (per window: comma list, last value repeats)")
    g.add_argument("--merge-thr", type=float, default=1e-8,
                   help="several windows: keep merged vectors with s^2 > merge_thr * s0^2")
    g.add_argument("--nmodes", default="400", help="modes per slab (largest occupations); per window: comma list")
    g.add_argument("--neig", type=int, help="eigenpairs computed per slab (default 2 * largest nmodes)")
    g.add_argument("--floor", type=float, default=0.0, help="drop modes with occupation < floor * largest")
    g.add_argument("--save-rho", action="store_true", help="keep the slab density matrices (<out>/rho_*.npz, large)")
    g.add_argument("--basis", default="l2", choices=["l2", "l1"],
                   help="l2: direct level 2 (slab rho of the open device, default); l1: local density matrices per "
                        "block (level 1, options below)")
    g.add_argument("--modes", help="saved basis directory (slab<s>.npy, slab<s>_orbitals.npy): skip the basis build")
    g.add_argument("--clean-slabs", default="", help="slabs cleaned against the lead bands, e.g. 0,-1 (default none)")
    g.add_argument("--win", default="-0.6,0.45", help="spurious-band window - Ec_lead [eV]")
    g.add_argument("--nk", type=int, default=25, help="k points on [0, pi] for the lead bands")
    g.add_argument("--kstep", type=int, default=2, help="every kstep-th k point is used in the optimization")
    g.add_argument("--maxit", type=int, default=150, help="max vectors added per slab")
    g.add_argument("--tol", type=float, default=0.005, help="spurious: > tol [eV] from every exact lead state")
    g.add_argument("--map-tol", type=float, default=0.6, help="max displacement [A] of a slab atom matched to the lead")
    g = p.add_argument_group("level 1 (--basis l1)")
    g.add_argument("--l1-partition", default="slab", choices=["slab", "tile"],
                   help="blocks: the device slabs (size set by --cells) or tiles of each slab")
    g.add_argument("--tile-orbs", type=int, default=2500, help="tile partition: orbitals per tile (transverse bisection)")
    g.add_argument("--tile-xsplit", type=int, default=1, help="cut every block into this many pieces along transport")
    g.add_argument("--l1-boundary", default="obc", choices=["obc", "cap"],
                   help="embedding closed by CAP + exact lead Sigma on the lead faces (obc) or by the CAP only (cap)")
    g.add_argument("--l1-emb", type=float, default=12.0, help="embedding radius around the block atoms [A]")
    g.add_argument("--l1-cap", default="3,0.3", help="CAP: shell width [A], strength [eV]")
    g.add_argument("--l1-mu", help="local rho window - Ec_lead [eV] (default: the first --mu window)")
    g.add_argument("--l1-npts", type=int, help="local rho contour points (default: the first --npts)")
    g.add_argument("--l1-cut", type=float, default=1e-6, help="keep eigenvectors of the local rho with occupation > cut")
    g.add_argument("--l1-store-cut", type=float, default=1e-10,
                   help="cache vectors down to this occupation (later runs with any cut >= it reuse the blocks)")
    g.add_argument("--l1-cache", help="block cache directory (default <out>/level1; 'none' = off)")
    g.add_argument("--l1-gpp", default="schur", choices=["schur", "full"],
                   help="G_PP by the Schur complement of the environment (schur) or one LU of the whole embedding")
    g.add_argument("--l1-env-prec", default="double", choices=["double", "single"],
                   help="schur: environment LU precision (single: complex64 + one refinement step)")
    g.add_argument("--l1-workers", type=int, default=1, help="contour points of a block over this many processes")
    g.add_argument("--l1-threads", type=int, default=0, help="BLAS threads per process (needs threadpoolctl; 0: unchanged)")
    g.add_argument("--l1-only", metavar="K,N", help="build only the blocks i with i %% N == K (into the cache), then exit")
    g = p.add_argument_group("transmission")
    g.add_argument("--transmission", action="store_true", help="mode-space T at --dE")
    g.add_argument("--exact", action="store_true", help="also the exact T (sparse direct solve) and the error")
    g.add_argument("--dE", default="0.02,0.05,0.15,0.3", help="energies - Ec: list or lo:hi:step [eV]")
    g.add_argument("--Ec", type=float, help="reference edge for --dE (default Ec_lead)")
    g.add_argument("--eta", type=float, default=0.0, help="imaginary part of E for the lead self-energies [eV] (0: real "
                   "axis, qttools OBC; the folded-lead fallback then uses E + 1e-6 i)")
    g.add_argument("--obc", default="auto", choices=["auto", "qttools", "folded"],
                   help="real-axis lead OBC: qttools spectral OBC (QuaTrEx) with the folded lead as fallback (auto), "
                        "qttools only, or the folded lead only")
    g.add_argument("--psi-dir", help="--exact: store / reuse the exact solutions here (psi_<E>.npy, any basis)")
    g = p.add_argument_group("iterative solve (--fgmres)")
    g.add_argument("--fgmres", action="store_true", help="also solve A x = b to --fgmres-tol by FGMRES + CAP-Schwarz")
    g.add_argument("--fgmres-coarse", default="ms", choices=["ms", "none"],
                   help="ms: two-level with the mode-space coarse level, x0 = mode space; none: one-level, x0 = 0")
    g.add_argument("--fgmres-tol", type=float, default=1e-10, help="relative residual per channel")
    g.add_argument("--fgmres-maxit", type=int, default=150, help="max FGMRES iterations")
    g.add_argument("--pre", type=int, default=1, help="pre-smoothing steps per two-level cycle")
    g.add_argument("--post", type=int, default=1, help="post-smoothing steps per two-level cycle (keep >= 1)")
    g.add_argument("--coarse-eta", type=float, default=0.0,
                   help="coarse solve at E + i eta_c [eV] (damps narrow spurious mode-space resonances)")
    g.add_argument("--cap", default="16,2,2", help="Schwarz windows: width [A], absorbing strength [eV], ramp [A]")
    g.add_argument("--cap-spacing", type=float, help="largest distance between window centres [A] (default: the slab "
                   "interfaces and the slab middles)")
    g.add_argument("--cap-centres", default="all", choices=["all", "ends"],
                   help="all: interfaces + between them; ends: interfaces + the two lead-face windows")
    g.add_argument("--cap-split", default="", help="pieces per periodic transverse axis, e.g. 1,2 (default none)")
    g.add_argument("--cap-tov", type=float, default=2.5, help="transverse overlap of the split boxes [A]")
    g.add_argument("--cap-solver", default="dense64", choices=["dense64", "dense", "pardiso"], help="window LU")
    g.add_argument("--cap-dE0", type=float, default=0.0, help="factor the windows at E + dE0 [eV]")
    g.add_argument("--system", default="open", choices=["open", "closed"],
                   help="open: QTBM A = E S - H - Sigma, b = injected channels; closed: A0 = z S - H without the lead "
                        "self-energy (fine and coarse), b = unit vectors on the contact-slab orbitals, FGMRES only")
    g.add_argument("--rhs-slabs", default="0", help="closed: slabs whose orbitals get unit right-hand sides (0 = next "
                   "to the left lead, -1 = right; e.g. 0,-1)")
    g.add_argument("--rhs-face", action="store_true", help="closed: only the lead-face orbitals of those slabs")
    g.add_argument("--rhs-max", type=int, default=0, help="closed: only the first n of those columns (tests; 0 = all)")
    g.add_argument("--rhs-chunk", type=int, default=64, help="closed: right-hand sides per FGMRES block (memory: "
                   "maxit x chunk device vectors)")
    g.add_argument("--closed-eta", type=float, default=0.0, help="closed: z = E + i eta [eV] (the closed device has "
                   "real eigenvalues: at eta = 0 A0 is singular there)")
    g.add_argument("--save-x", help="closed: save the solutions here (closed_<E>_<eta>.npz: X, columns, orbitals)")
    g.add_argument("--workers", type=int, default=4, help="Schwarz windows factored / solved concurrently (threads)")
    p.add_argument("--sigma-cache", help="directory for lead self-energies and lead band edges (default <out>/sigma; 'none' = off)")
    c, _ = p.parse_known_args(argv)
    if c.config:
        p.set_defaults(**load_config(p, c.config))
    a = p.parse_args(argv)
    for k in ("folder", "out"):
        if not getattr(a, k):
            p.error(f"{k} is required (command line or --config)")
    if a.system == "closed" and not a.fgmres:
        p.error("--system closed needs --fgmres")

    t0 = time.time()
    os.makedirs(os.path.join(a.out, "modes"), exist_ok=True)
    save_config(os.path.join(a.out, "config.toml"), a)
    cache = None if a.sigma_cache == "none" else (a.sigma_cache or os.path.join(a.out, "sigma"))
    inp = read_quatrex(a.folder, cells=a.cells, trim=a.trim, potential=a.potential, cache_dir=cache, log=log)
    meta = inp["meta"]
    Ec, Ev = meta["edges"]["Ec_lead"], meta["edges"]["Ev_lead"]
    dev = Device(inp, sigma_cache=cache, obc=a.obc)
    nsl = len(dev.ns)
    log(f"device {dev.n} orbitals, slabs {dev.ns}; Ec {Ec:.4f}, Ev {Ev:.4f} eV")

    if a.modes:
        V = []
        for s in range(nsl):
            orb = np.load(os.path.join(a.modes, f"slab{s}_orbitals.npy"))
            assert np.array_equal(orb, inp["perm"][dev.slab_orbs[s]]), \
                f"slab {s} of {a.modes} does not match this device (--cells / --trim?)"
            V.append(np.load(os.path.join(a.modes, f"slab{s}.npy")))
        info = dict(N=[int(v.shape[1]) for v in V])
        log(f"basis from {a.modes}: N {info['N']}, total {sum(info['N'])}")
    else:
        V, info = build_basis(a, inp, dev, Ec, t0)
        if V is None:
            log(f"--l1-only {a.l1_only}: blocks built, {time.time() - t0:.0f} s")
            return

    # ------------------------------------------------------------------ transmission
    if not (a.transmission or a.fgmres):
        return
    EcT = Ec if a.Ec is None else a.Ec
    c = Coarse(dev, V)
    sw = None
    if a.fgmres:
        w_, s_, d_ = (float(x) for x in a.cap.split(","))
        sw = CapSchwarz(dev, inp, w_, s_, d_, spacing=a.cap_spacing, centres=a.cap_centres,
                        split=[int(x) for x in a.cap_split.split(",")] if a.cap_split else None, tov=a.cap_tov,
                        solver=a.cap_solver, workers=a.workers)
    if a.system == "closed":
        closed_solves(a, dev, c, sw, EcT, info)
        log(f"done, {time.time() - t0:.0f} s")
        return
    ftr = os.path.join(a.out, "transmission.json")
    tr = json.load(open(ftr)) if os.path.exists(ftr) else {}
    if tr.get("Ec", EcT) != EcT or tr.get("N", info["N"]) != info["N"]:
        tr = {}
    tr.update(Ec=EcT, N=info["N"])
    res = tr.setdefault("E", {})
    for dE in energies(a.dE):
        key = f"{dE:.4f}"
        E = EcT + dE
        r = res.setdefault(key, {})
        t = time.time()
        SL, SR = dev.sigma(E + 1j * a.eta)
        ts = time.time() - t
        B = dev.inject(SL)
        GR = 1j * (SR - SR.conj().T)
        b = np.zeros((dev.n, B.shape[1]), complex)
        b[dev.bL] = B
        Tof = lambda x: float(np.real(np.einsum("ic,ij,jc->", x[dev.bR].conj(), GR, x[dev.bR])))
        t = time.time()
        Xms = c.prolong(c.Ac(E, SL, SR).solve(c.restrict(b)))
        r.update(E=E, channels=int(B.shape[1]), T_ms=Tof(Xms), t_ms=time.time() - t, t_sigma=ts)
        A, X = None, None
        if a.exact and ("T_exact" not in r or "err_psi" not in r or a.fgmres):
            A = dev.A(E, SL, SR)
            fpsi = a.psi_dir and os.path.join(a.psi_dir, f"psi_{E:+.8f}_{a.eta:.2e}.npy")
            X = np.load(fpsi) if fpsi and os.path.exists(fpsi) else None
            if X is not None and (X.shape != b.shape or np.linalg.norm(b - A @ X) > 1e-8 * np.linalg.norm(b)):
                log(f"  {fpsi} does not solve this system, recomputed")
                X = None
            if X is None:
                t = time.time()
                cs = CSolver(A)
                X = cs.solve(b)
                r.update(res_exact=cs.res, t_exact=time.time() - t)
                cs.free()
                if fpsi:
                    os.makedirs(a.psi_dir, exist_ok=True)
                    np.save(fpsi, X)
            r["T_exact"] = Tof(X)
            r.update({"err_" + k: v for k, v in psi_errors(dev, c, A, b, X, Xms).items()})
        if r.get("T_exact", 0) > 0:
            r["err"] = (r["T_ms"] - r["T_exact"]) / r["T_exact"]
        if a.fgmres:
            A = dev.A(E, SL, SR) if A is None else A
            r["fgmres"] = fg = solve_fgmres(a, dev, c, sw, A, b, E, SL, SR, Xms, Tof)
            if r.get("T_exact", 0) > 0:
                fg["err_T"] = (fg["T"] - r["T_exact"]) / r["T_exact"]
            if X is not None:
                fg["err_psi"] = float(np.linalg.norm(fg.pop("x") - X) / np.linalg.norm(X))
            fg.pop("x", None)
            log(f"  FGMRES ({a.fgmres_coarse}): its {min(fg['its'])}..{max(fg['its'])}, residual {max(fg['res']):.1e}, "
                f"T {fg['T']:.6f}" + (f" (err {fg['err_T']:+.1e})" if "err_T" in fg else "")
                + (f", psi err {fg['err_psi']:.1e}" if "err_psi" in fg else "")
                + f"; factor {fg['t_factor']:.0f} s + iterations {fg['t_iter']:.0f} s")
        log(f"E = Ec {dE:+.3f}: {r['channels']} channels, T_MS {r['T_ms']:.5f}"
            + (f", T_exact {r['T_exact']:.5f} ({100 * r['err']:+.4f}%)" if "err" in r else "")
            + (f"; psi error {r['err_psi']:.2e} (max channel {r['err_psi_max']:.2e}, S {r['err_psi_S']:.2e}, "
               f"last slab {r['err_psi_last']:.2e}, best in U {r['err_psi_best']:.2e}), residual {r['err_res']:.1e}"
               if "err_psi" in r else "")
            + f"; sigma {ts:.0f} s, MS {r['t_ms']:.1f} s" + (f", exact {r['t_exact']:.0f} s" if "t_exact" in r else ""))
        tr["E"] = dict(sorted(res.items(), key=lambda kv: float(kv[0])))
        res = tr["E"]
        json.dump(tr, open(ftr, "w"), indent=1)
        plot(ftr.replace(".json", ".png"), res, f"{meta['name']}: mode space N {sum(info['N'])} vs exact")
    errs = [abs(r["err"]) for r in res.values() if "err" in r]
    if errs:
        log(f"max |T_MS - T| / T = {100 * max(errs):.4f}% over {len(errs)} energies")
    errs = [r["err_psi"] for r in res.values() if "err_psi" in r]
    if errs:
        log(f"max |psi_MS - psi| / |psi| = {max(errs):.2e} over {len(errs)} energies")
    log(f"done, {time.time() - t0:.0f} s -> {ftr}")


if __name__ == "__main__":
    main()
