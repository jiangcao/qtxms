"""QuaTrEx device folder -> mode-space basis (saved as .npy) -> optional transmission vs exact.

  1. read the QuaTrEx folder (quatrex.read_quatrex): blocks of --cells contact cells (default: the coupling range),
     optional --trim to block tridiagonal; leads = the first / last block; lead band edges Ev, Ec
  2. level 2 (direct): slab density matrices of the open device, contour integral over [Ec + mu1, Ec + mu2] with the
     exact lead self-energies; per slab the --nmodes eigenvectors with the largest occupation, from an eigensolve of
     the --neig largest (default 2 nmodes; the smallest occupations are near rounding level, and which vectors come
     out there depends on the solve size), optionally only those with occupation > --floor * largest
  3. optional cleanup (off by default; not needed for T): the level-2 bases of the slabs listed in --clean-slabs
     (e.g. 0,-1 = next to the leads) are put on the lead cell, and spurious lead bands in [Ec + win1, Ec + win2] are
     removed by appending optimized vectors (clean.RMS). Its convergence is sensitive to rounding-level changes of
     the basis (the number of added vectors varies between runs)
  4. saved: <out>/modes/slab<s>.npy (slab orbitals x modes, orthonormal columns), slab<s>_orbitals.npy (row ->
     orbital index in the QuaTrEx hamiltonian.h5), basis.json (sizes, occupations, cleanup, device info)
     --modes DIR: use a saved basis (DIR/slab<s>.npy, checked against slab<s>_orbitals.npy) instead of 1-4
  5. --transmission: T of the mode-space solution x = U A_c^-1 U^H b (QTBM injection, Galerkin A_c with the exact
     lead Sigma projected) at the --dE energies, with --exact also the exact T (sparse direct solve) and the error

    python -m qtx_modespace DEVICE_FOLDER --out ms_out --cells 3 --trim --nmodes 400 \\
        --transmission --dE 0:0.5:0.05 --exact
"""
import argparse, json, os, time
import numpy as np

from .quatrex import read_quatrex, lead_bands
from .core import Device, Coarse, CSolver, injection, rho_contour, top_modes, log
from .clean import LeadCell, clean_slab


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


def build_basis(a, inp, dev, Ec, t0):
    nsl, meta = len(dev.ns), inp["meta"]
    neig = a.neig or 2 * a.nmodes
    assert neig >= a.nmodes, "--neig < --nmodes"
    # ------------------------------------------------------------------ level 2
    mu = [Ec + float(x) for x in a.mu.split(",")]
    feig = os.path.join(a.out, f"eig_mu{mu[0]:.4f}_{mu[1]:.4f}_n{a.npts}_top{neig}.npz")
    if os.path.exists(feig):
        z = np.load(feig)
        eig = [(z[f"l{s}"], z[f"X{s}"]) for s in range(nsl)]
        log(f"slab modes from {feig}")
    else:
        frho = os.path.join(a.out, f"rho_mu{mu[0]:.4f}_{mu[1]:.4f}_n{a.npts}.npz")
        if os.path.exists(frho):
            z = np.load(frho)
            rho = [z[f"r{s}"] for s in range(nsl)]
            log(f"slab density matrices from {frho}")
        else:
            t = time.time()
            rho = rho_contour(dev, mu[0], mu[1], a.npts, log=log)
            log(f"contour rho [{mu[0]:.4f}, {mu[1]:.4f}] eV, {a.npts} points: {time.time() - t:.0f} s")
            if a.save_rho:
                np.savez(frho, **{f"r{s}": r for s, r in enumerate(rho)})
        t = time.time()
        eig = []
        for s in range(nsl):
            eig.append(top_modes(rho[s], neig))
            rho[s] = None
        np.savez(feig, **{f"l{s}": e[0] for s, e in enumerate(eig)}, **{f"X{s}": e[1] for s, e in enumerate(eig)})
        log(f"slab eigenpairs (top {neig}): {time.time() - t:.0f} s")
    eig = [(lam[:a.nmodes], X[:, :a.nmodes]) for lam, X in eig]
    if a.floor > 0:
        eig = [(lam[lam > a.floor * lam[0]], X[:, lam > a.floor * lam[0]]) for lam, X in eig]
    V = [X for _, X in eig]
    log(f"level 2: N {[v.shape[1] for v in V]}; smallest kept occupation / largest "
        f"{[f'{lam[-1] / lam[0]:.1e}' for lam, _ in eig]}")

    # ------------------------------------------------------------------ cleanup of the slabs next to the leads
    info = dict(device=dict(meta, n=dev.n, slabs=dev.ns), mu=mu, npts=a.npts, nmodes=a.nmodes,
                neig=neig, floor=a.floor,
                occupations={s: dict(first=float(lam[0]), last=float(lam[-1])) for s, (lam, _) in enumerate(eig)},
                clean={}, args=vars(a))
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


def main(argv=None):
    p = argparse.ArgumentParser(prog="qtx_modespace", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="QuaTrEx device folder (quatrex_config.toml, inputs/)")
    p.add_argument("--out", required=True)
    g = p.add_argument_group("device")
    g.add_argument("--cells", type=int, help="contact cells per block (default: smallest covering the coupling range)")
    g.add_argument("--trim", action="store_true", help="drop H, S entries coupling non-neighbouring blocks")
    g.add_argument("--potential", action="store_true", help="add inputs/potential.npy as (V S + S V) / 2")
    g = p.add_argument_group("basis")
    g.add_argument("--mu", default="-0.15,0.6", help="contour window - Ec_lead [eV]")
    g.add_argument("--npts", type=int, default=24, help="contour points")
    g.add_argument("--nmodes", type=int, default=400, help="modes per slab (largest occupations)")
    g.add_argument("--neig", type=int, help="eigenpairs computed per slab (default 2 * nmodes)")
    g.add_argument("--floor", type=float, default=0.0, help="drop modes with occupation < floor * largest")
    g.add_argument("--save-rho", action="store_true", help="keep the slab density matrices (<out>/rho_*.npz, large)")
    g.add_argument("--modes", help="saved basis directory (slab<s>.npy, slab<s>_orbitals.npy): skip the basis build")
    g.add_argument("--clean-slabs", default="", help="slabs cleaned against the lead bands, e.g. 0,-1 (default none)")
    g.add_argument("--win", default="-0.6,0.45", help="spurious-band window - Ec_lead [eV]")
    g.add_argument("--nk", type=int, default=25, help="k points on [0, pi] for the lead bands")
    g.add_argument("--kstep", type=int, default=2, help="every kstep-th k point is used in the optimization")
    g.add_argument("--maxit", type=int, default=150, help="max vectors added per slab")
    g.add_argument("--tol", type=float, default=0.005, help="spurious: > tol [eV] from every exact lead state")
    g.add_argument("--map-tol", type=float, default=0.6, help="max displacement [A] of a slab atom matched to the lead")
    g = p.add_argument_group("transmission")
    g.add_argument("--transmission", action="store_true", help="mode-space T at --dE")
    g.add_argument("--exact", action="store_true", help="also the exact T (sparse direct solve) and the error")
    g.add_argument("--dE", default="0.02,0.05,0.15,0.3", help="energies - Ec: list or lo:hi:step [eV]")
    g.add_argument("--Ec", type=float, help="reference edge for --dE (default Ec_lead)")
    g.add_argument("--eta", type=float, default=1e-6, help="imaginary part of E for the lead self-energies [eV]")
    p.add_argument("--sigma-cache", help="directory for lead self-energies (default <out>/sigma; 'none' = off)")
    a = p.parse_args(argv)

    t0 = time.time()
    os.makedirs(os.path.join(a.out, "modes"), exist_ok=True)
    inp = read_quatrex(a.folder, cells=a.cells, trim=a.trim, potential=a.potential, log=log)
    meta = inp["meta"]
    Ec, Ev = meta["edges"]["Ec_lead"], meta["edges"]["Ev_lead"]
    cache = None if a.sigma_cache == "none" else (a.sigma_cache or os.path.join(a.out, "sigma"))
    dev = Device(inp, sigma_cache=cache)
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

    # ------------------------------------------------------------------ transmission
    if not a.transmission:
        return
    EcT = Ec if a.Ec is None else a.Ec
    c = Coarse(dev, V)
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
        B = injection(SL)
        GR = 1j * (SR - SR.conj().T)
        b = np.zeros((dev.n, B.shape[1]), complex)
        b[dev.bL] = B
        Tof = lambda x: float(np.real(np.einsum("ic,ij,jc->", x[dev.bR].conj(), GR, x[dev.bR])))
        t = time.time()
        r.update(E=E, channels=int(B.shape[1]), T_ms=Tof(c.prolong(c.Ac(E, SL, SR).solve(c.restrict(b)))),
                 t_ms=time.time() - t, t_sigma=ts)
        if a.exact and "T_exact" not in r:
            t = time.time()
            cs = CSolver(dev.A(E, SL, SR))
            r.update(T_exact=Tof(cs.solve(b)), res_exact=cs.res, t_exact=time.time() - t)
            cs.free()
        if r.get("T_exact", 0) > 0:
            r["err"] = (r["T_ms"] - r["T_exact"]) / r["T_exact"]
        log(f"E = Ec {dE:+.3f}: {r['channels']} channels, T_MS {r['T_ms']:.5f}"
            + (f", T_exact {r['T_exact']:.5f} ({100 * r['err']:+.4f}%)" if "err" in r else "")
            + f"; sigma {ts:.0f} s, MS {r['t_ms']:.1f} s" + (f", exact {r['t_exact']:.0f} s" if "t_exact" in r else ""))
        tr["E"] = dict(sorted(res.items(), key=lambda kv: float(kv[0])))
        res = tr["E"]
        json.dump(tr, open(ftr, "w"), indent=1)
        plot(ftr.replace(".json", ".png"), res, f"{meta['name']}: mode space N {sum(info['N'])} vs exact")
    errs = [abs(r["err"]) for r in res.values() if "err" in r]
    if errs:
        log(f"max |T_MS - T| / T = {100 * max(errs):.4f}% over {len(errs)} energies")
    log(f"done, {time.time() - t0:.0f} s -> {ftr}")


if __name__ == "__main__":
    main()
