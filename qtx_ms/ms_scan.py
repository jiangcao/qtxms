"""Mode-space transmission of a saved basis on a fine energy grid, against the exact T.

The basis is read from the per-slab mode files written by run_l2_clean (<modes>_slab<s>.npy, slab-local orbitals x
modes, orthonormal columns). Per energy: T_MS from x = U A_c^-1 U^H b (Galerkin, exact Sigma projected) and the exact T
(sparse direct solve) unless it is already in the exact results file (exact_T --json layout; new values are merged
into it). Energies are visited coarse-first (multiples of --first, then the rest), so a partial run covers the window.

    python -m qtx_ms.ms_scan --inputs inputs/mos2_hfo2 \\
        --modes results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead --dE 0:0.5:0.01
"""
import argparse, json, os, time
import numpy as np

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb
from qtx_ms.run_l2_clean import Coarse

log = mb.log


def grid(spec, first):
    lo, hi, st = (float(x) for x in spec.split(":"))
    dE = np.round(np.arange(lo, hi + 0.5 * st, st), 6)
    coarse = np.isclose(np.round(dE / first) * first, dE)
    return list(dE[coarse]) + list(dE[~coarse])


def plot(fpng, res, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = sorted(res["E"], key=float)
    d = np.array([float(x) for x in k])
    Tms = np.array([res["E"][x]["T_ms"] for x in k])
    Tex = np.array([res["E"][x].get("T_exact", np.nan) for x in k])
    Rms = np.array([res["E"][x].get("res_ms", np.nan) for x in k])
    Rex = np.array([res["E"][x].get("res_exact", np.nan) for x in k])
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7, 8.5), sharex=True, height_ratios=[2, 1, 1])
    a1.plot(d, Tex, "k-", lw=1.5, label="exact")
    a1.plot(d, Tms, "o", ms=3.5, color="#d9622b", label=f"mode space, N {res['N']}")
    a1.set_ylabel("T")
    a1.legend(frameon=False)
    a1.set_title(title, fontsize=10)
    ok = np.isfinite(Tex) & (Tex > 1e-8)
    a2.semilogy(d[ok], np.abs(Tms[ok] - Tex[ok]) / Tex[ok], "o-", ms=3, lw=0.8, color="#2b6cd9")
    a2.set_ylabel("|T_MS - T| / T")
    if np.isfinite(Rms).any():
        a3.semilogy(d, Rms, "o-", ms=3, lw=0.8, color="#d9622b", label="mode space |b - A x_MS| / |b|")
    if np.isfinite(Rex).any():
        a3.semilogy(d, Rex, "s-", ms=3, lw=0.8, color="k", label="exact solve")
    a3.set_ylabel("relative residual")
    a3.legend(frameon=False, fontsize=8)
    a3.set_xlabel("E - Ec (eV)")
    for a in (a1, a2, a3):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fpng, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--modes", required=True, help="mode file prefix: <prefix>_slab<s>.npy for every slab")
    p.add_argument("--dE", default="0:0.5:0.01", help="lo:hi:step, energies - Ec [eV]")
    p.add_argument("--first", type=float, default=0.05, help="visit multiples of this first")
    p.add_argument("--Ec", type=float, help="reference edge for --dE (default Ec_lead of the inputs)")
    p.add_argument("--exact", help="exact results json, read and extended (default results_ms/exact/<inputs name>.json)")
    p.add_argument("--no-exact", action="store_true", help="only T_MS (use exact values already in the file)")
    p.add_argument("--out", help="results json (default <modes>_scan.json; .png next to it)")
    p.add_argument("--sigma-cache", help="lead self-energy cache (default results_ms/sigma/<inputs name>; results_ms/sigma_qtx/<name> for QuaTrEx Sigma from qtx_ms.prep.quatrex_obc)")
    p.add_argument("--obc", default="auto", choices=["auto", "qttools", "folded"], help="lead OBC: qttools (QuaTrEx spectral OBC, real E; auto falls back to the folded-lead decimation when it is not available) or mgms")
    a = p.parse_args()

    name = os.path.basename(os.path.normpath(a.inputs))
    inp = read_inputs(a.inputs)
    Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
    dev = mb.Device(inp, obc=a.obc)
    dev.sigma_cache = a.sigma_cache or os.path.join("results_ms", "sigma", name)
    qtag = "_qtt" if dev.uses_qttools(Ec + 1e-6j) else ""        # Sigma of another OBC: own exact / scan files
    V = [np.load(f"{a.modes}_slab{s}.npy") for s in range(len(dev.ns))]
    assert [v.shape[0] for v in V] == list(dev.ns), f"{a.modes}: slab sizes {[v.shape[0] for v in V]} != {dev.ns}"
    c = Coarse(dev, V)
    fex = a.exact or os.path.join("results_ms", "exact", name + qtag + ".json")
    fout = a.out or f"{a.modes}_scan{qtag}.json"
    ex = json.load(open(fex)) if os.path.exists(fex) else {"Ec": Ec, "E": {}}
    res = json.load(open(fout)) if os.path.exists(fout) else {"E": {}}
    for f, z in ((fex, ex), (fout, res)):
        if z.get("Ec", Ec) != Ec:
            raise SystemExit(f"{f}: written with test edge {z['Ec']}, this run uses {Ec}")
    ex["Ec"] = res["Ec"] = Ec
    res.update(modes=a.modes, N=c.N, N_slab=[v.shape[1] for v in V])
    log(f"{name}: device {dev.n} orbitals, slabs {dev.ns}; OBC {'qttools' if qtag else 'folded lead'}; "
        f"basis {a.modes}: N {res['N_slab']}, total {c.N}; "
        f"Ec {Ec:.4f} eV")

    for dE in grid(a.dE, a.first):
        key = f"{dE:.4f}"
        E = Ec + dE
        r = res["E"].get(key, {})
        t = time.time()
        SL, SR = dev.sigma(E + 1e-6j)
        ts = time.time() - t
        B = dev.inject(SL)
        GR = 1j * (SR - SR.conj().T)
        b = np.zeros((dev.n, B.shape[1]), complex)
        b[dev.bL] = B
        Tof = lambda x: float(np.real(np.einsum("ic,ij,jc->", x[dev.bR].conj(), GR, x[dev.bR])))
        t = time.time()
        x = c.prolong(c.Ac(E, SL, SR).solve(c.restrict(b)))
        r.update(E=E, channels=int(B.shape[1]), T_ms=Tof(x), t_ms=time.time() - t, t_sigma=ts)
        r["res_ms"] = float(np.linalg.norm(b - dev.A(E, SL, SR) @ x) / np.linalg.norm(b))
        if "T_exact" in ex["E"].get(key, {}):
            r["T_exact"] = ex["E"][key]["T_exact"]
            if "res_exact" in ex["E"][key]:
                r["res_exact"] = ex["E"][key]["res_exact"]
        elif not a.no_exact:
            t = time.time()
            cs = mb.CSolver(dev.A(E, SL, SR))
            r.update(T_exact=Tof(cs.solve(b)), t_exact=time.time() - t, res_exact=cs.res)
            ex["E"][key] = dict(E=E, channels=r["channels"], T_exact=r["T_exact"], res_exact=cs.res,
                                source="ms_scan")
            cs.free()
            ex["E"] = dict(sorted(ex["E"].items(), key=lambda kv: float(kv[0])))
            json.dump(ex, open(fex, "w"), indent=1)
        if "T_exact" in r and r["T_exact"] > 0:
            r["err"] = (r["T_ms"] - r["T_exact"]) / r["T_exact"]
        log(f"E = Ec {dE:+.3f}: {r['channels']} channels, T_MS {r['T_ms']:.5f} (residual {r['res_ms']:.1e})"
            + (f", T_exact {r['T_exact']:.5f} ({100 * r['err']:+.4f}%)" if "err" in r else "")
            + f"; sigma {ts:.0f} s, MS {r['t_ms']:.1f} s" + (f", exact {r['t_exact']:.0f} s" if "t_exact" in r else ""))
        res["E"][key] = r
        res["E"] = dict(sorted(res["E"].items(), key=lambda kv: float(kv[0])))
        json.dump(res, open(fout, "w"), indent=1)
        plot(fout.replace(".json", ".png"), res, f"{name}: {os.path.basename(a.modes)} (N {c.N}) vs exact")
    log(f"done -> {fout}")


if __name__ == "__main__":
    main()
