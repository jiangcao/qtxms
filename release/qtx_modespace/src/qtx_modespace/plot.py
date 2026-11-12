"""Post-processing: plots of finished runs (needs matplotlib, the `plot` extra). The run itself (python -m
qtx_modespace) writes only numbers (transmission.json, basis.json, modes); nothing is plotted there.

    python -m qtx_modespace.plot OUT [OUT ...]        # OUT = a run's --out folder or its transmission.json

For every run: transmission.png next to transmission.json (T_MS and, where computed, T_exact and the relative error
against E - Ec). Runs without a transmission.json are skipped with a note.
"""
import argparse, json, os, sys
import numpy as np


def plot(fpng, res, title):
    """T_MS (and T_exact, relative error) vs E - Ec from the "E" dict of a transmission.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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


def plot_run(path, log=print):
    """transmission.png for one run folder / transmission.json; returns the png path or None."""
    fj = os.path.join(path, "transmission.json") if os.path.isdir(path) else path
    if not os.path.exists(fj):
        log(f"{path}: no transmission.json, skipped")
        return None
    tr = json.load(open(fj))
    res = {k: r for k, r in tr.get("E", {}).items() if "T_ms" in r}
    if not res:
        log(f"{fj}: no energies, skipped")
        return None
    name = tr.get("name") or os.path.basename(os.path.dirname(os.path.abspath(fj)))
    fpng = os.path.splitext(fj)[0] + ".png"
    N = tr.get("N")
    ex = any("T_exact" in r for r in res.values())
    plot(fpng, res, f"{name}: mode space" + (f" N {sum(N)}" if N else "") + (" vs exact" if ex else ""))
    log(f"{fj}: {len(res)} energies -> {fpng}")
    return fpng


def main(argv=None):
    p = argparse.ArgumentParser(prog="qtx-modespace-plot", description=__doc__.split("\n\n")[0])
    p.add_argument("out", nargs="+", help="run folders (--out of python -m qtx_modespace) or transmission.json files")
    a = p.parse_args(argv)
    done = [plot_run(o) for o in a.out]
    return 0 if any(done) else 1


if __name__ == "__main__":
    sys.exit(main())
