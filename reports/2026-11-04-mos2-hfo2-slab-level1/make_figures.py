"""Slab level 1 (local rho per slab, CAP+OBC embedding, 24-pt contour, window Ec - 0.15 ... + 0.6 eV) on the
2808-atom MoS2/HfO2 device: T and |T_MS1 - T| / T vs E for the local-rho cuts 1e-5 ... 1e-10 (N1 in the legend). MS1 only: the level-2
cut is not tuned yet.
    uv run python reports/2026-11-04-mos2-hfo2-slab-level1/make_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CUTS = ["1e-5", "1e-6", "1e-7", "1e-8", "1e-9", "1e-10"]
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]      # sequential blue, light -> dark
NTOT = 22848


def load(t, level):
    d = json.load(open(os.path.join(ROOT, f"results_ms/mos2_hfo2_slabl1_tobc24_t{t}/results_mu-0.15_0.6.json")))
    E = d["E"]
    k = sorted(E, key=float)
    dE = np.array([float(x) for x in k])
    T = np.array([E[x][f"T_MS{level}"] for x in k])
    Tx = np.array([E[x]["T_exact"] for x in k])
    return dE, T, Tx, sum(d["N1"]), sum(d["N2"])


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)


for level in (1,):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 8.0), sharex=True, height_ratios=[1, 1.15])
    fig.patch.set_facecolor(SURF)
    for a in (a1, a2):
        style(a)
    dE, _, Tx, _, _ = load(CUTS[0], level)
    a1.plot(dE, Tx, color=INK, lw=2.2, label="exact (N 22 848)", zorder=2)
    for t, c in zip(CUTS, RAMP):
        dE, T, Tx, N1, N2 = load(t, level)
        N = N1 if level == 1 else N2
        lab = (f"cut {t}: N1 {N1} ({100 * N1 / NTOT:.1f}%)" if level == 1 else
               f"cut {t}: N1 {N1}, N2 {N2} ({100 * N2 / NTOT:.1f}%)")
        a1.plot(dE, T, "o", ms=3.2, color=c, mec="none", zorder=3, label=lab)
        e = np.abs(T - Tx) / Tx
        a2.semilogy(dE, e, "-o", ms=3.2, lw=1.3, color=c, mec="none", label=lab)
        print(f"MS{level} {t}: N {N} median {np.median(e):.1e} max {e.max():.1e}")
    a1.set_ylabel("transmission T", color=INK)
    a1.set_title(f"MoS2/HfO2 (2808 atoms), slab level 1{' + level 2' if level == 2 else ''}: MS{level} vs exact",
                 loc="left", fontsize=11, color=INK)
    a1.legend(frameon=False, fontsize=8, loc="upper left")
    a2.set_ylabel(f"|T_MS{level} − T| / T", color=INK)
    a2.set_xlabel("E − Ec (eV)", color=INK)
    a2.text(0.99, 0.98, "colours as in the legend above", transform=a2.transAxes, ha="right", va="top",
            fontsize=8, color=MUTED)
    a2.set_ylim(1e-9, 2)
    fig.tight_layout()
    f = os.path.join(OUT, f"slab_level1_MS{level}_T_and_error.png")
    fig.savefig(f, dpi=150, facecolor=SURF)
    plt.close(fig)
    print("->", f)


# error summary vs basis size: median and max over the 51 energies (MS1 only; level 2 not tuned yet)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
fig.patch.set_facecolor(SURF)
style(ax)
for level, c, m in ((1, "#2a78d6", "o"),):
    rows = [load(t, level) for t in CUTS]
    N = np.array([r[3] if level == 1 else r[4] for r in rows])
    med = np.array([np.median(np.abs(r[1] - r[2]) / r[2]) for r in rows])
    mx = np.array([np.max(np.abs(r[1] - r[2]) / r[2]) for r in rows])
    Nx = np.array([r[3] for r in rows])
    ax.semilogy(Nx, med, "-" + m, color=c, ms=6, lw=1.6, label=f"MS{level} median")
    ax.semilogy(Nx, mx, "--" + m, color=c, ms=6, lw=1.2, mfc=SURF, label=f"MS{level} max")
for t, r in zip(CUTS, [load(t, 1) for t in CUTS]):
    ax.annotate(f"cut {t}", (r[3], 0.03), xycoords=("data", "axes fraction"), fontsize=7, color=MUTED, ha="center")
ax.set_xlim(800, 3900); ax.set_ylim(1e-7, 0.5)
ax.set_xlabel("N1 (level-1 basis size; local-rho cut annotated)", color=INK)
ax.set_ylabel("|T_MS − T| / T over Ec + 0 … 0.5 eV", color=INK)
ax.set_title("error vs level-1 basis size (N = 22 848)", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
f = os.path.join(OUT, "slab_level1_error_vs_N1.png")
fig.savefig(f, dpi=150, facecolor=SURF); plt.close(fig)
print("->", f)
