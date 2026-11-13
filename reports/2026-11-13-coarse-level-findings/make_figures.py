"""Figures for reports/2026-11-13-coarse-level-findings/summary.md.

Run from the repo root:  python reports/2026-11-13-coarse-level-findings/make_figures.py
Fig. 2 parses results_ms/gpu/slow/sweep_recycle_2808a.log; the others use numbers copied from the logs
listed in summary.md (slow_2808a*.log, slow_6nm_t015_m128.log / _tov6.log, pou_ms_6nm_d4.log, tile_iter_2808a_*.log).
"""
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]   # categorical order, fixed
MK = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


# ---- Fig. 1: 2808a, slow-mode enrichment rounds vs mode-space bases ---------------------------------------
# its = max over columns; round 0 = the base alone (two smoothing steps for "none")
def fig1():
    E = ["Ec+0.05", "Ec+0.2", "Ec+0.3"]
    data = {  # base: per energy list of its per round
        "none (slow modes only)": [[41, 15, 14, 13], [69, 23, 17, 15], [107, 37, 23, 18]],
        "slab level 1, cut 1e-2 (N 454)": [[30, 16], [49, 21], [79, 31]],
        "slab level 1, cut 1e-8 (N 2373)": [[29, 20], [47, 25], [150, 38]],
        "lead CMS (N 980)": [[33, 23], [50, 32], [80, 46]],
    }
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for k, ax in enumerate(axs):
        for i, (lab, d) in enumerate(data.items()):
            y = d[k]
            ax.plot(range(len(y)), y, marker=MK[i], color=C[i], label=lab)
        ax.set_title(E[k])
        ax.set_xlabel("enrichment round (64 slow modes each)")
        ax.set_xticks(range(4))
        ax.set_yscale("log")
        ax.set_yticks([10, 20, 50, 100, 150])
        ax.set_yticklabels(["10", "20", "50", "100", "150"])
    axs[0].set_ylabel("FGMRES iterations (max over columns)")
    axs[2].annotate("stall", (0, 150), (0.35, 120), color=INK2, fontsize=9)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(" ", fontsize=4)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig1_slow_modes_2808a.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "fig1_slow_modes_2808a.png")


# ---- Fig. 2: energy sweep with recycled coarse spaces (parsed) ---------------------------------------------
def fig2():
    log = (ROOT / "results_ms/gpu/slow/sweep_recycle_2808a.log").read_text().splitlines()
    its = defaultdict(dict)
    e = None
    for line in log:
        m = re.match(r"Ec \+([\d.]+):", line)
        if m:
            e = float(m.group(1))
            continue
        m = re.match(r"\s+(\w+)\s+N\s+\d+: .*its (\d+)\.\.(\d+)", line)
        if m and e is not None:
            its[m.group(1)][e] = int(m.group(3))
    show = [("none", "no coarse level"), ("fixed", "slow modes fixed at Ec+0"),
            ("sol", "fixed + previous solutions"), ("ritz", "fixed + recycled Ritz vectors"),
            ("rebuild", "slow modes rebuilt at every energy")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    for i, (key, lab) in enumerate(show):
        es = sorted(its[key])
        ax.plot(es, [its[key][x] for x in es], marker=MK[i], markersize=4, color=C[i], label=lab)
    ax.axhline(100, color=INK2, lw=1, ls="--")
    ax.text(0.005, 102, "maxit 100 (not converged)", color=INK2, fontsize=8)
    ax.set_xlabel("E − Ec (eV)")
    ax.set_ylabel("FGMRES iterations (max over columns)")
    ax.set_title("2808a, 31 energies, 128 slow modes")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    tot = [sum(its[k].values()) for k, _ in show]
    y = np.arange(len(show))[::-1]
    ax2.barh(y, tot, color=[C[i] for i in range(len(show))], height=0.6)
    for yy, t in zip(y, tot):
        ax2.text(t + 20, yy, str(t), va="center", color=INK, fontsize=9)
    ax2.set_yticks(y)
    ax2.set_yticklabels([lab for _, lab in show], fontsize=8.5)
    ax2.set_xlabel("total iterations over the sweep")
    ax2.set_xlim(0, max(tot) * 1.2)
    ax2.grid(axis="y", visible=False)
    save(fig, "fig2_sweep_recycle_2808a.png")


# ---- Fig. 3: 6 nm, Ec+0.15, slow modes only ---------------------------------------------------------------
def fig3():
    N = [128, 256, 384, 512, 640, 768, 896, 1024]
    theta = [1.28, 1.32, 2.15, 1.00, 1.08, 0.91, 2.11, 1.61]          # largest cycle |theta| (m128 run)
    smin = [2.8e-2, 4.7e-3, 7.3e-4, 8.4e-4, 8.8e-4, 8.4e-4, 5.0e-4, 5.6e-4]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.7))
    ax = axs[0]
    ax.plot([256, 512, 768, 1024], [100, 30, 29, 31], marker="o", color=C[0], label="window shell 10.7 Å")
    ax.plot([512, 768, 1024], [36, 33, 33], marker="s", color=C[1], label="window shell 6 Å")
    ax.annotate("not converged\n(res 2.5e-6)", (256, 100), (330, 85), color=INK2, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("coarse size N (slow modes)")
    ax.set_ylabel("FGMRES iterations (max of 12 columns)")
    ax.set_title("iterations to 1e-10")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8.5)
    ax = axs[1]
    ax.plot(N, smin, marker="o", color=C[0])
    ax.set_yscale("log")
    ax.set_xlabel("coarse size N")
    ax.set_ylabel("σ_min(A_c)")
    ax.set_title("smallest coarse singular value")
    ax = axs[2]
    labs = ["exact\nstreamed LU", "slow modes\nshell 10.7 Å", "slow modes\nshell 6 Å"]
    parts = {"window LU factor": [0, 79, 55], "slow-mode build": [0, 98, 73], "FGMRES": [0, 203, 182],
             "exact solve": [150, 0, 0]}
    left = np.zeros(3)
    y = np.arange(3)[::-1]
    for i, (k, v) in enumerate(parts.items()):
        ax.barh(y, v, left=left, color=[C[1], C[2], C[0], INK2][i], height=0.55, label=k,
                edgecolor="#fcfcfb", linewidth=2)
        left += v
    for yy, t in zip(y, left):
        ax.text(t + 6, yy, f"{t:.0f} s", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlabel("wall time per energy (s), N 512")
    ax.set_xlim(0, 440)
    ax.set_title("cost vs exact (12 of 40 channels)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
    save(fig, "fig3_slow_modes_6nm.png")


# ---- Fig. 4: PoU slab basis on 6 nm, stand-alone MS T error ------------------------------------------------
def fig4():
    E = [0.02, 0.05, 0.15, 0.3]
    rows = [("block-diagonal slab L1, 1e-8 (N 38 054)", [7.3e-3, 6.6e-4, 1.4e-4, 8.7e-5]),
            ("\"hard\", 1e-8 (N 22 648)", [1.1e-1, 1.2e-3, 2.2e-3, 4.0e-6]),
            ("χ·v, 1e-8 (N 41 225)", [6.6e-2, 3.9e-5, 9.8e-5, 1.1e-5]),
            ("\"hard\", 1e-6 (N 10 592)", [176, 0.49, 1.2e-2, 4.8e-6]),
            ("χ·v, 1e-6 (N 18 633)", [5.5, 4.2e-4, 2.6e-4, 7.9e-5])]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (lab, v) in enumerate(rows):
        ax.plot(E, v, marker=MK[i], color=C[i], label=lab, ls="-" if "1e-8" in lab else "--")
    ax.set_yscale("log")
    ax.set_xticks(E)
    ax.set_xlabel("E − Ec (eV)")
    ax.set_ylabel("|T_MS − T| / T")
    ax.set_title("6 nm MoS2/HfO2, stand-alone mode space")
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, "fig4_pou_6nm.png")


# ---- Fig. 5: iterated tile bases, 2808a, old 8 tiles ------------------------------------------------------
def fig5():
    ref_err, ref_tr, ref_miss = 2.5e-4, 1.008, 6.3e-4
    it0 = (max(7.4e-3, 7.8e-3, 3.4e-3), 0.998, 2.2e-3)
    runs = {
        "neighbours in their basis": [(1.7e-2, 1.45, 1.1e-2), (2.0e-2, 3.53, 5.2e-2),
                                      (7.9e-2, 4.39, 7.0e-2)],
        "hybrid (6 Å fine shell + basis)": [(3.1e-3, 1.004, 3.3e-3), (4.6e-3, 1.040, 1.2e-2),
                                            (2.1e-1, 1.025, 8.5e-3)],
        "neighbours in basis + outer CAP": [(2.8e-4, 1.450, 2.0e-3), (3.7e-3, 3.279, 2.7e-3),
                                            (3.2e-3, 3.895, 2.0e-3)],
    }
    titles = ["max |T err| (Ec+0.05 / 0.15 / 0.2)", "max tile trace, local / exact ρ",
              "missed exact occupation"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.7))
    for k, ax in enumerate(axs):
        for i, (lab, v) in enumerate(runs.items()):
            ax.plot(range(4), [it0[k]] + [x[k] for x in v], marker=MK[i], color=C[i], label=lab)
        ax.axhline([ref_err, ref_tr, ref_miss][k], color=INK2, ls="--", lw=1.2, label="reference 12 Å CAP shell")
        ax.set_xticks(range(4))
        ax.set_xlabel("iteration (0 = 6 Å CAP shell)")
        ax.set_title(titles[k])
        if k != 1:
            ax.set_yscale("log")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig5_tile_iter_2808a.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "fig5_tile_iter_2808a.png")


# ---- Fig. 6: y2 tiles, cluster-PoU vs reference: T error and spurious coarse states -----------------------
def fig6():
    rows = [("reference 12 Å CAP (N 2055)", [9.4e-4, 1.8e-3, 4.5e-6], [15, 19, 21]),
            ("outer CAP, iteration 2 (N 2382)", [1.1e-4, 1.1e-4, 1.7e-5], [23, 24, 28]),
            ("PoU 8 Å χ·v (N 7874)", [5.3e-5, 1.5e-4, 5.1e-3], [24, 28, 28]),
            ("PoU 8 Å χρχ (N 2279)", [8.4e-4, 1.1e-4, 1.7e-3], [23, 24, 28]),
            ("PoU 15 Å χ·v (N 8300)", [1.4e-4, 1.2e-4, 2.1e-4], [33, 32, 38]),
            ("PoU 15 Å χρχ (N 2640)", [5.0e-2, 4.6e-4, 4.0e-3], [29, 31, 33])]
    E = [0.05, 0.15, 0.2]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for i, (lab, t, s) in enumerate(rows):
        axs[0].plot(E, t, marker=MK[i], color=C[i], label=lab)
        axs[1].plot(E, s, marker=MK[i], color=C[i], label=lab)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("|T_MS − T| / T")
    axs[1].set_ylabel("# singular values of A_c < 3e-2")
    for ax in axs:
        ax.set_xticks(E)
        ax.set_xlabel("E − Ec (eV)")
    axs[0].set_title("T error (2808a, 2 y strips per slab, cut 1e-6)")
    axs[1].set_title("near-zero coarse states (spurious)")
    axs[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig6_tile_pou_2808a.png")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
