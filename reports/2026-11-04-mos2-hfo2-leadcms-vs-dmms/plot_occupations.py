"""Occupation spectrum (sorted eigenvalues of the level-1 local density matrix) per slab, slab partition, CAP+OBC,
24-pt contour (results_ms/mos2_hfo2_slabl1_tobc24/tiles_obc_1e-10/tile_obc_00<s>.npz); cuts 1e-5 ... 1e-10 dashed;
the five slab-1 modes of fig7 marked."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID, SURF, CUT = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a29b"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
fig, ax = plt.subplots(figsize=(8.2, 5.2))
ax.set_facecolor(SURF)
for s in range(4):
    z = np.load(os.path.join(ROOT, f"results_ms/mos2_hfo2_slabl1_tobc24/tiles_obc_1e-10/tile_obc_{s:03d}.npz"))
    lam = z["s"]
    i = np.arange(1, len(lam) + 1)
    ok = lam > 1e-14
    ax.semilogy(i[ok], lam[ok], color=COL[s], lw=2, label=f"slab {s} ({len(lam)} orbitals, tr ρ = {lam.sum():.0f})")
    if s == 1:
        for t in (1.0, 1e-2, 1e-4, 1e-6, 1e-8):
            j = int(np.argmin(np.abs(np.log10(lam[:1159]) - np.log10(t))))
            ax.plot(j + 1, lam[j], "o", ms=7, color=COL[1], mec=SURF, mew=1.5, zorder=5)
            ax.annotate(f"#{j + 1}", (j + 1, lam[j]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
for c in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
    ax.axhline(c, color=CUT, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(2490, c * 1.25, f"cut {c:.0e}", fontsize=7.5, color=MUTED, ha="right")
ax.set_xlim(0, 2500); ax.set_ylim(1e-14, 10)
ax.set_xlabel("eigenvalue index i (sorted)"); ax.set_ylabel("occupation λᵢ")
ax.set_title("MoS2/HfO2: level-1 local density matrix per slab (CAP+OBC, 24 contour points)", loc="left", fontsize=10.5, color=INK)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp_ in ("top", "right"):
    ax.spines[sp_].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1, 0.93))
ax.text(0.01, 0.02, "dots: slab-1 modes of fig7 (occupation 1, 1e-2, 1e-4, 1e-6, 1e-8)", transform=ax.transAxes, fontsize=8, color=MUTED)
fig.tight_layout()
f = os.path.join(OUT, "fig8_occupation_vs_index.png")
fig.savefig(f, dpi=150, facecolor=SURF)
print("->", f)
