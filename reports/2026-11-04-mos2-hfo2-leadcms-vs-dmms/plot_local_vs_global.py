"""Occupation spectrum per slab: level 1 (local rho, slab block + 12 A CAP+OBC embedding) vs level 2 (global rho of the
whole open device, RGF), same window Ec - 0.15 ... + 0.6 eV, 24 contour points, both Euclidean eigenvalues of the slab
block. Level 2 from results_ms/mos2_hfo2_l2direct/rho_0.3543_1.1043_24.npz (eigenvalues cached next to this script)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID, SURF, CUT = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a29b"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
fc = os.path.join(OUT, "global_rho_eigvals.npz")
if os.path.exists(fc):
    g = dict(np.load(fc))
else:
    z = np.load(os.path.join(ROOT, "results_ms/mos2_hfo2_l2direct/rho_0.3543_1.1043_24.npz"))
    g = {}
    for s in range(4):
        R = z[f"r{s}"]
        g[f"l{s}"] = np.linalg.eigvalsh(0.5 * (R + R.conj().T))[::-1]
        print("slab", s, "global trace", g[f"l{s}"].sum())
    np.savez(fc, **g)
fig, axs = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True, sharey=True)
for s, ax in enumerate(axs.ravel()):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    lo = np.load(os.path.join(ROOT, f"results_ms/mos2_hfo2_slabl1_tobc24/tiles_obc_1e-10/tile_obc_{s:03d}.npz"))["s"]
    gl = g[f"l{s}"]
    for lam, ls, lab in ((lo, "-", "level 1 (local ρ)"), (gl, "--", "level 2 (global ρ)")):
        i = np.arange(1, len(lam) + 1); ok = lam > 1e-15
        ax.semilogy(i[ok], lam[ok], ls, color=COL[s], lw=2, label=f"{lab}, tr {lam.sum():.0f}")
    for c in (1e-6, 1e-10):
        ax.axhline(c, color=CUT, lw=0.8, ls=(0, (4, 3)), zorder=1)
        nl, ng = int((lo > c).sum()), int((gl > c).sum())
        ax.text(1990, c * 1.4, f"cut {c:.0e}: local {nl}, global {ng}", fontsize=7.5, color=MUTED, ha="right")
    ax.set_title(f"slab {s} ({len(lo)} orbitals)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.86))
    ax.set_xlim(0, 2000); ax.set_ylim(1e-15, 10)
for ax in axs[1]:
    ax.set_xlabel("eigenvalue index i (sorted)")
for ax in axs[:, 0]:
    ax.set_ylabel("occupation λᵢ")
fig.suptitle("MoS2/HfO2: occupation spectrum of the slab blocks, local vs global density matrix "
             "(window Ec − 0.15 … + 0.6 eV, 24 contour points)", x=0.02, ha="left", fontsize=10.5, color=INK)
fig.tight_layout()
f = os.path.join(OUT, "fig10_local_vs_global_spectrum.png")
fig.savefig(f, dpi=150, facecolor=SURF)
print("->", f)
