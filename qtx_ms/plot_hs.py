"""Original H, S of the device next to the mode-space H, S (direct level 2: U = diag(V_s), V_s from the slab
density matrix cached by run_ms --direct-l2).

Row 1: |H|, |S| of the device (max |entry| per pixel, log scale), slabs 0-1 zoomed, eigenvalues of S_00.
Row 2: the same for H_MS = U^H H U, S_MS = U^H S U; last panel: eigenvalues of the isolated slab 0,
(H_00, S_00) vs (V^H H_00 V, V^H S_00 V), near the band edges.

    python -m mgms.plot_hs --inputs inputs/mos2_20nm --rho results_ms/mos2_20nm_l2direct/rho_....npz --l2-thr 1e-7
"""
import argparse, os
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from mgms.prep.inputs import read_inputs
from mgms import msbasis as mb

INK, MUTED, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834"
CMAP = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def binned(M, npx):
    """npx x npx image of max |M_ij| per pixel (NaN where empty)."""
    M = sp.coo_matrix(M)
    n = M.shape[0]
    f = npx / n
    img = np.zeros((npx, npx))
    np.maximum.at(img, ((M.row * f).astype(int), (M.col * f).astype(int)), np.abs(M.data))
    img[img == 0] = np.nan
    return img


def show(ax, M, npx, title, bounds=None, vmin=-6, vmax=1):
    n = M.shape[0]
    npx = min(npx, n)
    im = ax.imshow(np.log10(binned(M, npx)), cmap=CMAP, vmin=vmin, vmax=vmax, extent=(0, n, n, 0),
                   interpolation="nearest")
    if bounds is not None:
        for b in bounds[1:-1]:
            ax.axhline(b, color=GRID, lw=0.5); ax.axvline(b, color=GRID, lw=0.5)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color(GRID)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--rho", required=True)
    p.add_argument("--l2-thr", type=float, default=1e-7)
    p.add_argument("--out", default="results_ms/figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(os.path.normpath(a.inputs))
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp)
    Ec, Ev = inp["meta"]["edges"]["Ec_lead"], inp["meta"]["edges"]["Ev_lead"]
    z = np.load(a.rho)
    V = mb.level2_basis([z[f"r{s}"] for s in range(len(dev.ns))], a.l2_thr)
    U = sp.block_diag([sp.csr_matrix(v) for v in V], format="csr")
    Hm = (U.conj().T @ dev.H @ U).toarray(); Sm = (U.conj().T @ dev.S @ U).toarray()
    Hm[np.abs(Hm) < 1e-12] = 0; Sm[np.abs(Sm) < 1e-12] = 0
    N2 = [v.shape[1] for v in V]
    bo, bm = dev.sb, np.concatenate([[0], np.cumsum(N2)])
    b0 = dev.sb[2]; m0 = bm[2]
    S00 = dev.S[:dev.sb[1], :dev.sb[1]].toarray(); H00 = dev.H[:dev.sb[1], :dev.sb[1]].toarray()
    v0 = V[0]
    hs, ss = v0.conj().T @ H00 @ v0, v0.conj().T @ S00 @ v0

    fig, axs = plt.subplots(2, 4, figsize=(17, 8.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.1]))
    fig.patch.set_facecolor(SURF)
    show(axs[0, 0], dev.H, 700, f"H, {dev.n} orbitals ({len(dev.ns)} slabs)", bo)
    show(axs[0, 1], dev.S, 700, "S", bo)
    show(axs[0, 2], dev.H[:b0, :b0], 500, "H, slabs 0-1", bo[:3])
    show(axs[1, 0], sp.csr_matrix(Hm), 700, f"H_MS = U^H H U, N = {U.shape[1]}", bm)
    show(axs[1, 1], sp.csr_matrix(Sm), 700, "S_MS = U^H S U", bm)
    im = show(axs[1, 2], sp.csr_matrix(Hm[:m0, :m0]), 500, "H_MS, slabs 0-1", bm[:3])
    cb = fig.colorbar(im, ax=axs[:, 2], shrink=0.6, pad=0.02)
    cb.set_label("log10 |entry| (eV for H)", fontsize=8, color=MUTED); cb.ax.tick_params(labelsize=7, colors=MUTED)

    ax = axs[0, 3]
    es, em = np.linalg.eigvalsh(S00), np.linalg.eigvalsh(ss)
    ax.semilogy(np.arange(len(es)) / len(es), es, color=INK, lw=1.4, label=f"S_00, {len(es)} orbitals")
    ax.semilogy(np.arange(len(em)) / len(em), em, color=BLUE, lw=0, marker="o", ms=3, label=f"V^H S_00 V, {len(em)} modes")
    ax.set_title(f"eigenvalues of the slab-0 overlap (cond {es[-1] / es[0]:.1e} vs {em[-1] / em[0]:.1e})",
                 fontsize=9, color=INK, loc="left")
    ax.set_xlabel("index / size", fontsize=8, color=MUTED)
    ax.legend(fontsize=8, frameon=False)
    ax = axs[1, 3]
    eo = sla.eigh(H00, S00, eigvals_only=True); ey = sla.eigh(hs, ss, eigvals_only=True)
    lo, hi = Ev - 0.6, Ec + 1.2
    for y, ev_, col, lab in ((1, eo, INK, f"orbitals ({len(eo)})"), (0, ey, BLUE, f"mode space ({len(ey)})")):
        e_ = ev_[(ev_ > lo) & (ev_ < hi)]
        ax.vlines(e_, y - 0.35, y + 0.35, color=col, lw=0.8)
        ax.text(lo + 0.02, y + 0.42, lab, color=MUTED, fontsize=8, va="bottom")
        gap = ((ev_ > Ev + 0.05) & (ev_ < Ec - 0.05)).sum()
        ax.text(hi - 0.02, y + 0.42, f"{gap} in the gap", color=col, fontsize=8, va="bottom", ha="right")
    ax.set_ylim(-0.6, 1.9); ax.set_yticks([]); ax.set_xlim(lo, hi)
    for v, lab in ((Ec, "Ec"), (Ev, "Ev")):
        ax.axvline(v, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.text(v, 1.85, f" {lab}", color=MUTED, fontsize=8, va="top")
    ax.axvspan(Ec, Ec + 0.395, color=ORANGE, alpha=0.08, lw=0)
    ax.set_title("eigenvalues of the isolated slab 0 (level-2 window shaded)", fontsize=9, color=INK, loc="left")
    ax.set_xlabel("E (eV)", fontsize=8, color=MUTED)
    for ax in axs[:, 3]:
        ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.6); ax.tick_params(colors=MUTED, labelsize=7)
        for s in ax.spines.values():
            s.set_color(GRID)
    fig.suptitle(f"{name}: original vs mode-space H, S (direct level 2, thr {a.l2_thr:g})", fontsize=11,
                 color=INK, x=0.01, ha="left")
    f = os.path.join(a.out, f"{name}_hs_l2thr{a.l2_thr:g}.png")
    fig.savefig(f, dpi=130, facecolor=SURF, bbox_inches="tight")
    print("->", f)


if __name__ == "__main__":
    main()
