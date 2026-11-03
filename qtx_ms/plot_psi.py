"""Left-injected scattering states of QuaTrEx and mgms (psi_<E>.npz of qtx_ms.prep.compare_qtbm) in the device.

Density per atom: Mulliken n_a = sum_cols sum_{i in a} Re(conj(psi_i) (S psi)_i) over the device orbitals (S of the
device; lead orbitals are not in psi_m). Summed over columns this is diag of psi psi^H = G B B^H G^H, independent of the
column basis (QuaTrEx Bloch modes vs mgms Gamma_L eigenchannels).
Row 1: side view (x-z), row 2: top view (x-y), QuaTrEx | mgms, shared log colour scale.
Row 3: density per lead cell along x (totals, QuaTrEx columns) | relative difference mgms - QuaTrEx per cell.
With --modes: exact (qtx_ms psi_m) vs mode space x_MS of that basis instead (plot_residual.ms_solution: same Sigma cache
and injection).

    python -m qtx_ms.plot_psi --inputs inputs/mos2_hfo2 --psi results_ms/qtbm_cmp/mos2_hfo2/psi_+0.55434897.npz
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

from qtx_ms.prep.inputs import read_inputs

INK, MUTED, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834"
CMAP = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def mulliken(psi, S, atom, natoms):
    """Per-orbital Re(conj(psi) S psi) summed over columns, then per atom; also per column (atoms x cols)."""
    q = np.real(psi.conj() * (S @ psi))
    per_col = np.zeros((natoms, psi.shape[1]))
    np.add.at(per_col, atom, q)
    return per_col.sum(1), per_col


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--psi", required=True, help="psi_<E>.npz from qtx_ms.prep.compare_qtbm compare")
    p.add_argument("--out", help="png (default next to --psi)")
    p.add_argument("--dyn", type=float, default=1e-5, help="colour range below the largest atom density")
    p.add_argument("--modes", help="mode file prefix: compare exact psi_m with this mode-space solution")
    p.add_argument("--sigma-cache", help="with --modes: Sigma cache the psi file was computed with")
    p.add_argument("--eta", type=float, default=0.0, help="with --modes: Im z of that Sigma")
    a = p.parse_args()
    inp = read_inputs(a.inputs)
    c = np.load(a.psi)
    dev = c["dev"]
    E = float(c["E"])
    Ec = inp["meta"]["edges"]["Ec_lead"]
    xyz, labels, norb, blocks = inp["xyz"], inp["labels"], inp["norb"], inp["blocks"]
    natoms = len(xyz)
    atom_of = np.repeat(np.arange(natoms), norb)
    atom = atom_of[dev]
    Sd = inp["S"].tocsr()[dev][:, dev]
    nch = c["psi_m"].shape[1]
    if a.modes:                                                   # reference = exact, other = mode space
        from qtx_ms.plot_residual import ms_solution
        _, _, _, _, x, cs, T = ms_solution(inp, c, a.modes, a.sigma_cache, a.eta)
        nq, nq_col = mulliken(c["psi_m"], Sd, atom, natoms)
        nm, _ = mulliken(x, Sd, atom, natoms)
        tags = (f"exact ({nch} injected channels)", f"mode space (N {cs.N})")
        short, col_lab = ("exact", "mode space"), "exact, single channel"
        sub = f"; T exact {T(c['psi_m']):.6f}, mode space {T(x):.6f}"
    else:
        nq, nq_col = mulliken(c["psi_q"][dev], Sd, atom, natoms)
        nm, _ = mulliken(c["psi_m"], Sd, atom, natoms)
        tags = (f"QuaTrEx ({c['psi_q'].shape[1]} Bloch modes)", f"mgms ({nch} Γ_L channels)")
        short, col_lab, sub = ("QuaTrEx", "qtx_ms"), "QuaTrEx, single mode", ""
    on = np.isin(np.arange(natoms), atom)                         # device atoms
    vmax = max(nq.max(), nm.max())
    norm = LogNorm(vmax * a.dyn, vmax)
    oxide = np.isin(labels, ["Hf", "O"])
    for tag, n in zip(short, (nq, nm)):
        print(f"{tag}: total {n.sum():.4e}, on Hf/O {n[oxide].sum() / n.sum():.2%}, "
              f"negative atoms {int((n[on] < 0).sum())}")
    d = np.abs(nm - nq)
    print(f"max |n_{short[1]} - n_{short[0]}| / max n {d.max() / vmax:.2e}, sum |dn| / sum n {d.sum() / nq.sum():.2e}")

    edges = [xyz[blocks == b, 0].min() for b in np.unique(blocks)[1:]]
    fig, ax = plt.subplots(3, 2, figsize=(13, 10.5), facecolor=SURF,
                           gridspec_kw=dict(height_ratios=[1, 1, 1.05]), constrained_layout=True)
    for j, (tag, n) in enumerate(zip(tags, (nq, nm))):
        for i, (k, lab) in enumerate(((2, "z [Å]"), (1, "y [Å]"))):
            A = ax[i, j]
            A.scatter(xyz[:, 0], xyz[:, k], s=3, c=GRID, lw=0)
            o = np.where(on)[0]
            o = o[np.argsort(n[o])]                                 # largest on top
            sc = A.scatter(xyz[o, 0], xyz[o, k], s=9, c=np.clip(n[o], norm.vmin, None), cmap=CMAP, norm=norm, lw=0)
            for x in edges:
                A.axvline(x - 0.8, color=MUTED, lw=0.6, ls=":")
            A.set_aspect("equal")
            A.set_facecolor(SURF)
            A.set_xlabel("x (transport) [Å]", color=MUTED)
            A.set_ylabel(lab, color=MUTED)
            A.set_title(f"{tag}, {'side' if k == 2 else 'top'} view", color=INK, loc="left", fontsize=10)
    fig.colorbar(sc, ax=ax[:2, :], shrink=0.6, label="Mulliken density per atom [1/eV]")

    # profiles per lead cell along x
    L = inp["meta"]["source"]["cell_length"]
    x0 = xyz[:, 0].min() - 0.5 * L / 5
    cell = np.floor((xyz[:, 0] - x0) / L).astype(int)
    nc = cell.max() + 1
    xc = x0 + (np.arange(nc) + 0.5) * L
    prof = lambda n: np.bincount(cell[on], n[on], minlength=nc)
    Pq, Pm = prof(nq), prof(nm)
    keep = np.bincount(cell[on], minlength=nc) > 0
    A = ax[2, 0]
    A.semilogy(xc[keep], Pq[keep], color=BLUE, lw=2, label=f"{short[0]}, total")
    A.semilogy(xc[keep], Pm[keep], color=ORANGE, lw=2, ls="--", label=f"{short[1]}, total")
    for m in range(nq_col.shape[1]):
        A.semilogy(xc[keep], prof(nq_col[:, m])[keep], color=BLUE, lw=0.8, alpha=0.6,
                   label=col_lab if m == 0 else None)
    A.set_ylabel("density per cell [1/eV]", color=MUTED)
    A.legend(frameon=False, fontsize=9)
    A = ax[2, 1]
    A.plot(xc[keep], (Pm[keep] - Pq[keep]) / Pq[keep], color=INK, lw=2)
    A.axhline(0, color=MUTED, lw=0.6)
    A.set_ylabel(f"({short[1]} − {short[0]}) / {short[0]} per cell", color=MUTED)
    for A in ax[2]:
        A.set_xlabel("x (transport) [Å]", color=MUTED)
        A.set_facecolor(SURF)
        A.grid(color=GRID, lw=0.6)
        for x in edges:
            A.axvline(x - 0.8, color=MUTED, lw=0.6, ls=":")
        for s in A.spines.values():
            s.set_color(GRID)
    fig.suptitle(f"Left-injected states at E = Ec + {E - Ec:.3f} eV{sub}", color=INK, x=0.01, ha="left")
    out = a.out or os.path.splitext(a.psi)[0] + ("_ms.png" if a.modes else ".png")
    fig.savefig(out, dpi=130, facecolor=SURF)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
