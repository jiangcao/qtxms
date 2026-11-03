"""Mode-space residual r = b - A x_MS and error e = psi - x_MS of the left-injected states in the device (companion of
plot_psi; psi from qtx_ms.prep.compare_qtbm compare, same Sigma cache and injection).

Per atom: |r|^2 and |e|^2 summed over the atom's orbitals and the injected columns (Euclidean; r lives in the dual space
of psi, so no S). Row 1: side view (x-z), row 2: top view (x-y), residual | error, own log colour scales.
Row 3: per lead cell along x: |r|^2, |e|^2 and |psi|^2 | |e|^2 / |psi|^2 per cell.

    python -m qtx_ms.plot_residual --inputs inputs/mos2_hfo2 --psi results_ms/qtbm_cmp/mos2_hfo2_eta0/psi_+0.55434897.npz \\
        --modes results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead --sigma-cache results_ms/sigma_qtx_eta0/mos2_hfo2 --eta 0
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb
from qtx_ms.run_l2_clean import Coarse
from qtx_ms.plot_psi import INK, MUTED, GRID, SURF, BLUE, ORANGE, CMAP


def ms_solution(inp, c, modes, sigma_cache, eta=0.0, tol=1e-4, obc="folded"):
    """Mode-space solution x_MS for the exact psi_m of a psi file c (compare_qtbm, or exact_T --save-psi: then its obc
    and eta are used) with the same Sigma cache and injection; returns dev, E, b, A, x, Coarse, T(.)."""
    E = float(c["E"])
    if "obc" in c:
        obc, eta = str(c["obc"]), float(c["eta"])
    dev = mb.Device(inp, obc=obc)                              # the cached Sigma and its injection, as in psi_m
    dev.sigma_cache = sigma_cache
    assert np.array_equal(dev.dev, c["dev"])
    SL, SR = dev.sigma(E + 1j * eta)
    B = dev.inject(SL, tol)
    assert B.shape[1] == c["psi_m"].shape[1], f"{B.shape[1]} channels here, {c['psi_m'].shape[1]} in the psi file"
    b = np.zeros((dev.n, B.shape[1]), complex)
    b[dev.bL] = B
    V = [np.load(f"{modes}_slab{s}.npy") for s in range(len(dev.ns))]
    cs = Coarse(dev, V)
    x = cs.prolong(cs.Ac(E, SL, SR).solve(cs.restrict(b)))
    A = dev.A(E, SL, SR)
    assert np.linalg.norm(b - A @ c["psi_m"]) < 1e-8 * np.linalg.norm(b), "psi does not solve this system"
    GR = 1j * (SR - SR.conj().T)
    T = lambda y: float(np.real(np.einsum("ic,ij,jc->", y[dev.bR].conj(), GR, y[dev.bR])))
    return dev, E, b, A, x, cs, T


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--psi", required=True, help="psi_<E>.npz from qtx_ms.prep.compare_qtbm compare (exact psi_m)")
    p.add_argument("--modes", required=True, help="mode file prefix: <prefix>_slab<s>.npy for every slab")
    p.add_argument("--sigma-cache", required=True, help="Sigma cache the psi file was computed with")
    p.add_argument("--eta", type=float, default=0.0, help="Im z of that Sigma")
    p.add_argument("--tol", type=float, default=1e-4, help="Gamma eigenvalue threshold (as compare_qtbm)")
    p.add_argument("--dyn", type=float, default=1e-6, help="colour range below the largest atom value")
    p.add_argument("--out", help="png (default <psi>_residual.png)")
    a = p.parse_args()
    inp = read_inputs(a.inputs)
    c = np.load(a.psi)
    Ec = inp["meta"]["edges"]["Ec_lead"]
    dev, E, b, A, x, cs, T = ms_solution(inp, c, a.modes, a.sigma_cache, a.eta, a.tol)
    psi = c["psi_m"]
    r, e = b - A @ x, psi - x
    print(f"N {cs.N}; |r|/|b| {np.linalg.norm(r) / np.linalg.norm(b):.3e}, |e|/|psi| "
          f"{np.linalg.norm(e) / np.linalg.norm(psi):.3e}; T exact {T(psi):.6f}, MS {T(x):.6f} "
          f"({(T(x) - T(psi)) / T(psi):+.2e})")

    xyz, norb, blocks = inp["xyz"], inp["norb"], inp["blocks"]
    natoms = len(xyz)
    atom = np.repeat(np.arange(natoms), norb)[dev.dev]
    per_atom = lambda y: np.bincount(atom, (np.abs(y) ** 2).sum(1), minlength=natoms)
    nr, ne, npsi = per_atom(r), per_atom(e), per_atom(psi)
    on = np.isin(np.arange(natoms), atom)
    edges = [xyz[blocks == k, 0].min() for k in np.unique(blocks)[1:]]

    fig, ax = plt.subplots(3, 2, figsize=(13, 10.5), facecolor=SURF,
                           gridspec_kw=dict(height_ratios=[1, 1, 1.05]), constrained_layout=True)
    for j, (tag, n, lab) in enumerate(((f"residual |b - A x_MS|²", nr, "|r|² per atom"),
                                       (f"error |ψ - x_MS|²", ne, "|e|² per atom"))):
        norm = LogNorm(n.max() * a.dyn, n.max())
        for i, (k, ylab) in enumerate(((2, "z [Å]"), (1, "y [Å]"))):
            A_ = ax[i, j]
            A_.scatter(xyz[:, 0], xyz[:, k], s=3, c=GRID, lw=0)
            o = np.where(on)[0]
            o = o[np.argsort(n[o])]
            sc = A_.scatter(xyz[o, 0], xyz[o, k], s=9, c=np.clip(n[o], norm.vmin, None), cmap=CMAP, norm=norm, lw=0)
            for xe in edges:
                A_.axvline(xe - 0.8, color=MUTED, lw=0.6, ls=":")
            A_.set_aspect("equal")
            A_.set_facecolor(SURF)
            A_.set_xlabel("x (transport) [Å]", color=MUTED)
            A_.set_ylabel(ylab, color=MUTED)
            A_.set_title(f"{tag}, {'side' if k == 2 else 'top'} view", color=INK, loc="left", fontsize=10)
        fig.colorbar(sc, ax=ax[:2, j], shrink=0.7, label=lab, location="bottom")

    L = inp["meta"]["source"]["cell_length"]
    x0 = xyz[:, 0].min() - 0.5 * L / 5
    cell = np.floor((xyz[:, 0] - x0) / L).astype(int)
    nc = cell.max() + 1
    xc = x0 + (np.arange(nc) + 0.5) * L
    prof = lambda n: np.bincount(cell[on], n[on], minlength=nc)
    keep = np.bincount(cell[on], minlength=nc) > 0
    Pr, Pe, Pp = prof(nr), prof(ne), prof(npsi)
    A_ = ax[2, 0]
    A_.semilogy(xc[keep], Pp[keep], color=MUTED, lw=1.2, label="|ψ|²")
    A_.semilogy(xc[keep], Pr[keep], color=BLUE, lw=2, label="|r|² (residual)")
    A_.semilogy(xc[keep], Pe[keep], color=ORANGE, lw=2, ls="--", label="|e|² (error)")
    A_.set_ylabel("per cell", color=MUTED)
    A_.legend(frameon=False, fontsize=9)
    A_ = ax[2, 1]
    A_.semilogy(xc[keep], Pe[keep] / Pp[keep], color=INK, lw=2)
    A_.set_ylabel("|e|² / |ψ|² per cell", color=MUTED)
    for A_ in ax[2]:
        A_.set_xlabel("x (transport) [Å]", color=MUTED)
        A_.set_facecolor(SURF)
        A_.grid(color=GRID, lw=0.6)
        for xe in edges:
            A_.axvline(xe - 0.8, color=MUTED, lw=0.6, ls=":")
        for s in A_.spines.values():
            s.set_color(GRID)
    fig.suptitle(f"Mode space (N {cs.N}) residual and error at E = Ec + {E - Ec:.3f} eV: "
                 f"|r|/|b| {np.linalg.norm(r) / np.linalg.norm(b):.2e}, |e|/|ψ| "
                 f"{np.linalg.norm(e) / np.linalg.norm(psi):.2e}, T error {(T(x) - T(psi)) / T(psi):+.1e}",
                 color=INK, x=0.01, ha="left")
    out = a.out or os.path.splitext(a.psi)[0] + "_residual.png"
    fig.savefig(out, dpi=130, facecolor=SURF)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
