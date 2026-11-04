"""Exact vs mode-space states on an energy grid: transmission, residual, error, and charge (Mulliken, per atom).

Per energy: Sigma from the cache, injection from both leads (dev.inject: qttools modes or Gamma eigenchannels),
exact psi (sparse direct, one factorization for both leads) and the mode-space x_MS = U A_c^-1 U^H b of a saved basis.
Also |(I - U U^H) psi| / |psi| (best approximation in the basis span, total and per slab) next to the Galerkin error.
Spectral density per atom n_a(E) = sum over the L and R columns of Re(conj(psi) S psi) on the atom's orbitals (equal
weights, i.e. the local DOS of the open device; occupations / Fermi factors can be applied later from the saved
per-atom arrays). One file per energy (<out>/E_<dE>.npz), so several workers can share a grid; `summary` collects
them into <out>/summary.json and <out>/summary.png (errors vs E and of the energy-integrated density).

    python -m qtx_ms.density_scan run --inputs inputs/mos2_hfo2 --modes results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead \\
        --dE 0:0.5:0.01 --worker 0/2
    python -m qtx_ms.density_scan summary --inputs inputs/mos2_hfo2 --out results_ms/density_scan/mos2_hfo2_n400
"""
import argparse, glob, json, os, time
import numpy as np

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb
from qtx_ms.run_l2_clean import Coarse
from qtx_ms.ms_scan import grid

log = mb.log


def per_atom(y, S, atom, natoms):
    return np.bincount(atom, np.real(y.conj() * (S @ y)).sum(1), minlength=natoms)


def run(a, inp, out, Ec):
    name = os.path.basename(os.path.normpath(a.inputs))
    dev = mb.Device(inp, obc=a.obc)
    dev.sigma_cache = a.sigma_cache or os.path.join("results_ms", "sigma", name)
    V = [np.load(f"{a.modes}_slab{s}.npy") for s in range(len(dev.ns))]
    cs = Coarse(dev, V)
    natoms = len(inp["xyz"])
    atom = np.repeat(np.arange(natoms), inp["norb"])[dev.dev]
    Sd = inp["S"].tocsr()[dev.dev][:, dev.dev]
    k, nw = (int(x) for x in a.worker.split("/"))
    Es = grid(a.dE, a.first)[k::nw]
    log(f"{name}: N {cs.N}, worker {k}/{nw}: {len(Es)} energies, OBC {a.obc}, Sigma cache {dev.sigma_cache}")
    for dE in Es:
        f = os.path.join(out, f"E_{dE:.4f}.npz")
        if os.path.exists(f) and "proj_out" in np.load(f) and (not a.save_psi or "psi" in np.load(f)):
            continue
        t0 = time.time()
        E = Ec + dE
        SL, SR = dev.sigma(E + 1e-6j)
        BL, BR = dev.inject(SL), dev.inject(SR)
        if BL.shape[1] == 0 or BR.shape[1] == 0:
            log(f"E = Ec {dE:+.3f}: no propagating channel, skipped")
            continue
        b = np.zeros((dev.n, BL.shape[1] + BR.shape[1]), complex)
        b[dev.bL, :BL.shape[1]] = BL
        b[dev.bR, BL.shape[1]:] = BR
        A = dev.A(E, SL, SR)
        x = cs.prolong(cs.Ac(E, SL, SR).solve(cs.restrict(b)))
        tms = time.time() - t0
        s = mb.CSolver(A)
        psi = s.solve(b)
        res_ex = s.res
        s.free()
        nL = BL.shape[1]
        q = psi - cs.prolong(cs.restrict(psi))                  # part of psi outside span(U) (U orthonormal per slab)
        sb = dev.sb
        proj_slab = [np.linalg.norm(q[sb[i]:sb[i + 1]]) / np.linalg.norm(psi[sb[i]:sb[i + 1]]) for i in range(len(dev.ns))]
        GR = 1j * (SR - SR.conj().T)
        T = lambda y: float(np.real(np.einsum("ic,ij,jc->", y[dev.bR, :nL].conj(), GR, y[dev.bR, :nL])))
        np.savez(f, dE=dE, E=E, nL=nL, nR=BR.shape[1], T=T(psi), T_ms=T(x),
                 res_ms=np.linalg.norm(b - A @ x) / np.linalg.norm(b), res_exact=res_ex,
                 err=np.linalg.norm(psi - x) / np.linalg.norm(psi),
                 proj_out=np.linalg.norm(q) / np.linalg.norm(psi), proj_slab=proj_slab,
                 n=per_atom(psi, Sd, atom, natoms), n_ms=per_atom(x, Sd, atom, natoms), t_ms=tms,
                 t=time.time() - t0, **(dict(psi=psi) if a.save_psi else {}))
        c = np.load(f)
        log(f"E = Ec {dE:+.3f}: channels {nL}/{BR.shape[1]}, T {c['T']:.5f} ({(c['T_ms'] - c['T']) / c['T']:+.1e}), "
            f"res {float(c['res_ms']):.1e}, |e|/|psi| {float(c['err']):.1e}, |(I-P)psi|/|psi| {float(c['proj_out']):.1e}, Q {(c['n_ms'].sum() - c['n'].sum()) / c['n'].sum():+.1e}; "
            f"{time.time() - t0:.0f} s")


def summary(a, inp, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qtx_ms.plot_psi import INK, MUTED, GRID, SURF, BLUE, ORANGE
    fs = sorted(glob.glob(os.path.join(out, "E_*.npz")), key=lambda f: float(f.split("_")[-1][:-4]))
    C = [np.load(f) for f in fs]
    dE = np.array([float(c["dE"]) for c in C])
    labels, blocks = inp["labels"], inp["blocks"]
    on = C[0]["n"] != 0
    regions = {"MoS2": np.isin(labels, ["Mo", "S"]) & on, "HfO2": np.isin(labels, ["Hf", "O"]) & on}
    regions.update({f"block {b}": (blocks == b) & on for b in inp["meta"]["transport"]["device_blocks"]})
    rel = lambda m, n, msk: (m[msk].sum() - n[msk].sum()) / n[msk].sum()
    rows = []
    for c in C:
        n, m = c["n"], c["n_ms"]
        big = n > 1e-3 * n.max()
        ra = np.abs(m - n)[big] / n[big]
        rows.append(dict(dE=float(c["dE"]), channels=[int(c["nL"]), int(c["nR"])], T=float(c["T"]),
                         T_err=float((c["T_ms"] - c["T"]) / c["T"]) if c["T"] > 0 else None,
                         res_ms=float(c["res_ms"]), err=float(c["err"]), Q_err=float(rel(m, n, on)),
                         region_err={k: float(rel(m, n, v)) for k, v in regions.items()},
                         atom_err=dict(median=float(np.median(ra)), q99=float(np.quantile(ra, 0.99)), max=float(ra.max()))))
    # energy-integrated (equal weights on the grid) spectral density per atom
    N, M = sum(c["n"] for c in C), sum(c["n_ms"] for c in C)
    big = N > 1e-3 * N.max()
    ri = np.abs(M - N)[big] / N[big]
    integ = dict(energies=len(C), Q_err=float(rel(M, N, on)), region_err={k: float(rel(M, N, v)) for k, v in regions.items()},
                 atom_err=dict(median=float(np.median(ri)), q99=float(np.quantile(ri, 0.99)), max=float(ri.max())))
    json.dump(dict(rows=rows, integrated=integ), open(os.path.join(out, "summary.json"), "w"), indent=1)
    log(f"{len(C)} energies; integrated over the grid: Q {integ['Q_err']:+.1e}, per atom median "
        f"{integ['atom_err']['median']:.1e}, 99% {integ['atom_err']['q99']:.1e}, max {integ['atom_err']['max']:.1e}; "
        + ", ".join(f"{k} {v:+.1e}" for k, v in integ["region_err"].items()))

    g = lambda key: np.array([r[key] if r[key] is not None else np.nan for r in rows])
    fig, ax = plt.subplots(4, 1, figsize=(8, 11), sharex=True, facecolor=SURF, constrained_layout=True,
                           gridspec_kw=dict(height_ratios=[1.2, 1, 1, 1]))
    ax[0].plot(dE, g("T"), color=INK, lw=1.5)
    ax[0].set_ylabel("T (exact)", color=MUTED)
    ax[1].semilogy(dE, np.abs(g("T_err")), "o-", ms=3, lw=1, color=BLUE, label="|T_MS − T| / T")
    ax[1].semilogy(dE, g("err"), "s-", ms=3, lw=1, color=ORANGE, label="|ψ − x_MS| / |ψ|")
    ax[1].semilogy(dE, g("res_ms"), "^-", ms=3, lw=1, color=MUTED, label="|b − A x_MS| / |b|")
    ax[2].semilogy(dE, np.abs(g("Q_err")), "o-", ms=3, lw=1.5, color=INK, label="total charge")
    for (k, col) in (("MoS2", BLUE), ("HfO2", ORANGE)):
        ax[2].semilogy(dE, [abs(r["region_err"][k]) for r in rows], "-", lw=1, color=col, label=k)
    ax[2].semilogy(dE, [max(abs(r["region_err"][k]) for k in r["region_err"] if k.startswith("block")) for r in rows],
                   "--", lw=1, color=MUTED, label="worst slab")
    for q, ls in (("median", "-"), ("q99", "--"), ("max", ":")):
        ax[3].semilogy(dE, [r["atom_err"][q] for r in rows], ls, lw=1.5, color=INK, label=f"per atom, {q}")
        ax[3].axhline(integ["atom_err"][q], color=BLUE, ls=ls, lw=1)
    ax[3].plot([], [], color=BLUE, label="energy-integrated (lines)")
    ax[1].set_ylabel("relative error", color=MUTED)
    ax[2].set_ylabel("|charge error| (relative)", color=MUTED)
    ax[3].set_ylabel("|n_MS − n| / n per atom", color=MUTED)
    ax[3].set_xlabel("E − Ec (eV)", color=MUTED)
    for A in ax:
        A.set_facecolor(SURF)
        A.grid(color=GRID, lw=0.6)
        for s in A.spines.values():
            s.set_color(GRID)
        if A is not ax[0]:
            A.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(f"{os.path.basename(os.path.normpath(out))}: exact vs mode space, L + R injection, "
                 f"{len(C)} energies", color=INK, x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(out, "summary.png"), dpi=130, facecolor=SURF)
    log(f"-> {out}/summary.json, summary.png")

    # how much of psi is in the basis span: residual, Galerkin error, best approximation (total and per slab)
    P = [c for c in C if "proj_out" in c]
    if P:
        d = np.array([float(c["dE"]) for c in P])
        ps = np.array([c["proj_slab"] for c in P])
        fig, ax = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True, facecolor=SURF, constrained_layout=True)
        ax[0].semilogy(d, [float(c["res_ms"]) for c in P], "^-", ms=3.5, lw=1.2, color=MUTED, label="residual |b − A x_MS| / |b|")
        ax[0].semilogy(d, [float(c["err"]) for c in P], "s-", ms=3.5, lw=1.5, color=ORANGE, label="error |ψ − x_MS| / |ψ|")
        ax[0].semilogy(d, [float(c["proj_out"]) for c in P], "o-", ms=3.5, lw=1.5, color=BLUE,
                       label="outside the span |(I − UU†)ψ| / |ψ|")
        ax[0].set_ylabel("relative", color=MUTED)
        ax[0].legend(frameon=False, fontsize=8)
        cols = [BLUE, ORANGE, INK, MUTED]
        for i in range(ps.shape[1]):
            ax[1].semilogy(d, ps[:, i], lw=1.5, color=cols[i % 4], ls="-" if i < 4 else "--", label=f"slab {i}")
        ax[1].set_ylabel("|(I − U_s U_s†)ψ_s| / |ψ_s|", color=MUTED)
        ax[1].set_xlabel("E − Ec (eV)", color=MUTED)
        ax[1].legend(frameon=False, fontsize=8, ncol=4)
        for A in ax:
            A.set_facecolor(SURF)
            A.grid(color=GRID, lw=0.6)
            for s in A.spines.values():
                s.set_color(GRID)
        fig.suptitle(f"{os.path.basename(os.path.normpath(out))}: exact ψ vs the mode-space span, {len(P)} energies",
                     color=INK, x=0.01, ha="left", fontsize=10)
        fig.savefig(os.path.join(out, "span.png"), dpi=130, facecolor=SURF)
        log(f"-> {out}/span.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["run", "summary"])
    p.add_argument("--inputs", required=True)
    p.add_argument("--modes", help="mode file prefix: <prefix>_slab<s>.npy for every slab (run)")
    p.add_argument("--dE", default="0:0.5:0.01", help="lo:hi:step, energies - Ec [eV]")
    p.add_argument("--first", type=float, default=0.05, help="visit multiples of this first")
    p.add_argument("--Ec", type=float, help="reference edge for --dE (default Ec_lead of the inputs)")
    p.add_argument("--sigma-cache", help="lead self-energy cache (default results_ms/sigma/<inputs name>)")
    p.add_argument("--obc", default="folded", choices=["auto", "qttools", "folded"], help="lead OBC (default folded: the "
                   "cached folded-lead Sigma at E + 1e-6 i)")
    p.add_argument("--worker", default="0/1", help="k/n: this process takes every n-th energy from the k-th")
    p.add_argument("--out", help="default results_ms/density_scan/<inputs name>_<basename of --modes>")
    p.add_argument("--save-psi", action="store_true", help="also store the exact psi (device order, L then R columns)")
    a = p.parse_args()
    inp = read_inputs(a.inputs)
    name = os.path.basename(os.path.normpath(a.inputs))
    out = a.out or os.path.join("results_ms", "density_scan", f"{name}_{os.path.basename(a.modes)}")
    os.makedirs(out, exist_ok=True)
    Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
    if a.stage == "run":
        run(a, inp, out, Ec)
    else:
        summary(a, inp, out)


if __name__ == "__main__":
    main()
