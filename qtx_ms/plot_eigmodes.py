"""Plot OBC eigenmodes (full-size obc_eig eigenpairs and reduced level-1 eigenpairs of the same contour): weight per atom
|psi|^2 (Mulliken q_a = Re sum_{i in a} psi_i^* (S psi)_i) on the device, x-z and x-y projections, profile along x and the
share on each species.

  python -m qtx_ms.plot_eigmodes --inputs inputs/mos2_hfo2 --fine results_ms/obc_eig/c1_Ec+0.017 \
      --U1 results_ms/mos2_hfo2_vcycle/tobc_t1e-6_lc --center 0.52135 --a 0.015 --b 0.01 --out results_ms/obc_eig/figures/c1
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import msbasis as mb
from . import obc_eig as oe
from .obc_eig_l1 import SigmaInterp, project_sigma, fine_vec, apply_T


def atom_weights(dev, inp, v):
    atom = np.repeat(np.arange(len(inp["norb"])), inp["norb"])[dev.dev]
    q = np.real(v.conj() * (dev.S @ v))
    w = np.bincount(atom, q, minlength=len(inp["norb"]))
    return w / w.sum()


def plot_modes(fname, inp, dev, modes, Ec):
    xyz, lab = inp["xyz"], np.asarray(inp["labels"])
    atoms = np.unique(np.repeat(np.arange(len(inp["norb"])), inp["norb"])[dev.dev])
    x0 = xyz[atoms, 0].min()
    sb_x = []
    ob = np.repeat(inp["blocks"], inp["norb"])
    for b in dev.dev_blocks[1:]:
        sb_x.append(xyz[np.repeat(np.arange(len(inp["norb"])), inp["norb"])[ob == b], 0].min() - x0)
    n = len(modes)
    fig, ax = plt.subplots(n, 3, figsize=(16, 3.1 * n), squeeze=False, gridspec_kw=dict(width_ratios=[2.2, 2.2, 1.6]))
    sp = sorted(set(lab[atoms]))
    for i, (title, w) in enumerate(modes):
        wa = np.clip(w[atoms], 0, None)
        r = xyz[atoms]
        o = np.argsort(wa)
        sz = 2 + 120 * wa[o] / wa.max()
        for j, (c1, c2) in enumerate(((0, 2), (0, 1))):
            a = ax[i, j]
            a.scatter(r[:, c1] - x0, r[:, c2], s=0.3, c="0.85", lw=0)
            sc = a.scatter(r[o, c1] - x0, r[o, c2], s=sz, c=np.log10(wa[o] + 1e-12), cmap="viridis",
                           vmin=np.log10(wa.max()) - 3, vmax=np.log10(wa.max()), lw=0)
            for xb in sb_x:
                a.axvline(xb, color="r", lw=0.6, ls="--")
            a.set_xlabel("x (A)"); a.set_ylabel("z (A)" if c2 == 2 else "y (A)")
            a.set_aspect("equal")
        plt.colorbar(sc, ax=ax[i, 1], label="log10 weight / atom", fraction=0.03)
        bins = np.arange(0, r[:, 0].max() - x0 + 1, 1.0)
        a = ax[i, 2]
        for s in sp:
            m = lab[atoms] == s
            h, _ = np.histogram(r[m, 0] - x0, bins, weights=wa[m])
            a.semilogy(0.5 * (bins[1:] + bins[:-1]), h + 1e-9, label=f"{s} {100 * wa[m].sum():.0f}%")
        for xb in sb_x:
            a.axvline(xb, color="r", lw=0.6, ls="--")
        a.set_ylim(1e-6, 1); a.set_xlabel("x (A)"); a.set_ylabel("weight per 1 A")
        a.legend(fontsize=7, ncol=2)
        ax[i, 0].set_title(title, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--fine", required=True, help="eig dir(s), comma-separated")
    p.add_argument("--U1")
    p.add_argument("--center", type=float); p.add_argument("--a", type=float); p.add_argument("--b", type=float)
    p.add_argument("--npts", type=int, default=24)
    p.add_argument("--probes", type=int, default=300)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from .prep.inputs import read_inputs
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc="folded")
    Ec = inp["meta"]["edges"]["Ec_lead"]
    modes = []
    for fd in a.fine.split(","):   # several contour dirs -> one figure, sorted by Re E
        f = np.load(os.path.join(fd, "eigpairs.npz"))
        tag = os.path.basename(os.path.normpath(fd)).split("_")[0]
        for j, (l, v) in enumerate(zip(f["lam"], f["vec"].T)):
            miss = f["missing"][j] if "missing" in f else np.nan
            modes.append((l.real, f"full OBC eigenmode ({tag} #{j}): E = Ec{l.real - Ec:+.4f} {l.imag:+.1e}i eV, "
                          f"missing in span(U1) {miss:.1e}", atom_weights(dev, inp, v)))
    modes = [m[1:] for m in sorted(modes, key=lambda m: m[0])]
    fine0 = a.fine.split(",")[0]
    if a.U1:
        name = os.path.basename(os.path.normpath(a.inputs))
        nprop, nR = (tuple(x) for x in json.load(open(os.path.join(fine0, f"channels_{a.center:+.8f}.json"))))
        sig = oe.ContinuedSigma(dev, cache=os.path.join("results_ms", "sigma", name), nprop=nprop, nprop_R=nR)
        zs, ws = oe.ellipse(a.center, a.a, a.b, a.npts)
        SLs, SRs = zip(*[sig(z) for z in zs])
        sint = SigmaInterp(zs, ws, SLs, SRs)
        Us = oe.load_U1(a.U1, len(dev.ns))
        blk1 = mb.slab_blocks(dev, Us)
        V = np.random.default_rng(1).normal(size=(sum(u.shape[1] for u in Us), a.probes)) + 0j
        Xs = [mb.BlockTridiag(blk1, z, *project_sigma(dev, Us, SL, SR)).solve(V) for z, SL, SR in zip(zs, SLs, SRs)]
        lam, C, s, k = oe.beyn(Xs, zs, ws, 1, 1e-8, a.center, max(a.a, a.b))
        ins = ((lam.real - a.center) / a.a) ** 2 + (lam.imag / a.b) ** 2 < 1
        for l, c in sorted(zip(lam[ins], C[:, ins].T), key=lambda t: t[0].real):
            v = fine_vec(Us, c); v /= np.linalg.norm(v)
            r = np.linalg.norm(apply_T(dev, l, *sint(l), v))
            kind = "spurious" if r > 2 else "physical"
            modes.append((f"level-1 eigenmode ({kind}): E = Ec{l.real - Ec:+.4f} {l.imag:+.1e}i eV, |T(E) v| {r:.2f} eV",
                          atom_weights(dev, inp, v)))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plot_modes(f"{a.out}_eigmodes.png", inp, dev, modes, Ec)
    print(f"-> {a.out}_eigmodes.png ({len(modes)} modes)")


if __name__ == "__main__":
    main()
