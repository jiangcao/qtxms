"""Figures: lead CMS vs density-matrix mode space (DMMS) on the 2808-atom MoS2/HfO2 device (inputs/mos2_hfo2).

  uv run python reports/2026-11-04-mos2-hfo2-leadcms-vs-dmms/make_figures.py [bands] [scan] [modes]

bands: exact lead bands | lead CMS (N 84 per cell) | after spurious-band removal (RMS, N 100)
scan : T, |T_MS - T| / T, MS residual |b - A x_MS| / |b| vs E on the 10 meV grid Ec + 0 ... 0.5 eV
       (lead CMS, lead CMS + RMS: results_ms/mos2_hfo2_leadcms/modes/*_scan.json;
        slab level 2 (direct: global rho on the fine device), top 400 per slab = cut ~5e-14
        (lambda_400 1e-14 ... 6e-14), + end-slab lead RMS, N 1636:
        results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead_scan.json)
contours: the Beyn contours c1 / c2 / c3 (complex E plane) next to the lead bands, eigenvalues found
modes: full-size OBC nonlinear eigenmodes T(lambda) v = 0 (Beyn, contours c1 / c2 / c3, results_ms/obc_eig/c*),
       one mode per figure (weight per atom, x-z / x-y projections, profile along x per species)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from qtx_ms.prep.inputs import read_inputs                       # noqa: E402
from qtx_ms import msbasis as mb                                   # noqa: E402

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = [("cms", "lead CMS (N 336)", "#2a78d6", "o"),
          ("cms_rms", "lead CMS + RMS (N 400)", "#eb6834", "s"),
          ("dmms", "slab level 2, cut 5e-14 (N 1636)", "#1baf7a", "D")]
SCANS = dict(cms="results_ms/mos2_hfo2_leadcms/modes/cms_scan.json",
             cms_rms="results_ms/mos2_hfo2_leadcms/modes/cms_rms_scan.json",
             dmms="results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead_scan.json")
inp = read_inputs("inputs/mos2_hfo2")
Ec, Ev = inp["meta"]["edges"]["Ec_lead"], inp["meta"]["edges"]["Ev_lead"]


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def bands():
    from qtx_ms.lead_rms import LeadCell, RMS, plot_bands
    d = "results_ms/mos2_hfo2_leadcms"
    cell = LeadCell(inp["lead"])
    ks = np.linspace(0, np.pi, 25)
    Eex = cell.bands(ks, cache=os.path.join(d, "bands_25.npz"))
    Phi = np.load(os.path.join(d, "phi_mu0_0.4_thr1e-05_nk32.npy"))
    w1, w2 = Ec - 0.8, Ec + 0.4
    rms = RMS(cell, Phi, ks, (w1, w2))
    e0 = [rms.eig(q)[0] for q in ks]
    rms.Cs = np.load(os.path.join(d, "rms_mu0_0.4_thr1e-05_nk32_win-0.8_0.4.npz"))["C"]
    e1 = [rms.eig(q)[0] for q in ks]
    f = os.path.join(OUT, "fig1_lead_bands_cms_rms.png")
    plot_bands(f, ks, Eex, [(f"lead CMS, N {Phi.shape[1]}", e0), (f"+ {rms.Cs.shape[1]} RMS vectors (N "
               f"{Phi.shape[1] + rms.Cs.shape[1]})", e1)], {"Ec": Ec, "Ev": Ev}, (Ev - 0.6, Ec + 1.0),
               f"MoS2/HfO2 lead cell ({cell.n} orbitals)",
               [(w1, w2, "spurious window"), (Ec, Ec + 0.4, "basis window")], legend_loc="center")
    print("->", f)


def load_scans():
    out = {}
    for k, f in SCANS.items():
        r = json.load(open(f))["E"]
        dE = np.array(sorted(float(x) for x in r))
        g = lambda key: np.array([r[f"{x:.4f}"].get(key, np.nan) for x in dE])
        out[k] = dict(dE=dE, T=g("T_ms"), Tx=g("T_exact"), res=g("res_ms"))
    return out


def scan():
    S = load_scans()
    ref = S["dmms"]
    # T vs E
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style(ax)
    ax.plot(ref["dE"], ref["Tx"], color=INK, lw=2, label="exact", zorder=2)
    for k, lab, c, m in SERIES:
        ax.plot(S[k]["dE"], S[k]["T"], m, ms=5, color=c, mec=SURF, mew=0.8, label=lab, zorder=3)
    ax.set_xlabel("E − Ec (eV)"); ax.set_ylabel("transmission T")
    ax.set_title("MoS2/HfO2 (2808 atoms): transmission", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_transmission.png"), dpi=150); plt.close(fig)
    # relative T error
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style(ax)
    for k, lab, c, m in SERIES:
        e = np.abs(S[k]["T"] - S[k]["Tx"]) / S[k]["Tx"]
        ax.semilogy(S[k]["dE"], e, "-" + m, ms=4.5, lw=1.2, color=c, mec=SURF, mew=0.8, label=lab)
        print(f"{k:8s} T error median {np.nanmedian(e):.2e} max {np.nanmax(e):.2e}  "
              f"residual median {np.nanmedian(S[k]['res']):.2e}")
    ax.set_xlabel("E − Ec (eV)"); ax.set_ylabel("|T_MS − T| / T")
    ax.set_title("relative transmission error", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_transmission_error.png"), dpi=150); plt.close(fig)
    # residual
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style(ax)
    for k, lab, c, m in SERIES:
        ax.semilogy(S[k]["dE"], S[k]["res"], "-" + m, ms=4.5, lw=1.2, color=c, mec=SURF, mew=0.8, label=lab)
    ax.set_xlabel("E − Ec (eV)"); ax.set_ylabel("|b − A x_MS| / |b|")
    ax.set_title("mode-space residual on the full device", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_residual.png"), dpi=150); plt.close(fig)
    print("-> fig2..4")


def modes():
    from qtx_ms.plot_eigmodes import atom_weights, plot_modes
    dev = mb.Device(inp, obc="folded")
    allm = []
    for c in ("c1_Ec+0.017", "c2_Ec+0.085", "c3_Ec+0.184"):
        f = np.load(os.path.join("results_ms/obc_eig", c, "eigpairs.npz"))
        for j, (l, v) in enumerate(zip(f["lam"], f["vec"].T)):
            allm.append((l, c.split("_")[0], j, f["missing"][j], v))
    allm.sort(key=lambda t: t[0].real)
    for i, (l, tag, j, miss, v) in enumerate(allm):
        title = (f"OBC eigenmode {i + 1}/{len(allm)} (contour {tag} #{j}): E = Ec {l.real - Ec:+.4f} {l.imag:+.1e}i eV, "
                 f"missing in span(U1 tiles 1e-6) {miss:.1e}")
        fn = os.path.join(OUT, f"fig5_obc_mode{i + 1:02d}_Ec{l.real - Ec:+.4f}.png")
        plot_modes(fn, inp, dev, [(title, atom_weights(dev, inp, v))], Ec)
        print("->", fn)


def contours():
    """Beyn contours c1 / c2 / c3 in the complex E plane (Re E vertical) next to the exact lead bands, shared E axis;
    OBC eigenvalues found inside each contour; lead band extrema at k = 0 and pi (branch points of Sigma) dotted."""
    from qtx_ms.lead_rms import LeadCell
    from qtx_ms.obc_eig import ellipse
    cell = LeadCell(inp["lead"])
    ks = np.linspace(0, np.pi, 25)
    Eex = cell.bands(ks, cache="results_ms/mos2_hfo2_leadcms/bands_25.npz")
    cols = {"c1": "#2a78d6", "c2": "#eb6834", "c3": "#1baf7a"}
    lo, hi = Ec - 0.06, Ec + 0.30
    fig, (ab, ac) = plt.subplots(1, 2, figsize=(10.5, 6.2), sharey=True, gridspec_kw=dict(width_ratios=[1.3, 1]))
    for a in (ab, ac):
        style(a)
    sel = ((Eex > lo - 0.1) & (Eex < hi + 0.1)).any(0)
    for b in np.where(sel)[0]:
        ab.plot(ks / np.pi, Eex[:, b], color=INK, lw=1.0)
    edges = np.unique(np.round(np.concatenate([Eex[0], Eex[-1]]), 4))
    edges = edges[(edges > lo) & (edges < hi)]
    for e in edges:
        for a in (ab, ac):
            a.axhline(e, color="#a3a29b", lw=0.6, ls=(0, (2, 2)), zorder=1)
    for a in (ab, ac):
        a.axhline(Ec, color=MUTED, lw=0.9, ls=(0, (5, 3)), zorder=1)
    ab.text(0.99, Ec - 0.003, "Ec (lead)", color=MUTED, fontsize=8, ha="right", va="top")
    ab.set_xlim(0, 1); ab.set_xlabel("k (π / cell length)"); ab.set_ylabel("E (eV)")
    ab.set_title("exact lead bands (dotted: band extrema at k = 0, π)", loc="left", fontsize=10, color=INK)
    for c in ("c1_Ec+0.017", "c2_Ec+0.085", "c3_Ec+0.184"):
        tag = c.split("_")[0]
        d = json.load(open(os.path.join("results_ms/obc_eig", c, "eig.json")))
        ct = d["contour"]
        z, _ = ellipse(ct["center"], ct["a"], ct["b"], ct["npts"])
        th = np.linspace(0, 2 * np.pi, 400)
        ac.plot(ct["b"] * np.sin(th) * 1e3, ct["center"] + ct["a"] * np.cos(th), color=cols[tag], lw=2,
                label=f"{tag}: Ec{ct['center'] - Ec:+.3f} ± {ct['a']:.3f} eV, Im ± {1e3 * ct['b']:.0f} meV, "
                      f"{ct['npts']} pts, {len(d['lam_re'])} eigenvalue{'s' * (len(d['lam_re']) > 1)}")
        ac.plot(z.imag * 1e3, z.real, "o", ms=3.5, color=cols[tag], mec=SURF, mew=0.6)
        for i, (lr, li) in enumerate(sorted(zip(d["lam_re"], d["lam_im"]))):
            ac.plot(li * 1e3, lr, "*", ms=11, color=INK, mec=SURF, mew=0.6, zorder=5)
            side = 1 if i % 2 == 0 else -1                                  # alternate sides: no overlaps
            ac.text(li * 1e3 + 2.0 * side, lr, f"Ec{lr - Ec:+.4f} {1e3 * li:+.2f}i meV", fontsize=7, color=INK,
                    ha="left" if side > 0 else "right", va="center",
                    bbox=dict(fc=SURF, ec="none", pad=0.5, alpha=0.85), zorder=6)
    ac.axvline(0, color=MUTED, lw=0.8)
    ac.plot([], [], "*", ms=11, color=INK, label="OBC eigenvalue (Beyn)")
    ac.set_xlabel("Im E (meV)")
    ac.set_title("Beyn contours in the complex E plane", loc="left", fontsize=10, color=INK)
    ac.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.04), borderaxespad=0)
    ac.set_ylim(lo, hi)
    fig.tight_layout()
    f = os.path.join(OUT, "fig6_obc_contours_lead_bands.png")
    fig.savefig(f, dpi=150, facecolor=SURF); plt.close(fig)
    print("->", f)


if __name__ == "__main__":
    what = sys.argv[1:] or ["bands", "scan", "modes", "contours"]
    for w in what:
        globals()[w]()
