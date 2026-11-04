"""Figures: lead CMS vs density-matrix mode space (DMMS) on the 2808-atom MoS2/HfO2 device (inputs/mos2_hfo2).

  uv run python reports/2026-11-04-mos2-hfo2-leadcms-vs-dmms/make_figures.py [bands] [scan] [modes]

bands: exact lead bands | lead CMS (N 84 per cell) | after spurious-band removal (RMS, N 100)
scan : T, |T_MS - T| / T, MS residual |b - A x_MS| / |b| vs E on the 10 meV grid Ec + 0 ... 0.5 eV
       (lead CMS, lead CMS + RMS: results_ms/mos2_hfo2_leadcms/modes/*_scan.json;
        DMMS = direct level 2, global rho, n400 per slab + end-slab lead RMS, N 1636:
        results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead_scan.json)
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
          ("dmms", "DMMS (N 1636)", "#1baf7a", "D")]
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


if __name__ == "__main__":
    what = sys.argv[1:] or ["bands", "scan", "modes"]
    for w in what:
        globals()[w]()
