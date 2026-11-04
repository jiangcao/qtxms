"""Lead CMS (no spurious-band removal) at different occupation cuts: T and |T_MS - T|/T vs E on the 10 meV grid,
exact and slab level 2 (cut ~5e-14) for reference. Old: window Ec+0...0.4 eV, 32 k, cut 1e-5; sweep: window Ec-0.15...+0.6 eV, 64 k."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S = [("results_ms/mos2_hfo2_leadcms/modes/cms_scan.json", "CMS cut 1e-5 (N 336, old window)", "#2a78d6", "o"),
     ("results_ms/mos2_hfo2_leadcms_sweep/t1e-7/modes/cms_scan.json", "CMS cut 1e-7 (N 816)", "#eb6834", "s"),
     ("results_ms/mos2_hfo2_leadcms_sweep/t1e-9/modes/cms_scan.json", "CMS cut 1e-9 (N 980)", "#1baf7a", "^"),
     ("results_ms/mos2_hfo2_leadcms_sweep/t1e-11/modes/cms_scan.json", "CMS cut 1e-11 (N 1164)", "#eda100", "v")]
REF = "results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead_scan.json"


def load(f):
    r = json.load(open(os.path.join(ROOT, f)))["E"]
    k = sorted(r, key=float)
    return (np.array([float(x) for x in k]), np.array([r[x]["T_ms"] for x in k]), np.array([r[x]["T_exact"] for x in k]))


fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 8.4), sharex=True, height_ratios=[1.1, 1])
for a in (a1, a2):
    a.set_facecolor(SURF); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
dE, Td, Tx = load(REF)
a1.plot(dE, Tx, color=INK, lw=2.2, label="exact", zorder=2)
for f, lab, c, m in S:
    d, T, X = load(f)
    a1.plot(d, T, "-" + m, ms=4, lw=0.9, color=c, mec=SURF, mew=0.6, label=lab, zorder=3)
    e = np.abs(T - X) / X
    a2.semilogy(d, e, "-" + m, ms=4, lw=0.9, color=c, mec=SURF, mew=0.6, label=lab)
    print(f"{lab}: median {np.median(e):.2f} max {e.max():.2f}")
a2.semilogy(dE, np.abs(Td - Tx) / Tx, "-D", ms=3.5, lw=0.9, color="#6250d6", mec=SURF, mew=0.6, label="slab level 2, cut 5e-14 (N 1636), reference")
a1.set_ylabel("transmission T"); a1.legend(frameon=False, fontsize=8.5)
a1.set_title("MoS2/HfO2: lead CMS at different occupation cuts", loc="left", fontsize=11, color=INK)
a2.set_ylabel("|T_MS − T| / T"); a2.set_xlabel("E − Ec (eV)"); a2.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
a2.set_ylim(1e-8, 20)
fig.tight_layout()
f = os.path.join(OUT, "fig9_cms_cuts_T_err.png")
fig.savefig(f, dpi=150, facecolor=SURF)
print("->", f)
