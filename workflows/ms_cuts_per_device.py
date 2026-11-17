"""Per-device plots of the slab level-1 cut scans at cut 1e-8: T vs E, |T_MS - T|/T vs E and psi error (err_psi) vs E.
One figure per device, one row per variant (the cleaned-end-slab run <d>_l1_<cut>_clean is not plotted):
  no-clean  results_ms/gpu/ms_cuts/<d>_l1_1e-8            (flat band, end slabs not cleaned)
  gate      results_ms/gpu/ms_cuts_gate/<d>_l1_1e-8_gate  (OFF state, basis rebuilt at the gate potential)
  flat      results_ms/gpu/ms_cuts_gate/<d>_l1_1e-8_flat  (OFF state, flat-band basis reused)
    python workflows/ms_cuts_per_device.py [OUT] [CUT]     (OUT default reports/2026-11-17-slab-level1-cuts, CUT 1e-8)"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "reports", "2026-11-17-slab-level1-cuts")
CUT = sys.argv[2] if len(sys.argv) > 2 else "1e-8"
R1 = os.path.join(REPO, "results_ms", "gpu", "ms_cuts")
R2 = os.path.join(REPO, "results_ms", "gpu", "ms_cuts_gate")
ROWS = [("no-clean (flat band)", R1, "", "#1f77b4", False),
        ("gate (OFF, basis at gate)", R2, "_gate", "#d62728", True),
        ("flat (OFF, flat-band basis)", R2, "_flat", "#9467bd", True)]
DEVS = ["2808a", "9072a", "21600a"]


def load(path):
    if not os.path.exists(path):
        return None
    t = json.load(open(path))
    E = sorted(t["E"], key=float)
    g = lambda k: np.array([t["E"][e].get(k, np.nan) for e in E], float)  # noqa: E731
    return dict(x=np.array([float(e) for e in E]), Te=g("T_exact"), Tm=g("T_ms"), psi=g("err_psi"), N=sum(t["N"]))


os.makedirs(OUT, exist_ok=True)
for d in DEVS:
    fig, ax = plt.subplots(len(ROWS), 3, figsize=(14, 3.2 * len(ROWS)), sharex=True)
    for i, (lab, root, suf, col, logT) in enumerate(ROWS):
        r = load(os.path.join(root, f"{d}_l1_{CUT}{suf}", "transmission.json"))
        if r is None:
            for a in ax[i]:
                a.text(0.5, 0.5, "no data", ha="center", va="center", transform=a.transAxes)
            ax[i, 0].set_ylabel(f"{lab}\nT")
            continue
        ax[i, 0].plot(r["x"], r["Te"], "-", color="0.55", lw=4, alpha=0.6, label="exact", zorder=1)
        neg = r["Tm"] <= 0
        ax[i, 0].plot(r["x"], np.where(neg, np.nan, r["Tm"]), "o", ms=4.5, mfc="none", mew=1.2, color=col,
                      label=f"MS cut {CUT} (N {r['N']})", zorder=3)
        if logT:
            ax[i, 0].set_yscale("log")
            if neg.any():   # T_MS <= 0 cannot sit on a log axis: mark at the bottom
                ax[i, 0].plot(r["x"][neg], np.full(neg.sum(), np.nanmin(r["Te"][r["Te"] > 0])), "v", color=col,
                              ms=6, label="MS T <= 0", zorder=3)
        rel = np.abs(r["Tm"] - r["Te"]) / np.maximum(np.abs(r["Te"]), 1e-300)
        ax[i, 1].semilogy(r["x"], rel, "o-", ms=3.5, lw=1, color=col)
        ax[i, 2].semilogy(r["x"], r["psi"], "o-", ms=3.5, lw=1, color=col)
        ax[i, 0].set_ylabel(f"{lab}\nT")
        ax[i, 1].set_ylabel("|T_MS - T| / T")
        ax[i, 2].set_ylabel("err_psi")
        ax[i, 0].legend(fontsize=8)
        for a in ax[i]:
            a.grid(alpha=0.3, which="both", lw=0.5)
    for a in ax[-1]:
        a.set_xlabel("E - Ec_lead [eV]")
    ax[0, 0].set_title("transmission")
    ax[0, 1].set_title("relative T error")
    ax[0, 2].set_title("psi error")
    fig.suptitle(f"{d}: slab level 1 (CAP + lead Σ), cut {CUT}, stand-alone MS")
    fig.tight_layout()
    f = os.path.join(OUT, f"ms_cuts_{d}.png")
    fig.savefig(f, dpi=120)
    plt.close(fig)
    print("->", f)
