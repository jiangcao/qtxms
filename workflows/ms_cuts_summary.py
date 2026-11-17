"""Summary of run_ms_cuts.sh: basis size N, device orbitals n, n/N, mode-space T error over the energy grid per device
and slab level-1 cut; figures ms_cuts[_clean|_gate|_flat].png, one per case (T_exact, T_MS and |dT|/T vs E - Ec). Table (summary.txt), summary.json, the figure and
the per-run transmission.json / basis.json (OUT/<device>_l1_<cut>[_clean|_gate|_flat]/) go to OUT (default reports/2026-11-17-slab-level1-cuts).
    python workflows/ms_cuts_summary.py [RUNS [OUT]]      (RUNS default results_ms/gpu/ms_cuts)"""
import glob
import json
import os
import shutil
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "results_ms", "gpu", "ms_cuts")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "reports", "2026-11-17-slab-level1-cuts")
os.makedirs(OUT, exist_ok=True)
lines = []


def out(line):
    print(line)
    lines.append(line)

DEVS = ["2808a", "9072a", "21600a"]
CUTS = ["1e-4", "1e-6", "1e-8"]
SUFS = ["", "_clean", "_gate", "_flat"]   # _clean: end slabs cleaned; _gate / _flat (run_ms_cuts_gate.sh): OFF-state
RUNS = [c + s for s in SUFS for c in CUTS]  # device with the basis rebuilt at the gate / the flat-band basis reused
STY = {"": ("--", "o-", None), "_clean": (":", "s:", "none"), "_gate": ("--", "o-", None), "_flat": (":", "s:", "none")}
NDEV = {}


def load(d, c):
    f = os.path.join(R, f"{d}_l1_{c}", "transmission.json")
    if not os.path.exists(f):
        return None
    t = json.load(open(f))
    E = sorted(t["E"], key=float)
    x = np.array([float(e) for e in E])
    Te = np.array([t["E"][e]["T_exact"] for e in E])
    Tm = np.array([t["E"][e]["T_ms"] for e in E])
    psi = np.array([t["E"][e].get("err_psi", np.nan) for e in E])
    log = os.path.join(R, f"{d}_l1_{c}.log")
    n = None
    if os.path.exists(log):
        for line in open(log):
            if line.startswith("basis: N") and "device" in line:
                n = int(line.split("(device ")[1].split()[0])
    fb = os.path.join(R, f"{d}_l1_{c}", "basis.json")
    if n is None and os.path.exists(fb):
        n = json.load(open(fb)).get("device", {}).get("n")
    n = NDEV.setdefault(d, n) if n else NDEV.get(d)      # --modes runs (_flat) log no device size: same device
    return dict(x=x, Te=Te, Tm=Tm, psi=psi, N=t["N"], n=n)


rows = []
out(f"{'device':7s} {'cut':11s} {'n':>7s} {'N':>6s} {'n/N':>6s} {'N/slab (interior)':>18s} {'pts':>4s} "
      f"{'med|dT|/T':>10s} {'max|dT|/T':>10s} {'max@':>6s} {'max|dT|':>9s} {'med psi':>8s}")
res = {}
for d in DEVS:
    for c in sorted(RUNS, key=lambda c: c.endswith("_flat")):
        r = load(d, c)
        if r is None:
            continue
        res[d, c] = r
        os.makedirs(os.path.join(OUT, f"{d}_l1_{c}"), exist_ok=True)       # per-run json files into the report
        for j in ("transmission.json", "basis.json"):
            src, dst = os.path.join(R, f"{d}_l1_{c}", j), os.path.join(OUT, f"{d}_l1_{c}", j)
            if os.path.exists(src) and not os.path.samefile(os.path.dirname(src), os.path.dirname(dst)):
                shutil.copy2(src, dst)
        N = sum(r["N"])
        ok = r["Te"] > (1e-10 if c.endswith(("_gate", "_flat")) else 1e-6)   # rel. error where T is not ~0;
        # OFF state: tunnelling T below the barrier is physical down to ~1e-9
        rel = np.abs(r["Tm"] - r["Te"]) / np.maximum(r["Te"], 1e-300)
        i = np.argmax(np.where(ok, rel, -1))
        inner = r["N"][1:-1] if len(r["N"]) > 2 else r["N"]
        out(f"{d:7s} {c:11s} {r['n'] or 0:7d} {N:6d} {(r['n'] or 0) / N:6.1f} {min(inner):8d}..{max(inner):<8d} "
              f"{len(r['x']):4d} {np.median(rel[ok]):10.2e} {rel[i]:10.2e} {r['x'][i]:6.2f} "
              f"{np.max(np.abs(r['Tm'] - r['Te'])):9.2e} {np.nanmedian(r['psi']):8.1e}")
        rows.append(dict(device=d, cut=c, n=r["n"], N=N, N_slab=r["N"], med_rel=float(np.median(rel[ok])),
                         max_rel=float(rel[i]), E_max=float(r["x"][i])))
json.dump(rows, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit()
devs = [d for d in DEVS if any((d, c) in res for c in RUNS)]
col = {"1e-4": "#d1495b", "1e-6": "#edae49", "1e-8": "#00798c"}
CASE = {"": "no-clean (flat band, end slabs not cleaned)", "_clean": "clean (flat band, end slabs cleaned)",
        "_gate": "gate (OFF state, basis rebuilt at the gate)", "_flat": "flat (OFF state, flat-band basis reused)"}
for suf in SUFS:           # one figure per case: ms_cuts.png (no-clean), ms_cuts_clean.png, ms_cuts_gate.png, ...
    if not any((d, c + suf) in res for d in devs for c in CUTS):
        continue
    logT = suf in ("_gate", "_flat")
    fig, ax = plt.subplots(2, len(devs), figsize=(4.2 * len(devs), 6.5), sharex=True, squeeze=False)
    for j, d in enumerate(devs):
        first = True
        for cc in CUTS:
            r = res.get((d, cc + suf))
            if r is None:
                continue
            if first:
                ax[0, j].plot(r["x"], r["Te"], "-", color="0.55", lw=4, alpha=0.6, label="exact", zorder=1)
                first = False
            ax[0, j].plot(r["x"], np.where(r["Tm"] > 0, r["Tm"], np.nan) if logT else r["Tm"], "o--", ms=3, lw=1,
                          color=col[cc], label=f"cut {cc} (N {sum(r['N'])})", zorder=3)
            ax[1, j].semilogy(r["x"], np.abs(r["Tm"] - r["Te"]) / np.maximum(np.abs(r["Te"]), 1e-300), "o-", ms=3,
                              lw=1, color=col[cc])
        if logT:
            ax[0, j].set_yscale("log")
        n = next((res[d, c]["n"] for c in RUNS if (d, c) in res), None)
        ax[0, j].set_title(f"{d} (n {n})")
        ax[0, j].legend(fontsize=8)
        ax[1, j].set_xlabel("E - Ec_lead [eV]")
        for a in ax[:, j]:
            a.grid(alpha=0.3, which="both", lw=0.5)
    ax[0, 0].set_ylabel("T")
    ax[1, 0].set_ylabel("|T_MS - T| / T")
    fig.suptitle(f"Slab level 1 (CAP + lead Sigma), stand-alone MS: {CASE[suf]}")
    fig.tight_layout()
    f = os.path.join(OUT, f"ms_cuts{suf}.png")
    fig.savefig(f, dpi=130)
    plt.close(fig)
    print("->", f)
print("->", OUT)
