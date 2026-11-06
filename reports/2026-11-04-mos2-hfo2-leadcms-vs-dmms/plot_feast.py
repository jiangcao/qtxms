"""fig17: level-1 local basis of slab 0 (mos2_hfo2, 3483 orbitals, CAP+OBC embedding 9744 orbitals, 24-pt arc
Ec-0.15...+0.6, one dense LU per point): density-matrix integral (identity RHS on P, eigh of rho_PP) vs randomized
range finder (Omega on P, Nystrom) vs FEAST probes Q = sum_e w_e G(z_e) R with R = S_e Y (feast), S_e Y_P (feastP) or
S_e^1/2 Y (feastL), S-orthonormalized by Cholesky(S_PP) + SVD. (a) missed exact occupation sum_i lam_i |(1 - Pi_V) x_i|^2
when the method's own cut is used, (b) same with the top-k_dm(cut) vectors, (c) MS T error (only slab 0 replaced,
other slabs = slab level 1 cut 1e-8) on Ec + 0:0.5:0.05.  Data: results_ms/feast_bench (qtx_ms/feast_bench.py)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "results_ms/feast_bench/"
b = json.load(open(D + "mos2_hfo2_b0.json")); bl = json.load(open(D + "L/mos2_hfo2_b0.json"))
meth = dict(b["methods"]); meth.update(bl["methods"])
lab = {"rf": "range finder (Ω on P)", "feast": "FEAST, R = S_e Y", "feastP": "FEAST, R = S_e Y_P",
       "feastL": "FEAST, R = S_e^1/2 Y"}
C = {"rf": "#2a78d6", "feast": "#eb6834", "feastP": "#1baf7a", "feastL": "#9b59b6"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for g in ["rf", "feast", "feastP", "feastL"]:
    M = [r["M"] for r in meth[g]]
    for cut, ls in (("1e-06", "--"), ("1e-08", "-")):
        ax[0].semilogy(M, [r["cuts"][cut]["miss_self"] for r in meth[g]], ls, marker="o", ms=4, color=C[g],
                       label=f"{lab[g]}" if cut == "1e-08" else None)
        ax[1].semilogy(M, [r["cuts"][cut]["miss_count"] for r in meth[g]], ls, marker="o", ms=4, color=C[g],
                       label=f"{lab[g]}" if cut == "1e-08" else None)
for a, t in zip(ax[:2], ["(a) own cut: missed occupation", "(b) top-k_dm vectors: missed occupation"]):
    a.set_xlabel("probes M"); a.set_title(t, fontsize=10); a.set_ylim(1e-16, 1)
    a.text(0.98, 0.62, "solid cut 1e-8 (k_dm 550)\ndashed cut 1e-6 (k_dm 319)", transform=a.transAxes, ha="right",
           va="top", fontsize=8, color="#52514e")
ax[0].set_ylabel("Σ λ_i |(1 − Π_V) x_i|²  (trace ρ_PP 84.7)"); ax[0].legend(frameon=False, fontsize=8, loc="lower left")
sets = [("dm", 0, "1e-6"), ("rf", 400, "1e-6"), ("feast", 400, "1e-6"), ("feastL", 400, "1e-6"), None,
        ("dm", 0, "1e-8"), ("rf", 700, "1e-8"), ("rf", 900, "1e-8"), ("feast", 900, "1e-8"), ("feast", 1600, "1e-8"),
        ("feast", 3000, "1e-8"), ("feastP", 900, "1e-8"), ("feastP", 3000, "1e-8"), ("feastL", 900, "1e-8"),
        ("feastL", 3000, "1e-8")]
x, names = 0, []
for s in sets:
    if s is None:
        x += 0.6; continue
    g, M, c = s
    d = json.load(open(f"{D}scan_mos2_hfo2_b0_{g}_M{M}_c{c}.json"))["E"]
    e = np.array([abs(v["T_ms"] - v["T_exact"]) / v["T_exact"] for v in d.values() if v.get("T_exact")])
    col = "#52514e" if g == "dm" else C[g]
    ax[2].bar(x, e.max(), 0.7, color=col, alpha=0.35); ax[2].bar(x, np.median(e), 0.7, color=col)
    names.append((x, ("ρ_PP eigh" if g == "dm" else f"{g} M{M}") + f"  {c}")); x += 1
ax[2].set_yscale("log"); ax[2].set_xticks([p for p, _ in names]); ax[2].set_xticklabels([n for _, n in names],
                                                                                          rotation=60, ha="right", fontsize=7.5)
ax[2].set_ylabel("|ΔT|/T  (dark: median, light: max of 11 E)"); ax[2].set_title("(c) MS T error, slab 0 replaced", fontsize=10)
for a in ax:
    a.grid(color="#e5e5e5"); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
fl = b["flops"]
fig.suptitle(f"Slab 0 local ρ: density matrix vs range finder vs FEAST probes  —  24 LUs {fl['lu']:.1e} flops "
             f"(332 s), solves {fl['per_col']:.1e} flops/column: dm 3483 cols {b['dm']['flops_solve']:.1e} (296 s), "
             f"M 3000 257–286 s", fontsize=10)
fig.tight_layout(); fig.savefig("reports/2026-11-04-mos2-hfo2-leadcms-vs-dmms/fig17_feast.png", dpi=140)
