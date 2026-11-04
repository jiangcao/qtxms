"""fig12: OFF state (0.4 eV gate barrier over slabs 1-2), flat-band bases reused: T and T error vs E."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results_ms/gate_off/scan_results_ms_mos2_hfo2_"
S = [("slabl1_tobc24_modes_t1e-6", "slab level 1, cut 1e-6 (N 1305)"),
     ("slabl1_tobc24_modes_t1e-8", "slab level 1, cut 1e-8 (N 2373)"),
     ("slabl1_tobc24_modes_t1e-10", "slab level 1, cut 1e-10 (N 3658)"),
     ("vcycle_tobc_t1e-6_lc_U1", "tile level 1, cut 1e-6 (N 2301)"),
     ("l2direct_modes_mu-0.15_0.6_n400_lead", "slab level 2, cut 5e-14 (N 1636)")]
C = ["#4C78A8", "#72B7B2", "#54A24B", "#E45756", "#B279A2"]
fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
for (k, lab), c in zip(S, C):
    E = json.load(open(R + k + ".json"))["E"]; ks = sorted(E, key=float)
    x = np.array([float(q) for q in ks]); T = np.array([E[q]["T_ms"] for q in ks])
    ax[0].semilogy(x, T, "-", color=c, lw=1.5, label=lab)
    ke = [q for q in ks if "T_exact" in E[q]]
    xe = np.array([float(q) for q in ke]); Te = np.array([E[q]["T_exact"] for q in ke])
    ax[1].semilogy(xe, np.abs(np.array([E[q]["T_ms"] for q in ke]) / Te - 1), "o-", color=c, ms=4, lw=1.2, label=lab)
ax[0].semilogy(xe, Te, "ko", ms=5, label="exact")
ax[0].set_ylabel("T"); ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_title("OFF state, 0.4 eV barrier over slabs 1-2, bases built for flat band\nMS on 10 meV grid, exact on 20 meV grid")
ax[1].axhline(1e-2, color="0.6", lw=0.8, ls="--")
ax[1].set_ylabel("|T_MS / T_exact - 1|"); ax[1].set_xlabel("E - Ec (eV)")
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("reports/2026-11-04-mos2-hfo2-leadcms-vs-dmms/fig12_gate_off.png", dpi=140)
