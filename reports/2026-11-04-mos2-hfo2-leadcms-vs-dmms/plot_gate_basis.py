"""fig13: OFF state (0.4 eV barrier over slabs 1-2): bases built at the gate potential vs the flat-band bases."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G, F = "results_ms/gate_off/gate_basis/scan_", "results_ms/gate_off/scan_results_ms_mos2_hfo2_"
S = [("slabl1_t1e-6", "slabl1_tobc24_modes_t1e-6", "slab level 1, cut 1e-6"),
     ("slabl1_t1e-8", "slabl1_tobc24_modes_t1e-8", "slab level 1, cut 1e-8"),
     ("slabl1_t1e-10", "slabl1_tobc24_modes_t1e-10", "slab level 1, cut 1e-10"),
     ("l2_n400", "l2direct_modes_mu-0.15_0.6_n400_lead", "slab level 2, cut 5e-14")]
C = ["#4C78A8", "#72B7B2", "#54A24B", "#B279A2"]
def load(f):
    d = json.load(open(f + ".json")); E = d["E"]; ks = sorted(E, key=float)
    x = np.array([float(q) for q in ks]); T = np.array([E[q]["T_ms"] for q in ks])
    ke = [q for q in ks if "T_exact" in E[q]]
    return d["N"], x, T, np.array([float(q) for q in ke]), np.array([E[q]["T_exact"] for q in ke]), \
        np.abs(np.array([E[q]["T_ms"] / E[q]["T_exact"] - 1 for q in ke]))
fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
for (g, f, lab), c in zip(S, C):
    Ng, x, T, xe, Te, eg = load(G + g)
    Nf, _, Tf, _, _, ef = load(F + f)
    ax[0].semilogy(x, T, "-", color=c, lw=1.6, label=f"{lab}, gate basis (N {Ng})")
    ax[1].semilogy(xe, eg, "o-", color=c, ms=4, lw=1.4, label=f"{lab}, gate basis (N {Ng})")
    ax[1].semilogy(xe, ef, ":", color=c, lw=1.2, label=f"{lab}, flat-band basis (N {Nf})")
ax[0].semilogy(xe, Te, "ko", ms=5, label="exact")
ax[0].set_ylabel("T"); ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_title("OFF state, 0.4 eV barrier over slabs 1-2: basis built at the gate potential\n"
                "MS on 10 meV grid, exact on 20 meV grid; dotted: flat-band bases (fig12)")
ax[1].axhline(1e-2, color="0.6", lw=0.8, ls="--")
ax[1].set_ylabel("|T_MS / T_exact - 1|"); ax[1].set_xlabel("E - Ec (eV)"); ax[1].legend(fontsize=7, ncol=2, loc="upper right")
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("reports/2026-11-04-mos2-hfo2-leadcms-vs-dmms/fig13_gate_basis.png", dpi=140)
