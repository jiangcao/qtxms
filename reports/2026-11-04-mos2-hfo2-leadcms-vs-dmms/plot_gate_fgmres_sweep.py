"""fig16: OFF state (0.4 eV barrier over slabs 1-2), energy sweep Ec + 0:0.5:0.05, block FGMRES to 1e-10 with the
CAP-Schwarz smoother (16 Å windows, ramp 2 Å): smoother only (one-level, x0 = 0) vs two-level with the flat-band slab level 1
cut 1e-8 coarse solve, pre + post smoothing or pre-smoothing only. Data: results_ms/gate_off/fgmres (run_mg_gate3.sh)."""
import json, os, re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "results_ms/gate_off/fgmres/"
Es = np.round(np.arange(0, 0.501, 0.05), 2)
nc = json.load(open(D + "nocoarse.json"))["E"]
for e in Es:
    f = D + f"nocoarse_{e:.2f}.json"
    if os.path.exists(f):
        nc.update(json.load(open(f))["E"])
def two(tag, e):
    x = json.load(open(f"{D}mgb_{tag}_{0.5043489720738419 + e:+.8f}.json"))["runs"][0]
    return max(x["its"]), x["t_factor"] + x["t_iter"]
rows = {"smoother only (one-level, x0 = 0)": [(max(nc[f"{e:.4f}"]["its"]), nc[f"{e:.4f}"]["t_factor"] + nc[f"{e:.4f}"]["t_iter"]) for e in Es],
        "+ coarse MS, pre + post smoothing": [two("flat_t1e-8", e) for e in Es],
        "+ coarse MS, pre-smoothing only": [two("flat_t1e-8_nopost", e) for e in Es]}
tex = []
for e in Es:
    t = re.search(r"total (\d+) s", open(D + f"exact_{e:.2f}.log").read())
    tex.append(float(t.group(1)))
Tex = [nc[f"{e:.4f}"]["T_exact"] for e in Es]
C = ["#2a78d6", "#eb6834", "#1baf7a"]; Mk = ["o", "s", "^"]
fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.4, 1.4]))
ax[0].semilogy(Es, Tex, "k.-", lw=1.4); ax[0].set_ylabel("T exact")
for (lab, v), c, mk in zip(rows.items(), C, Mk):
    v = np.array(v)
    ax[1].plot(Es, v[:, 0], "-", marker=mk, color=c, lw=2, ms=6, label=lab)
    ax[2].plot(Es, v[:, 1], "-", marker=mk, color=c, lw=2, ms=6, label=lab)
ax[2].plot(Es, tex, "--", color="#52514e", lw=1.5, label="sparse direct LU (6 threads)")
ax[1].set_ylabel("FGMRES iterations\n(max over columns, tol 1e-10)"); ax[2].set_ylabel("wall time per energy [s]\n(window LU + iterations)")
ax[2].set_xlabel("E − Ec [eV]")
for a in ax:
    a.axvline(0.4, color="#bbbbbb", lw=1, ls=":"); a.grid(color="#e5e5e5"); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].text(0.405, 1e-6, "barrier top", fontsize=8, color="#52514e")
ax[1].legend(frameon=False, fontsize=9); ax[2].legend(frameon=False, fontsize=9)
ax[1].set_ylim(0, None); ax[2].set_ylim(0, None)
fig.suptitle("OFF state (0.4 eV barrier): FGMRES with CAP-Schwarz 16 Å windows, energy sweep", fontsize=11)
fig.tight_layout(); fig.savefig("reports/2026-11-04-mos2-hfo2-leadcms-vs-dmms/fig16_gate_fgmres_sweep.png", dpi=140)
