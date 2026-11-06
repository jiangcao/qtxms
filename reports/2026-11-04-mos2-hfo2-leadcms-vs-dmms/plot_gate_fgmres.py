"""fig15: OFF state (0.4 eV barrier over slabs 1-2), block FGMRES to 1e-10 with the CAP-Schwarz smoother (fig14 windows,
ramp 2 Å): one-level (smoother only, x0 = 0) vs two-level with a slab level 1 mode-space coarse solve (flat-band or
gate-potential basis, coarse shift E + i eta_c, window factors reused at E + dE0). Data: results_ms/gate_off/fgmres."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "results_ms/gate_off/fgmres/"
EK = ["+0.55434897", "+0.70434897", "+0.95434897"]; EL = ["Ec+0.05", "Ec+0.20", "Ec+0.45"]
cfg = [("one-level (no coarse)", None),
       ("flat-band cut 1e-6", "flat_t1e-6"), ("flat-band cut 1e-8", "flat_t1e-8"),
       ("flat 1e-8, η_c 1e-3", "flat_t1e-8_eta1e-3"), ("flat 1e-8, η_c 1e-2", "flat_t1e-8_eta1e-2"),
       ("gate cut 1e-8", "gate_t1e-8"), ("gate 1e-8, η_c 1e-3", "gate_t1e-8_eta1e-3"),
       ("gate 1e-8, η_c 1e-2", "gate_t1e-8_eta1e-2"),
       ("flat 1e-8, LU at E−0.05", "flat_t1e-8_dE0-0.05"), ("flat 1e-8, LU at E+0.05", "flat_t1e-8_dE00.05"),
       ("flat 1e-8, LU at E+0.10", "flat_t1e-8_dE00.10")]
nc = json.load(open(D + "nocoarse.json"))["E"]
its, tim, conv = np.zeros((len(cfg), 3)), np.zeros((len(cfg), 3)), np.ones((len(cfg), 3), bool)
for i, (_, k) in enumerate(cfg):
    for j, e in enumerate(EK):
        if k is None:
            r = nc[["0.0500", "0.2000", "0.4500"][j]]
            its[i, j], tim[i, j], conv[i, j] = max(r["its"]), r["t_factor"] + r["t_iter"], max(r["res"]) < 2e-10
        else:
            r = json.load(open(f"{D}mgb_{k}_{e}.json"))["runs"][0]
            its[i, j], tim[i, j], conv[i, j] = max(r["its"]), r["t_factor"] + r["t_iter"], max(r["res"]) < 2e-10
C = ["#2a78d6", "#eb6834", "#1baf7a"]
fig, ax = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
y = np.arange(len(cfg)); h = 0.26
for a, v, xl in [(ax[0], its, "FGMRES iterations (max over injection columns)"),
                 (ax[1], tim, "wall time per energy [s] (window LU + iterations)")]:
    for j in range(3):
        b = a.barh(y + (j - 1) * h, v[:, j], h * 0.9, color=C[j], label=EL[j])
        for i in np.where(~conv[:, j])[0]:
            b[i].set_hatch("///"); b[i].set_edgecolor("white")
            a.text(v[i, j], y[i] + (j - 1) * h, " stalled (res 1e-7)", va="center", fontsize=8,
                   color="#52514e")
    a.set_xlabel(xl); a.grid(axis="x", color="#e5e5e5"); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].set_yticks(y); ax[0].set_yticklabels([c[0] for c in cfg]); ax[0].invert_yaxis()
for yy in (0.5, 4.5, 7.5): [a.axhline(yy, color="#bbbbbb", lw=0.8) for a in ax]
ax[1].axvspan(390, 470, color="#dddddd", zorder=0)
ax[1].text(430, -0.75, "sparse LU (6 threads)\n390–470 s", ha="center", va="bottom", fontsize=8, color="#52514e")
fig.legend(*ax[0].get_legend_handles_labels(), title="energy (T exact 2.9e-5 / 7.3e-4 / 1.11)", loc="upper center",
           ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.95))
fig.suptitle("OFF state (0.4 eV barrier): FGMRES with CAP-Schwarz 16 Å windows, tol 1e-10 — smoother only vs + mode-space "
             "coarse solve", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig("reports/2026-11-04-mos2-hfo2-leadcms-vs-dmms/fig15_gate_fgmres.png", dpi=140)
