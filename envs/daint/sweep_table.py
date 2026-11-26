"""Timing table of a hier sweep: python envs/daint/sweep_table.py RESULT.json [LOG] -> per energy T vs the 2026-11-18 exact T, its, time per part."""
import json, re, sys, glob
r = json.load(open(sys.argv[1]))
ref = {}
for f in glob.glob("results_ms/gpu/ms_ycut/12nm_exactT_beyn*/transmission.json"):
    d = json.load(open(f))
    for k, v in d["E"].items():
        ref[round(float(v.get("dE", k)) if isinstance(v, dict) else float(k), 3)] = v["T_exact"] if isinstance(v, dict) and "T_exact" in v else v
parts = ["load", "fine operator", "S0", "A_k", "level smoothers", "bottom LU", "solve"]
print(f"{'dE':>5} {'T':>16} {'vs ref':>9} {'cols':>4} {'its':>8} {'total':>6} " + " ".join(f"{p[:9]:>9}" for p in parts))
tot = dict.fromkeys(parts + ["energy"], 0.0)
for k, v in r["E"].items():
    e = round(float(k), 3)
    rv = ref.get(e)
    dv = f"{(v['T'] - rv) / rv:+.1e}" if rv else "-"
    its = f"{min(v['its'])}..{max(v['its'])}" if v["its"] else "-"
    print(f"{e:5.2f} {v['T']:16.10f} {dv:>9} {len(v['its']):4d} {its:>8} {v['t_energy']:6.1f} " + " ".join(f"{v['t'].get(p, 0):9.1f}" for p in parts)
          + ("" if v["converged"] else "  NOT CONVERGED"))
    for p in parts:
        tot[p] += v["t"].get(p, 0)
    tot["energy"] += v["t_energy"]
print("sum", f"{tot['energy']:.0f} s:", ", ".join(f"{p} {tot[p]:.0f}" for p in parts), "; setup", {k: round(v, 1) for k, v in r["t_setup"].items()})
if len(sys.argv) > 2:
    L = open(sys.argv[2]).read()
    print(*re.findall(r"^obc: \d+ of .*$", L, re.M), sep="\n")
    ts = [float(x) for x in re.findall(r"^obc: Beyn [LR] at .*?: ([\d.]+) s", L, re.M)]
    if ts:
        print(f"Beyn tasks on this log: {len(ts)}, {sum(ts):.0f} s in all, {min(ts):.1f}..{max(ts):.1f} s each")
    print(*re.findall(r"^setup .*$|^done.*$", L, re.M), sep="\n")
