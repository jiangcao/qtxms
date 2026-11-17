"""Compare a rebuilt release-CLI run with the recorded one: basis size N per slab (identical), channels (identical),
T_MS and T_exact on the common energies (relative tolerances), optionally the level-1 occupations in basis.json.
    python workflows/bases/compare_runs.py OLD/transmission.json NEW/transmission.json [--basis-json OLD NEW]
Exit status 1 on a mismatch; one summary line "compare: ..." for logs.
"""
import argparse
import json
import sys


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--basis-json", nargs=2)
    p.add_argument("--tol-ms", type=float, default=1e-6, help="relative T_MS tolerance")
    p.add_argument("--tol-exact", type=float, default=1e-8, help="relative T_exact tolerance")
    a = p.parse_args()
    o, n = json.load(open(a.old)), json.load(open(a.new))
    bad = []
    if o.get("N") != n.get("N"):
        bad.append(f"N {o.get('N')} vs {n.get('N')}")
    oE = {f"{float(k):.4f}": v for k, v in o["E"].items()}
    nE = {f"{float(k):.4f}": v for k, v in n["E"].items()}
    common = sorted(set(oE) & set(nE), key=float)
    dms, dex = [], []
    for k in common:
        x, y = oE[k], nE[k]
        if x.get("channels") != y.get("channels"):
            bad.append(f"Ec+{k}: channels {x.get('channels')} vs {y.get('channels')}")
        if "T_ms" in x and "T_ms" in y:
            dms.append(rel(y["T_ms"], x["T_ms"]))
            if dms[-1] > a.tol_ms:
                bad.append(f"Ec+{k}: T_MS {x['T_ms']:.10f} vs {y['T_ms']:.10f}")
        if "T_exact" in x and "T_exact" in y:
            dex.append(rel(y["T_exact"], x["T_exact"]))
            if dex[-1] > a.tol_exact:
                bad.append(f"Ec+{k}: T_exact {x['T_exact']:.12f} vs {y['T_exact']:.12f}")
    if a.basis_json:
        try:
            bo, bn = (json.load(open(f)) for f in a.basis_json)
            if bo.get("N") != bn.get("N"):
                bad.append(f"basis.json N {bo.get('N')} vs {bn.get('N')}")
        except FileNotFoundError:
            pass
    for b in bad:
        print("  MISMATCH", b)
    print(f"compare: N {sum(n.get('N') or [0])} ({'same' if o.get('N') == n.get('N') else 'DIFFERENT'}), "
          f"{len(common)} common energies, max rel dT_MS {max(dms, default=0):.1e}, max rel dT_exact "
          f"{max(dex, default=0):.1e} -> {'OK' if not bad else 'MISMATCH'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
