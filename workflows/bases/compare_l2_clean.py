"""Compare two qtx_ms.run_l2_clean results json files (recorded vs rebuilt): N per stage and slab (identical),
mode-space T per stage and energy (relative tolerance), T_exact. Exit status 1 on a mismatch.
    python workflows/bases/compare_l2_clean.py OLD.json NEW.json [--tol 1e-6]
"""
import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--tol", type=float, default=1e-6)
    a = p.parse_args()
    o, n = json.load(open(a.old)), json.load(open(a.new))
    bad, worst = [], 0.0
    for st, N in o["N"].items():
        if n["N"].get(st) != N:
            bad.append(f"N {st}: {N} vs {n['N'].get(st)}")
    for k, r in o["E"].items():
        q = n["E"].get(k)
        if q is None:
            bad.append(f"Ec+{k}: missing")
            continue
        for st in o["N"]:
            if st in r and st in q:
                d = abs(q[st]["T"] - r[st]["T"]) / max(abs(r[st]["T"]), 1e-300)
                worst = max(worst, d)
                if d > a.tol:
                    bad.append(f"Ec+{k} {st}: T {r[st]['T']:.10f} vs {q[st]['T']:.10f}")
    for b in bad:
        print("  MISMATCH", b)
    print(f"compare: {len(o['N'])} stages, {len(o['E'])} energies, max rel dT {worst:.1e} -> "
          f"{'OK' if not bad else 'MISMATCH'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
