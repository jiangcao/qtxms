"""Compare QuaTrEx transmissions of a mode-space input set and its full-orbital reference (to_quatrex).

    python -m qtx_modespace.compare_quatrex QTX_MS QTX_EXACT [--ref basis.json] [--out compare.json]

Reads <dir>/outputs/transmission_<l>_<r>.npy (k x energies) and electron_energies.npy of both runs; --ref adds the
T_exact / T_ms of a qtx_modespace basis.json or transmission.json (own solver) at the same energies.
"""
import argparse, glob, json, os
import numpy as np


def load(d):
    E = np.load(os.path.join(d, "outputs", "electron_energies.npy")).real
    fs = sorted(glob.glob(os.path.join(d, "outputs", "transmission_*.npy")))
    T = {os.path.basename(f)[13:-4]: np.load(f).real.sum(0) for f in fs}
    return E, T


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ms")
    p.add_argument("exact")
    p.add_argument("--ref", help="basis.json / transmission.json with per-energy T_exact, T_ms")
    p.add_argument("--out")
    a = p.parse_args(argv)
    Em, Tm = load(a.ms)
    Ee, Te = load(a.exact)
    assert np.allclose(Em, Ee), "different energy grids"
    ref = {}
    if a.ref:
        r = json.load(open(a.ref))
        r = r.get("transmission") or r.get("E") or {}          # basis.json | transmission.json
        ref = {round(v["E"], 6): v for v in r.values() if isinstance(v, dict) and "E" in v}
    Ec = json.load(open(os.path.join(a.ms, "mode_space.json")))["Ec"]
    res = []
    for lab in Tm:
        print(f"contacts {lab}: QuaTrEx mode space vs QuaTrEx full orbitals")
        print(f"{'E-Ec':>7s} {'T_MS(qtx)':>11s} {'T_exact(qtx)':>13s} {'err':>10s}" + ("   T_exact(own) T_MS(own)" if ref else ""))
        for i, E in enumerate(Em):
            tm, te = Tm[lab][i], Te[lab][i]
            row = dict(contacts=lab, E=float(E), dE=float(E - Ec), T_ms=float(tm), T_exact=float(te),
                       err=float((tm - te) / te) if te > 1e-8 else None)
            line = f"{E - Ec:7.3f} {tm:11.5f} {te:13.5f} " + (f"{100 * row['err']:+9.4f}%" if row["err"] is not None else f"{'-':>10s}")
            v = ref.get(round(float(E), 6))
            if v:
                row.update(T_exact_own=v.get("T_exact"), T_ms_own=v.get("T_ms"))
                line += f"   {v.get('T_exact', float('nan')):12.5f} {v.get('T_ms', float('nan')):9.5f}"
            print(line)
            res.append(row)
        errs = [abs(r["err"]) for r in res if r["contacts"] == lab and r["err"] is not None]
        if errs:
            print(f"max |T_MS - T_exact| / T_exact = {100 * max(errs):.4f}% over {len(errs)} energies")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
