"""OFF-state potential of a transistor on a QuaTrEx device folder: a new folder whose inputs/potential.npy (eV per atom, electron
potential energy) is

    V(x) = 0 (source) | Vb on the gate [x1, x2] | -Vds (drain),

joined by smoothstep ramps s(t) = 3t^2 - 2t^3 of length `ramp` outside the gate ([x1 - ramp, x1] up, [x2, x2 + ramp] down), so
V is exactly flat away from the gate: the leads and their neighbouring device blocks see a constant potential (checked over
`flat` Å at both ends). The other files are links to the source folder. Then prepare with the potential (the right lead becomes
the left one shifted by -Vds S):

    python -m qtx_ms.prep.offstate examples/12nm_experiment --out examples/12nm_off --Vb 0.2 --Vds 0.4
    python -m qtx_modespace.prep examples/12nm_off --out cache_daint/12nm_off/prep --potential ...

Default gate: the middle third of the device along the transport axis, ramps 1/12 of its length.
"""
import argparse, json, os
import numpy as np


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def profile(x, Vb, Vds, x1, x2, ramp):
    """Electron potential energy [eV] at the positions x [Å]."""
    up, down = smooth((x - (x1 - ramp)) / ramp), smooth((x - x2) / ramp)
    return Vb * up * (1.0 - down) - Vds * down                         # exactly 0 / Vb / -Vds on the flat parts


def read_x(f, axis=0):
    with open(f) as fh:
        n = int(fh.readline().split()[0])
        fh.readline()
        return np.array([float(fh.readline().split()[1 + axis]) for _ in range(n)])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="QuaTrEx device folder (inputs/structure.xyz, hamiltonian.h5, overlap.h5, quatrex_config.toml)")
    p.add_argument("--out", required=True)
    p.add_argument("--Vb", type=float, required=True, help="barrier height above the source [eV]")
    p.add_argument("--Vds", type=float, required=True, help="drain bias [eV]: the drain sits at -Vds")
    p.add_argument("--x1", type=float, default=None, help="gate start [Å, structure coordinates] (default: 1/3 of the length)")
    p.add_argument("--x2", type=float, default=None, help="gate end [Å] (default: 2/3 of the length)")
    p.add_argument("--ramp", type=float, default=None, help="ramp length [Å] (default: 1/12 of the length)")
    p.add_argument("--flat", type=float, default=40.0, help="V must be constant over this length [Å] at both ends (lead + next block)")
    p.add_argument("--axis", type=int, default=0)
    a = p.parse_args(argv)
    src, out = os.path.abspath(a.folder), os.path.abspath(a.out)
    x = read_x(os.path.join(src, "inputs", "structure.xyz"), a.axis)
    x0, L = x.min(), np.ptp(x)
    x1 = x0 + L / 3 if a.x1 is None else a.x1
    x2 = x0 + 2 * L / 3 if a.x2 is None else a.x2
    ramp = L / 12 if a.ramp is None else a.ramp
    assert x1 - ramp >= x0 + a.flat and x2 + ramp <= x.max() - a.flat, "the ramps reach into the flat ends (leads)"
    V = profile(x, a.Vb, a.Vds, x1, x2, ramp)
    lo, hi = x < x0 + a.flat, x > x.max() - a.flat
    assert np.ptp(V[lo]) == 0 and np.ptp(V[hi]) == 0
    os.makedirs(os.path.join(out, "inputs"), exist_ok=True)
    for f in ("quatrex_config.toml",) + tuple(os.path.join("inputs", g) for g in ("structure.xyz", "hamiltonian.h5", "overlap.h5")):
        d = os.path.join(out, f)
        if os.path.exists(os.path.join(src, f)) and not os.path.lexists(d):
            os.symlink(os.path.join(src, f), d)
    np.save(os.path.join(out, "inputs", "potential.npy"), V)
    info = dict(kind="off-state", Vb=a.Vb, Vds=a.Vds, x1=float(x1), x2=float(x2), ramp=float(ramp), length=float(L), source=src,
                V_source=float(V[lo][0]), V_drain=float(V[hi][0]), convention="electron potential energy per atom, H += (V S + S V) / 2")
    json.dump(info, open(os.path.join(out, "potential.json"), "w"), indent=1)
    print(f"OFF state: {len(x)} atoms over {L:.1f} Å; source {info['V_source']:+.3f} eV up to x = {x1 - ramp:.1f}, barrier {a.Vb:+.3f} eV "
          f"on [{x1:.1f}, {x2:.1f}] Å, drain {info['V_drain']:+.3f} eV from x = {x2 + ramp:.1f} Å (ramps {ramp:.1f} Å) -> {out}")


if __name__ == "__main__":
    main()
