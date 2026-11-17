"""OFF-state copy of a QuaTrEx device folder for the release CLI: links hamiltonian.h5 / overlap.h5 / structure.xyz and
quatrex_config.toml, and writes inputs/potential.npy (per atom, eV) = a gate barrier
    V(x) = Vb [tanh((x - x1) / w) - tanh((x - x2) / w)] / 2,   x1 = f1 L, x2 = f2 L   (L = cell length along transport)
set to exactly 0 where |V| < 1e-6 eV (>= 13 A from the barrier edges: the lead blocks keep V = 0, same lead Sigma). Defaults Vb 0.4 eV over the middle third, w 2 A = the earlier
2808a OFF state (inputs/mos2_hfo2_gate0.4: x 32.5 ... 63.3 A of 95.0 A), scaled with the device length.
Prepare with `python -m qtx_modespace.prep OUT --potential` (H += (V S + S V) / 2).
    python workflows/gate_device.py examples/device_MoS2HfO2_9072a_78912o OUT [--Vb 0.4 --f1 0.3333 --f2 0.6667 --w 2]"""
import argparse
import os

import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("src")
p.add_argument("out")
p.add_argument("--Vb", type=float, default=0.4)
p.add_argument("--f1", type=float, default=1 / 3)
p.add_argument("--f2", type=float, default=2 / 3)
p.add_argument("--w", type=float, default=2.0)
p.add_argument("--axis", type=int, default=0, help="transport axis of structure.xyz (0 = x)")
a = p.parse_args()
src = os.path.abspath(a.src)
os.makedirs(os.path.join(a.out, "inputs"), exist_ok=True)
for f in ("quatrex_config.toml",):
    d = os.path.join(a.out, f)
    if not os.path.lexists(d):
        os.symlink(os.path.join(src, f), d)
for f in ("hamiltonian.h5", "overlap.h5", "structure.xyz"):
    d = os.path.join(a.out, "inputs", f)
    if not os.path.lexists(d):
        os.symlink(os.path.join(src, "inputs", f), d)
lines = open(os.path.join(src, "inputs", "structure.xyz")).read().split("\n")
n = int(lines[0])
lat = lines[1].split('Lattice="')[1].split('"')[0].split()
L = float(lat[4 * a.axis])
x = np.array([float(l.split()[1 + a.axis]) for l in lines[2:2 + n]])
x1, x2 = a.f1 * L, a.f2 * L
V = a.Vb * 0.5 * (np.tanh((x - x1) / a.w) - np.tanh((x - x2) / a.w))
V[np.abs(V) < 1e-6] = 0.0
np.save(os.path.join(a.out, "inputs", "potential.npy"), V)
print(f"{n} atoms, L {L:.2f} A: barrier {a.Vb} eV over x {x1:.2f} ... {x2:.2f} A (w {a.w} A); "
      f"V > 0 on x {x[V > 0].min():.1f} ... {x[V > 0].max():.1f} A -> {a.out}")
