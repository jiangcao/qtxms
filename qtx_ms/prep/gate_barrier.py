"""Gate barrier on a generic input set: H += (V S + S V) / 2 (QuaTrEx convention), V per atom from its transport
coordinate, V(x) = Vb * [tanh((x - x1) / w) - tanh((x - x2) / w)] / 2, zero on the lead blocks (asserted small at
the device faces). Default x1 / x2: the outer faces of the device slabs given by --slabs (default the middle two).
Everything else is linked from the source set (same lead, same Sigma cache can be reused).

    python -m qtx_ms.prep.gate_barrier --inputs inputs/mos2_hfo2 --Vb 0.4 --out inputs/mos2_hfo2_gate0.4
"""
import argparse, json, os
import numpy as np
import scipy.sparse as sp
from qtx_ms.prep.inputs import read_inputs

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--inputs", required=True)
p.add_argument("--out", required=True)
p.add_argument("--Vb", type=float, required=True, help="barrier height [eV]")
p.add_argument("--slabs", default="1,2", help="device slabs under the gate (0-based device slab indices)")
p.add_argument("--w", type=float, default=2.0, help="ramp width [A]")
a = p.parse_args()
inp = read_inputs(a.inputs)
meta = inp["meta"]
ax = meta.get("axis", 0)
x = inp["xyz"][:, ax]
dblk = meta["transport"]["device_blocks"]
sl = [dblk[int(s)] for s in a.slabs.split(",")]
x1 = x[np.isin(inp["blocks"], sl)].min()
x2 = x[np.isin(inp["blocks"], sl)].max()
Va = a.Vb * 0.5 * (np.tanh((x - x1) / a.w) - np.tanh((x - x2) / a.w))
lead = np.isin(inp["blocks"], [meta["transport"]["lead_left_block"], meta["transport"]["lead_right_block"]])
print(f"barrier {a.Vb} eV over x = {x1:.2f} ... {x2:.2f} A (blocks {sl}), ramp {a.w} A; max |V| on lead blocks "
      f"{np.abs(Va[lead]).max():.1e} eV -> set to 0")
Va[lead] = 0.0
V = sp.diags(np.repeat(Va, inp["norb"]))
S = inp["S"]
H = (inp["H"] + 0.5 * (V @ S + S @ V)).tocsr()
H = (0.5 * (H + H.T.conj())).tocsr()
src = os.path.abspath(a.inputs)
os.makedirs(a.out, exist_ok=True)
for f in os.listdir(src):
    if f in ("H.npz", "meta.json") or f.startswith("meta.json."):
        continue
    dst = os.path.join(a.out, f)
    if not os.path.exists(dst):
        os.symlink(os.path.join(src, f), dst)
sp.save_npz(os.path.join(a.out, "H.npz"), H)
meta = dict(meta, gate=dict(Vb=a.Vb, x1=float(x1), x2=float(x2), w=a.w, slabs=a.slabs, source=src,
                            convention="H += (V S + S V) / 2"))
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
np.save(os.path.join(a.out, "potential_atoms.npy"), Va)
print(f"-> {a.out}")
