"""Write slab-partition level-1 bases (one tile per slab, run_ms --tile-orbs 100000) as ms_scan mode files
<prefix>_slab<s>.npy (slab-local orbitals x modes), keeping occupations > cut.

    python -m qtx_ms.slab_l1_modes --inputs inputs/mos2_hfo2 --tiles results_ms/mos2_hfo2_slabl1_tobc24/tiles_obc_1e-10 \
        --cut 1e-8 --out results_ms/mos2_hfo2_slabl1_tobc24/modes/t1e-8
"""
import argparse, os
import numpy as np
from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--inputs", required=True)
p.add_argument("--tiles", required=True)
p.add_argument("--cut", type=float, required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
inp = read_inputs(a.inputs)
dev = mb.Device(inp, obc="folded")
blocks, _, _ = mb.partition(dev, cores=False, tile_orbs=100000)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
for i, b in enumerate(blocks):
    z = np.load(os.path.join(a.tiles, f"tile_obc_{i:03d}.npz"))
    assert np.array_equal(z["atoms"], b["atoms"])
    k = int(np.sum(z["s"][:z["U"].shape[1]] > a.cut))
    X = np.zeros((dev.ns[b["slab"]], k), z["U"].dtype)
    X[b["orbs"]] = z["U"][:, :k]
    np.save(f"{a.out}_slab{b['slab']}.npy", X)
    print(f"slab {b['slab']}: {k} modes")
