"""Add the transverse periodicity of a QuaTrEx supercell (qtx_ms.prep.quatrex.detect_periodic) to the meta.json of
input sets written before it was detected; H, S and all other files are left unchanged.

    python -m qtx_ms.prep.set_periodic inputs/mos2_hfo2 [inputs/mos2_hfo2_trim_c3 ...]
"""
import json, os, sys
from pathlib import Path

from qtx_ms.prep.inputs import read_inputs
from qtx_ms.prep.quatrex import read_xyz, detect_periodic

for d in sys.argv[1:]:
    inp = read_inputs(d)
    meta = inp["meta"]
    _, _, lat = read_xyz(Path(meta["source"]["folder"]) / "inputs" / "structure.xyz")
    per, info = detect_periodic(inp["xyz"], inp["norb"], inp["H"], inp["S"], lat, meta["axis"])
    print(f"{d}: periodic {per} {info} (was {meta.get('periodic')})")
    meta["periodic"] = per or None
    f = os.path.join(d, "meta.json")
    json.dump(meta, open(f, "w"), indent=1)
