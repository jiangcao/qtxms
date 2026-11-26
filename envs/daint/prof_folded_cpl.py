"""Folded Sigma of the 12 nm lead at one contour point: sparse against dense couplings with the SAME surface functions, and the surface
functions recomputed (reproducibility of the decimation). python -u prof_folded_cpl.py STORE PREP SIGMA_DIR"""
import glob, os, sys
import numpy as np
from qtx_ms import devstore, gpu

gpu.enable()
inp, dev = devstore.load(sys.argv[1], source=sys.argv[2])
rel = lambda x, y: float(np.abs(x - y).max() / np.abs(y).max())
for f in (sorted(glob.glob(sys.argv[3] + "/sig_*"))[-1], sorted(glob.glob(sys.argv[3] + "/sig_*"))[len(glob.glob(sys.argv[3] + "/sig_*")) // 2]):
    re, im = (float(x) for x in os.path.basename(f)[4:-4].split("_"))
    z, c = complex(re, im), np.load(f)
    print(os.path.basename(f), flush=True)
    g = [gpu.asnumpy(x) for x in dev.lead.surface_gfs(z)]
    g2 = [gpu.asnumpy(x) for x in dev.lead.surface_gfs(z)]
    for k, (nm, cc) in enumerate((("L", dev.cL), ("R", dev.cR))):
        S = dev._coupled(z, g[k], cc)
        Hld, Sld, Hdl, Sdl = (M.toarray() for M in cc)
        D = (z * Sdl - Hdl) @ g[k] @ (z * Sld - Hld)
        print(f"  {nm}: sparse vs dense, same g {rel(S, D):.1e}; g recomputed {rel(g2[k], g[k]):.1e}; Sigma of g2 vs g {rel(dev._coupled(z, g2[k], cc), S):.1e}; "
              f"this vs cached file {rel(S, c['S' + nm]):.1e}; dense vs cached {rel(D, c['S' + nm]):.1e}; file z exact {z}", flush=True)
