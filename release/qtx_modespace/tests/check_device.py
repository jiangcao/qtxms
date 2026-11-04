"""Regression of the contour recursion on a real device (not part of pytest: minutes per point at ~8k-orbital slabs).

At contour points of the default window (Ec - 0.15 ... Ec + 0.6 eV) the diagonal blocks G_ss from core.g_diag are
compared with the 0.1.0 recursion (tests/ref010.g_diag_010), with timings. Use a Sigma cache that already
holds those points (e.g. the sigma/ of an earlier basis run), otherwise Sigma is computed.

    PYTHONPATH=src python tests/check_device.py DEVICE_FOLDER --sigma-cache OUT/sigma --points 0
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from qtx_modespace.quatrex import read_quatrex
from qtx_modespace import core
from ref010 import g_diag_010

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("folder")
p.add_argument("--sigma-cache", required=True)
p.add_argument("--mu", default="-0.15,0.6")
p.add_argument("--npts", type=int, default=24)
p.add_argument("--points", default="0", help="contour point indices (comma list)")
a = p.parse_args()

inp = read_quatrex(a.folder, cache_dir=a.sigma_cache)
dev = core.Device(inp, sigma_cache=a.sigma_cache)
Ec = inp["meta"]["edges"]["Ec_lead"]
mu1, mu2 = (Ec + float(v) for v in a.mu.split(","))
blk = core.slab_blocks(dev)
sym = core.is_real(blk)
blk_r = {k: [np.ascontiguousarray(np.real(M)) for M in v] for k, v in blk.items()} if sym else blk
iL, iR = dev.bL, dev.bR - dev.sb[-2]
c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
x, _ = np.polynomial.legendre.leggauss(a.npts)
th = 0.5 * np.pi * (x + 1)
print(f"slabs {dev.ns}, real H and S (symmetric recursion): {sym}", flush=True)
worst = 0.0
for ip in (int(v) for v in a.points.split(",")):
    z = c + r * np.exp(1j * th[ip])
    t = time.time(); SL, SR = dev.sigma(z); ts = time.time() - t
    t = time.time(); new = core.g_diag(blk_r, z, SL, SR, iL, iR, sym); tn = time.time() - t
    t = time.time(); ref = g_diag_010(blk, z, SL, SR, iL, iR); to = time.time() - t
    scale = max(np.abs(g).max() for g in ref)
    err = [float(np.abs(gn - gr).max() / scale) for gn, gr in zip(new, ref)]
    worst = max(worst, max(err))
    print(f"point {ip} z = {z:.6f}: sigma {ts:.0f} s, new {tn:.0f} s, 0.1.0 {to:.0f} s (x{to / tn:.2f}); "
          f"max |G_new - G_010| / max|G| per slab {', '.join(f'{e:.1e}' for e in err)}", flush=True)
print("PASS" if worst < 1e-10 else "FAIL", f"(worst {worst:.1e})")
sys.exit(0 if worst < 1e-10 else 1)
