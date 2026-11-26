"""Phases of the Beyn Sigma at 12 nm (left lead, bases built first) and how many fixed-point steps it needs: Sigma and B after 1, 2, ... steps
(fp_maxit = k) against 30 steps without the stall test. python -u prof_fp.py STORE PREP"""
import sys, time
import numpy as np
from qtx_ms import devstore, dist, gpu
from qtx_ms.hier.obc import OBC

gpu.enable()
inp, dev = devstore.load(sys.argv[1], source=sys.argv[2])
Ec = inp["meta"]["edges"]["Ec_lead"]
o = OBC(dev, dist.Comm(), None, log=lambda *a: print(*a, flush=True), model="beyn")
o._beyn(complex(Ec + 0.15, 0.0))                                          # leads, bases, kernels
bl, (Hld, Sld, Hdl, Sdl) = o._leads["L"]
rel = lambda x, y: float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-300))


def run(z, k=None, stall=True):
    bl.fp_maxit, bl.fp_stall = (k or 300), stall
    t = time.time()
    S, B = bl.sigma(z, "L", z * Sdl - Hdl, z * Sld - Hld)
    return S, B, time.time() - t


for z in (Ec + 0.15 + 0j, Ec + 0.30 + 0j, Ec + 0.50 + 0j, Ec + 0.6 + 0.0028j, Ec + 0.6 + 0.0149j):
    R, BR, _ = run(z, 30, False)
    h30 = bl.info["fp_hist"]
    S, B, t = run(z)
    print(f"z = Ec {z.real - Ec:+.2f} {z.imag:+.4f}i: default {t:.1f} s, {bl.info['fp_it']} steps, modes {bl.info['modes']}, incoming {bl.info['channels']}; "
          f"Sigma vs 30 steps {rel(S, R):.1e}, B B^H {rel(B @ B.conj().T, BR @ BR.conj().T) if B.shape[1] else 0:.1e}")
    print("   phases " + ", ".join(f"{k} {v:.2f}" for k, v in bl.info["phase_t"].items()) + " s")
    print("   change per step " + " ".join(f"{x:.0e}" for x in h30[:14]))
    err = []
    for k in range(1, 9):
        Sk, Bk, tk = run(z, k, False)
        err.append(f"{k}: {rel(Sk, R):.0e} / {rel(Bk @ Bk.conj().T, BR @ BR.conj().T) if Bk.shape[1] else 0:.0e} ({tk:.1f} s)")
    print("   Sigma / B B^H error after k steps  " + "  ".join(err), flush=True)
