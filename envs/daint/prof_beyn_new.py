"""hier.obc Beyn Sigma at one real energy of the 12 nm device with the coupling-basis products on the GPU: time, and Sigma, B against the
cached files of job 5457257. python -u prof_beyn_new.py STORE PREP SIGMA_DIR"""
import sys, time
import numpy as np
from qtx_ms import devstore, dist, gpu
from qtx_ms import msbasis as mb
from qtx_ms.hier.obc import OBC

gpu.enable()
inp, dev = devstore.load(sys.argv[1], source=sys.argv[2])
Ec = inp["meta"]["edges"]["Ec_lead"]
o = OBC(dev, dist.Comm(), None, log=lambda *a: print(*a, flush=True), model="beyn-real")
t = time.time()
for e in (0.15, 0.30):
    z = complex(Ec + e, 1e-6)
    (SL, SR), B = o._beyn(z)
    f = lambda p: np.load(f"{sys.argv[3]}/{p}_{z.real:+.8f}_{z.imag:.8e}.npz")
    c, b = f("bsig"), f("bmode")["B"]
    rel = lambda x, y: float(np.abs(x - y).max() / np.abs(y).max())
    print(f"Ec {e:+.2f}: {time.time() - t:.1f} s since the start; Sigma L {rel(SL, c['SL']):.1e}, R {rel(SR, c['SR']):.1e}, "
          f"B B^H {rel(B @ B.conj().T, b @ b.conj().T):.1e}; reconstruction errors {[['%.0e' % x for x in o._leads[s][0]._bases[s]['err']] for s in 'LR']}", flush=True)
import glob, os
f = sorted(glob.glob(sys.argv[3] + "/sig_*"))[-1]                       # a folded contour point: the sparse couplings in Device._sigma
re, im = (float(x) for x in os.path.basename(f)[4:-4].split("_"))
c, t = np.load(f), time.time()
SL, SR = dev._sigma(complex(re, im))[0]
print(f"folded {os.path.basename(f)}: {time.time() - t:.1f} s, Sigma L {rel(SL, c['SL']):.1e}, R {rel(SR, c['SR']):.1e}", flush=True)
