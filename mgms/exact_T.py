"""Exact QTBM transmission of a prepared input set (sparse direct solve, no mode space): checks a new input set.

    python -m mgms.exact_T --inputs inputs/mos2_hfo2 --dE 0.05,0.15
"""
import argparse, os, time
import numpy as np
from mgms.prep.inputs import read_inputs
from mgms import msbasis as mb

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--inputs", required=True)
p.add_argument("--dE", default="0.05,0.15", help="energies - Ec_lead [eV]")
p.add_argument("--Ec", type=float, help="reference edge for --dE (default Ec_lead of the inputs; set it to compare "
               "input sets at the same absolute energies)")
a = p.parse_args()
inp = read_inputs(a.inputs)
dev = mb.Device(inp)
dev.sigma_cache = os.path.join("results_ms", "sigma", os.path.basename(os.path.normpath(a.inputs)))
Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
print(f"device {dev.n} orbitals, slabs {dev.ns}, |bL| {len(dev.bL)}, |bR| {len(dev.bR)}, "
      f"separate right lead {dev.leadR is not dev.lead}", flush=True)
for dE in (float(x) for x in a.dE.split(",")):
    E = Ec + dE
    t = time.time()
    SL, SR = dev.sigma(E + 1e-6j)
    ts = time.time() - t
    B = mb.injection(SL)[0]
    nR = mb.injection(SR)[0].shape[1]
    GR = 1j * (SR - SR.conj().T)
    b = np.zeros((dev.n, B.shape[1]), complex)
    b[dev.bL] = B
    cs = mb.CSolver(dev.A(E, SL, SR))
    psi = cs.solve(b)
    T = float(np.real(np.einsum("ic,ij,jc->", psi[dev.bR].conj(), GR, psi[dev.bR])))
    print(f"E = Ec {dE:+.3f}: channels L {B.shape[1]} / R {nR}, T = {T:.5f}, residual {cs.res:.1e} "
          f"(GMRES its {cs.its}, perturbed pivots {cs.npert}); sigma {ts:.0f} s, total {time.time() - t:.0f} s",
          flush=True)
    cs.free()
