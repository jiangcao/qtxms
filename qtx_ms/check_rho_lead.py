"""Check of the contour slab density matrix on the pristine lead: for the infinite chain of lead cells,
rho_00 = -(1/pi) Im int_[mu1, mu2] (E S00 - H00 - Sigma_L - Sigma_R)^-1 dE (contour as msbasis.rho_contour)
must equal the k integral of the window Bloch projectors, int dk/2pi sum_{E_n(k) in window} psi psi^H.

    python -m qtx_ms.check_rho_lead --inputs inputs/mos2_20nm [--npts 24]
"""
import argparse, time
import numpy as np
import scipy.linalg as sla
from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb
from qtx_ms.lead_rms import LeadCell

p = argparse.ArgumentParser()
p.add_argument("--inputs", required=True)
p.add_argument("--mu", default="0,0.395")
p.add_argument("--npts", type=int, default=24)
p.add_argument("--nk", type=int, default=400)
a = p.parse_args()
inp = read_inputs(a.inputs)
Ec = inp["meta"]["edges"]["Ec_lead"]
mu1, mu2 = (Ec + float(x) for x in a.mu.split(","))
cell = LeadCell(inp["lead"])
lead = mb.FoldedLead(inp["lead"]["H00"], inp["lead"]["H01"], inp["lead"]["S00"], inp["lead"]["S01"])
H00, H01, S00, S01 = cell.H00, cell.H01, cell.S00, cell.S01
oL, oR = lead.oL, lead.oR
print("lead faces", len(lead.Lf), len(lead.Rf), "interior", len(lead.I), "oL", len(oL), "oR", len(oR), flush=True)
c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
x, w = np.polynomial.legendre.leggauss(a.npts)
th, w = 0.5 * np.pi * (x + 1), 0.5 * np.pi * w
acc = np.zeros_like(H00, dtype=complex)
t0 = time.time()
for t_, w_ in zip(th, w):
    z = c + r * np.exp(1j * t_)
    dz = -1j * r * np.exp(1j * t_) * w_
    gL, gR = lead.surface_gfs(z)
    # cell 0 couples to cell -1 through H01^H (left) and to cell +1 through H01 (right)
    A01 = z * S01 - H01                                   # (0, +1)
    A10 = A01.conj().T if False else (z * S01.conj().T - H01.conj().T)
    SigR = A01[:, oR] @ gR @ A10[oR, :]
    SigL = A10[:, oL] @ gL @ A01[oL, :]
    acc += np.linalg.inv(z * S00 - H00 - SigL - SigR) * dz
rho_c = -((acc - acc.conj().T) / 2j) / np.pi
print(f"contour: {time.time() - t0:.0f} s", flush=True)
rho_k = np.zeros_like(acc)
for k in np.linspace(-np.pi, np.pi, a.nk, endpoint=False):
    e, Z = sla.eigh(*cell.hk(k))
    s = (e > mu1) & (e < mu2)
    rho_k += Z[:, s] @ Z[:, s].conj().T / a.nk
print(f"tr(rho S00): contour {np.trace(rho_c @ S00).real:.4f}, k integral {np.trace(rho_k @ S00).real:.4f}")
print(f"|rho_c - rho_k| / |rho_k| = {np.linalg.norm(rho_c - rho_k) / np.linalg.norm(rho_k):.2e}")
