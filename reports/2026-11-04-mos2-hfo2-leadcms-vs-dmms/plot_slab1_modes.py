"""Level-1 local density matrix of slab 1 (slab partition, CAP+OBC, 24-pt contour, results_ms/mos2_hfo2_slabl1_tobc24/
tiles_obc_1e-10/tile_obc_001.npz): the eigenvectors whose occupation is closest to 1, 1e-2, 1e-4, 1e-6, 1e-8, weight
|u_i|^2 summed per atom (eigenvectors orthonormal in the Euclidean metric), log colour map, one row per mode."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb
from qtx_ms.plot_eigmodes import plot_modes

inp = read_inputs("inputs/mos2_hfo2")
dev = mb.Device(inp, obc="folded")
Ec = inp["meta"]["edges"]["Ec_lead"]
z = np.load("results_ms/mos2_hfo2_slabl1_tobc24/tiles_obc_1e-10/tile_obc_001.npz")
U, s, atoms = z["U"], z["s"], z["atoms"]
norb = inp["norb"]
aidx = np.repeat(np.arange(len(atoms)), norb[atoms])            # row -> position in the slab atom list
modes = []
for target in (1.0, 1e-2, 1e-4, 1e-6, 1e-8):
    i = int(np.argmin(np.abs(np.log10(s[:U.shape[1]]) - np.log10(target))))
    wa = np.bincount(aidx, np.abs(U[:, i]) ** 2, minlength=len(atoms))
    w = np.zeros(len(norb)); w[atoms] = wa
    modes.append((f"slab 1 local-rho eigenvector #{i + 1} of {len(s)}: occupation {s[i]:.2e} (target {target:g})", w))
    print(target, i + 1, s[i])
f = os.path.join(OUT, "fig7_slab1_local_rho_modes.png")
plot_modes(f, inp, dev, modes, Ec)
print("->", f)
