"""Synthetic non-orthogonal TB test device (stand-in for real ab initio files).

Rough 'Si' wire on a cubic lattice (2 orbitals/atom: s-like valence, p-like conduction, gap ~2 eV),
H-passivated sidewalls (1 orbital), 1st+2nd-neighbour hopping, NN overlap.  Atoms are written in
shuffled order, H and S as 1-based upper-triangle triplets, to exercise the reader.
"""
import numpy as np
import scipy.sparse as sp

rng = np.random.default_rng(3)
a, nx, ny, nz, nideal, p_remove = 2.7, 60, 8, 5, 5, 0.35
present = np.ones((nx, ny, nz), bool)
for i in range(nideal, nx - nideal):           # sidewall roughness: remove surface atoms
    for j in range(ny):
        for k in (0, nz - 1):
            if rng.random() < p_remove:
                present[i, j, k] = False
    for k in range(1, nz - 1):
        for j in (0, ny - 1):
            if rng.random() < p_remove:
                present[i, j, k] = False

atoms, kind = [], []
for i, j, k in zip(*np.nonzero(present)):
    atoms.append(np.array([i, j, k], float) * a); kind.append("Si")
# passivate missing neighbours in the transverse directions
for i, j, k in zip(*np.nonzero(present)):
    for d in ((0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        jj, kk = j + d[1], k + d[2]
        if not (0 <= jj < ny and 0 <= kk < nz and present[i, jj, kk]):
            atoms.append(np.array([i, j, k], float) * a + 1.5 * np.array(d)); kind.append("H")
atoms, kind = np.array(atoms), np.array(kind)
perm = rng.permutation(len(atoms))
atoms, kind = atoms[perm], kind[perm]

norb = np.where(kind == "Si", 2, 1)
off = np.concatenate([[0], np.cumsum(norb)])
n = off[-1]
onsite = {"Si": [-3.0, 3.0], "H": [-4.0]}
rows, cols, hv, sv = [], [], [], []


def add(i, j, h, s):
    rows.append(i); cols.append(j); hv.append(h); sv.append(s)


for A in range(len(atoms)):
    for o, e in enumerate(onsite[kind[A]]):
        add(off[A] + o, off[A] + o, e, 1.0)
from scipy.spatial import cKDTree
pairs = cKDTree(atoms).query_pairs(a * np.sqrt(2) + 0.01)
for A, Bq in pairs:
    d = np.linalg.norm(atoms[A] - atoms[Bq])
    ka, kb = kind[A], kind[Bq]
    if ka == "Si" and kb == "Si":
        nn = d < a + 0.01
        hss, hpp, hsp = (-0.3, -0.3, 0.1) if nn else (-0.05, -0.05, 0.0)
        sss, spp = (0.04, 0.04) if nn else (0.0, 0.0)
        for (oa, ob, h, s) in ((0, 0, hss, sss), (1, 1, hpp, spp), (0, 1, hsp, 0), (1, 0, hsp, 0)):
            add(off[A] + oa, off[Bq] + ob, h, s)
    elif "H" in (ka, kb) and d < 1.6:
        si, hh = (A, Bq) if ka == "Si" else (Bq, A)
        add(off[si] + 0, off[hh], -0.5, 0.05)
        add(off[si] + 1, off[hh], 0.3, 0.0)

r, c, h, s = map(np.array, (rows, cols, hv, sv))
up = np.where(r <= c, 1, 0).astype(bool)          # store upper triangle only
r2, c2 = np.where(up, r, c), np.where(up, c, r)
with open("test_H.txt", "w") as f:
    f.write("# i j Re(H) [eV], 1-based, upper triangle\n")
    np.savetxt(f, np.c_[r2 + 1, c2 + 1, h], fmt="%d %d %.10f")
nz_s = s != 0
with open("test_S.txt", "w") as f:
    f.write("# i j S, 1-based, upper triangle\n")
    np.savetxt(f, np.c_[r2[nz_s] + 1, c2[nz_s] + 1, s[nz_s]], fmt="%d %d %.10f")
with open("test.xyz", "w") as f:
    f.write(f"{len(atoms)}\nsynthetic rough wire, Angstrom\n")
    for kk, x in zip(kind, atoms):
        f.write(f"{kk} {x[0]:.5f} {x[1]:.5f} {x[2]:.5f}\n")
print(f"{len(atoms)} atoms ({(kind == 'Si').sum()} Si, {(kind == 'H').sum()} H), n = {n} orbitals")
