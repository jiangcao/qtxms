"""Generic transport input set: what every TB model (empirical or ab initio) hands to the solvers.

<dir>/device.xyz        atoms in orbital order: label x y z [A]   (labels only identify species)
<dir>/norb.txt          orbitals per atom, one integer per line (orbitals of an atom are consecutive)
<dir>/blocks.txt        transport block (slab) index per atom
<dir>/H.npz, S.npz      device H [eV] and overlap S (scipy.sparse.save_npz); S = I for orthogonal TB
<dir>/lead_H00.npz ...  lead_{H,S}{00,01}: periodic lead cell and its coupling to the next cell along +x
<dir>/leadR_H00.npz ... optional right lead (same orbital order as the right lead block), e.g. under bias;
                        without it both leads are lead_*
<dir>/meta.json         units, transport axis, block length, device / lead blocks, band edges, model info
<dir>/perm.npy          optional: for each orbital of H, its index in the source model's files (e.g. QuaTrEx
                        hamiltonian.h5), when the preparation reordered orbitals

Model-specific extras may sit next to these (e.g. structure.npz of the Si sheet with bonds and species
codes); solvers that only need the generic part must not depend on them. The files are readable by
mgms.abtb as well (read_xyz, parse_norb with norb.txt, read_sparse with the .npz files).
"""
import json
import numpy as np
import scipy.sparse as sp
from pathlib import Path

LEAD_KEYS = ("H00", "H01", "S00", "S01")


def write_inputs(out, labels, xyz, norb, blocks, H, S=None, lead=None, meta=None, lead_right=None, perm=None):
    """lead: dict with H00, H01 (and S00, S01; default I, 0); lead_right: same for a different right lead.
    meta: JSON-serialisable dict. perm: source orbital index of every orbital (saved as perm.npy)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = int(np.sum(norb))
    assert H.shape == (n, n), f"H is {H.shape}, sum(norb) = {n}"
    with open(out / "device.xyz", "w") as f:
        f.write(f"{len(labels)}\n{(meta or {}).get('name', '')}\n")
        for s, r in zip(labels, xyz):
            f.write(f"{s:<5s} {r[0]:14.8f} {r[1]:14.8f} {r[2]:14.8f}\n")
    np.savetxt(out / "norb.txt", np.asarray(norb, int), fmt="%d")
    np.savetxt(out / "blocks.txt", np.asarray(blocks, int), fmt="%d")
    sp.save_npz(out / "H.npz", sp.csr_matrix(H))
    sp.save_npz(out / "S.npz", sp.csr_matrix(S) if S is not None else sp.identity(n, format="csr"))
    for pre, ld in (("lead", lead), ("leadR", lead_right)):
        if ld is not None:
            m = ld["H00"].shape[0]
            ld = dict(dict(S00=sp.identity(m, format="csr"), S01=sp.csr_matrix((m, m))), **ld)
            for k in LEAD_KEYS:
                sp.save_npz(out / f"{pre}_{k}.npz", sp.csr_matrix(ld[k]))
    if perm is not None:
        assert len(perm) == n, f"perm has {len(perm)} entries, {n} orbitals"
        np.save(out / "perm.npy", np.asarray(perm, np.int64))
    json.dump(meta or {}, open(out / "meta.json", "w"), indent=1)


def read_inputs(d):
    """dict(labels, xyz, norb, blocks, H, S, lead (dict or None), lead_right (dict or None), perm (or None), meta)."""
    d = Path(d)
    with open(d / "device.xyz") as f:
        lines = f.read().splitlines()
    nat = int(lines[0].split()[0])
    w = [ln.split() for ln in lines[2:2 + nat]]
    labels = np.array([x[0] for x in w])
    xyz = np.array([[float(v) for v in x[1:4]] for x in w])
    lead = ({k: sp.load_npz(d / f"lead_{k}.npz").tocsr() for k in LEAD_KEYS}
            if (d / "lead_H00.npz").exists() else None)
    lead_right = ({k: sp.load_npz(d / f"leadR_{k}.npz").tocsr() for k in LEAD_KEYS}
                  if (d / "leadR_H00.npz").exists() else None)
    return dict(labels=labels, xyz=xyz, norb=np.loadtxt(d / "norb.txt", dtype=int).ravel(),
                blocks=np.loadtxt(d / "blocks.txt", dtype=int).ravel(),
                H=sp.load_npz(d / "H.npz").tocsr(), S=sp.load_npz(d / "S.npz").tocsr(), lead=lead, lead_right=lead_right,
                perm=np.load(d / "perm.npy") if (d / "perm.npy").exists() else None, meta=json.load(open(d / "meta.json")))
