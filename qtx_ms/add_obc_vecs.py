"""Append eigenvectors of the full open-boundary eigenproblem (obc_eig eigpairs.npz) to a level-1 basis, slab by slab:
each eigenvector restricted to slab s, orthogonalized against U1_s (twice), SVD over all eigenvectors, kept above --tol
(relative to the normalized slab parts). Saves <out>_U1_slab<s>.npy (ms_scan --modes <out>_U1).
Full-size eigenvectors: an oracle / diagnostic, not a scalable construction.

  python -m qtx_ms.add_obc_vecs --inputs inputs/mos2_hfo2 --U1 results_ms/mos2_hfo2_vcycle/tobc_t1e-6_lc \
      --eig results_ms/obc_eig/c1_Ec+0.017 --out results_ms/obc_eig/add/tobc_t1e-6_lc_c1
"""
import argparse
import json
import os

import numpy as np

from . import msbasis as mb
from .obc_eig_l1 import enrich_slabs

log = mb.log


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--U1", required=True)
    p.add_argument("--eig", required=True, help="comma list of obc_eig output dirs (eigpairs.npz)")
    p.add_argument("--max-im", type=float, default=None, help="only eigenvalues with |Im lambda| below this [eV]")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--nmax", type=int, default=400)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from .prep.inputs import read_inputs
    dev = mb.Device(read_inputs(a.inputs), obc="folded")
    Us = [np.load(f"{a.U1}_U1_slab{s}.npy") for s in range(len(dev.ns))]
    V, lam = [], []
    for d in a.eig.split(","):
        z = np.load(os.path.join(d, "eigpairs.npz"))
        sel = np.ones(len(z["lam"]), bool) if a.max_im is None else np.abs(z["lam"].imag) < a.max_im
        V.append(z["vec"][:, sel]); lam += list(z["lam"][sel])
        log(f"{d}: {sel.sum()} of {len(z['lam'])} eigenvectors")
    V = np.hstack(V)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new, added = enrich_slabs(dev, Us, V, a.tol, a.nmax)
    for s, u in enumerate(new):
        np.save(f"{a.out}_U1_slab{s}.npy", u)
    N = [u.shape[1] for u in new]
    log(f"N1 {N} = {sum(N)} (+{added}) -> {a.out}_U1_slab*.npy")
    json.dump(dict(args=vars(a), N1=N, added=added, lam=[[l.real, l.imag] for l in lam]), open(f"{a.out}.json", "w"))


if __name__ == "__main__":
    main()
