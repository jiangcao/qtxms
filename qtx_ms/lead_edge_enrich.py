"""Lead band-edge (subband onset) enrichment of a level-1 basis.

The tile bases miss the states within ~ hbar^2 (pi / L_emb)^2 / 2m* of each lead subband edge: their envelopes vary
on the device scale (wavelength / decay length >> the tile embedding), so the local spectral function of an embedded
tile does not resolve them. They are lead Bloch states near the band extrema. Here:
  1. band extrema (k_e, E_e) of the lead cell in [Ec + emin, Ec + emax] from inputs/<name>/lead_bands.npz
  2. Bloch states psi_n(k) of H(k) psi = E S(k) psi at k = k_e + j dk (|j| <= nside) with |E_n(k) - E_e| < delta
     (and psi(-k) = conj(psi(k)) for a real lead): standing / slowly evanescent envelopes over a slab length
  3. every device slab: lead-cell atoms matched to the slab atoms (lead_rms.slab_cell_map; oxide / unmatched atoms get
     zero), orthogonalized against U1_s (twice), SVD, kept above --tol (normalized candidates), at most --nmax
  4. saves <out>_U1_slab<s>.npy (use with ms_scan --modes <out>_U1)

  python -m qtx_ms.lead_edge_enrich --inputs inputs/mos2_hfo2 --U1 results_ms/mos2_hfo2_vcycle/tobc_t1e-6_lc \
      --emax 0.3 --delta 0.05 --out results_ms/lead_edge/tobc_t1e-6_lc_d0.05
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.linalg as sla

from . import msbasis as mb
from .lead_rms import LeadCell, slab_cell_map, to_slab

log = mb.log


def band_extrema(k, E, lo, hi):
    """(k index, energy) of the local extrema of every sorted band E[:, b] in [lo, hi] (endpoints k = 0, pi count)."""
    out = []
    nk = len(k)
    for b in range(E.shape[1]):
        e = E[:, b]
        if e.max() < lo or e.min() > hi:
            continue
        for i in range(nk):
            l = e[i - 1] if i > 0 else e[1]
            r = e[i + 1] if i < nk - 1 else e[-2]
            if ((e[i] <= l and e[i] <= r) or (e[i] >= l and e[i] >= r)) and lo <= e[i] <= hi:
                out.append((i, float(e[i])))
    return sorted(set(out))


def edge_states(cell, kgrid, ext, delta, nside, dk, elo, ehi, log=log):
    """Bloch vectors (cell orbitals x n) near the extrema ext = [(k index, E_e)], and their (k, E)."""
    real = not (np.iscomplexobj(cell.H00) and np.abs(cell.H00.imag).max() > 0) and \
        not (np.iscomplexobj(cell.H01) and np.abs(cell.H01.imag).max() > 0)
    want = {}
    for i, Ee in ext:
        for j in range(-nside, nside + 1):
            q = float(np.clip(kgrid[i] + j * dk, 0, np.pi))
            want.setdefault(round(q, 10), []).append(Ee)
    vecs, info = [], []
    t0 = time.time()
    for q, Ees in sorted(want.items()):
        H, S = cell.hk(q)
        e, V = sla.eigh(H, S, subset_by_value=(elo, ehi), driver="gvx", check_finite=False)
        Ees = np.array(Ees)
        keep = np.array([np.abs(Ees - x).min() < delta for x in e], bool) if len(e) else np.zeros(0, bool)
        for x, v in zip(e[keep], V[:, keep].T):
            vecs.append(v); info.append((q, float(x)))
            if real and 0 < q < np.pi:
                vecs.append(v.conj()); info.append((-q, float(x)))
    log(f"  {len(want)} k points, {len(vecs)} Bloch vectors near the edges ({'real' if real else 'complex'} lead), "
        f"{time.time() - t0:.0f} s")
    return (np.array(vecs).T if vecs else np.zeros((cell.n, 0), complex)), info


def enrich(dev, Us, Psi, maps, tol, nmax):
    new, added, svs = [], [], []
    for s, (U, cmap) in enumerate(zip(Us, maps)):
        C = to_slab(Psi, cmap, dev.ns[s])
        nrm = np.linalg.norm(C, axis=0)
        C = C[:, nrm > 1e-12] / nrm[nrm > 1e-12]
        for _ in range(2):
            C = C - U @ (U.conj().T @ C)
        W, sv, _ = np.linalg.svd(C, full_matrices=False)
        k = int(min((sv > tol).sum(), nmax))
        new.append(np.hstack([U, W[:, :k]])); added.append(k); svs.append(sv)
        log(f"  slab {s}: {C.shape[1]} candidates, outside span(U1) sv max {sv[0] if len(sv) else 0:.2e}, "
            f"# > 0.3 / 0.1 / 0.03 / 0.01: {[(sv > x).sum() for x in (0.3, 0.1, 0.03, 0.01)]} -> +{k}")
    return new, added, svs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--U1", required=True, help="level-1 prefix (<prefix>_U1_slab<s>.npy)")
    p.add_argument("--emin", type=float, default=-0.01, help="extrema window lower end - Ec [eV]")
    p.add_argument("--emax", type=float, default=0.3, help="extrema window upper end - Ec [eV]")
    p.add_argument("--delta", type=float, default=0.05, help="keep Bloch states within delta of an extremum [eV]")
    p.add_argument("--nside", type=int, default=3, help="k points on each side of an extremum")
    p.add_argument("--dk", type=float, default=None, help="k step (default: half the lead_bands grid step)")
    p.add_argument("--tol", type=float, default=0.05, help="SVD keep threshold of the normalized candidates")
    p.add_argument("--nmax", type=int, default=400)
    p.add_argument("--map-tol", type=float, default=0.6)
    p.add_argument("--out", required=True, help="output prefix")
    a = p.parse_args(argv)

    from .prep.inputs import read_inputs
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc="folded")
    Ec = inp["meta"]["edges"]["Ec_lead"]
    cell = LeadCell(inp["lead"])
    Us = [np.load(f"{a.U1}_U1_slab{s}.npy") for s in range(len(dev.ns))]
    lb = np.load(os.path.join(a.inputs, "lead_bands.npz"))
    kg, Eg = lb["k"], np.sort(lb["E"], axis=1)
    ext = band_extrema(kg, Eg, Ec + a.emin, Ec + a.emax)
    dk = a.dk or 0.5 * (kg[1] - kg[0])
    log(f"{a.inputs}: lead cell {cell.n}, Ec {Ec:.5f}; U1 {a.U1} N1 {[u.shape[1] for u in Us]}; "
        f"{len(ext)} band extrema in [Ec{a.emin:+}, Ec{a.emax:+}] at k indices {sorted({i for i, _ in ext})}")
    log("  extrema - Ec (meV): " + " ".join(f"{1e3 * (e - Ec):.0f}" for _, e in ext))
    fpsi = f"{a.out}_psi.npz"
    if os.path.exists(fpsi):
        z = np.load(fpsi)
        Psi, info = z["Psi"], json.loads(str(z["info"]))
    else:
        Psi, info = edge_states(cell, kg, ext, a.delta, a.nside, dk, Ec + a.emin - a.delta, Ec + a.emax + a.delta)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.savez(fpsi, Psi=Psi, info=json.dumps(info))
    maps = []
    for s in range(len(dev.ns)):
        cmap, mi = slab_cell_map(inp, dev, s, tol=a.map_tol, return_info=True)
        maps.append(cmap)
        log(f"  slab {s}: lead-cell atoms unmatched {mi['missing']}, max displacement {mi['max_dist']:.3f} A")
    new, added, svs = enrich(dev, Us, Psi, maps, a.tol, a.nmax)
    for s, u in enumerate(new):
        np.save(f"{a.out}_U1_slab{s}.npy", u)
    N = [u.shape[1] for u in new]
    log(f"enriched N1 {N} = {sum(N)} (+{added}, was {sum(u.shape[1] for u in Us)}) -> {a.out}_U1_slab*.npy")
    json.dump(dict(args=vars(a), N1=N, added=added, n_psi=int(Psi.shape[1]), extrema=ext,
                   sv=[x[:50].tolist() for x in svs]), open(f"{a.out}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
