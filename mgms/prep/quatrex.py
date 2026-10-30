"""Read a QuaTrEx / deviceBuilder device folder into the generic input set (mgms.prep.inputs).

Folder layout (as written by deviceBuilder, e.g. examples/device_MoS2HfO2_2808a_29088o):
    quatrex_config.toml        [device] transport_direction, num_orbitals_per_atom, contacts (origin, lattice
                               vectors, mid_gap_energy)
    inputs/structure.xyz       extended xyz (Lattice="..."), atoms in orbital order
    inputs/hamiltonian.h5      group "[0, 0, 0]": coo row / col / data, attrs shape    (eV)
    inputs/overlap.h5          same for S
    inputs/potential.npy       optional, per orbital or per atom (eV); used only with --potential
Only the R = [0, 0, 0] block is used (kpoint_grid 1 1 1: no transverse periodicity).
With --potential, as in QuaTrEx (electron/solver.py _apply_potential): H += (V S + S V) / 2.

Blocks: the contact lattice vector a (along the transport axis) defines primitive cells, counted from the left
contact origin. A transport block is m cells with m * a >= the coupling range of H and S (max distance along
the transport axis between coupled orbitals), so H, S are block tridiagonal. The first and last blocks are the
leads (the QuaTrEx contacts are the device's own end cells). The lead cell is the first block and its coupling
to its periodic image in the second block (matched atoms only; other atoms there, e.g. oxide, are device atoms).
The right lead block is reordered into the lead-cell atom order. If the potential differs between the contacts
(bias), the right lead is the left one shifted by dV S (written as leadR_*). Deviations of the end blocks from
these lead cells are reported. Band edges from the left-lead bands around mid_gap_energy.

    python -m mgms.prep.quatrex examples/device_MoS2HfO2_2808a_29088o --out inputs/mos2_hfo2
"""
import argparse, json, time, tomllib
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from .inputs import write_inputs
from .mos2 import lead_bands


def read_xyz(f):
    lines = Path(f).read_text().splitlines()
    n = int(lines[0].split()[0])
    lat = None
    if "Lattice=" in lines[1]:
        lat = np.array(lines[1].split('Lattice="')[1].split('"')[0].split(), float).reshape(3, 3)
    w = [ln.split() for ln in lines[2:2 + n]]
    return np.array([x[0] for x in w]), np.array([[float(v) for v in x[1:4]] for x in w]), lat


def read_h5(f, key="[0, 0, 0]"):
    import h5py
    with h5py.File(f) as h:
        keys = list(h.keys())
        g = h[key]
        shape = tuple(int(s) for s in g.attrs["shape"])
        M = sp.coo_matrix((g["data"][()], (g["row"][()], g["col"][()])), shape=shape).tocsr()
    if keys != [key]:
        print(f"  {Path(f).name}: blocks {keys}; only {key} used")
    return M


def symmetrize(M, name):
    """Full Hermitian matrix; completes a stored triangle."""
    d = abs(M - M.conj().T).max()
    if d < 1e-10 * abs(M).max():
        return 0.5 * (M + M.conj().T)
    lo, up = sp.tril(M, -1), sp.triu(M, 1)
    if lo.nnz == 0 or up.nnz == 0:
        print(f"  {name}: one triangle stored, completing")
        return (sp.triu(M) + sp.triu(M, 1).conj().T).tocsr()
    raise ValueError(f"{name} is not Hermitian (max |M - M^H| = {d:.2e})")


def coupling_range(M, x, off):
    """Max |x_i - x_j| over the nonzeros of M (orbital coordinates x)."""
    r, c = M.nonzero()
    return float(np.abs(x[r] - x[c]).max())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="QuaTrEx device folder (quatrex_config.toml, inputs/)")
    p.add_argument("--out", required=True)
    p.add_argument("--cells", type=int, help="primitive cells per block (default: smallest covering the coupling range)")
    p.add_argument("--potential", action="store_true",
                   help="add inputs/potential.npy to H as (V S + S V) / 2 (default: H as stored, no potential)")
    p.add_argument("--cut", type=float, default=0.0, help="drop |H|, |S| entries below cut (default keep all)")
    p.add_argument("--nk", type=int, default=41, help="k points for the lead band edges")
    a = p.parse_args()
    t0 = time.time()
    root = Path(a.folder)
    cfg = tomllib.loads((root / "quatrex_config.toml").read_text())
    dc = cfg["device"]
    ax = "abc".index(dc["transport_direction"])
    norb_sp = dc["num_orbitals_per_atom"]
    cL = next(c for c in dc["contacts"] if c["name"] == "left")
    cR = next(c for c in dc["contacts"] if c["name"] == "right")
    acell = float(np.linalg.norm(cL["lattice_vectors"][ax]))
    x0 = float(cL["origin"][ax])
    mid = float(cL.get("mid_gap_energy", 0.0))
    if tuple(dc.get("kpoint_grid", [1, 1, 1])) != (1, 1, 1):
        print(f"  kpoint_grid {dc['kpoint_grid']}: transverse k not supported, using Gamma ([0, 0, 0] block)")

    labels, xyz, lat = read_xyz(root / "inputs" / "structure.xyz")
    norb = np.array([norb_sp[s] for s in labels])
    off = np.concatenate([[0], np.cumsum(norb)])
    H = read_h5(root / "inputs" / "hamiltonian.h5")
    S = read_h5(root / "inputs" / "overlap.h5")
    n = int(off[-1])
    assert H.shape == (n, n) and S.shape == (n, n), f"H {H.shape}, S {S.shape}, sum(norb) {n}"
    H, S = symmetrize(H, "H"), symmetrize(S, "S")
    if a.cut > 0:
        H.data[np.abs(H.data) < a.cut] = 0; S.data[np.abs(S.data) < a.cut] = 0
        H.eliminate_zeros(); S.eliminate_zeros()
    print(f"{root.name}: {len(labels)} atoms {dict(zip(*np.unique(labels, return_counts=True)))}, {n} orbitals; "
          f"H nnz {H.nnz}, S nnz {S.nnz}; {time.time() - t0:.0f} s")

    pot = root / "inputs" / "potential.npy"
    V = None
    if pot.exists() and a.potential:
        V = np.load(pot)
        if len(V) == len(labels):
            V = np.repeat(V, norb)
        assert len(V) == n, f"potential has {len(V)} entries, {len(labels)} atoms, {n} orbitals"
        if np.abs(V).max() > 0:
            D = sp.diags(V)
            H = (H + 0.5 * (D @ S + S @ D)).tocsr()
        print(f"  potential: [{V.min():.3f}, {V.max():.3f}] eV added as (V S + S V) / 2")

    xo = np.repeat(xyz[:, ax], norb)
    rng = max(coupling_range(H, xo, off), coupling_range(S, xo, off))
    cell = np.floor((xyz[:, ax] - x0) / acell + 1e-9).astype(int)
    ncell = cell.max() + 1
    assert cell.min() == 0, "atoms before the left contact origin"
    m = a.cells or int(np.ceil(rng / acell - 1e-9))
    nb = ncell // m
    assert nb >= 4, f"{ncell} cells, {m} per block: too few blocks"
    blocks = np.minimum(cell // m, nb - 1)                    # remainder cells join the last device block
    if ncell % m:                                            # keep the right lead exactly m cells
        blocks = np.where(cell >= ncell - m, nb - 1, np.minimum(cell // m, nb - 2))
    # atoms of block b that are images of the lead block (block 0) shifted by `shift` cells, in block-0 order
    lead_at = np.where(blocks == 0)[0]

    def images(b, shift, tol=0.1):
        cand = np.where(blocks == b)[0]
        d = xyz[cand][None] - xyz[lead_at][:, None]
        d[..., ax] -= shift * acell
        dist = np.linalg.norm(d, axis=2)
        j = dist.argmin(1)
        ok = (dist[np.arange(len(lead_at)), j] < tol) & (labels[cand[j]] == labels[lead_at])
        return cand[j], ok

    # reorder the right lead block into the lead-cell atom order (the solvers map lead orbitals by position in
    # the block), the other atoms keep their order
    rimg, rok = images(nb - 1, ncell - m)
    assert rok.all() and (blocks == nb - 1).sum() == len(lead_at), "right lead block is not a copy of the left one"
    order = np.concatenate([np.where(blocks < nb - 1)[0], rimg])
    labels, xyz, norb, blocks, cell = labels[order], xyz[order], norb[order], blocks[order], cell[order]
    oo = np.concatenate([np.arange(off[i], off[i + 1]) for i in order])
    H, S = H[oo][:, oo].tocsr(), S[oo][:, oo].tocsr()
    if V is not None:
        V = V[oo]
    off = np.concatenate([[0], np.cumsum(norb)])
    lead_at = np.where(blocks == 0)[0]
    ob = np.repeat(blocks, norb)
    O = [np.where(ob == b)[0] for b in range(nb)]
    print(f"  transport axis {'xyz'[ax]}, cell {acell:.4f} A, {ncell} cells; coupling range {rng:.3f} A -> "
          f"{m} cells per block, {nb} blocks, orbitals per block {[len(o) for o in O]}")

    P = (abs(H) + abs(S)).tocsr()
    r, c = P.nonzero()
    assert np.abs(ob[r] - ob[c]).max() <= 1, "H, S not block tridiagonal with these blocks (use a larger --cells)"
    orbs = lambda at: np.concatenate([np.arange(off[i], off[i + 1]) for i in at])
    # lead cell: block 0 and its coupling to its periodic image in block 1 (MoS2 atoms shifted by m cells; other
    # atoms of block 1, e.g. oxide, are device atoms and enter only through the device-lead coupling)
    img1, ok1 = images(1, m)
    assert ok1.all(), f"block 1 lacks images of {np.sum(~ok1)} lead atoms (use larger --cells?)"
    o0, o1 = O[0], orbs(img1)
    lead = dict(H00=H[o0][:, o0], H01=H[o0][:, o1], S00=S[o0][:, o0], S01=S[o0][:, o1])
    checks = {}
    rb = O[nb - 1]
    checks["block1_image_vs_H00"] = dict(max_dH=float(abs(H[o1][:, o1] - lead["H00"]).max()),
                                         max_dS=float(abs(S[o1][:, o1] - lead["S00"]).max()))
    checks["right_vs_H00"] = dict(max_dH=float(abs(H[rb][:, rb] - lead["H00"]).max()),
                                  max_dS=float(abs(S[rb][:, rb] - lead["S00"]).max()))
    imgr, okr = images(nb - 2, ncell - 2 * m)
    if okr.all():
        o2 = orbs(imgr)
        checks["right_coupling_vs_H01"] = dict(max_dH=float(abs(H[o2][:, rb] - lead["H01"]).max()),
                                               max_dS=float(abs(S[o2][:, rb] - lead["S01"]).max()))
    lead_right, dV = None, 0.0
    if V is not None:
        print(f"  potential on the leads: left [{V[o0].min():.3f}, {V[o0].max():.3f}], right "
              f"[{V[rb].min():.3f}, {V[rb].max():.3f}] eV")
        dV = float(V[rb].mean() - V[o0].mean())
    if abs(dV) > 1e-6:           # right lead = left lead rigidly shifted by the potential difference (bias)
        lead_right = dict(H00=lead["H00"] + dV * lead["S00"], H01=lead["H01"] + dV * lead["S01"],
                          S00=lead["S00"], S01=lead["S01"])
        checks["right_vs_shifted_H00"] = dict(max_dH=float(abs(H[rb][:, rb] - lead_right["H00"]).max()))
        print(f"  right lead = left lead + {dV:+.4f} eV * S")
    print(f"  lead periodicity (max |difference|): {checks}")

    t = time.time()
    ks, E = lead_bands(lead["H00"], lead["H01"], lead["S00"], lead["S01"], nk=a.nk)
    Ev, Ec = float(E[E < mid].max()), float(E[E > mid].min())
    print(f"  lead {len(O[0])} orbitals: Ev {Ev:.4f}, Ec {Ec:.4f} eV, gap {Ec - Ev:.3f} eV (mid-gap {mid}), "
          f"{time.time() - t:.0f} s")

    meta = dict(name=f"{root.name} (QuaTrEx / deviceBuilder)", model="QuaTrEx input, non-orthogonal",
                units=dict(energy="eV", length="Angstrom"), orthogonal=False, axis=ax,
                block_length=float(m * acell), lead_cell_length=float(m * acell), nblocks=int(nb), periodic=None,
                transport=dict(device_blocks=list(range(1, nb - 1)), lead_left_block=0, lead_right_block=int(nb - 1),
                               lead_cell=f"first block ({m} contact cells) and its coupling to the second"),
                edges=dict(Ev_lead=Ev, Ec_lead=Ec, Eref=mid, right_lead_shift=dV),
                source=dict(folder=str(root.resolve()), cells=int(ncell), cell_length=acell, cells_per_block=int(m),
                            coupling_range=rng, potential=None if V is None else
                            dict(min=float(V.min()), max=float(V.max()), convention="H += (V S + S V) / 2"),
                            lead_checks=checks, contacts=dict(left=cL, right=cR)))
    write_inputs(a.out, labels, xyz, norb, blocks, H, S, lead=lead, meta=meta, lead_right=lead_right)
    np.savez(Path(a.out) / "lead_bands.npz", k=ks, E=E)
    print(f"-> {a.out}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
