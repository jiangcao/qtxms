"""Mode-space input set for the QuaTrEx production code: a saved basis applied to a QuaTrEx device.

    python -m qtx_modespace.to_quatrex DEVICE_FOLDER --modes DIR --out QTX_DIR [--cells 3 --trim] --dE 0:0.5:0.05
    python -m qtx_modespace.to_quatrex DEVICE_FOLDER --exact --out QTX_EXACT [--cells 3 --trim] ...   # reference
    cd QTX_DIR && quatrex run quatrex_config.toml        # -> outputs/transmission_*.npy

Device (code order of read_quatrex): blocks 0 … nb-1 of --cells contact cells; 0 and nb-1 are the lead blocks, the
device slabs 1 … nb-2 carry the basis (DIR/slab<s>.npy, rows checked against slab<s>_orbitals.npy). The written
device is

    [PL | lead block 0 | U_1 ... U_S | lead block nb-1 | PR]

with H_red = U^H H U, S_red = U^H S U, U = diag(I, I, U_1 … U_S, I, I). PL, PR are one extra copy of each lead block
outside the device (same atoms shifted by one block; PL-block 0 coupling = block 0 to its images in block 1,
block nb-1 - PR coupling = images of block nb-1 in block nb-2 to block nb-1). They are part of the semi-infinite leads,
so T is unchanged, and they keep QuaTrEx's real-space contact discovery on real atoms: each contact is defined as one
whole lead block (lattice vector = block length), whose only periodic image in the device is the lead block, so the
QuaTrEx lead is exactly (H00, H01) of the lead block. Reduced orbitals are written as one pseudo-atom per slab
(species U<s>, N_s orbitals) at the slab centre. With --exact all U_s = I (same padding): the full-orbital
reference for the same leads.

--potential adds inputs/potential.npy to H (as read_quatrex) before the projection; the written set has no potential
file. The QuaTrEx config is written for the current github.com/quatrex/quatrex (formalism "wf" = QTBM), energies
Ec + dE in inputs/electron_energies.npy.
"""
import argparse, json, os, time, tomllib
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from .quatrex import read_quatrex
from .cli import energies
from .core import log


def images(xyz, labels, cand, ref, shift, tol=0.1):
    """for each atom of ref, the atom of cand at its position + shift (same species)"""
    d = np.linalg.norm(xyz[cand][None] - (xyz[ref] + shift)[:, None], axis=2)
    j = d.argmin(1)
    assert (d[np.arange(len(ref)), j] < tol).all() and (labels[cand[j]] == labels[ref]).all(), "images not found"
    return cand[j]


def in_cell(xyz, origin, lv, n=0):
    """atoms in repetition n (transport) of the contact cell, as in quatrex.contact.discovery"""
    f = (xyz - origin) @ np.linalg.inv(lv)
    ok = (f[:, 0] >= n) & (f[:, 0] < n + 1)
    for i in (1, 2):
        ok &= (f[:, i] >= 0) & (f[:, i] < 1)
    return np.where(ok)[0]


def write_h5(f, M):
    import h5py
    M = sp.triu(sp.csr_matrix(M)).tocoo()
    with h5py.File(f, "w") as h:
        g = h.create_group("[0, 0, 0]")
        g.attrs["format"] = "coo"
        g.attrs["shape"] = np.array(M.shape)
        g.create_dataset("row", data=M.row.astype(np.int32))
        g.create_dataset("col", data=M.col.astype(np.int32))
        g.create_dataset("data", data=M.data)


def toml_val(v):
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(toml_val(x) for x in v) + "]"
    return repr(float(v)) if isinstance(v, (float, np.floating)) else str(v)


def main(argv=None):
    p = argparse.ArgumentParser(prog="qtx_modespace.to_quatrex", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="QuaTrEx device folder")
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--modes", help="basis directory (slab<s>.npy, slab<s>_orbitals.npy)")
    g.add_argument("--exact", action="store_true", help="identity basis: full-orbital reference set")
    p.add_argument("--cells", type=int)
    p.add_argument("--trim", action="store_true")
    p.add_argument("--potential", action="store_true")
    p.add_argument("--cache", help="directory for the lead band edges (lead_edges.json; e.g. the --sigma-cache)")
    p.add_argument("--dE", default="0:0.5:0.05", help="energies - Ec: list or lo:hi:step [eV]")
    p.add_argument("--Ec", type=float, help="reference edge (default Ec_lead)")
    p.add_argument("--solver", default="pardiso", help="QuaTrEx electron.solver.direct_solver")
    p.add_argument("--threads", type=int, default=4, help="QuaTrEx compute.blas_num_threads")
    a = p.parse_args(argv)

    t0 = time.time()
    root = Path(a.folder)
    cfg = tomllib.loads((root / "quatrex_config.toml").read_text())
    dc = cfg["device"]
    inp = read_quatrex(root, cells=a.cells, trim=a.trim, potential=a.potential, cache_dir=a.cache, log=log)
    meta = inp["meta"]
    ax, m, nb = meta["axis"], meta["cells_per_block"], meta["nblocks"]
    H, S, xyz, labels, norb, blocks = inp["H"], inp["S"], inp["xyz"], inp["labels"], inp["norb"], inp["blocks"]
    Lb = meta["block_length"]
    e = np.zeros(3); e[ax] = 1.0
    off = np.concatenate([[0], np.cumsum(norb)])
    orbs = lambda at: np.concatenate([np.arange(off[i], off[i + 1]) for i in at])
    at_b = [np.where(blocks == b)[0] for b in range(nb)]
    O = [orbs(at) for at in at_b]

    # ---------------------------------------------------------------- basis per device slab
    U = []
    for s, b in enumerate(range(1, nb - 1)):
        if a.exact:
            U.append(None)
            continue
        orb = np.load(os.path.join(a.modes, f"slab{s}_orbitals.npy"))
        assert np.array_equal(orb, inp["perm"][O[b]]), f"slab {s}: basis rows do not match (--cells / --trim?)"
        U.append(np.load(os.path.join(a.modes, f"slab{s}.npy")))
    N = [len(O[b]) if u is None else u.shape[1] for b, u in zip(range(1, nb - 1), U)]

    # ---------------------------------------------------------------- lead pads
    i1 = images(xyz, labels, at_b[1], at_b[0], Lb * e)                    # block 0 -> block 1
    iR = images(xyz, labels, at_b[nb - 2], at_b[nb - 1], -Lb * e)         # block nb-1 -> block nb-2
    o0, oR, o1, oRm = O[0], O[nb - 1], orbs(i1), orbs(iR)
    blk = lambda M, r, c: sp.csr_matrix(M[r][:, c])
    # device blocks (code order) in the reduced basis: diagonal D[k], upper C[k] between k and k+1
    # k = 0 PL, 1 block 0, 2 … nb-1 slabs, nb block nb-1, nb+1 PR
    def red(M, r, c, Ur, Uc):
        """Ur^H M[r, c] Uc (U = None: identity)"""
        X = M[r][:, c]
        if Uc is not None:
            X = np.asarray(X @ Uc)
        if Ur is not None:
            X = Ur.conj().T @ X if isinstance(X, np.ndarray) else np.asarray((X.T @ Ur.conj()).T)
        return sp.csr_matrix(X)
    Uall = [None, None] + U + [None, None]
    rows = [o0, o0] + [O[b] for b in range(1, nb - 1)] + [oR, oR]
    out = {}
    for nm, M in (("hamiltonian", H), ("overlap", S)):
        n = len(rows)
        B = [[None] * n for _ in range(n)]
        for k in range(n):
            B[k][k] = red(M, rows[k], rows[k], Uall[k], Uall[k])
        B[0][1] = blk(M, o0, o1)                                          # PL -> block 0 = H01 of the lead
        for k in range(1, n - 2):
            B[k][k + 1] = red(M, rows[k], rows[k + 1], Uall[k], Uall[k + 1])
        B[n - 2][n - 1] = blk(M, oRm, oR)                                 # block nb-1 -> PR
        for k in range(n - 1):
            B[k + 1][k] = B[k][k + 1].conj().T
        R = sp.bmat(B, format="csr")
        d = abs(R - R.conj().T).max()
        assert d < 1e-8 * abs(R).max(), f"{nm} not Hermitian ({d:.1e})"
        out[nm] = 0.5 * (R + R.conj().T)
    n_red = out["hamiltonian"].shape[0]

    # ---------------------------------------------------------------- structure: real atoms of the leads + pseudo-atoms
    at0, atR = at_b[0], at_b[nb - 1]
    nper = {str(k): int(v) for k, v in dc["num_orbitals_per_atom"].items()}
    lab = list(labels[at0]) * 2
    pos = [xyz[at0] - Lb * e, xyz[at0]]
    for s, b in enumerate(range(1, nb - 1)):
        if U[s] is None:                                    # --exact: the slab's real atoms
            lab += list(labels[at_b[b]])
            pos.append(xyz[at_b[b]])
        else:
            lab.append(f"U{s}")
            pos.append(xyz[at_b[b]].mean(0)[None])
            nper[f"U{s}"] = int(N[s])
    lab += list(labels[atR]) * 2
    pos = np.vstack(pos + [xyz[atR], xyz[atR] + Lb * e])
    assert sum(nper[x] for x in lab) == n_red

    # ---------------------------------------------------------------- contacts: one lead block each
    cL = next(c for c in dc["contacts"] if c["name"] == "left")
    cR = next(c for c in dc["contacts"] if c["name"] == "right")
    lvL, lvR = np.array(cL["lattice_vectors"], float), np.array(cR["lattice_vectors"], float)
    lvL[0], lvR[0] = Lb * e, -Lb * e
    oL = np.array(cL["origin"], float); oL[ax] -= Lb
    xr_last, xr_lead = xyz[at_b[nb - 2], ax].max(), xyz[atR, ax].min()
    oR_ = np.array(cR["origin"], float); oR_[ax] = 0.5 * (xr_last + xr_lead) + 2 * Lb
    nL, nR = len(at0), len(atR)
    ia = np.arange(len(lab))
    sets = dict(PL=ia[:nL], B0=ia[nL:2 * nL], BR=ia[-2 * nR:-nR], PR=ia[-nR:])
    for nm, o, lv, (c0, c1) in (("left", oL, lvL, ("PL", "B0")), ("right", oR_, lvR, ("PR", "BR"))):
        for rep, key in ((0, c0), (1, c1)):
            got = in_cell(pos, o, lv, rep)
            assert np.array_equal(np.sort(got), sets[key]), f"{nm} contact cell {rep}: {len(got)} atoms, " \
                                                            f"expected the {len(sets[key])} of {key}"
    log(f"contacts: one block each ({Lb:.3f} A); left origin {np.round(oL, 4).tolist()}, right origin "
        f"{np.round(oR_, 4).tolist()}; cell and first repetition checked")

    # ---------------------------------------------------------------- write
    od = Path(a.out)
    (od / "inputs").mkdir(parents=True, exist_ok=True)
    write_h5(od / "inputs" / "hamiltonian.h5", out["hamiltonian"])
    write_h5(od / "inputs" / "overlap.h5", out["overlap"])
    lat = np.diag([pos[:, 0].max() - pos[:, 0].min() + 1.0, *[float(np.linalg.norm(cL["lattice_vectors"][i]))
                                                               for i in (1, 2)]])
    with open(od / "inputs" / "structure.xyz", "w") as f:
        f.write(f"{len(lab)}\nLattice=\"{' '.join(f'{x:.6f}' for x in lat.ravel())}\"\n")
        for s_, r in zip(lab, pos):
            f.write(f"{s_} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n")
    Ec = meta["edges"]["Ec_lead"] if a.Ec is None else a.Ec
    dEs = energies(a.dE)
    np.save(od / "inputs" / "electron_energies.npy", Ec + np.array(dEs))

    obc = dict(cfg.get("electron", {}).get("obc", {}))
    obc.setdefault("algorithm", "spectral"); obc.setdefault("nevp_solver", "full")
    L = ['simulation_dir = "."', 'input_dir = "inputs"', 'output_dir = "outputs"', 'formalism = "wf"', "",
         "[device]", f'transport_direction = "{"abc"[ax]}"', "kpoint_grid = [1, 1, 1]", "",
         "[device.num_orbitals_per_atom]"] + [f'"{k}" = {v}' for k, v in nper.items()]
    for c, o, lv in ((cL, oL, lvL), (cR, oR_, lvR)):
        L += ["", "[[device.contacts]]", f'name = "{c["name"]}"', f"origin = {toml_val(list(o))}",
              f"lattice_vectors = {toml_val([list(v) for v in lv])}", f'transport_direction = "{"abc"[ax]}"']
        for k in ("fermi_level", "temperature", "mid_gap_energy"):
            if k in c:
                L.append(f"{k} = {toml_val(c[k])}")
    L += ["", "[electron]", f'solver.direct_solver = "{a.solver}"', "", "[electron.obc]"] \
        + [f"{k} = {toml_val(v)}" for k, v in obc.items()] \
        + ["", "[compute]", f"blas_num_threads = {a.threads}"]
    (od / "quatrex_config.toml").write_text("\n".join(L) + "\n")
    info = dict(source=str(root.resolve()), exact=a.exact, modes=a.modes, cells=m, trim=a.trim,
                potential=a.potential, N_slabs=N, n_lead=len(o0), n_total=int(n_red),
                order=["PL", "lead block 0"] + [f"slab {s}" for s in range(len(U))] + [f"lead block {nb - 1}", "PR"],
                sizes=[len(o0), len(o0)] + N + [len(oR), len(oR)], Ec=Ec, dE=dEs, trimmed=meta.get("trimmed"))
    json.dump(info, open(od / "mode_space.json", "w"), indent=1, default=str)
    log(f"{od}: {n_red} orbitals {info['sizes']}; H nnz {out['hamiltonian'].nnz}; {len(dEs)} energies; "
        f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
