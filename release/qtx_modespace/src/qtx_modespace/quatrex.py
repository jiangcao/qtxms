"""Read a QuaTrEx / deviceBuilder device folder into a block-tridiagonal device with periodic leads.

Folder layout (as written by deviceBuilder):
    quatrex_config.toml        [device] transport_direction, num_orbitals_per_atom, contacts (origin, lattice
                               vectors, mid_gap_energy)
    inputs/structure.xyz       extended xyz (Lattice="..."), atoms in orbital order
    inputs/hamiltonian.h5      group "[0, 0, 0]": coo row / col / data, attrs shape    (eV)
    inputs/overlap.h5          same for S
    inputs/potential.npy       optional, per orbital or per atom (eV); used only with potential=True
Only the R = [0, 0, 0] block is used (no transverse k).

Blocks: the contact lattice vector a (transport axis) defines primitive cells counted from the left contact origin.
A block is m cells; m defaults to the smallest number with m * a >= the coupling range of H and S. With trim=True,
H and S entries coupling non-neighbouring blocks are dropped (block-tridiagonal truncation; the largest dropped
|H|, |S| are reported), which allows a smaller m. The first and last blocks are the leads (the QuaTrEx contacts are
the device's own end cells); the lead cell is the first block and its coupling to its periodic image in the second
block. The right lead block is reordered into the lead-cell atom order; `perm` maps every orbital of the returned
matrices to its index in the QuaTrEx files.
"""
import hashlib, json, time, tomllib
from pathlib import Path
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp


def read_xyz(f):
    lines = Path(f).read_text().splitlines()
    n = int(lines[0].split()[0])
    w = [ln.split() for ln in lines[2:2 + n]]
    return np.array([x[0] for x in w]), np.array([[float(v) for v in x[1:4]] for x in w])


def read_lattice(f):
    """Lattice="..." of an extended xyz comment line as a 3 x 3 array (rows = vectors), or None."""
    with open(f) as fh:
        fh.readline(); c = fh.readline()
    return np.array(c.split('Lattice="')[1].split('"')[0].split(), float).reshape(3, 3) if 'Lattice="' in c else None


def read_h5(f, key="[0, 0, 0]", log=print):
    import h5py
    with h5py.File(f) as h:
        keys = list(h.keys())
        g = h[key]
        shape = tuple(int(s) for s in g.attrs["shape"])
        M = sp.coo_matrix((g["data"][()], (g["row"][()], g["col"][()])), shape=shape).tocsr()
    if keys != [key]:
        log(f"  {Path(f).name}: blocks {keys}; only {key} used")
    return M


def hermitian(M, name, log=print):
    """Full Hermitian matrix; completes a stored triangle."""
    d = abs(M - M.conj().T).max()
    if d < 1e-10 * abs(M).max():
        return (0.5 * (M + M.conj().T)).tocsr()
    lo, up = sp.tril(M, -1), sp.triu(M, 1)
    if lo.nnz == 0 or up.nnz == 0:
        log(f"  {name}: one triangle stored, completing")
        return (sp.triu(M) + sp.triu(M, 1).conj().T).tocsr()
    raise ValueError(f"{name} is not Hermitian (max |M - M^H| = {d:.2e})")


def lead_bands(lead, ks):
    H0, H1, S0, S1 = (lead[k].toarray() for k in ("H00", "H01", "S00", "S01"))
    out = []
    for k in ks:
        ph = np.exp(1j * k)
        H, S = H0 + H1 * ph + H1.conj().T / ph, S0 + S1 * ph + S1.conj().T / ph
        out.append(sla.eigvalsh(0.5 * (H + H.conj().T), 0.5 * (S + S.conj().T)))
    return np.array(out)


def lead_edges(lead, mid, nk, cache_dir=None, log=print):
    """Lead band edges (Ev, Ec) around mid from nk k points on [0, pi]. With cache_dir they are stored in
    cache_dir/lead_edges.json under a hash of the lead H00, H01, S00, S01, mid and nk, and reused."""
    h = hashlib.sha1(np.array([mid, nk], float).tobytes())
    for k in ("H00", "H01", "S00", "S01"):
        M = sp.csr_matrix(lead[k]).copy()
        M.sum_duplicates(); M.sort_indices()
        for x in (np.array(M.shape), M.indptr, M.indices, M.data):
            h.update(np.ascontiguousarray(x).tobytes())
    key = h.hexdigest()
    f = Path(cache_dir) / "lead_edges.json" if cache_dir else None
    db = json.loads(f.read_text()) if f and f.exists() else {}
    if key in db:
        log(f"  lead band edges from {f}")
        return db[key]["Ev"], db[key]["Ec"]
    E = lead_bands(lead, np.linspace(0, np.pi, nk))
    Ev, Ec = float(E[E < mid].max()), float(E[E > mid].min())
    if f:
        f.parent.mkdir(parents=True, exist_ok=True)
        db = json.loads(f.read_text()) if f.exists() else {}
        db[key] = dict(Ev=Ev, Ec=Ec, mid=mid, nk=nk)
        f.write_text(json.dumps(db, indent=1))
    return Ev, Ec


def detect_periodic(xyz, norb, H, S, lat, ax, tol=1e-3):
    """[{"axis": k, "length": L}] for the transverse axes k of the (orthorhombic) supercell Lattice along which
    |H| + |S| > tol couples two atoms more than L / 2 apart, i.e. through the periodic image (QuaTrEx [0, 0, 0] blocks
    of Gamma-point supercells), and per such axis the largest minimum-image |d_k| (must stay below L / 2)."""
    if lat is None or np.abs(lat - np.diag(np.diag(lat))).max() > 1e-8:
        return [], {}
    atom = np.repeat(np.arange(len(norb)), norb)
    P = (abs(H) + abs(S)).tocoo()
    k = P.data > tol
    ia, ja = atom[P.row[k]], atom[P.col[k]]
    keep = ia < ja
    D = xyz[ja[keep]] - xyz[ia[keep]]
    per, info = [], {}
    for t in range(3):
        if t == ax:
            continue
        L = float(lat[t, t])
        if len(D) and np.abs(D[:, t]).max() > 0.5 * L:
            per.append(dict(axis=t, length=L))
            Dm = D[:, t] - L * np.round(D[:, t] / L)
            info[t] = dict(raw_max=float(np.abs(D[:, t]).max()), image_max=float(np.abs(Dm).max()),
                           wrapped=int((np.abs(D[:, t]) > 0.5 * L).sum()))
    return per, info


def read_quatrex(folder, cells=None, trim=False, potential=False, nk=41, cache_dir=None, log=print):
    """Device dict: labels, xyz, norb, blocks, H, S, lead, lead_right, perm, meta (axis, periodic, block_length,
    transport = device / lead blocks, edges = lead band edges Ev_lead, Ec_lead; cached in cache_dir, see
    lead_edges)."""
    t0 = time.time()
    root = Path(folder)
    dc = tomllib.loads((root / "quatrex_config.toml").read_text())["device"]
    ax = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}[dc["transport_direction"]]   # deviceBuilder writes a|b|c or x|y|z
    norb_sp = dc["num_orbitals_per_atom"]
    cL = next(c for c in dc["contacts"] if c["name"] == "left")
    acell = float(np.linalg.norm(cL["lattice_vectors"][ax]))
    x0 = float(cL["origin"][ax])
    mid = float(cL.get("mid_gap_energy", 0.0))
    if tuple(dc.get("kpoint_grid", [1, 1, 1])) != (1, 1, 1):
        log(f"  kpoint_grid {dc['kpoint_grid']}: transverse k not supported, using the [0, 0, 0] block")

    labels, xyz = read_xyz(root / "inputs" / "structure.xyz")
    lat = read_lattice(root / "inputs" / "structure.xyz")
    def end_cells(a_):                      # atoms in the first and the last primitive cell
        c_ = np.floor((xyz[:, ax] - x0) / a_ + 1e-9).astype(int)
        return (c_ == 0).sum(), (c_ == c_.max()).sum()
    if lat is not None and len(set(end_cells(acell))) > 1:
        # contact vectors are rounded (3.168 for 3.16758): over ~100 cells the cell edges drift onto atom planes and
        # the last cell is cut. Then take the period from the supercell length (only then: keeps older devices' blocks).
        Lx = float(np.linalg.norm(lat[ax]))
        nc = round(Lx / acell)
        if nc and abs(Lx / acell - nc) < 0.05 and len(set(end_cells(Lx / nc))) == 1:
            log(f"  end cells {end_cells(acell)} atoms: contact cell {acell:.6f} A -> {Lx / nc:.6f} A "
                  f"(structure Lattice {Lx:.4f} A / {nc} cells)")
            acell = Lx / nc
    norb = np.array([norb_sp[s] for s in labels])
    off = np.concatenate([[0], np.cumsum(norb)])
    n = int(off[-1])
    H = hermitian(read_h5(root / "inputs" / "hamiltonian.h5", log=log), "H", log)
    S = hermitian(read_h5(root / "inputs" / "overlap.h5", log=log), "S", log)
    assert H.shape == (n, n) and S.shape == (n, n), f"H {H.shape}, S {S.shape}, sum(norb) {n}"
    counts = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    log(f"{root.name}: {len(labels)} atoms {counts}, {n} orbitals; "
        f"H nnz {H.nnz}, S nnz {S.nnz}")

    periodic, pinfo = detect_periodic(xyz, norb, H, S, lat, ax)
    if periodic:
        log(f"  periodic transverse axes (couplings through the supercell image): "
            f"{[(p['axis'], round(p['length'], 4)) for p in periodic]}, {pinfo}")

    V = None
    pot = root / "inputs" / "potential.npy"
    if potential and pot.exists():
        V = np.load(pot)
        if len(V) == len(labels):
            V = np.repeat(V, norb)
        assert len(V) == n, f"potential has {len(V)} entries, {len(labels)} atoms, {n} orbitals"
        D = sp.diags(V)
        H = (H + 0.5 * (D @ S + S @ D)).tocsr()
        log(f"  potential [{V.min():.3f}, {V.max():.3f}] eV added as (V S + S V) / 2")

    xo = np.repeat(xyz[:, ax], norb)
    rng = 0.0
    for M in (H, S):
        r, c = M.nonzero()
        rng = max(rng, float(np.abs(xo[r] - xo[c]).max()))
    cell = np.floor((xyz[:, ax] - x0) / acell + 1e-9).astype(int)
    ncell = int(cell.max()) + 1
    assert cell.min() == 0, "atoms before the left contact origin"
    m = cells or int(np.ceil(rng / acell - 1e-9))
    nb = ncell // m
    assert nb >= 4, f"{ncell} cells, {m} per block: too few blocks"
    blocks = np.minimum(cell // m, nb - 1)                    # remainder cells join the last device block
    if ncell % m:                                            # keep the right lead exactly m cells
        blocks = np.where(cell >= ncell - m, nb - 1, np.minimum(cell // m, nb - 2))
    lead_at = np.where(blocks == 0)[0]

    def images(b, shift, tol=0.1):
        """atoms of block b that are images of the lead block shifted by `shift` cells, in lead-block order"""
        cand = np.where(blocks == b)[0]
        d = xyz[cand][None] - xyz[lead_at][:, None]
        d[..., ax] -= shift * acell
        dist = np.linalg.norm(d, axis=2)
        j = dist.argmin(1)
        ok = (dist[np.arange(len(lead_at)), j] < tol) & (labels[cand[j]] == labels[lead_at])
        return cand[j], ok

    rimg, rok = images(nb - 1, ncell - m)
    assert rok.all() and (blocks == nb - 1).sum() == len(lead_at), "right lead block is not a copy of the left one"
    order = np.concatenate([np.where(blocks < nb - 1)[0], rimg])
    labels, xyz, norb, blocks = labels[order], xyz[order], norb[order], blocks[order]
    perm = np.concatenate([np.arange(off[i], off[i + 1]) for i in order])     # new orbital -> QuaTrEx orbital
    H, S = H[perm][:, perm].tocsr(), S[perm][:, perm].tocsr()
    if V is not None:
        V = V[perm]
    off = np.concatenate([[0], np.cumsum(norb)])
    lead_at = np.where(blocks == 0)[0]
    ob = np.repeat(blocks, norb)
    O = [np.where(ob == b)[0] for b in range(nb)]
    log(f"  transport axis {'xyz'[ax]}, cell {acell:.4f} A, {ncell} cells; coupling range {rng:.3f} A -> {m} cells "
        f"per block, {nb} blocks, orbitals per block {[len(o) for o in O]}")

    trimmed = None
    if trim:
        trimmed = {}
        for nm in ("H", "S"):
            M = sp.coo_matrix(H if nm == "H" else S)
            far = np.abs(ob[M.row] - ob[M.col]) > 1
            trimmed[nm] = dict(n=int(far.sum()), max=float(np.abs(M.data[far]).max()) if far.any() else 0.0)
            M = sp.csr_matrix((M.data[~far], (M.row[~far], M.col[~far])), shape=M.shape)
            H, S = (M, S) if nm == "H" else (H, M)
        log(f"  trimmed to block-tridiagonal: dropped {trimmed}")
    r, c = (abs(H) + abs(S)).nonzero()
    assert np.abs(ob[r] - ob[c]).max() <= 1, "H, S not block tridiagonal with these blocks (larger cells or trim)"
    orbs = lambda at: np.concatenate([np.arange(off[i], off[i + 1]) for i in at])
    img1, ok1 = images(1, m)
    assert ok1.all(), f"block 1 lacks images of {np.sum(~ok1)} lead atoms (use more cells per block)"
    o0, o1, rb = O[0], orbs(img1), O[nb - 1]
    lead = dict(H00=H[o0][:, o0], H01=H[o0][:, o1], S00=S[o0][:, o0], S01=S[o0][:, o1])
    checks = dict(right_vs_lead=dict(max_dH=float(abs(H[rb][:, rb] - lead["H00"]).max()),
                                     max_dS=float(abs(S[rb][:, rb] - lead["S00"]).max())))
    lead_right, dV = None, 0.0
    if V is not None:
        dV = float(V[rb].mean() - V[o0].mean())
    if abs(dV) > 1e-6:           # right lead = left lead rigidly shifted by the potential difference (bias)
        lead_right = dict(H00=lead["H00"] + dV * lead["S00"], H01=lead["H01"] + dV * lead["S01"],
                          S00=lead["S00"], S01=lead["S01"])
        log(f"  right lead = left lead + {dV:+.4f} eV * S")
    log(f"  lead periodicity (max |difference|): {checks}")
    Ev, Ec = lead_edges(lead, mid, nk, cache_dir, log=log)
    log(f"  lead {len(o0)} orbitals: Ev {Ev:.4f}, Ec {Ec:.4f} eV, gap {Ec - Ev:.3f} eV; {time.time() - t0:.0f} s")
    meta = dict(name=root.name, axis=ax, periodic=periodic or None, block_length=float(m * acell), cells_per_block=int(m), nblocks=int(nb),
                coupling_range=rng, trimmed=trimmed, lead_checks=checks,
                transport=dict(device_blocks=list(range(1, nb - 1)), lead_left_block=0, lead_right_block=int(nb - 1)),
                edges=dict(Ev_lead=Ev, Ec_lead=Ec, mid_gap=mid, right_lead_shift=dV))
    return dict(labels=labels, xyz=xyz, norb=norb, blocks=blocks, H=H, S=S, lead=lead, lead_right=lead_right,
                perm=perm, meta=meta)
