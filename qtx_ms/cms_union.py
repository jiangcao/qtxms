"""Coupled mode space (CMS) bases for the rough Si nanosheet with oxide shell (mgms.prep.si_sheet inputs).

conventional : per device slab s, the lowest M eigenvectors of the closed slab H_ss above mid-gap
               (one eigenproblem per slab).
union        : ONE eigenproblem. Reference slab = regular Si cross-section whose channel is the union of the
               channel sites of all device slabs (lattice sites taken modulo one Si cell along x, so the
               reference wire is translation invariant), surrounded by the same oxide surrogate (oxide disorder
               of the lead template). Its lowest M slab eigenvectors are copied onto every device slab atom by
               atom through the lattice: channel Si -> channel Si, oxide Si -> oxide Si (same site), O -> O
               (same bond). A "missing" channel Si (union site that is oxide Si in this slab) takes the s, p
               coefficients of the reference channel Si damped by exp(-kappa r), r = its distance to the
               slab's channel and kappa the decay constant of the reference modes into the oxide (fit_kappa);
               union-zero sets them to 0. Other unmatched atoms (O on former Si-Si bonds) get 0. The copied
               vectors are re-orthonormalized per slab (SVD, singular values below `tol` dropped).
nominal      : as union, but the reference is the flat lead block (nominal cross-section), for comparison.
*-bloch      : union / nominal reference, but the reference modes are the Bloch states of the periodic
               reference wire in an energy window at nk wave vectors, compressed by SVD (ordered by singular
               value), instead of closed-slab eigenvectors; only possible because the reference is periodic.

Atoms are matched by keys in block-local lattice coordinates (units a/4): Si -> ("si", X - 8 b, Y, Z);
bridging O -> ("ob", site_i, site_j); non-bridging O -> ("on", site, direction signs).
"""
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from .prep import structure as stc
from .prep.structure import SI_C, SI_OX, OX, A_SI, NX4_BLOCK

A4 = A_SI / 4


def _is_diamond(c):
    X, Y, Z = c[..., 0], c[..., 1], c[..., 2]
    even = (X % 2 == 0) & (Y % 2 == 0) & (Z % 2 == 0) & ((X + Y + Z) % 4 == 0)
    odd = (X % 2 == 1) & (Y % 2 == 1) & (Z % 2 == 1) & ((X + Y + Z) % 4 == 3)
    return even | odd


def lattice_coords(pos, species):
    """Integer diamond-lattice coordinates [a/4] of the Si atoms: nearest diamond site among the 8 corners of
    the a/4 grid cube around each atom (channel Si sit exactly on it, oxide Si are displaced ~0.15 A)."""
    si = np.isin(species, (SI_C, SI_OX))
    c = np.full((len(pos), 3), -10 ** 6, int)
    q = pos[si] / A4
    corners = np.floor(q).astype(int)[:, None, :] + np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).reshape(3, -1).T[None]
    d = np.linalg.norm(corners - q[:, None, :], axis=2)
    d[~_is_diamond(corners)] = np.inf
    k = np.argmin(d, 1)
    best = d[np.arange(len(q)), k]
    assert best.max() < 0.8, f"Si atom {best.max():.2f} a/4 from the nearest diamond site"
    c[si] = corners[np.arange(len(q)), k]
    return c


def atom_keys(pos, species, bonds, block, x4_origin=0, period=None):
    """Block-local matching key per atom (see module doc). x4_origin: lattice x [a/4] of block 0;
    period: ring length [a/4] for structures periodic in x (local x taken in (-period/2, period/2])."""
    c = lattice_coords(pos, species)
    loc = c.copy()
    nbr = [[] for _ in range(len(pos))]
    for i, j in bonds:
        nbr[i].append(j); nbr[j].append(i)
    keys = [None] * len(pos)

    def lx(x, x0):
        return x - x0 if not period else (x - x0 + period // 2) % period - period // 2

    for i, s in enumerate(species):
        x0 = x4_origin + NX4_BLOCK * block[i]
        if s in (SI_C, SI_OX):
            keys[i] = ("si", lx(c[i, 0], x0), c[i, 1], c[i, 2])
        elif s == OX:
            si = [j for j in nbr[i] if species[j] in (SI_C, SI_OX)]
            sites = sorted((lx(c[j, 0], x0), c[j, 1], c[j, 2]) for j in si)
            if len(si) == 2:
                keys[i] = ("ob",) + tuple(sites)
            else:
                dv = pos[i] - c[si[0]] * A4
                if period:
                    dv[0] -= period * A4 * np.round(dv[0] / (period * A4))
                d = np.sign(np.round(dv, 1)).astype(int)
                keys[i] = ("on", sites[0], tuple(d))
    return keys, loc


def union_mask(dev_pos, dev_species, dev_block, blocks):
    """Set of (X mod 4, Y, Z) of the channel Si sites in the given blocks (union over x of one Si cell)."""
    sel = (dev_species == SI_C) & np.isin(dev_block, blocks)
    c = lattice_coords(dev_pos[sel], dev_species[sel])
    return {(x % 4, y, z) for x, y, z in c}


def reference_slab(meta, mask, P, model):
    """Regular reference: 3-block ring (periodic in x) whose channel is `mask` (or the flat nominal faces if
    mask is None), oxide disorder of the lead template (lead_seed, block periodic). Returns (Structure of the
    middle block, its keys, H of the middle block)."""
    s = meta["structure"]
    nt, hw = tuple(s["nt"]), tuple(s["half_width"])
    dis = stc.Disorder(np.random.default_rng(s["lead_seed"]), NX4_BLOCK, nt)
    ch = None if mask is None else (lambda co: np.array([(x % 4, y, z) in mask for x, y, z in co]))
    reg = stc.build_si_region(3 * stc.CELLS_PER_BLOCK, nt, hw, dis, None, 0.0, periodic_x=True, channel=ch)
    n_si = reg["n_si"]
    pos = reg["pos"]
    block = np.empty(len(pos), int)
    block[:n_si] = reg["coords"][:, 0] // NX4_BLOCK
    block[n_si:] = np.floor(pos[n_si:, 0] / stc.L_BLOCK).astype(int) % 3
    st = stc.Structure(pos, reg["species"], reg["bonds"], block, -np.ones(len(pos), int), 3,
                       dict(Lx=3 * stc.CELLS_PER_BLOCK * A_SI))
    H = model.build_H(st, P).tocsr()
    o = np.repeat(block, st.norb)
    mid = np.where(o == 1)[0]
    keys, _ = atom_keys(pos, st.species, st.bonds, block, period=3 * NX4_BLOCK)
    am, am2 = np.where(block == 1)[0], np.where(block == 2)[0]
    k2 = {keys[i]: i for i in am2}
    perm = np.array([k2[keys[i]] for i in am])                 # block-2 atoms in block-1 order
    off = st.offsets
    o2 = np.concatenate([np.arange(off[i], off[i + 1]) for i in perm])
    H11 = H[mid][:, mid].tocsr()
    assert abs(H[o2][:, o2] - H11).max() < 1e-10, "reference ring is not block periodic"
    return dict(species=st.species[am], norb=st.norb[am], keys=[keys[i] for i in am], H=H11,
                H12=H[mid][:, o2].tocsr(), n_channel=int((st.species[am] == SI_C).sum()), pos=pos[am],
                chan_pos=pos[st.species == SI_C], Lx=3 * stc.CELLS_PER_BLOCK * A_SI)


def bloch_modes(ref, Elo, Ehi, nk=12, chunk=64):
    """Basis from the Bloch states of the periodic reference wire, H(k) = H11 + H12 e^ik + H12^T e^-ik,
    k in [0, pi] (nk points), energies in [Elo, Ehi]: real and imaginary parts of all states, compressed by
    one SVD. Returns (Q, singular values, states per k); columns of Q ordered by singular value."""
    import scipy.sparse.linalg as spla
    H11, H12 = ref["H"], ref["H12"]
    n = H11.shape[0]
    c, cols, cnt = 0.5 * (Elo + Ehi), [], []
    for k in np.linspace(0, np.pi, nk):
        Hk = (H11 + H12 * np.exp(1j * k) + H12.T * np.exp(-1j * k)).tocsc()
        lu = spla.splu((Hk - c * sp.identity(n, format="csc")).tocsc())
        op = spla.LinearOperator((n, n), matvec=lu.solve, dtype=complex)
        m = chunk
        while True:                                    # widen until both window edges are passed
            w, V = spla.eigsh(Hk, k=min(m, n - 2), sigma=c, which="LM", OPinv=op, tol=1e-10)
            if (w.min() < Elo and w.max() > Ehi) or m >= n - 2:
                break
            m *= 2
        sel = (w >= Elo) & (w <= Ehi)
        cols.append(V[:, sel]); cnt.append(int(sel.sum()))
    C = np.hstack(cols)
    Q, sv, _ = np.linalg.svd(np.hstack([C.real, C.imag]), full_matrices=False)
    return Q, sv, cnt


def slab_modes(H, M, Emid):
    """Lowest M eigenpairs of the (dense) slab Hamiltonian above Emid."""
    Hd = H.toarray() if sp.issparse(H) else np.asarray(H)
    lo = count_below(Hd, Emid)
    return sla.eigh(Hd, subset_by_index=(lo, min(lo + M, len(Hd)) - 1), driver="evr")


def count_below(Hd, E):
    """Number of eigenvalues of the symmetric Hd below E (Sylvester inertia of the LDL^T of Hd - E)."""
    _, d, _ = sla.ldl(Hd - E * np.eye(len(Hd)), lower=True)
    return int((_block_diag_eigs(d) < 0).sum())


def _block_diag_eigs(D):
    n = len(D)
    out, i = [], 0
    while i < n:
        if i + 1 < n and D[i + 1, i] != 0:
            out.extend(np.linalg.eigvalsh(D[i:i + 2, i:i + 2])); i += 2
        else:
            out.append(D[i, i]); i += 1
    return np.array(out)


def oxide_depth(pos, chan_pos, Lx=None):
    """Distance [A] from each atom to the nearest channel Si (minimum image along x if Lx)."""
    from scipy.spatial import cKDTree
    if Lx:
        chan_pos = np.vstack([chan_pos + [k * Lx, 0, 0] for k in (-1, 0, 1)])
    return cKDTree(chan_pos).query(pos)[0]


def fit_kappa(ref, Q, rmin=1.0, rmax=6.0):
    """Decay constant kappa [1/A] of the reference modes into the oxide: |psi|^2 per atom (mean over the
    columns of Q) ~ exp(-2 kappa r), r = depth into the oxide, least squares on log weight for r in
    [rmin, rmax]. Returns (kappa, depth bins, mean log weight per bin)."""
    w = np.add.reduceat(Q ** 2, np.concatenate([[0], np.cumsum(ref["norb"])[:-1]]), axis=0).mean(1)
    r = oxide_depth(ref["pos"], ref["chan_pos"], ref["Lx"])
    ox = (ref["species"] != SI_C) & (r >= rmin) & (r <= rmax) & (w > 0)
    slope = np.polyfit(r[ox], np.log(w[ox]), 1)[0]
    bins = np.arange(0, rmax + 1.0, 1.0)
    idx = np.digitize(r, bins)
    prof = [float(np.log(w[(idx == b) & (w > 0)]).mean()) if ((idx == b) & (w > 0)).any() else None
            for b in range(1, len(bins))]
    return -0.5 * slope, bins, prof


def orbital_map(ref, keys, species, norb, depth=None, kappa=None):
    """For the slab atoms (keys, species, norb): slab-local orbital -> reference orbital index or -1, and a
    weight per orbital. Atoms match when the key and the species agree (weight 1). With kappa, a missing
    channel Si (union channel site that is oxide Si in this slab) takes the s, p coefficients of the
    reference channel Si times exp(-kappa * depth), depth = its distance to the slab's channel; otherwise
    (kappa None) it gets 0. Unmatched other atoms get 0."""
    roff = np.concatenate([[0], np.cumsum(ref["norb"])])
    rk = {(k, s): i for i, (k, s) in enumerate(zip(ref["keys"], ref["species"]))}
    out, fac = [], []
    stats = dict(matched=0, missing_si=0, other=0)
    for a, (k, s, no) in enumerate(zip(keys, species, norb)):
        i = rk.get((k, s))
        if i is None:
            j = rk.get((k, SI_C)) if s == SI_OX else None
            if j is not None and kappa is not None:
                out.append(np.arange(roff[j], roff[j] + no)); fac.append(np.full(no, np.exp(-kappa * depth[a])))
            else:
                out.append(np.full(no, -1)); fac.append(np.zeros(no))
            stats["missing_si" if j is not None else "other"] += 1
        else:
            assert ref["norb"][i] == no
            out.append(np.arange(roff[i], roff[i] + no)); fac.append(np.ones(no)); stats["matched"] += 1
    return np.concatenate(out), np.concatenate(fac), stats


def project_modes(Q, omap, fac=None, tol=1e-3):
    """Copy reference modes Q onto a slab through omap (unmatched rows 0, rows scaled by fac) and
    orthonormalize (SVD, keep singular values > tol). Returns (V, singular values)."""
    V = np.zeros((len(omap), Q.shape[1]))
    ok = omap >= 0
    V[ok] = Q[omap[ok]] * (1.0 if fac is None else fac[ok, None])
    W, s, _ = np.linalg.svd(V, full_matrices=False)
    return W[:, s > tol], s
