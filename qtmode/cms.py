"""Two-level coupled-mode-space (CMS) reduction of a periodic Si nanosheet supercell.

Level 1 (real space -> block modes):
  * core blocks: channel Si at least d_core from any oxide atom, one block per transport slab.
    Basis = bulk sp3s* Bloch states in an energy window (k_x commensurate with the supercell,
    k_y, k_z oversampled for the core size), restricted to the block, made real, compressed.
  * interface tiles: all other orbitals (interface Si + oxide) of a slab, binned in (y, z).
    Basis = dominant eigenvectors of the local spectral function A_PP(E) at a few energies,
    computed in an embedding region with a complex absorbing potential (CAP) at its edge.
  H1 = U^T H U with U block-diagonal and orthonormal; all block-block couplings kept.

Level 2 (block modes -> slab modes):
  For each transport slab, diagonalize its diagonal block of H1 and keep the eigenvectors with
  energies in a window. H2 = V^T H1 V, again with all slab-slab couplings kept.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
try:
    import pypardiso
except ImportError:          # pragma: no cover
    from . import _pardiso_compat as pypardiso   # SciPy SuperLU fallback

from mgms.prep.structure import A_SI, SI_C, VB
from mgms.prep import structure as _structure
from mgms.prep.hamiltonian import TBParams, sk_block


def _nsi():
    """Orbitals per channel-Si atom of the selected model (5 for sp3s*, 10 for sp3d5s*)."""
    return int(_structure.NORB[SI_C])


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ geometry helpers
def periodic_tree(pos, Lx):
    """KD tree with periodic x (period Lx), open y, z. Lx=None: open in x as well."""
    Lx = 1e6 if Lx is None else Lx
    p = pos.copy()
    p[:, 0] %= Lx
    lo = p[:, 1:].min(0) - 1.0
    p[:, 1:] -= lo
    box = np.array([Lx, 1e5, 1e5])
    return cKDTree(p, boxsize=box), lo


def _query_points(pos, Lx, lo):
    Lx = 1e6 if Lx is None else Lx
    q = pos.copy()
    q[:, 0] %= Lx
    q[:, 1:] -= lo
    return q


def perimeter_coordinate(dy, dz, hy, hz):
    """Arc length of the radial projection of (dy, dz) onto the nominal channel rectangle
    (half widths hy, hz), counter-clockwise from the middle of the +y face. Period 4(hy + hz)."""
    t = np.maximum(np.abs(dy) / hy, np.abs(dz) / hz)
    py, pz = dy / t, dz / t
    s = np.empty_like(dy)
    right = (np.abs(py - hy) < 1e-9) & (pz >= 0)
    top = np.abs(pz - hz) < 1e-9
    left = np.abs(py + hy) < 1e-9
    bottom = np.abs(pz + hz) < 1e-9
    s[:] = 4 * hy + 4 * hz + pz                       # +y face, lower half (pz < 0)
    s[bottom] = 2 * hy + 3 * hz + (py[bottom] + hy)
    s[left] = hy + hz + (hy) + (hz - pz[left])
    s[top] = hz + (hy - py[top])
    s[right] = pz[right]
    return s


def partition(st, d_core=6.0, tile=40.0, tiling="box", tile_orbs=2500):
    """Returns dict with per-atom core flag, slab, and the list of blocks.
    Each block: dict(kind='core'|'tile', slab, atoms, orbs).
    tiling="box": original 40 A bins in y from the box edge, split into top/bottom halves.
    tiling="perimeter": the interface shell of each slab is ordered by its perimeter coordinate
    around the channel and cut at equal-orbital quantiles into round(n / tile_orbs) tiles."""
    Lx = st.meta.get("Lx")
    spc, pos = st.species, st.pos
    chan = spc == SI_C
    tree, lo = periodic_tree(pos[~chan], Lx)
    dmin, _ = tree.query(_query_points(pos, Lx, lo), k=1)
    core = chan & (dmin >= d_core)
    slab = st.block.copy()
    off = st.offsets
    blocks = []
    nslab = st.nblocks
    for s in range(nslab):
        a = np.where(core & (slab == s))[0]
        blocks.append(dict(kind="core", slab=s, atoms=a))
    yc, zc = st.meta["center"]
    hy, hz = st.meta["half_width"] if not np.isscalar(st.meta["half_width"]) else (st.meta["half_width"],) * 2
    norb = st.norb
    for s in range(nslab):
        a = np.where(~core & (slab == s))[0]
        if tiling == "box":
            key = np.c_[np.floor(pos[a, 1] / tile).astype(int), (pos[a, 2] > zc).astype(int)]
            keys, inv = np.unique(key, axis=0, return_inverse=True)
            for t in range(len(keys)):
                blocks.append(dict(kind="tile", slab=s, atoms=a[inv.ravel() == t],
                                   key=tuple(keys[t])))
        else:
            sc = perimeter_coordinate(pos[a, 1] - yc, pos[a, 2] - zc, hy, hz)
            o = np.argsort(sc)
            a, sc = a[o], sc[o]
            cum = np.cumsum(norb[a])
            nt = max(1, int(round(cum[-1] / tile_orbs)))
            cut = np.searchsorted(cum, cum[-1] * np.arange(1, nt) / nt)
            for t, idx in enumerate(np.split(np.arange(len(a)), cut)):
                blocks.append(dict(kind="tile", slab=s, atoms=np.sort(a[idx]),
                                   key=(t, float(sc[idx[0]]), float(sc[idx[-1]]))))
    for b in blocks:
        b["orbs"] = np.concatenate([off[i] + np.arange(off[i + 1] - off[i]) for i in b["atoms"]])
    return dict(core=core, dmin=dmin, slab=slab, blocks=blocks)


# ------------------------------------------------------------------ core: bulk Bloch basis
def _bulk_hk_batch(P, ks):
    """Primitive-cell H(k) for many k: (nk, 2 nsi, 2 nsi). Convention psi(r) = c e^{ik.r}.
    sp3s* (TBParams) or sp3d5s* (TB5Params, no spin-orbit, shifted so the bulk VBM is 0)."""
    v = VB * A_SI / 4
    if hasattr(P, "si"):                                  # sp3d5s*
        from mgms.prep import sp3d5s as T5
        ns = T5.NORB
        on = np.kron(np.eye(2), T5.onsite(P.si)) + P.shift * np.eye(2 * ns)
        Ms = [T5.sk_block(d, P.si) for d in v]
    else:
        ns = 5
        on = np.diag(np.concatenate([P.onsite(SI_C)] * 2))
        Ms = [sk_block(SI_C, SI_C, d, P) for d in v]
    Hk = np.repeat(on[None].astype(complex), len(ks), 0)
    for d, M in zip(v, Ms):
        ph = np.exp(1j * ks @ d)
        Hk[:, :ns, ns:] += M[None] * ph[:, None, None]
    Hk[:, ns:, :ns] = np.conj(np.transpose(Hk[:, :ns, ns:], (0, 2, 1)))
    return Hk


def core_kgrid(Lx, Wy, Wz, oversample=2.0):
    a = A_SI
    nkx = int(round(Lx / a)) * 2                     # commensurate: k_x = 2 pi m / Lx in [-2pi/a, 2pi/a)
    kx = (np.arange(nkx) - nkx // 2) * 2 * np.pi / Lx
    dky, dkz = np.pi / (oversample * Wy), np.pi / (oversample * Wz)
    ky = np.arange(-2 * np.pi / a, 2 * np.pi / a, dky)
    kz = np.arange(-2 * np.pi / a, 2 * np.pi / a, dkz)
    return np.array(np.meshgrid(kx, ky, kz, indexing="ij")).reshape(3, -1).T


def core_bloch_states(P, K, ewin, chunk=20000):
    """Bulk eigenpairs with E in ewin: returns k (m,3), coefficient vectors c (m,10), energies."""
    ks, cs, es = [], [], []
    for c0 in range(0, len(K), chunk):
        kk = K[c0:c0 + chunk]
        E, V = np.linalg.eigh(_bulk_hk_batch(P, kk))
        sel = np.argwhere((E > ewin[0]) & (E < ewin[1]))
        ks.append(kk[sel[:, 0]])
        cs.append(V[sel[:, 0], :, sel[:, 1]])
        es.append(E[sel[:, 0], sel[:, 1]])
    return np.vstack(ks), np.vstack(cs), np.concatenate(es)


def core_basis(st, atoms, kst, cst, thr=1e-2, chunk=4000):
    """Real orthonormal basis of the Bloch states restricted to `atoms` (channel Si, ideal sites).
    Gram matrix C = Re(B B^H) = [Re B, Im B][Re B, Im B]^T, so its eigenvectors are the left
    singular vectors of the real basis set and sqrt(eigenvalues) its singular values."""
    a4 = A_SI / 4
    pos = st.pos[atoms]
    sub = (np.rint(pos[:, 0] / a4).astype(int) % 2)   # sublattice 1 has odd a/4 coordinates
    ns = _nsi(); n = len(atoms) * ns
    C = np.zeros((n, n))
    for c0 in range(0, len(kst), chunk):
        k, c = kst[c0:c0 + chunk], cst[c0:c0 + chunk]
        ph = np.exp(1j * (k @ pos.T))                            # (m, natom)
        amp = np.where(sub[None, :, None] == 0, c[:, None, :ns], c[:, None, ns:]) * ph[:, :, None]
        B = amp.reshape(len(k), -1).T                            # (n, m)
        C += (B @ B.conj().T).real
    lam, Q = np.linalg.eigh(C)
    lam, Q = lam[::-1], Q[:, ::-1]
    S = np.sqrt(np.clip(lam, 0, None))
    m = S > thr * S[0]
    return Q[:, m], S


# ------------------------------------------------------------------ tiles: local spectral basis
def embedding(st, tree_info, atoms, d_emb, d_cap, eta_cap):
    """Atoms within d_emb of the tile (periodic x) and the CAP strength on each (per atom)."""
    tree, lo, Lx = tree_info
    q = _query_points(st.pos[atoms], Lx, lo)
    near = tree.query_ball_point(q, d_emb)
    emb = np.unique(np.concatenate([np.asarray(n, int) for n in near]))
    emb = np.union1d(emb, atoms)
    t_tile = cKDTree(q, boxsize=tree.boxsize)
    d, _ = t_tile.query(_query_points(st.pos[emb], Lx, lo), k=1)
    w = eta_cap * np.clip((d - (d_emb - d_cap)) / d_cap, 0, None) ** 2
    return emb, w


def tile_spectral_modes(H, st, tile_atoms, emb_atoms, w_cap, energies, rank=300, n_power=1,
                        rng=None):
    """Dominant eigenpairs of A_PP(E) (P = tile orbitals) for each energy.
    A = -Im G / pi, G = (E - H_emb + i W)^-1 is complex symmetric. Randomized range finder."""
    rng = np.random.default_rng(0) if rng is None else rng
    off = st.offsets
    orbs = lambda at: np.concatenate([off[i] + np.arange(off[i + 1] - off[i]) for i in at])
    oe = orbs(emb_atoms)
    nP = off[tile_atoms + 1] - off[tile_atoms]
    pos_in_emb = np.searchsorted(emb_atoms, tile_atoms)
    eoff = np.concatenate([[0], np.cumsum(off[emb_atoms + 1] - off[emb_atoms])])
    pidx = np.concatenate([eoff[j] + np.arange(n) for j, n in zip(pos_in_emb, nP)])
    W = np.repeat(w_cap, off[emb_atoms + 1] - off[emb_atoms])
    He = H[oe][:, oe].tocsr()
    n, m = He.shape[0], len(pidx)
    r = min(rank, m)
    out = []
    for E in energies:
        # real-equivalent form of (E - H + iW) x = b:  [[E-H, -W], [W, E-H]] [xr; xi] = [br; bi]
        D = (sp.identity(n, format="csr") * E - He)
        Wd = sp.diags(W)
        M = sp.bmat([[D, -Wd], [Wd, D]], format="csr")
        solver = pypardiso.PyPardisoSolver(mtype=11)
        solver.factorize(M)

        def apply_A(X):                                   # A_PP X for real X (m, r)
            R = np.zeros((2 * n, X.shape[1]))
            R[pidx] = X
            Z = solver.solve(M, R)
            for _ in range(1):                            # refinement (perturbed pivots)
                Z += solver.solve(M, R - M @ Z)
            return -Z[n:][pidx] / np.pi

        Y = apply_A(rng.standard_normal((m, r)))
        for _ in range(n_power):
            Y = apply_A(np.linalg.qr(Y)[0])
        Q = np.linalg.qr(Y)[0]
        B = Q.T @ apply_A(Q)
        lam, X = np.linalg.eigh(0.5 * (B + B.T))
        o = np.argsort(-lam)
        out.append((E, lam[o], Q @ X[:, o]))
        solver.free_memory(everything=True)
    return out, pidx


# ------------------------------------------------------------------ assembly
def assemble(H, n, block_orbs, bases):
    """U (n x N) sparse block-diagonal from per-block bases; H1 = U^T H U (dense)."""
    rows, cols, vals, c0 = [], [], [], 0
    owner = []
    for b, (orbs, Q) in enumerate(zip(block_orbs, bases)):
        k = Q.shape[1]
        R, C = np.meshgrid(orbs, np.arange(c0, c0 + k), indexing="ij")
        rows.append(R.ravel()); cols.append(C.ravel()); vals.append(Q.ravel())
        owner.append(np.full(k, b))
        c0 += k
    U = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, c0))
    H1 = (U.T @ (H @ U)).toarray()
    return U, 0.5 * (H1 + H1.T), np.concatenate(owner)


def assemble_blocks(H, block_orbs, bases):
    """Dense-block assembly. Returns U (sparse), H1 = U^T H U and G = U^T H^2 U (both dense),
    and the block owning each column. Only block pairs that share rows of H U are visited."""
    H = H.tocsc()
    N = sum(Q.shape[1] for Q in bases)
    starts = np.concatenate([[0], np.cumsum([Q.shape[1] for Q in bases])])
    D, R = [], []
    for orbs, Q in zip(block_orbs, bases):
        Hc = H[:, orbs]
        rows = np.unique(Hc.indices)
        D.append(np.asarray(Hc[rows] @ Q)); R.append(rows)        # (H U)_j on its row support
    H1 = np.zeros((N, N)); G = np.zeros((N, N))
    nb = len(bases)
    for i in range(nb):
        for j in range(i, nb):
            com, ii, jj = np.intersect1d(R[i], R[j], assume_unique=True, return_indices=True)
            if len(com) == 0:
                continue
            si, sj = slice(starts[i], starts[i + 1]), slice(starts[j], starts[j + 1])
            g = D[i][ii].T @ D[j][jj]
            G[si, sj] = g; G[sj, si] = g.T
            _, a, b = np.intersect1d(block_orbs[i], R[j], assume_unique=True, return_indices=True)
            if len(a):
                h = bases[i][a].T @ D[j][b]
                H1[si, sj] = h; H1[sj, si] = h.T
    del D
    perm = np.concatenate(block_orbs)                  # row r of block_diag -> orbital perm[r]
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    U = sp.block_diag(bases, format="csr")[inv]
    owner = np.repeat(np.arange(nb), np.diff(starts))
    H1 += H1.T; H1 *= 0.5
    G += G.T; G *= 0.5
    return U, H1, G, owner


def folded_ritz(Hr, Fr, k):
    """Lowest k eigenvectors of the compressed folded operator Fr = W^T (H - s)^2 W, then
    Rayleigh-Ritz with Hr = W^T H W on their span. Interlacing keeps spurious levels out."""
    import scipy.linalg as sla
    mu, Y = sla.eigh(Fr, subset_by_index=(0, k - 1), driver="evr")
    h = Y.T @ Hr @ Y
    e, Z = np.linalg.eigh(0.5 * (h + h.T))
    return e, Y @ Z, mu


def slab_modes_folded(F1, slab_of_col, nslab, mu_max):
    """Level 2 with folded selection: slab modes = eigenvectors of F1_ss with mu < mu_max."""
    V_blocks, cols, info = [], [], []
    for s in range(nslab):
        idx = np.where(slab_of_col == s)[0]
        mu, X = np.linalg.eigh(F1[np.ix_(idx, idx)])
        m = mu < mu_max
        V_blocks.append(X[:, m]); cols.append(idx); info.append((len(idx), int(m.sum())))
    N2 = sum(v.shape[1] for v in V_blocks)
    V = np.zeros((F1.shape[0], N2))
    c0 = 0
    for idx, X in zip(cols, V_blocks):
        V[idx, c0:c0 + X.shape[1]] = X
        c0 += X.shape[1]
    return V, info


def slab_modes(H1, slab_of_col, nslab, ewin):
    """Level 2: eigenvectors of each slab's diagonal block of H1 with energies in ewin."""
    V_blocks, cols, info = [], [], []
    for s in range(nslab):
        idx = np.where(slab_of_col == s)[0]
        e, X = np.linalg.eigh(H1[np.ix_(idx, idx)])
        m = (e > ewin[0]) & (e < ewin[1])
        V_blocks.append(X[:, m]); cols.append(idx); info.append((len(idx), int(m.sum())))
    N1 = H1.shape[0]
    N2 = sum(v.shape[1] for v in V_blocks)
    V = np.zeros((N1, N2))
    c0 = 0
    for idx, X in zip(cols, V_blocks):
        V[idx, c0:c0 + X.shape[1]] = X
        c0 += X.shape[1]
    H2 = V.T @ H1 @ V
    return V, 0.5 * (H2 + H2.T), info


def slab_modes_embedded(F1, slab_of_col, nslab, k_targets=120, w_thr=1e-6):
    """Level 2, embedded (DMRG-like): for each slab s, take the superblock s-1, s, s+1, find the
    k_targets lowest eigenvectors of the compressed folded operator F1 on it (the states of the
    superblock nearest sigma; F1 includes the exact leakage out of the superblock), and keep the
    eigenvectors of the slab's reduced density matrix rho_s = Y_s Y_s^T with weight > w_thr."""
    import scipy.linalg as sla
    V_blocks, cols, info = [], [], []
    for s in range(nslab):
        sup = sorted({(s - 1) % nslab, s, (s + 1) % nslab})
        idx = np.where(np.isin(slab_of_col, sup))[0]
        k = min(k_targets, len(idx))
        mu, Y = sla.eigh(F1[np.ix_(idx, idx)], subset_by_index=(0, k - 1), driver="evr")
        mine = slab_of_col[idx] == s
        Ys = Y[mine]
        lam, X = np.linalg.eigh(Ys @ Ys.T)
        m = lam > w_thr * lam.max()
        V_blocks.append(X[:, m]); cols.append(idx[mine])
        info.append((int(mine.sum()), int(m.sum()), float(np.sqrt(mu[-1]))))
    N2 = sum(v.shape[1] for v in V_blocks)
    V = np.zeros((F1.shape[0], N2))
    c0 = 0
    for idx, X in zip(cols, V_blocks):
        V[idx, c0:c0 + X.shape[1]] = X
        c0 += X.shape[1]
    return V, info


def slab_modes_sweep(F1, slab_of_col, nslab, k_targets=60, w_thr=1e-8, n_sweeps=4, init=None,
                     log_fn=None):
    """Level 2 by finite-system sweeps (White 1992, one particle) on the folded operator.
    Superblock for slab s = slab s in its full level-1 basis (+) every other slab in its current
    level-2 basis. Its k_targets lowest eigenvectors of W^T F1 W (states nearest sigma, exact
    leakage included) give rho_s = Y_s Y_s^T; its dominant eigenvectors replace slab s's basis.
    init: per-slab bases (columns in slab-local level-1 coordinates); default from the
    embedded superblock s-1, s, s+1."""
    import scipy.linalg as sla
    cols = [np.where(slab_of_col == s)[0] for s in range(nslab)]
    if init is None:
        V0, _ = slab_modes_embedded(F1, slab_of_col, nslab, k_targets=2 * k_targets, w_thr=1e-6)
        init = []
        for s in range(nslab):
            X = V0[cols[s]]
            init.append(X[:, np.abs(X).sum(0) > 0])
    B = [b.copy() for b in init]
    hist = []
    for sw in range(n_sweeps):
        for s in list(range(nslab)) + list(range(nslab - 2, 0, -1)):
            Ws = [None if t == s else B[t] for t in range(nslab)]       # None = identity
            dims = [len(cols[t]) if t == s else B[t].shape[1] for t in range(nslab)]
            st_ = np.concatenate([[0], np.cumsum(dims)])
            Fs = np.zeros((st_[-1], st_[-1]))
            for t in range(nslab):                 # superblock assembled from slab-pair blocks
                for u in range(t, nslab):
                    blk = F1[np.ix_(cols[t], cols[u])]
                    if Ws[t] is not None:
                        blk = Ws[t].T @ blk
                    if Ws[u] is not None:
                        blk = blk @ Ws[u]
                    Fs[st_[t]:st_[t + 1], st_[u]:st_[u + 1]] = blk
                    Fs[st_[u]:st_[u + 1], st_[t]:st_[t + 1]] = blk.T
            parts = [(cols[t], None) for t in range(nslab)]
            mu, Y = sla.eigh(0.5 * (Fs + Fs.T), subset_by_index=(0, k_targets - 1), driver="evr")
            off = st_[s]
            Ys = Y[off:off + len(cols[s])]
            lam, X = np.linalg.eigh(Ys @ Ys.T)
            m = lam > w_thr * lam.max()
            B[s] = X[:, m][:, ::-1]
            disc = 1 - lam[m].sum() / lam.sum()
        hist.append(np.sqrt(np.clip(mu, 0, None)))
        if log_fn:
            log_fn(f"    sweep {sw + 1}: sqrt(mu) of the lowest targets {np.round(hist[-1][:3], 5)}, "
                   f"slab sizes {[b.shape[1] for b in B]}")
    N2 = sum(b.shape[1] for b in B)
    V = np.zeros((F1.shape[0], N2))
    c0 = 0
    for s in range(nslab):
        V[cols[s], c0:c0 + B[s].shape[1]] = B[s]
        c0 += B[s].shape[1]
    return V, [(len(cols[s]), B[s].shape[1]) for s in range(nslab)], hist


# ------------------------------------------------------------------ core: shared (union) basis
def union_core_region(nt, half_width, d_core=6.0, margin=4.5):
    """Device-independent superset of the core sites of any 2-cell slab of this cross-section:
    all diamond sites with |y - yc| <= hy - d_core + margin and |z - zc| <= hz - d_core + margin.
    margin covers outward excursions of the rough interface (default 3 x rms 1.5 A).
    Returns block-local integer coordinates (units a/4) and positions."""
    from mgms.prep.structure import diamond_sites, NX4_BLOCK, _nyz, _hyz
    ny, nz = _nyz(nt)
    hy, hz = _hyz(half_width)
    a4 = A_SI / 4
    c, _ = diamond_sites(NX4_BLOCK, 4 * ny, 4 * nz)
    yc, zc = (4 * ny - 1) / 2 * a4, (4 * nz - 1) / 2 * a4
    r = c * a4
    m = (np.abs(r[:, 1] - yc) <= hy - d_core + margin) & (np.abs(r[:, 2] - zc) <= hz - d_core + margin)
    return c[m], r[m]


def core_basis_positions(pos, kst, cst, thr=1e-3, chunk=4000):
    """As core_basis, for explicit ideal lattice positions (sublattice from the a/4 parity)."""
    a4 = A_SI / 4
    sub = (np.rint(pos[:, 0] / a4).astype(int) % 2)
    ns = _nsi(); n = len(pos) * ns
    C = np.zeros((n, n))
    for c0 in range(0, len(kst), chunk):
        k, c = kst[c0:c0 + chunk], cst[c0:c0 + chunk]
        ph = np.exp(1j * (k @ pos.T))
        amp = np.where(sub[None, :, None] == 0, c[:, None, :ns], c[:, None, ns:]) * ph[:, :, None]
        B = amp.reshape(len(k), -1).T
        C += (B @ B.conj().T).real
    lam, Q = np.linalg.eigh(C)
    lam, Q = lam[::-1], Q[:, ::-1]
    S = np.sqrt(np.clip(lam, 0, None))
    m = S > thr * S[0]
    return Q[:, m], S


def restrict_union_basis(st, atoms, slab, ucoords, QU, SU, thr=1e-2, s_min=1e-2):
    """Basis of one core block from the shared union basis: keep union vectors above `thr`,
    take the rows of the block's sites, re-orthonormalize (SVD), drop directions whose
    restriction has singular value < s_min (they live almost entirely outside the block)."""
    from mgms.prep.structure import NX4_BLOCK
    a4 = A_SI / 4
    lut = {tuple(c): i for i, c in enumerate(ucoords)}
    c = np.rint(st.pos[atoms] / a4).astype(int)
    c[:, 0] -= NX4_BLOCK * slab
    try:
        rows_u = np.array([lut[tuple(x)] for x in c])
    except KeyError as e:
        raise ValueError(f"core site {e} outside the union region: increase margin")
    ns = _nsi()
    rows = (rows_u[:, None] * ns + np.arange(ns)[None]).ravel()
    Qk = QU[:, SU[:QU.shape[1]] > thr * SU[0]]
    U, s, _ = np.linalg.svd(Qk[rows], full_matrices=False)
    return U[:, s > s_min], s


def bloch_gram_kernel(P, kx, ky, kz, ewin, dmax):
    """Displacement kernel of the Bloch-state Gram matrix on a tensor-product k grid:
    K[dx, dy, dz, a, b] = Re sum_{k, bands in ewin} c_a(k) c_b(k)^* e^{i k . d a/4},
    for integer displacements |d_i| <= dmax_i (units a/4). a, b = sublattice*5 + orbital.
    Then C[(i,o),(j,o')] = K[c_i - c_j, s_i*5+o, s_j*5+o'] for any set of ideal sites."""
    a4 = A_SI / 4
    K = np.array(np.meshgrid(kx, ky, kz, indexing="ij")).reshape(3, -1).T
    nb = 2 * _nsi()
    W = np.zeros((len(K), nb, nb), complex)
    for c0 in range(0, len(K), 20000):
        E, V = np.linalg.eigh(_bulk_hk_batch(P, K[c0:c0 + 20000]))
        sel = ((E > ewin[0]) & (E < ewin[1])).astype(float)        # (nk, nb bands)
        Vs = V * sel[:, None, :]
        W[c0:c0 + 20000] = Vs @ np.conj(np.transpose(V, (0, 2, 1)))
    W = W.reshape(len(kx), len(ky), len(kz), nb * nb)
    d = [np.arange(-m, m + 1) for m in dmax]
    Ex, Ey, Ez = (np.exp(1j * np.outer(k, dd * a4)) for k, dd in zip((kx, ky, kz), d))
    T = np.einsum("xyzp,zc->xycp", W, Ez, optimize=True)
    T = np.einsum("xycp,yb->xbcp", T, Ey, optimize=True)
    T = np.einsum("xbcp,xa->abcp", T, Ex, optimize=True)
    return T.real.reshape(len(d[0]), len(d[1]), len(d[2]), nb, nb), int((W.reshape(-1, nb * nb)[:, ::nb + 1].real.sum()))


def core_basis_union(P, ucoords, Kgrid_axes, ewin, thr=1e-3):
    """Union core basis from the displacement kernel (no loop over Bloch states)."""
    kx, ky, kz = Kgrid_axes
    c = ucoords
    dmax = tuple(int(np.ptp(c[:, i])) for i in range(3))
    Kr, nst = bloch_gram_kernel(P, kx, ky, kz, ewin, dmax)
    sub = c[:, 0] % 2
    n = len(c); ns = _nsi()
    C = np.zeros((n, ns, n, ns))
    for si in (0, 1):
        for sj in (0, 1):
            I, J = np.where(sub == si)[0], np.where(sub == sj)[0]
            for c0 in range(0, len(I), 256):
                ii = I[c0:c0 + 256]
                dd = c[ii][:, None, :] - c[J][None, :, :] + np.array(dmax)
                blk = Kr[dd[..., 0], dd[..., 1], dd[..., 2]][..., si * ns:si * ns + ns, sj * ns:sj * ns + ns]
                C[ii[:, None], :, J[None, :], :] = np.transpose(blk, (0, 1, 2, 3))
    import scipy.linalg as sla
    C = C.reshape(n * ns, n * ns)                      # symmetric by construction (K(-d) = K(d)^T)
    lam, Q = sla.eigh(C, overwrite_a=True, check_finite=False, driver="evr")
    del C
    lam, Q = lam[::-1], Q[:, ::-1]
    S = np.sqrt(np.clip(lam, 0, None))
    m = S > thr * S[0]
    return np.ascontiguousarray(Q[:, m]), S, nst
