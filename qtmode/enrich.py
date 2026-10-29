"""Residual enrichment of the level-1 block bases against cut-induced (spurious) Ritz states of H1.

One round: Ritz pairs (lam, c) of H1 = U^T H U with lam - ec in lwin whose full-space residual
r = (H - lam) U c exceeds rthr (eV). Per block, the part of r orthogonal to the block basis is
compressed by SVD (keep s > smin, eV) and appended, so U stays block-diagonal. Inside the band
window the residual does not separate spurious from physical Ritz states (they hybridise), so in
practice this is a Davidson step on the whole window; it improves both (CLAUDE.md 4.5).
"""
import numpy as np

from .lsq import assemble_slab_blocks


def h1_dense(blk):
    """Dense H1 from the slab blocks of assemble_slab_blocks (columns grouped by slab)."""
    N = sum(len(i) for i in blk["idx"])
    H1 = np.zeros((N, N))
    for s, i in enumerate(blk["idx"]):
        H1[np.ix_(i, i)] = blk["HD"][s]
        if s + 1 < len(blk["idx"]):
            j = blk["idx"][s + 1]
            H1[np.ix_(i, j)] = blk["HU"][s]; H1[np.ix_(j, i)] = blk["HU"][s].T
    return H1


def enrich_round(H, orbs, bases, sob, ec, lwin=(-0.45, 0.40), rthr=1.0, smin=0.1):
    """Returns ({block index: added orthonormal columns}, stats). `bases` is not modified."""
    blk = assemble_slab_blocks(H, orbs, bases, sob)
    lam, C = np.linalg.eigh(h1_dense(blk))
    sel = np.where((lam > ec + lwin[0]) & (lam < ec + lwin[1]))[0]
    X = blk["U"].tocsc() @ C[:, sel]
    del blk, C
    R = H.tocsr() @ X - X * lam[sel]
    rn = np.linalg.norm(R, axis=0)
    spur = rn > rthr
    stats = dict(N1=int(len(lam)), n_window=int(len(sel)), n_spurious=int(spur.sum()),
                 n_gap=int(((lam > 0.3) & (lam < ec - 0.02)).sum()),
                 res_hist=np.histogram(rn, bins=[0, 0.1, 0.3, 0.5, 1, 2, 3, 5, 100])[0].tolist())
    Rs = R[:, spur]
    del R, X
    added = {}
    for b, Q in enumerate(bases):
        Rb = Rs[orbs[b]]
        for _ in range(2):
            Rb = Rb - Q @ (Q.T @ Rb)
        Ub, sv, _ = np.linalg.svd(Rb, full_matrices=False)
        keep = sv > smin
        if keep.any():
            Z = Ub[:, keep]
            Z = Z - Q @ (Q.T @ Z)
            added[b] = np.linalg.qr(Z)[0]
    stats["added"] = int(sum(z.shape[1] for z in added.values()))
    return added, stats
