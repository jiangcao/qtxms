"""Removal of spurious (cut-induced) states from a level-1 model by leakage.

For W with orthonormal columns, P = W W^T, H1 = W^T H W and G = W^T H^2 W, the leakage operator
    L = W^T H (1 - P) H W = G - H1^2
gives the exact Ritz residual of any Ritz pair (lam, W y) of H1:  ||(H - lam) W y||^2 = y^T L y.
A Ritz value at distance d from the spectrum of H therefore needs y^T L y >= d^2. In particular a
Ritz value deep in the band gap of the real-space H (a spurious state) has a large leakage, while
any direction that represents true states of H in the energy range of interest has a small one.

H1 couples only neighbouring slabs, so the slab-diagonal block of L is local:
    L_ss = G_ss - sum_{t = s-1, s, s+1} H1_st H1_ts.
clean_by_leakage diagonalizes every L_ss, keeps the directions with leakage <= r_max^2, transforms
H1 and G to the kept basis, and repeats (removing a direction from one slab raises the leakage of
its neighbours) until nothing more is removed.

With a smooth potential u(x) that is nearly constant over a basis block, (1 - P) diag(u) W ~ 0, so
the residual bound carries over to H + u: the cleaned model cannot have states more than ~r_max
inside the local gap at any gate voltage.
"""
import numpy as np


def block_transform(M, cur, newC):
    """C^T M C for block-diagonal C (blocks newC on index sets cur) and slab-block-tridiagonal M."""
    ns = len(cur)
    st = np.concatenate([[0], np.cumsum([c.shape[1] for c in newC])])
    out = np.zeros((st[-1], st[-1]))
    for s in range(ns):
        for t in (s, s + 1):
            if t >= ns:
                continue
            b = newC[s].T @ M[np.ix_(cur[s], cur[t])] @ newC[t]
            out[st[s]:st[s + 1], st[t]:st[t + 1]] = b
            out[st[t]:st[t + 1], st[s]:st[s + 1]] = b.T
    return out


def slab_leakage(H1, G, idx, s):
    ns = len(idx)
    i = idx[s]
    L = G[np.ix_(i, i)].copy()
    for t in (s - 1, s, s + 1):
        if 0 <= t < ns:
            B = H1[np.ix_(i, idx[t])]
            L -= B @ B.T
    return 0.5 * (L + L.T)


def clean_by_leakage(H1, G, slab, r_max, max_iter=8, log=print):
    """Returns the per-slab transforms Cs (orthonormal columns; the cleaned basis of slab s is
    W_s Cs[s]) and a history of (iteration, removed per slab, max r per slab)."""
    ns = int(slab.max()) + 1
    idx = [np.where(slab == s)[0] for s in range(ns)]
    Cs = [np.eye(len(i)) for i in idx]                 # slab-local transforms from the original
    H, Gm = H1, G
    hist = []
    for it in range(max_iter):
        cur = [np.arange(sum(c.shape[1] for c in Cs[:s]), sum(c.shape[1] for c in Cs[:s + 1]))
               for s in range(ns)]
        removed, rmax, newC = [], [], []
        for s in range(ns):
            L = slab_leakage(H, Gm, cur, s)
            ell, Y = np.linalg.eigh(L)
            keep = ell <= r_max ** 2
            removed.append(int((~keep).sum()))
            rmax.append(float(np.sqrt(max(ell.max(), 0))))
            newC.append(Y[:, keep])
        hist.append((it, removed, rmax))
        log(f"    leakage iter {it}: removed {removed}, max r before {np.round(rmax, 3)} eV")
        if sum(removed) == 0:
            break
        H = block_transform(H, cur, newC)
        Gm = block_transform(Gm, cur, newC)
        Cs = [Cs[s] @ newC[s] for s in range(ns)]
    return Cs, hist


def clean_gap_states(H1, slab, band, w_min=0.01, max_iter=10, log=print):
    """Slab-local removal of spurious gap states.
    Repeat: find the Ritz pairs of the current model with eigenvalue in `band` (inside the gap of
    the real-space device); for every slab, remove from its basis the directions that carry more
    than w_min of the weight of some combination of those Ritz vectors (left singular vectors of
    their slab rows with sigma^2 > w_min); until no Ritz value is left in the band.
    Removing the slab-local ingredients (not just the global eigenvectors) is what keeps the gap
    clean when a slab-wise potential later recombines the remaining directions.
    Returns per-slab transforms Cs and the history [(iteration, n_gap, removed per slab)]."""
    import scipy.linalg as sla
    ns = int(slab.max()) + 1
    idx = [np.where(slab == s)[0] for s in range(ns)]
    Cs = [np.eye(len(i)) for i in idx]
    H = H1
    hist = []
    for it in range(max_iter):
        cur = [np.arange(sum(c.shape[1] for c in Cs[:s]), sum(c.shape[1] for c in Cs[:s + 1]))
               for s in range(ns)]
        lam, Y = sla.eigh(H, subset_by_value=band, driver="evr", check_finite=False)
        if len(lam) == 0:
            hist.append((it, 0, [0] * ns))
            log(f"    iter {it}: no Ritz values in {band}")
            break
        removed, newC = [], []
        for s in range(ns):
            A, sv, _ = np.linalg.svd(Y[cur[s]], full_matrices=False)
            rm = sv ** 2 > w_min
            removed.append(int(rm.sum()))
            # orthogonal complement of the removed directions in the slab's current basis
            Q, _ = np.linalg.qr(A[:, rm], mode="complete")
            newC.append(Q[:, rm.sum():])
        hist.append((it, int(len(lam)), removed))
        log(f"    iter {it}: {len(lam)} Ritz values in {band}; removed per slab {removed}")
        H = block_transform(H, cur, newC)
        Cs = [Cs[s] @ newC[s] for s in range(ns)]
    return Cs, hist
