"""Folded (minimal-residual) NEGF in a reduced basis.

Galerkin (the usual CMS solve) projects the open system:  U^T (E - H - Sigma) U psi = U^T b.
Every spurious Ritz level of U^T H U becomes a pole of that reduced Green's function.

The folded / least-squares solve instead minimises the true residual
        psi = argmin || (E - H - Sigma) U psi - b ||,
        U^T A^dag A U psi = U^T A^dag b,      A = E - H - Sigma.
U^T A^dag A U contains the exact compression U^T (H - E)^2 U (the folded operator used for the
spurious-free mode finding), so a cut-induced state x with a large residual ||(H - E) U x||
costs exactly that residual and cannot act as a resonance. With U = identity it is exact.

Everything is kept in transport-slab blocks: H1 = U^T H U and G = U^T H^2 U couple only
neighbouring slabs when every slab is thicker than two hopping ranges.
"""
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla


def assemble_slab_blocks(H, block_orbs, bases, block_slab, row_w=None):
    """Slab-block assembly of H1 = U^T H U and G = U^T H^2 U.

    block_slab[b] = transport slab of block b. Blocks are reordered by slab (stable), so within a
    slab the columns keep the order of the input blocks. Returns dict with
      U   : sparse basis (n x N), columns grouped by slab
      idx : list of column ranges per slab
      HD, HU : diagonal / upper (s, s+1) blocks of H1
      GD, GU : same for G
      WD, WU : same for G_w = (H U)^T diag(row_w) (H U)   (only if row_w is given)
    """
    H = H.tocsc()
    order = np.argsort(np.asarray(block_slab), kind="stable")
    block_orbs = [block_orbs[b] for b in order]
    bases = [bases[b] for b in order]
    bslab = np.asarray(block_slab)[order]
    nslab = int(bslab.max()) + 1
    k = np.array([Q.shape[1] for Q in bases])
    starts = np.concatenate([[0], np.cumsum(k)])
    sl_start = [int(starts[np.where(bslab == s)[0][0]]) for s in range(nslab)] + [int(starts[-1])]
    idx = [np.arange(sl_start[s], sl_start[s + 1]) for s in range(nslab)]
    ns = [len(i) for i in idx]
    HD = [np.zeros((n, n)) for n in ns]; GD = [np.zeros((n, n)) for n in ns]
    HU = [np.zeros((ns[s], ns[s + 1])) for s in range(nslab - 1)]
    GU = [np.zeros((ns[s], ns[s + 1])) for s in range(nslab - 1)]
    if row_w is not None:
        WD = [np.zeros((n, n)) for n in ns]
        WU = [np.zeros((ns[s], ns[s + 1])) for s in range(nslab - 1)]
    Dm, R = [], []
    for orbs, Q in zip(block_orbs, bases):
        Hc = H[:, orbs]
        rows = np.unique(Hc.indices)
        Dm.append(np.asarray(Hc[rows] @ Q)); R.append(rows)
    nb = len(bases)
    for i in range(nb):
        si = bslab[i]
        for j in range(i, nb):
            sj = bslab[j]
            com, ii, jj = np.intersect1d(R[i], R[j], assume_unique=True, return_indices=True)
            if len(com) == 0:
                continue
            assert sj - si <= 1, f"blocks {i},{j} in slabs {si},{sj} overlap: not block tridiagonal"
            g = Dm[i][ii].T @ Dm[j][jj]
            gw = None
            if row_w is not None:
                w = row_w[com]
                if np.any(w != 0):
                    gw = (Dm[i][ii] * w[:, None]).T @ Dm[j][jj]
            _, a, b = np.intersect1d(block_orbs[i], R[j], assume_unique=True, return_indices=True)
            h = bases[i][a].T @ Dm[j][b] if len(a) else None
            ri = slice(starts[i] - sl_start[si], starts[i + 1] - sl_start[si])
            rj = slice(starts[j] - sl_start[sj], starts[j + 1] - sl_start[sj])
            if si == sj:
                GD[si][ri, rj] = g; GD[si][rj, ri] = g.T
                if gw is not None:
                    WD[si][ri, rj] = gw; WD[si][rj, ri] = gw.T
                if h is not None:
                    HD[si][ri, rj] = h; HD[si][rj, ri] = h.T
            else:
                GU[si][ri, rj] = g
                if gw is not None:
                    WU[si][ri, rj] = gw
                if h is not None:
                    HU[si][ri, rj] = h
    del Dm
    for s in range(nslab):
        HD[s] = 0.5 * (HD[s] + HD[s].T); GD[s] = 0.5 * (GD[s] + GD[s].T)
    perm = np.concatenate(block_orbs)
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    U = sp.block_diag(bases, format="csr")[inv]
    out = dict(U=U, idx=idx, HD=HD, HU=HU, GD=GD, GU=GU, order=order)
    if row_w is not None:
        for s in range(nslab):
            WD[s] = 0.5 * (WD[s] + WD[s].T)
        out.update(WD=WD, WU=WU)
    return out


def boundary_terms(Hs, U, idx, bL, bR):
    """Per-lead pieces of the folded normal matrix (constant in E):
    P = U[b] restricted to the contact slab (|b| x n_s), K = U_s^T H[:, b] (n_s x |b|)."""
    Hs = Hs.tocsc()
    U = U.tocsr()
    out = []
    for b, s in ((bL, 0), (bR, len(idx) - 1)):
        Us = U[:, idx[s]]
        assert abs(abs(U[b]).sum() - abs(Us[b]).sum()) < 1e-8 * max(1.0, abs(Us[b]).sum())
        P = Us[b].toarray()
        K = np.asarray((Us.T @ Hs[:, b]).todense())
        out.append((P, K))
    return out


def project_blocks(blk, bnd, Vb):
    """Project slab blocks and boundary pieces onto block-diagonal level-2 bases Vb[s]."""
    n = len(Vb)
    out = dict(HD=[Vb[s].T @ blk["HD"][s] @ Vb[s] for s in range(n)],
               GD=[Vb[s].T @ blk["GD"][s] @ Vb[s] for s in range(n)],
               HU=[Vb[s].T @ blk["HU"][s] @ Vb[s + 1] for s in range(n - 1)],
               GU=[Vb[s].T @ blk["GU"][s] @ Vb[s + 1] for s in range(n - 1)])
    if "WD" in blk:
        out["WD"] = [Vb[s].T @ blk["WD"][s] @ Vb[s] for s in range(n)]
        out["WU"] = [Vb[s].T @ blk["WU"][s] @ Vb[s + 1] for s in range(n - 1)]
    for k_ in ("pi",):
        if k_ in blk:
            out[k_] = blk[k_]
    (PL, KL), (PR, KR) = bnd
    bnd2 = [(PL @ Vb[0], Vb[0].T @ KL), (PR @ Vb[-1], Vb[-1].T @ KR)]
    return out, bnd2


def _gamma_factor(S):
    Gm = 1j * (S - S.conj().T)
    lam, X = np.linalg.eigh(0.5 * (Gm + Gm.conj().T))
    m = lam > 1e-10 * max(lam.max(), 1e-30)
    return X[:, m] * np.sqrt(lam[m])


def _btd_solve(Dg, Up, B):
    """Block Thomas for a Hermitian block-tridiagonal matrix (lower = Up^H). B: list (None = 0)."""
    n = len(Dg)
    ncol = next(b.shape[1] for b in B if b is not None)
    lus, y = [], []
    for i in range(n):
        Di = Dg[i]
        bi = B[i] if B[i] is not None else np.zeros((Di.shape[0], ncol), complex)
        if i > 0:
            Li = sla.lu_solve(lus[-1], Up[i - 1]).conj().T            # Up^H S^{-1}  (S Hermitian)
            Di = Di - Li @ Up[i - 1]
            bi = bi - Li @ y[-1]
        lus.append(sla.lu_factor(Di, check_finite=False))
        y.append(bi)
    X = [None] * n
    X[-1] = sla.lu_solve(lus[-1], y[-1])
    for i in range(n - 2, -1, -1):
        X[i] = sla.lu_solve(lus[i], y[i] - Up[i] @ X[i + 1])
    return X


def transmission_folded(blk, bnd, E, SigL, SigR):
    """Caroli T from the minimal-residual solution. blk: HD/HU/GD/GU of H1 = U^T H U and
    G = U^T H^2 U (U orthonormal); bnd: boundary_terms."""
    n = len(blk["HD"])
    (PL, KL), (PR, KR) = bnd
    Dg = []
    for s in range(n):
        M = (blk["GD"][s] - 2 * E * blk["HD"][s]).astype(complex)
        M[np.diag_indices_from(M)] += E * E
        Dg.append(M)
    Up = [(blk["GU"][s] - 2 * E * blk["HU"][s]).astype(complex) for s in range(n - 1)]
    FL, FR = _gamma_factor(SigL), _gamma_factor(SigR)
    rhs = None
    for (P, K, S, s) in ((PL, KL, SigL, 0), (PR, KR, SigR, n - 1)):
        C = E * P.T - K                                  # U_s^T (E - H)[:, b]
        CS = C @ S @ P
        Dg[s] = Dg[s] - CS - CS.conj().T + P.T @ (S.conj().T @ S) @ P
        if s == 0:
            rhs = C @ FL - P.T @ (S.conj().T @ FL)       # U^T A^dag b, b = Gamma_L^{1/2} on bL
    B = [rhs] + [None] * (n - 1)
    X = _btd_solve(Dg, Up, B)
    Y = FR.conj().T @ (PR @ X[-1])                       # Gamma_R^{1/2 dag} G_RL Gamma_L^{1/2}
    return float(np.sum(np.abs(Y) ** 2))


def transmission_galerkin(blk, bnd, E, SigL, SigR):
    """Standard (Galerkin) reduced solve on the same blocks, for comparison."""
    n = len(blk["HD"])
    (PL, KL), (PR, KR) = bnd
    Dg = []
    for s in range(n):
        M = (-blk["HD"][s]).astype(complex)
        M[np.diag_indices_from(M)] += E
        Dg.append(M)
    Up = [(-blk["HU"][s]).astype(complex) for s in range(n - 1)]
    Dg[0] = Dg[0] - PL.T @ SigL @ PL
    Dg[-1] = Dg[-1] - PR.T @ SigR @ PR
    FL, FR = _gamma_factor(SigL), _gamma_factor(SigR)
    # Galerkin matrix is complex symmetric, not Hermitian: use a general block Thomas
    nn = len(Dg)
    lus, y = [], []
    bi0 = PL.T @ FL
    for i in range(nn):
        Di = Dg[i]
        bi = bi0 if i == 0 else np.zeros((Di.shape[0], FL.shape[1]), complex)
        if i > 0:
            Li = sla.lu_solve(lus[-1], Up[i - 1], trans=1).T             # Up^T S^{-1}  (S symmetric)
            Di = Di - Li @ Up[i - 1]
            bi = bi - Li @ y[-1]
        lus.append(sla.lu_factor(Di, check_finite=False)); y.append(bi)
    X = sla.lu_solve(lus[-1], y[-1])
    Y = FR.conj().T @ (PR @ X)
    return float(np.sum(np.abs(Y) ** 2))


# ---------------------------------------------------------------------------------------------
# Folded selection + Galerkin ("folded Ritz" for an open system)
#
# Plain least squares damps transmission when the basis is not complete (the residual of a
# propagating wave that the basis represents imperfectly is penalised along the whole device).
# The folded Ritz recipe of the sheet instead SELECTS a spurious-free subspace and then does an
# ordinary Galerkin (Rayleigh-Ritz) solve inside it. For an open device the right folded operator
# releases the rows of the lead-coupled orbitals b = bL u bR, where the open boundary condition
# acts:   F_int(E) = U^T (H - E) Pi_int (H - E) U,   Pi_int = 1 - sum_{i in b} e_i e_i^T.
# Every scattering state solves (H - E) psi = 0 on the interior rows, so its projection has a
# small F_int; a cut-induced state carries its full ~eV residual. Keep the eigenvectors of
# F_int(E) with sqrt(mu) <= rmax and solve Galerkin with Sigma in their span.
# ---------------------------------------------------------------------------------------------

def dense_from_blocks(D, Up):
    n = [d.shape[0] for d in D]
    off = np.concatenate([[0], np.cumsum(n)])
    M = np.zeros((off[-1], off[-1]), dtype=np.result_type(*D))
    for s in range(len(D)):
        M[off[s]:off[s + 1], off[s]:off[s + 1]] = D[s]
        if s < len(Up):
            M[off[s]:off[s + 1], off[s + 1]:off[s + 2]] = Up[s]
            M[off[s + 1]:off[s + 2], off[s]:off[s + 1]] = Up[s].conj().T
    return M, off


def folded_region(blk, E):
    """Folded operator restricted to the rows of a set of whole slabs (blk["pi"][s] = 1 or 0):
    F_w(E) = U^T (H - E) Pi (H - E) U = G_w - E (Pi H1 + H1 Pi) + E^2 Pi   (U per-slab block diagonal)."""
    n = len(blk["HD"]); pi = blk["pi"]
    Dg = []
    for s in range(n):
        M = blk["WD"][s] - 2 * pi[s] * E * blk["HD"][s]
        M[np.diag_indices_from(M)] += pi[s] * E * E
        Dg.append(M)
    Up = [blk["WU"][s] - (pi[s] + pi[s + 1]) * E * blk["HU"][s] for s in range(n - 1)]
    F, off = dense_from_blocks(Dg, Up)
    H1, _ = dense_from_blocks(blk["HD"], blk["HU"])
    return 0.5 * (F + F.T), H1, off


def folded_interior(blk, bnd, E):
    """F_int(E) (dense, real symmetric) and H1 (dense)."""
    if "pi" in blk:
        return folded_region(blk, E)
    n = len(blk["HD"])
    Dg = []
    for s in range(n):
        M = blk["GD"][s] - 2 * E * blk["HD"][s]
        M[np.diag_indices_from(M)] += E * E
        Dg.append(M)
    Up = [blk["GU"][s] - 2 * E * blk["HU"][s] for s in range(n - 1)]
    for (P, K), s in zip(bnd, (0, n - 1)):
        C = K - E * P.T                                   # U_s^T (H - E)[:, b]
        Dg[s] = Dg[s] - C @ C.T                           # release the boundary rows
    F, off = dense_from_blocks(Dg, Up)
    H1, _ = dense_from_blocks(blk["HD"], blk["HU"])
    return 0.5 * (F + F.T), H1, off


def transmission_folded_select(blk, bnd, E, SigL, SigR, rmax, return_info=False):
    """rmax: float or list of floats (one eigen-decomposition, one Galerkin solve per value)."""
    rl = np.atleast_1d(rmax)
    F, H1, off = folded_interior(blk, bnd, E)
    mu, Y = sla.eigh(F, subset_by_value=(-np.inf, float(rl.max()) ** 2), driver="evr")
    del F
    HY = Y.T @ H1 @ Y
    del H1
    (PL, KL), (PR, KR) = bnd
    nlast = len(blk["HD"]) - 1
    YL = PL @ Y[off[0]:off[1]]                            # |bL| x m
    YR = PR @ Y[off[nlast]:off[nlast + 1]]
    FL, FR = _gamma_factor(SigL), _gamma_factor(SigR)
    Ts, ms = [], []
    for r in rl:
        k = int(np.sum(mu <= r * r))
        A = (-HY[:k, :k]).astype(complex)
        A[np.diag_indices_from(A)] += E
        A -= YL[:, :k].T @ SigL @ YL[:, :k] + YR[:, :k].T @ SigR @ YR[:, :k]
        X = sla.solve(A, YL[:, :k].T @ FL, check_finite=False)
        Ts.append(float(np.sum(np.abs(FR.conj().T @ (YR[:, :k] @ X)) ** 2))); ms.append(k)
    T = Ts[0] if np.ndim(rmax) == 0 else Ts
    if return_info:
        return T, dict(m=ms, N=len(off) and int(off[-1]), mu=mu)
    return T


# ---------------------------------------------------------------------------------------------
# Folded stabilisation (consistent): A(E) = U^T (E - H - Sigma) U + tau * F_int(E)
# F_int(E) = U^T (H - E) Pi_int (H - E) U annihilates every exact interior solution at energy E,
# so a basis that contains the exact scattering states still reproduces them exactly; a
# cut-induced direction with interior residual r gets an extra tau r^2. For tau = -i gamma the
# spurious level turns into a pole of width ~ gamma r^2, while exact directions are untouched.
# ---------------------------------------------------------------------------------------------

def folded_interior_blocks(blk, bnd, E):
    n = len(blk["HD"])
    Dg = []
    for s in range(n):
        M = blk["GD"][s] - 2 * E * blk["HD"][s]
        M[np.diag_indices_from(M)] += E * E
        Dg.append(M)
    Up = [blk["GU"][s] - 2 * E * blk["HU"][s] for s in range(n - 1)]
    for (P, K), s in zip(bnd, (0, n - 1)):
        C = K - E * P.T
        Dg[s] = Dg[s] - C @ C.T
    return Dg, Up


def _sym_btd_solve(Dg, Up, B):
    """Block Thomas for a complex-symmetric block-tridiagonal matrix (lower = Up^T)."""
    n = len(Dg)
    ncol = next(b.shape[1] for b in B if b is not None)
    lus, y = [], []
    for i in range(n):
        Di = Dg[i]
        bi = B[i] if B[i] is not None else np.zeros((Di.shape[0], ncol), complex)
        if i > 0:
            Li = sla.lu_solve(lus[-1], Up[i - 1], trans=1).T
            Di = Di - Li @ Up[i - 1]
            bi = bi - Li @ y[-1]
        lus.append(sla.lu_factor(Di, check_finite=False)); y.append(bi)
    X = [None] * n
    X[-1] = sla.lu_solve(lus[-1], y[-1])
    for i in range(n - 2, -1, -1):
        X[i] = sla.lu_solve(lus[i], y[i] - Up[i] @ X[i + 1])
    return X


def transmission_stabilized(blk, bnd, E, SigL, SigR, taus, gam_thr=1e-6):
    """T for each tau in taus (tau = 0 is plain Galerkin)."""
    n = len(blk["HD"])
    (PL, KL), (PR, KR) = bnd
    FDg, FUp = folded_interior_blocks(blk, bnd, E)
    FL, FR = _gamma_factor(SigL), _gamma_factor(SigR)
    lamL = np.linalg.norm(FL, axis=0) ** 2; FL = FL[:, lamL > gam_thr * lamL.max()]
    out = []
    for tau in np.atleast_1d(taus):
        Dg = []
        for s in range(n):
            M = (-blk["HD"][s] + tau * FDg[s]).astype(complex)
            M[np.diag_indices_from(M)] += E
            Dg.append(M)
        Up = [(-blk["HU"][s] + tau * FUp[s]).astype(complex) for s in range(n - 1)]
        Dg[0] = Dg[0] - PL.T @ SigL @ PL
        Dg[-1] = Dg[-1] - PR.T @ SigR @ PR
        X = _sym_btd_solve(Dg, Up, [PL.T @ FL] + [None] * (n - 1))
        out.append(float(np.sum(np.abs(FR.conj().T @ (PR @ X[-1])) ** 2)))
    return out


# ---------------------------------------------------------------------------------------------
# Folded absorber restricted to the classically forbidden region
#   F_w(E) = U^T (H + V - E) W(E) (H + V - E) U,   W_ii(E) = 1 / (1 + exp((E - E_c - U(x_i)) / kw))
# W selects the rows where the transport energy lies in the LOCAL gap (under the barrier), which
# is where cut-induced levels act as fake tunnelling paths. Where the electron propagates
# (contacts, on-state channel, energies above the barrier top) W ~ 0 and the solve reduces to
# plain Galerkin, so propagating states with a moderate truncation residual are not penalised.
# ---------------------------------------------------------------------------------------------

def weighted_folded_blocks(Hs, block_orbs, bases, block_slab, w, wtol=1e-6):
    """Slab blocks (WD, WU) of (Hs U)^T diag(w) (Hs U); Hs = H + V - E (sparse). Rows with
    w < wtol are skipped; returns None if no row carries weight."""
    if not np.any(w >= wtol):
        return None
    Hs = Hs.tocsc()
    order = np.argsort(np.asarray(block_slab), kind="stable")
    block_orbs = [block_orbs[b] for b in order]
    bases = [bases[b] for b in order]
    bslab = np.asarray(block_slab)[order]
    nslab = int(bslab.max()) + 1
    k = np.array([Q.shape[1] for Q in bases])
    starts = np.concatenate([[0], np.cumsum(k)])
    sl_start = [int(starts[np.where(bslab == s)[0][0]]) for s in range(nslab)] + [int(starts[-1])]
    ns = [sl_start[s + 1] - sl_start[s] for s in range(nslab)]
    WD = [np.zeros((n, n)) for n in ns]
    WU = [np.zeros((ns[s], ns[s + 1])) for s in range(nslab - 1)]
    Dm, R = [], []
    for orbs, Q in zip(block_orbs, bases):
        Hc = Hs[:, orbs]
        rows = np.unique(Hc.indices)
        rows = rows[w[rows] >= wtol]
        if len(rows):
            Dm.append(np.asarray(Hc[rows] @ Q) * np.sqrt(w[rows])[:, None])
        else:
            Dm.append(np.zeros((0, Q.shape[1])))
        R.append(rows)
    nb = len(bases)
    for i in range(nb):
        if len(R[i]) == 0:
            continue
        si = bslab[i]
        for j in range(i, nb):
            if len(R[j]) == 0:
                continue
            sj = bslab[j]
            com, ii, jj = np.intersect1d(R[i], R[j], assume_unique=True, return_indices=True)
            if len(com) == 0:
                continue
            g = Dm[i][ii].T @ Dm[j][jj]
            ri = slice(starts[i] - sl_start[si], starts[i + 1] - sl_start[si])
            rj = slice(starts[j] - sl_start[sj], starts[j + 1] - sl_start[sj])
            if si == sj:
                WD[si][ri, rj] = g; WD[si][rj, ri] = g.T
            else:
                WU[si][ri, rj] = g
    for s in range(nslab):
        WD[s] = 0.5 * (WD[s] + WD[s].T)
    return WD, WU


def gamma_factors(SigL, SigR, gam_thr=1e-6):
    """(FL, FR) with Gamma = F F^H; FL keeps the channels above gam_thr. Depends only on E and the
    leads: compute once per energy and pass to transmission_with_F (the eigh costs ~20 s per call)."""
    FL, FR = _gamma_factor(SigL), _gamma_factor(SigR)
    lamL = np.linalg.norm(FL, axis=0) ** 2
    return FL[:, lamL > gam_thr * lamL.max()], FR


def transmission_with_F(blk, bnd, E, SigL, SigR, taus, F, gam_thr=1e-6, gam=None):
    """Like transmission_stabilized, with a given absorber F = (FD, FU) (None = Galerkin only).
    gam: precomputed gamma_factors(SigL, SigR, gam_thr) (computed here if None)."""
    n = len(blk["HD"])
    (PL, KL), (PR, KR) = bnd
    FL, FR = gam if gam is not None else gamma_factors(SigL, SigR, gam_thr)
    out = []
    for tau in np.atleast_1d(taus):
        use = (F is not None) and tau != 0
        Dg = []
        for s in range(n):
            M = (-blk["HD"][s]).astype(complex)
            if use:
                M = M + tau * F[0][s]
            M[np.diag_indices_from(M)] += E
            Dg.append(M)
        Up = [(-blk["HU"][s] + (tau * F[1][s] if use else 0)).astype(complex) for s in range(n - 1)]
        Dg[0] = Dg[0] - PL.T @ SigL @ PL
        Dg[-1] = Dg[-1] - PR.T @ SigR @ PR
        X = _sym_btd_solve(Dg, Up, [PL.T @ FL] + [None] * (n - 1))
        out.append(float(np.sum(np.abs(FR.conj().T @ (PR @ X[-1])) ** 2)))
    return out
