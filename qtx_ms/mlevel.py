"""Density-matrix hierarchy with repeated merging: a multilevel preconditioner for the QTBM system A psi = b
(user, 2026-11-15; the library form of results_ms/gpu/mlevel/mlevel.py, on the qtx_ms stack: read_inputs,
msbasis.Device, schwarz.CapSchwarz / SlowCoarse / fgmres_block; numpy or cupy via qtx_ms.gpu).

Level 0 (fine): A = E S - H - Sigma, smoother S0 = CAP-Schwarz (schwarz.CapSchwarz, x windows, optional transverse
  split), RAS over the window cores. The cores are the level-1 blocks.
Level 1 (local rho, per core): modes of the contour density matrix (Ec + mu1 .. Ec + mu2, npts) of each S0 window
  operator A_kk(z) + i s P S P, restricted to its core. Selection "dm": eigenvectors of the window rho above the cut,
  restricted to the core (SVD); "dmc": eigenvectors of the core block of rho above the cut.
Level k + 1 (one `merge` per further level): blocks = pairs of level-k blocks, neighbours along x (transport) or y
  (the transverse strips). Per pair C: window w = C + the level-k blocks coupled to it (halo), local operator in the
  LEVEL-k BASIS z S_k - H_k - Sigma_k(z) + CAP_k (CAP and Sigma projected through all the bases below), contour rho
  over its own energy window (a narrower window = the energy slice of that level), modes above the cut on C (same
  selection as level 1) = U_{k+1}[C] in level-k coordinates. S_{k+1}, H_{k+1}, A_{k+1} = Galerkin, block-sparse over
  the pair graph.
  halo = H: the window keeps only the coupled level-k blocks within H blocks of the pair along x (0: the x positions
  of the pair only; None: every coupled block). The couplings to the dropped blocks are cut like any other artificial
  cut (projected CAP); the operators A_k stay the full Galerkin products. halo_smooth "full" keeps the full windows
  for the smoother of A_k (halo only for rho). Per level the log lists the coupling blocks of A_k, S_k by distance
  (2-norms) and the window size / rho cost of every halo.
Smoothers: the smoother of A_k is formed in the level-k basis with the level-(k+1) blocks: RAS over the windows w
  (A_k[w, w] + CAP_k), writing the rows of C (l1_solver "rows": only those rows of the window inverse are kept).
Cycle (preconditioner of flexible block GMRES on the fine system, x0 = 0): on every level one smoothing step, gamma
  coarse corrections (1: V, 2: W), one smoothing step. Bottom solver of the deepest level m used:
    exact   dense LU of A_m ("exact level m", for every m = 1 .. n: the loss of each level)
    top     A_n: RAS over its own blocks (block + coupled blocks, projected CAP) + slow modes of that cycle
            (schwarz.SlowCoarse) in rounds of --slow m,q, the first N of them (--counts)
  fine  : two-level baseline, slow modes directly on the fine smoother.

Basis reuse over energies: the block bases of every level are energy independent (contour rho over the window from
  S, H, the projected CAP and Sigma(z)); only A_k(E) = E S_k - H_k - Sigma_k(E) and the smoothers depend on E. merge(...,
  bkey=K) keeps U_{k+1} per key in memory (host) and, with h.basis_dir set, in <basis_dir>/l<k>_<hash>.npz; a later
  merge with the same key skips the contour rho (driver: --basis-cache, default <dm-cache>/<tag>_levels).

Library: h = Hierarchy(dev, inp, sw); DM = h.window_dm(...); U = h.basis1(DM, kind, cut); h.level1(U, E, SL, SR);
h.merge(axis, cut, kind, mu1, mu2, npts, halo, bkey) per further level; prec, timer = h.cycle(m, bottom, A0, S0, gamma).

    python -m qtx_ms.mlevel --inputs inputs/mos2_hfo2_9072a --dE 0.15 --gpu --sigma-cache DIR --psi-dir DIR \\
        --dm-cache DIR [--cap 16,2,2 --cap-split 2,1 --cap-tov 6 --cap-solver rows64] --l1 dm:1e-2 \\
        --up y:1e-2,x:1e-2:0.05:0.25 [--halo 1 --halo-smooth same] [--sweep fine,exact,top --counts 32,128,256]
"""
import argparse, gc, hashlib, json, os, time
import numpy as np
import scipy.sparse as sp

from qtx_ms import gpu, msbasis as mb

log = mb.log
quiet = lambda *x: None


def _xp():
    return gpu.cp if gpu.on else np


def contour_points(mu1, mu2, npts):
    """Upper half circle over [mu1, mu2]: points z and weights dz (Gauss-Legendre in the angle, as rho_contour)."""
    c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    x, w = np.polynomial.legendre.leggauss(int(npts))
    th, w = 0.5 * np.pi * (x + 1), 0.5 * np.pi * w
    return c + r * np.exp(1j * th), -1j * r * np.exp(1j * th) * w


def offs(r):
    return np.concatenate([[0], np.cumsum(r)]).astype(int)


def rho_of(acc):
    """rho = -(1 / pi) Im of the contour integral of G (Hermitian part)."""
    rho = -((acc - acc.conj().T) / 2j) / np.pi
    return 0.5 * (rho + rho.conj().T)


def nrm2(B, its=12):
    """2-norm of a block by power iteration on B^H B (a lower bound, converged to a few % in 12 steps)."""
    if not min(B.shape):
        return 0.0
    xp = gpu.xp_of(B)
    v = xp.asarray(np.random.default_rng(1).standard_normal(B.shape[1]) + 0j)
    for _ in range(its):
        v = B.conj().T @ (B @ v)
        nv = float(xp.linalg.norm(v))
        if nv == 0:
            return 0.0
        v /= nv
    return float(xp.linalg.norm(B @ v))


EIG_STATS = []          # per rand_top_eig call: (n, l, k, rounds, fallback); summarized and reset by merge


def rand_top_eig(rho, cut, q=2, f=0.5, p=32, l0=256, lmax_frac=0.6, seed=0):
    """Eigenpairs of the Hermitian rho with eigenvalue > cut, by a randomized range finder (Halko, Martinsson, Tropp):
    Q = orth((rho)^(2q+1) Omega), grown in blocks until the Rayleigh-Ritz spectrum of Q^H rho Q has at most l - p
    values above f * cut (the subspace then reaches below the cut, so the kept vectors converge at the rate
    (f cut / lambda)^(2q+1)). Memory n x l instead of the ~11 n^2 workspace of the dense eigh. When l would pass
    lmax_frac * n: dense eigh (on the host when on the GPU: no workspace limit). Returns (ev, V) ascending, ev > cut."""
    xp = gpu.xp_of(rho)
    n = rho.shape[0]
    rs = xp.random.RandomState(seed)
    Q = xp.zeros((n, 0), complex)
    l, rounds = min(n, l0), 0

    def orth(Y):                                           # Y orthogonal to Q (block Gram-Schmidt twice), then QR
        for _ in range(2):
            if Q.shape[1]:
                Y -= Q @ (Q.conj().T @ Y)
        return xp.linalg.qr(Y)[0]

    while True:
        m = l - Q.shape[1]
        Y = rs.standard_normal((n, m)) + 1j * rs.standard_normal((n, m))
        Y = rho @ Y
        for _ in range(q):
            Y = rho @ orth(Y)
        Q = xp.concatenate([Q, orth(Y)], axis=1)
        del Y
        rounds += 1
        ev, W = xp.linalg.eigh(Q.conj().T @ (rho @ Q))
        nb = int((ev > f * cut).sum())
        if nb + p <= l or l >= n:
            break
        lnew = min(n, max(2 * l, nb + 2 * p))
        if lnew > lmax_frac * n:                               # the range is most of the window: dense eigh
            del Q, W
            gpu.free()
            if gpu.on:
                ev, V = np.linalg.eigh(gpu.asnumpy(rho))
                ev, V = xp.asarray(ev), xp.asarray(V)
            else:
                ev, V = np.linalg.eigh(rho)
            k = int((ev > cut).sum())
            EIG_STATS.append((n, n, k, rounds, True))
            return ev[n - k:], V[:, n - k:]
        l = lnew
    k = int((ev > cut).sum())
    EIG_STATS.append((n, l, k, rounds, False))
    return ev[l - k:], Q @ W[:, l - k:]


def select(rho, rows, cut, kind, eig="full"):
    """Modes on `rows` of a local rho: dm = eigenvectors of the whole window rho with occupation > cut, restricted to
    rows, SVD; dmc = eigenvectors of rho[rows, rows] with occupation > cut. Returns U (orthonormal columns).
    eig "rand": the dm eigenvectors by the randomized range finder (rand_top_eig), not the dense eigh."""
    xp = gpu.xp_of(rho)
    if kind == "dm":
        ev, V = rand_top_eig(rho, cut) if eig == "rand" else xp.linalg.eigh(rho)
        k = int((ev > cut).sum())
        if not k:
            return xp.zeros((len(rows), 0), complex)
        u, sv, _ = xp.linalg.svd(V[rows][:, -k:], full_matrices=False)
        return xp.ascontiguousarray(u[:, :int((sv > 1e-8 * sv[0]).sum())])
    ev, V = xp.linalg.eigh(rho[rows][:, rows])
    return xp.ascontiguousarray(V[:, ev > cut][:, ::-1])


def nystrom(Om, Y):
    """Nystrom of a Hermitian PSD rho from Y = rho Om: rho ~ F F^H, F = Y (Om^H Y)^(-1/2) (pseudo-inverse square root,
    eigenvalues below 1e-10 of the largest dropped). Returns X, lam (left singular vectors of F, squared singular values,
    descending) and F."""
    xp = gpu.xp_of(Y)
    C = Om.conj().T @ Y
    d, V = xp.linalg.eigh(0.5 * (C + C.conj().T))
    ok = d > 1e-10 * d.max()
    F = Y @ (V[:, ok] / xp.sqrt(d[ok]))
    X, sv, _ = xp.linalg.svd(F, full_matrices=False)
    return X, sv ** 2, F


class SchurRho:
    """Contour accumulation of the block P (core / pair rows) of a window Green's function by folding the rest of the
    window E into P (the single-shot msbasis tile_rho_contour "schur" path): per point
      G_PP(z) = S_P(z)^-1,  S_P = A_PP - A_PE A_EE^-1 A_EP,  A = M(z) (dense window matrix),
    only the m = |P| columns A_EP are solved with the environment. prec "single": A_EE LU in complex64 and `refine`
    steps of iterative refinement of X = A_EE^-1 A_EP in complex128; the Schur complement and its solve in complex128.
    probes r > 0: G_PP Omega and G_PP^H Omega (LU of S_P, two solves) instead of the inverse; rho_PP Omega -> Nystrom.
    add(z, dz, M) per point; result() -> (occ, V) of rho_PP (descending; V orthonormal, m x k)."""

    def __init__(self, P, n, prec="single", refine=1, probes=0, seed=0):
        xp = _xp()
        self.xp, self.prec, self.refine = xp, prec, int(refine)
        mask = np.zeros(n, bool)
        mask[np.asarray(gpu.asnumpy(P))] = True
        self.p, self.e = xp.asarray(np.where(mask)[0]), xp.asarray(np.where(~mask)[0])
        m = len(self.p)
        self.Om = None
        if probes and probes < m:
            rs = xp.random.RandomState(seed)
            self.Om = (rs.standard_normal((m, probes)) + 1j * rs.standard_normal((m, probes))) / np.sqrt(2)
            self.acc, self.acch = xp.zeros((m, probes), complex), xp.zeros((m, probes), complex)
        else:
            self.acc = xp.zeros((m, m), complex)

    def add(self, z, dz, M):
        xp, p, e = self.xp, self.p, self.e
        Me = M[e]
        AEE, AEP = Me[:, e], Me[:, p]
        del Me
        Mp = M[p]
        APP, APE = Mp[:, p], Mp[:, e]
        del Mp
        if len(e):
            if self.prec == "single":
                lu = gpu.lu_factor(AEE.astype(np.complex64))
                X = gpu.lu_solve(lu, AEP.astype(np.complex64)).astype(complex)
                for _ in range(self.refine):
                    X += gpu.lu_solve(lu, (AEP - AEE @ X).astype(np.complex64))
            else:
                lu = gpu.lu_factor(AEE)
                X = gpu.lu_solve(lu, AEP)
            del lu, AEE, AEP
            Sp = APP - APE @ X
            del X
        else:
            Sp = APP.copy()
        del APP, APE
        if self.Om is None:
            self.acc += dz * xp.linalg.inv(Sp)
        else:
            lu = gpu.lu_factor(Sp)
            self.acc += dz * gpu.lu_solve(lu, self.Om)
            self.acch += np.conj(dz) * gpu.lu_solve(lu, self.Om, trans=2)
            del lu
        del Sp

    def result(self):
        xp = self.xp
        if self.Om is None:
            ev, V = xp.linalg.eigh(rho_of(self.acc))
            o = xp.argsort(-ev)
            occ, V = ev[o], V[:, o]
        else:
            X, occ, _ = nystrom(self.Om, -((self.acc - self.acch) / 2j) / np.pi)
            V = X
        self.acc = self.acch = None
        return occ, V


class Timed:
    """Callable / operator wrapper: column applications and wall time (GPU synchronized)."""

    def __init__(self, f, name):
        self.f, self.name, self.n, self.t = f, name, 0, 0.0
        self.shape = getattr(f, "shape", None)

    def _run(self, g, X):
        t = time.time()
        y = g(X)
        if gpu.on:
            gpu.sync()
        self.t += time.time() - t
        self.n += X.shape[1] if X.ndim > 1 else 1
        return y

    def __call__(self, r):
        return self._run(self.f, r)

    def __matmul__(self, X):
        return self._run(lambda Y: self.f @ Y, X)


class BlockOp:
    """Block-sparse level matrix: dense blocks B[i][q] for the coupled blocks (i, cols[i][q]); A_k @ Y = one GEMM per
    block row."""

    def __init__(self, off, rows, cols, blocks, host=False):
        xp = _xp()
        self.off, self.N = off, int(off[-1])
        self.shape = (self.N, self.N)
        self.rows, self.cols, self.blocks = rows, cols, blocks      # per block row: row block, its columns, [B_ij]
        # host: the block rows live in pinned host memory and every product streams them (gpu.Streamer: the copy of
        # row i + 1 overlaps the GEMM of row i); take / dense / sub copy the rows they need
        self.host = bool(host) and gpu.on
        self.R, self.st = [], None
        for i in range(len(rows)):
            Ri = (gpu.pinned(np.hstack([gpu.asnumpy(b_) for b_ in blocks[i]])) if self.host
                  else xp.hstack(blocks[i]))
            co = offs([b_.shape[1] for b_ in blocks[i]])
            # the block list now points into the block row: A_k is stored once (in place: the caller's list too)
            blocks[i][:] = [Ri[:, co[q]:co[q + 1]] for q in range(len(co) - 1)]
            self.R.append(Ri)
        self.C = [xp.asarray(np.concatenate([np.arange(off[j], off[j + 1]) for j in cols[i]])) for i in range(len(rows))]
        if self.host:
            self.st = gpu.Streamer([(Ri,) for Ri in self.R])
        self.gb = sum(Ri.nbytes for Ri in self.R) / 2 ** 30

    def __matmul__(self, Y):
        xp = _xp()
        one = Y.ndim == 1
        Y = Y[:, None] if one else Y
        out = xp.empty((self.N, Y.shape[1]), complex)
        if self.st is not None:
            def fn(i, v):
                out[self.off[i]:self.off[i + 1]] = v[0] @ Y[self.C[i]]
            self.st.run(fn)
        else:
            for i in range(len(self.rows)):
                out[self.off[i]:self.off[i + 1]] = self.R[i] @ Y[self.C[i]]
        return out[:, 0] if one else out

    def dense(self):
        D = _xp().zeros(self.shape, complex)
        for i in range(len(self.rows)):
            D[self.off[i]:self.off[i + 1], self.C[i]] = _xp().asarray(self.R[i])
        return D

    def take(self, wr, wc):
        """Dense A_k[wr, wc] for the block lists wr, wc (blocks in the given order)."""
        wr, wc = [int(k) for k in wr], [int(k) for k in wc]
        loc = {k: i for i, k in enumerate(wc)}
        orow = offs([self.off[k + 1] - self.off[k] for k in wr])
        ocol = offs([self.off[k + 1] - self.off[k] for k in wc])
        D = _xp().zeros((orow[-1], ocol[-1]), complex)
        for a_, i in enumerate(wr):                              # from the block rows R (the block list may be freed)
            co = offs([self.off[int(j) + 1] - self.off[int(j)] for j in self.cols[i]])
            for q, j in enumerate(self.cols[i]):
                if int(j) in loc:
                    D[orow[a_]:orow[a_ + 1], ocol[loc[int(j)]]:ocol[loc[int(j)] + 1]] = _xp().asarray(
                        self.R[i][:, co[q]:co[q + 1]])
        return D

    def sub(self, w):
        """Dense A_k[w, w] for the sorted block list w, and the offsets of the blocks in it."""
        w = [int(k) for k in w]
        return self.take(w, w), offs([self.off[k + 1] - self.off[k] for k in w])


class GalerkinOp:
    """Level operator without stored blocks (user, 2026-11-19; levels 1 .. --free-levels): A_k = U_k^H A_{k-1} U_k
    exactly (Galerkin, Sigma included), so A_k @ Y = P_k^H (A_{k-1} (P_k Y)) through the operator of the level below
    (recursively down to the fine A: 13.8 GB sparse vs ~33 GB of dense A_1 blocks on the 6 nm, where level 1
    compresses only n / N1 ~ 2; A_2 ~10 GB). Dense pieces (take / sub: windows of the smoother of A_k, block pairs of
    the next Galerkin product, SlabLU) are assembled on demand from the level-k S, H blocks and the lead Sigma blocks
    sig1 (level-k coordinates, lead-face blocks only): E S - H - Sigma, the same entries as the stored A_k."""

    def __init__(self, h, L, A0, E, sig1):
        self.h, self.L, self.A0, self.E, self.sig1 = h, L, A0, E, sig1
        self.off, self.N = L.off, int(L.off[-1])
        self.shape = (self.N, self.N)
        self.rows, self.cols = list(range(len(L.r))), L.nbr
        self.R, self.blocks, self.host, self.st, self.gb = [], None, False, None, 0.0

    def __matmul__(self, Y):
        one = Y.ndim == 1
        Y = Y[:, None] if one else Y
        out = self.L.PH(self.A0 @ self.L.P(Y))           # A0: the operator of the level below (fine A for k = 1)
        return out[:, 0] if one else out

    def take(self, wr, wc):
        h, L = self.h, self.L
        wr, wc = [int(k) for k in wr], [int(k) for k in wc]
        D = self.E * h.dense(L, L.Sb, wr, wc)
        D -= h.dense(L, L.Hb, wr, wc)
        orow, ocol = offs(L.r[wr]), offs(L.r[wc])
        lr, lc = {k: q for q, k in enumerate(wr)}, {k: q for q, k in enumerate(wc)}
        for (i, j), B in self.sig1.items():
            if i in lr and j in lc:
                D[orow[lr[i]]:orow[lr[i] + 1], ocol[lc[j]]:ocol[lc[j] + 1]] -= B
        return D

    def sub(self, w):
        w = [int(k) for k in w]
        return self.take(w, w), offs([self.off[k + 1] - self.off[k] for k in w])

    def dense(self):
        return self.take(self.rows, self.rows)


class SlabLU:
    """Exact solve with a level operator A_k without its dense LU: block-tridiagonal LU over x slabs (RGF-style
    left-to-right elimination). A slab = `width` consecutive x positions of the level's blocks, width = the largest x
    distance of two coupled blocks, so that A_k is block tridiagonal over the slabs. Storage 3 sum n_s^2 (slab LU,
    L_s = A_{s+1,s} D_s^-1, A_{s,s+1}) instead of N^2, work sum n_s^3 instead of N^3."""

    def __init__(self, A, ix):
        xp = _xp()
        ix = np.asarray(ix)
        rk = np.searchsorted(np.unique(ix), ix)                  # x position of every block: 0, 1, ...
        self.width = wd = max(1, max(int(abs(rk[i] - rk[int(j)])) for i in range(len(rk)) for j in A.cols[i]))
        sl = rk // wd
        bs = [np.where(sl == s)[0] for s in range(int(sl.max()) + 1)]
        self.idx = [xp.asarray(np.concatenate([np.arange(A.off[k], A.off[k + 1]) for k in b_])) for b_ in bs]
        self.sizes = [len(i_) for i_ in self.idx]
        self.lu, self.L, self.Up = [], [], []
        Dt = A.take(bs[0], bs[0])
        for s in range(len(bs)):
            self.lu.append(gpu.lu_factor(Dt))
            del Dt
            if s < len(bs) - 1:
                Lo = A.take(bs[s + 1], bs[s])
                Lt = xp.ascontiguousarray(gpu.lu_solve(self.lu[s], xp.ascontiguousarray(Lo.T), trans=1).T)   # Lo D^-1
                del Lo
                Up = A.take(bs[s], bs[s + 1])
                Dt = A.take(bs[s + 1], bs[s + 1])
                Dt -= Lt @ Up
                self.L.append(Lt)
                self.Up.append(Up)
        self.gb = 3 * sum(float(n_) ** 2 for n_ in self.sizes) * 16 / 2 ** 30

    def __call__(self, r):
        xp = _xp()
        n = len(self.idx)
        Y = [r[i_].astype(complex) for i_ in self.idx]
        for s in range(1, n):
            Y[s] -= self.L[s - 1] @ Y[s - 1]
        x = xp.empty(r.shape, complex)
        X = gpu.lu_solve(self.lu[-1], Y[-1])
        x[self.idx[-1]] = X
        for s in range(n - 2, -1, -1):
            X = gpu.lu_solve(self.lu[s], Y[s] - self.Up[s] @ X)
            x[self.idx[s]] = X
        return x


class Lev:
    """One level: blocks (unions of level-1 cores), U[b] in the coordinates of the level below (level 1: the fine
    orbitals of the core), block graph G / nbr, grid (ix, iy) of the blocks, block-sparse S, H, A(E) (Sb, Hb, Ab, A),
    prolongation P to the level below and its adjoint PH, Sbelow = the smoother of the level below (levels >= 2)."""


class SubCoarse:
    """Galerkin coarse level on the first N columns of Q (= the SlowCoarse space after N / m rounds)."""

    def __init__(self, A, Q, N):
        xp = gpu.xp_of(Q)
        self.Q = xp.ascontiguousarray(Q[:, :N])
        self.lu = gpu.lu_factor(self.Q.conj().T @ xp.asarray(A @ self.Q))

    def __call__(self, r):
        return self.Q @ gpu.lu_solve(self.lu, self.Q.conj().T @ gpu.xp_of(self.Q).asarray(r))


def gcycle(A, smoother, coarse, gamma=1):
    """Preconditioner: pre smoothing, gamma coarse corrections (1: V, 2: W), post smoothing."""
    def prec(r):
        z = smoother(r)
        for _ in range(gamma):
            z = z + coarse(r - A @ z)
        return z + smoother(r - A @ z)
    return prec


class Hierarchy:
    """The levels above a CAP-Schwarz fine smoother `sw` (schwarz.CapSchwarz with x windows; its cores partition the
    device and are the level-1 blocks, ordered k = ix * ny + iy).
    l1_cap = (s1, d1): strength [eV] and ramp [Å] of the projected CAP of the level windows (default: those of sw);
    l1_ramp "dist" (distance to the nearest atom outside the window, minimum image) | "plane" (x planes);
    l1_lu "c64" | "c128": precision of the window factors of the level smoothers; l1_solver "rows" | "lu";
    batch_gb, mem_frac: GPU memory of the window accumulators of one rho batch (at most mem_frac of the free memory);
    halo_smooth "same" | "full": windows of the level smoothers (see the module docstring)."""

    def __init__(self, dev, inp, sw, l1_cap=None, l1_ramp="dist", l1_lu="c64", l1_solver="rows", batch_gb=24.0,
                 mem_frac=0.5, halo_smooth="same", host_blocks=False, stream_gb=None, l1_free=False, merge_eig="full",
                 sketch_prec="c128", core=None, log=log):
        assert l1_ramp in ("dist", "plane") and l1_lu in ("c64", "c128") and l1_solver in ("rows", "lu")
        assert halo_smooth in ("same", "full") and merge_eig in ("full", "rand", "sketch")
        self.dev, self.inp, self.sw, self.log = dev, inp, sw, log
        # "rand": dm eigenvectors of the merge windows by the randomized range finder (rho formed); "sketch": rho
        # Omega from a probe block (one LU + 2 solves per contour point, no window rho), Nystrom (_sketch_batch)
        self.merge_eig = merge_eig
        self.sketch_prec = sketch_prec
        # core = dict(prec, refine, probes): dmc blocks by Schur folding of the window into the core / pair (SchurRho)
        self.core = core
        # host_blocks: the S, H Galerkin blocks of every level live in host memory (read only to assemble A_k, the rho
        # windows and the Galerkin products of the next level); 6 nm y-cut level 1: ~44 GB off the GPU
        self.host_blocks = bool(host_blocks) and gpu.on
        # stream_gb: level operators A_k and level smoothers larger than this live in pinned host memory and are
        # streamed through the GPU at every application (copy overlapped with the GEMMs; None: everything resident)
        self.stream_gb = stream_gb if gpu.on else None
        # l1_free: no stored A_1; products through the fine operator, dense pieces from the S1, H1 blocks (GalerkinOp)
        self.free_levels = int(l1_free)                  # levels 1 .. free_levels without stored A_k (bool: 0 / 1)
        self.memlog = []
        self.xp = xp = _xp()
        self.n = n = dev.n
        self.S = dev.S.tocsr()
        self.x = sw.x
        self.s1, self.d1 = l1_cap if l1_cap else (sw.s, sw.d)
        self.l1_ramp, self.l1_solver, self.halo_smooth = l1_ramp, l1_solver, halo_smooth
        self.batch_gb, self.mem_frac = batch_gb, mem_frac
        self.cdt = np.complex64 if l1_lu == "c64" else np.complex128
        self.lub = 8 if l1_lu == "c64" else 16
        self.nw = nw = len(sw.idx)
        self.cores = [sw.idx[k][sw.core[k]] for k in range(nw)]     # RAS partition of the device (device indices)
        assert np.array_equal(np.sort(np.concatenate(self.cores)), np.arange(n)), "window cores must partition the device"
        self.cores_g = [xp.asarray(c) for c in self.cores]
        self.lab = np.empty(n, int)
        self.posin = np.empty(n, np.int64)                          # position of each orbital inside its core
        for k, c in enumerate(self.cores):
            self.lab[c] = k
            self.posin[c] = np.arange(len(c))
        assert nw < 2 ** 15
        self.lab16 = self.lab.astype(np.int16)        # int16: numpy's stable argsort is then a radix sort (O(nnz))
        # core coupling graph from the pattern of A (= of H, S and the lead Sigma blocks, which couple only within
        # bL / bR)
        Lm = sp.csr_matrix((np.ones(n, np.float32), (np.arange(n), self.lab)), shape=(n, nw))
        G = np.zeros((nw, nw), bool)
        for M_ in (dev.H.tocsr(), self.S):                          # pattern only: reuse the index arrays
            G |= (Lm.T @ sp.csr_matrix((np.ones(len(M_.indices), np.float32), M_.indices, M_.indptr), shape=(n, n))
                  @ Lm).toarray() > 0
        for b in (dev.bL, dev.bR):
            G[np.ix_(np.unique(self.lab[b]), np.unique(self.lab[b]))] = True
        self.G = G
        self.nbr = [np.where(G[i])[0] for i in range(nw)]
        self.pos = [{int(j): q for q, j in enumerate(self.nbr[i])} for i in range(nw)]
        self.atom = np.repeat(np.arange(len(inp["norb"])), inp["norb"])[dev.dev]
        self.geo = mb.Geometry(inp["xyz"], inp["meta"].get("periodic"))
        self.lead_cores = set(int(k) for b_ in (dev.bL, dev.bR) for k in np.unique(self.lab[b_]))
        nxw = len(sw.centres)
        assert nw % nxw == 0, "the windows are not x windows (x transverse strips)"
        self.ny = nw // nxw                                         # windows are ordered k = ix * ny + iy
        cx = np.array([self.x[c].mean() for c in self.cores]).reshape(-1, self.ny)
        assert np.ptp(cx, axis=1).max() < 3.0 and (np.diff(cx.mean(1)) > 0).all(), "window order is not (x, strip)"
        self.LV = {}
        self.ubank, self.basis_dir = {}, None                       # level bases by key: host copies / disk cache
        log(f"fine: n {n}, {nw} window cores, {min(len(c) for c in self.cores)}..{max(len(c) for c in self.cores)} "
            f"orbitals, core graph {int(G.sum())} coupled pairs ({G.sum(1).min()}..{G.sum(1).max()} per core); level "
            f"CAP {self.s1:g} eV over {self.d1:g} Å, ramp {l1_ramp}, LU {l1_lu}")

    # ------------------------------------------------------------------ helpers
    def mem(self, step):
        """Memory profile: one log line per step (GB): live = arrays alive after the step, peak = highest live during
        the step (since the previous line), pool = held by cupy, free = driver free. Kept in self.memlog."""
        m = gpu.mem_mark()
        if m is None:
            return
        g = 2 ** 30
        self.memlog.append(dict(step=step, **{k: v / g for k, v in m.items()}))
        self.log(f"    mem | {step:58s} live {m['live'] / g:6.2f}  peak {m['peak'] / g:6.2f}  pool {m['pool'] / g:6.2f}  "
                 f"free {m['free'] / g:6.2f}")

    def inventory(self):
        """Log the arrays held per level (GB; shared arrays counted once): block bases U, S / H / A blocks, the
        smoother of the level below; and the fine smoother."""
        if not gpu.on:
            return
        seen = set()

        def nb(o):
            if o is None or id(o) in seen:
                return 0
            seen.add(id(o))
            if isinstance(o, (list, tuple)):
                return sum(nb(v) for v in o)
            return int(getattr(o, "nbytes", 0))
        g = 2 ** 30
        self.log(f"    mem | held: fine smoother rows {self.sw.rows_gb:.2f}"
                 + (" (host, streamed)" if getattr(self.sw, "st", None) is not None else "")
                 + f", core index arrays {nb(self.cores_g) / g:.2f}")
        for k, L in sorted(self.LV.items()):
            sm = getattr(L, "Sbelow", None)
            self.log(f"    mem | held: level {k} (N {L.N}): U {nb(L.U) / g:.2f}, S {nb(L.Sb) / g:.2f}, H {nb(L.Hb) / g:.2f}"
                     f"{' (host)' if self.host_blocks else ''}, "
                     f"A list {nb(L.Ab) / g + nb(L.A.blocks) / g:.2f}, A operator (block rows) {nb(L.A.R) / g:.2f}"
                     + (" (host, streamed)" if getattr(L.A, "host", False) else "")
                     + (f" (none: A{k} through the level below)" if isinstance(L.A, GalerkinOp) else "")
                     + (f", smoother of A{k - 1} {sm.gb:.2f}" + (" (host, streamed)" if getattr(sm, "streamed", False)
                                                                  else "") if sm is not None else ""))

    def memstr(self):
        return f"GPU {gpu.mem_used() / 2 ** 30:.1f} GB" if gpu.on else "cpu"

    @staticmethod
    def gpu_free_gb():
        """Free GPU memory in GB (driver free + nothing cached after gpu.free()); inf on the CPU."""
        if not gpu.on:
            return np.inf
        gpu.free()
        return gpu.cp.cuda.Device().mem_info[0] / 2 ** 30

    def ramp(self, F):
        """CAP ramp of the projected level CAP on the orbital set F (a union of cores)."""
        x, d1 = self.x, self.d1
        if self.l1_ramp == "plane":
            xi = x[F]
            lo, hi = xi.min(), xi.max()
            r = np.zeros(len(F))
            if lo > x.min() + 1e-6:
                r = np.maximum(r, np.clip((lo + d1 - xi) / d1, 0, 1) ** 2)
            if hi < x.max() - 1e-6:
                r = np.maximum(r, np.clip((xi - (hi - d1)) / d1, 0, 1) ** 2)
            return r
        inside = np.zeros(self.n, bool)
        inside[F] = True
        aw = np.unique(self.atom[F])
        ao = np.unique(self.atom[~inside])
        if not len(ao):
            return np.zeros(len(F))
        dist, _ = self.geo.tree(ao).query(self.geo.points(aw), distance_upper_bound=d1 + 6.0)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        d0 = dist.min()                                       # the outermost window atoms sit at the cut: ramp 1
        ra = np.clip((d1 - (dist - d0)) / d1, 0, 1) ** 2
        return ra[np.searchsorted(aw, self.atom[F])]

    def core_blocks(self, M, U):
        """[[U_i^H M[core_i, core_j] U_j for j in nbr[i]] for i]: rows of core i only, nonzeros grouped by the core of
        their column (radix sort of int16 core labels): every nonzero of M is touched once."""
        xp, nw, nbr, cores = self.xp, self.nw, self.nbr, self.cores
        blocks = [[None] * len(nbr[i]) for i in range(nw)]
        M = M.tocsr()
        for i in range(nw):
            R = M[cores[i]].tocoo()                                     # rows of core i (row gather, cheap)
            lc = self.lab16[R.col]
            o = np.argsort(lc, kind="stable")
            lc, rr, cc, vv = lc[o], R.row[o], self.posin[R.col[o]], R.data[o]
            bnd = np.searchsorted(lc, np.arange(nw + 1))
            assert bnd[-1] == len(lc) and set(np.unique(lc)) <= set(nbr[i]), "M couples beyond the core graph"
            for q, j in enumerate(nbr[i]):
                s, e = bnd[j], bnd[j + 1]
                if e == s:
                    blocks[i][q] = self.keep(xp.zeros((U[i].shape[1], U[j].shape[1]), complex))
                    continue
                # stable sort: inside the group the rows are still in CSR order -> indptr from row counts, no COO sort
                ip = np.concatenate([[0], np.cumsum(np.bincount(rr[s:e], minlength=len(cores[i])))])
                Rij = sp.csr_matrix((vv[s:e], cc[s:e], ip), shape=(len(cores[i]), len(cores[j])))
                Rg = gpu.csr(Rij, np.complex128) if gpu.on else Rij
                blocks[i][q] = self.keep(U[i].conj().T @ (Rg @ U[j]))
        return blocks

    def streams(self, gb):
        """True when an operator of gb GB is streamed from pinned host memory instead of kept on the GPU."""
        return self.stream_gb is not None and gb > self.stream_gb

    def keep(self, B):
        """An S / H block as it is stored: on the host (pinned: the windows read them back at bus speed) with
        host_blocks, else unchanged."""
        return gpu.pinned(B) if self.host_blocks and gpu.on else B

    def sigma1(self, U, SL, SR):
        """{(i, j): U_i^H Sigma U_j} for the lead-face cores i, j (level-1 coordinates)."""
        xp, lab, posin = self.xp, self.lab, self.posin
        out = {}
        for Sg, b_ in ((SL, self.dev.bL), (SR, self.dev.bR)):
            Sg = xp.asarray(Sg)
            ks = [int(k) for k in np.unique(lab[b_])]
            sel = {k: np.where(lab[b_] == k)[0] for k in ks}
            Ub = {k: U[k][xp.asarray(posin[b_[sel[k]]])] for k in ks}
            for i in ks:
                for j in ks:
                    Sij = Sg[xp.asarray(sel[i])[:, None], xp.asarray(sel[j])[None, :]]
                    out[(i, j)] = out.get((i, j), 0) + Ub[i].conj().T @ (Sij @ Ub[j])
        return out

    def dense(self, L, blocks, rows, cols):
        """Dense [rows, cols] (block lists) of a block-sparse matrix of level L (blocks[i][q] <-> (i, L.nbr[i][q]))."""
        rows, cols = [int(v) for v in rows], [int(v) for v in cols]
        orow, ocol = offs(L.r[rows]), offs(L.r[cols])
        D = self.xp.zeros((orow[-1], ocol[-1]), complex)
        loc = {k: q for q, k in enumerate(cols)}
        for i_, i in enumerate(rows):
            for j, Bk in zip(L.nbr[i], blocks[i]):
                if int(j) in loc:
                    D[orow[i_]:orow[i_ + 1], ocol[loc[int(j)]]:ocol[loc[int(j)] + 1]] = self.xp.asarray(Bk)
        return D

    @staticmethod
    def cores_of(L, w):
        return [c for b in w for c in L.cores[int(b)]]

    def projright(self, L, w, Y):
        """Y (m x level-1 columns of the cores of the block list w, in order) -> Y @ (prolongations down to level 1),
        i.e. the columns in level-L coordinates of w. Block by block: no dense prolongation, peak = Y + its image."""
        if L.k == 1:
            return Y
        w = [int(b) for b in w]
        Y = self.projright(L.parent, [kb for b in w for kb in L.kids[b]], Y)
        o1 = offs([int(L.parent.r[list(L.kids[b])].sum()) for b in w])
        o2 = offs(L.r[w])
        Z = self.xp.empty((Y.shape[0], o2[-1]), complex)
        for q, b in enumerate(w):
            Z[:, o2[q]:o2[q + 1]] = Y[:, o1[q]:o1[q + 1]] @ L.U[b]
        return Z

    def projop(self, L, w, M1):
        """M1 (dense, level-1 coordinates of the cores of the block list w, in order) -> level-L coordinates of w."""
        if L.k == 1:
            return M1
        T = self.projright(L, w, M1)
        del M1
        return self.projright(L, w, T.conj().T).conj().T

    def cap_of(self, L, w):
        """Projected CAP i s1 U^H (P S P) U of the window w (block list of level L) in level-L coordinates (None: the
        window has no artificial cut)."""
        xp, lab, posin = self.xp, self.lab, self.posin
        cw = self.cores_of(L, w)
        U1 = self.LV[1].U
        o = offs([U1[k].shape[1] for k in cw])
        F = np.concatenate([self.cores[k] for k in cw])
        rp = self.ramp(F)
        on_ = np.where(rp > 0)[0]
        if not len(on_):
            return None
        Fo = F[on_]
        Pr = sp.diags(np.sqrt(rp[on_]))
        Wf = (1j * self.s1 * (Pr @ self.S[Fo][:, Fo] @ Pr)).tocsr()
        Uo = xp.zeros((len(on_), o[-1]), complex)
        lf = lab[Fo]
        for q, k in enumerate(cw):
            sel = np.where(lf == k)[0]
            if len(sel):
                Uo[xp.asarray(sel)[:, None], xp.arange(o[q], o[q + 1])[None, :]] = U1[k][xp.asarray(posin[Fo[sel]])]
        Wd = gpu.csr(Wf, np.complex128) if gpu.on else Wf
        # factored form: the CAP lives on the len(on_) ramp orbitals only, so project the (ramp orbitals x columns)
        # factor through the levels and form the window matrix once, in level-L coordinates
        Y = self.projright(L, w, Uo)
        del Uo
        return Y.conj().T @ (Wd @ Y)

    def sig_factor(self, L, w):
        """Factor of the lead Sigma of the window w (with a lead core): (lead-face cores of w, offsets of their level-1
        columns, Y = their rows of the prolongation of w, lead level-1 columns x level-L coordinates). Sigma couples
        only the lead-face cores (a few % of the level-1 columns), so no window-sized level-1 matrix is formed and
        Y serves every contour point."""
        xp = self.xp
        cw = self.cores_of(L, w)
        U1 = self.LV[1].U
        o = offs([U1[k].shape[1] for k in cw])
        qs = [q for q, k in enumerate(cw) if k in self.lead_cores]
        idx = np.concatenate([np.arange(o[q], o[q + 1]) for q in qs])
        Ef = xp.zeros((len(idx), o[-1]), complex)
        Ef[xp.arange(len(idx)), xp.asarray(idx)] = 1
        Y = self.projright(L, w, Ef)
        return [cw[q] for q in qs], offs([o[q + 1] - o[q] for q in qs]), Y

    def sig_of(self, fac, sg):
        """Sigma(z) of a window in its level coordinates from the level-1 blocks sg and the window's sig_factor."""
        ks, o, Y = fac
        M1 = self.xp.zeros((o[-1], o[-1]), complex)
        for a_, i in enumerate(ks):
            for b_, j in enumerate(ks):
                if (i, j) in sg:
                    M1[o[a_]:o[a_ + 1], o[b_]:o[b_ + 1]] = sg[(i, j)]
        return Y.conj().T @ (M1 @ Y)

    # ------------------------------------------------------------------ basis store
    def basis_get(self, key, k):
        """(U blocks as host arrays, trace per block) of the level-k bases stored under key, or None."""
        if key is None:
            return None
        if key in self.ubank:
            return self.ubank[key]
        if self.basis_dir:
            f = os.path.join(self.basis_dir, f"l{k}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz")
            if os.path.exists(f):
                c = np.load(f)
                assert str(c["key"]) == key, f"{f}: key mismatch"
                self.ubank[key] = ([c[f"u{i}"] for i in range(int(c["nb"]))], c["tr"])
                return self.ubank[key]
        return None

    def basis_put(self, key, k, U, tr):
        if key is None:
            return
        Uh = [gpu.asnumpy(u) for u in U]
        self.ubank[key] = (Uh, np.asarray(tr))
        if self.basis_dir:
            os.makedirs(self.basis_dir, exist_ok=True)
            f = os.path.join(self.basis_dir, f"l{k}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz")
            tmp = f + f".tmp{os.getpid()}.npz"
            np.savez(tmp, key=key, nb=len(Uh), tr=np.asarray(tr), **{f"u{i}": u for i, u in enumerate(Uh)})
            os.replace(tmp, f)

    # ------------------------------------------------------------------ level 1
    def window_dm(self, cache, tag, mu1, mu2, npts, keep=1e-4, batch_gb=16.0, core=None):
        """Per window k of the fine smoother: occupations (descending, > keep) and the core rows of the eigenvectors of
        the local contour density matrix over [mu1, mu2] (absolute energies) of A_kk(z) + i s P S P (the S0 window
        operator, - Sigma(z) on the lead faces). One file per window in cache/tag (cache None: nothing is stored),
        built in batches of batch_gb. Returns [(occ, vec)]. A file records its keep (files without one: 1e-4); a file
        built with a larger keep than asked for is rebuilt, a smaller one is filtered to occ > keep on load.
        core = dict(prec, refine, probes): only the core block rho_cc, the rest of the window folded into the core
        (SchurRho: no window inverse, no window eigh); stores the eigenpairs of rho_cc (dmc only), in <tag>_core_...."""
        xp, dev, sw, nw = self.xp, self.dev, self.sw, self.nw
        if core:
            tag = (f"{tag}_core_{core['prec'][0]}{core['refine'] if core['prec'] == 'single' else ''}"
                   + (f"_p{core['probes']}" if core.get("probes") else ""))
        d = os.path.join(cache, tag) if cache else None
        if d:
            os.makedirs(d, exist_ok=True)
        f = lambda k: os.path.join(d, f"w{k}.npz")
        DM = [None] * nw
        def kept(k):                                         # keep of the stored file (None: no file)
            if not (d and os.path.exists(f(k))):
                return None
            with np.load(f(k)) as c:
                return float(c["keep"]) if "keep" in c.files else 1e-4
        todo = [k for k in range(nw) if kept(k) is None or kept(k) > keep * (1 + 1e-12)]
        if todo:
            t = time.time()
            zs, dzs = contour_points(mu1, mu2, npts)
            H = dev.H.tocsr()
            caps = {}
            for k in todo:
                idx = sw.idx[k]
                P = sp.diags(np.sqrt(sw.ramp(k)))
                Skk = self.S[idx][:, idx]
                caps[k] = (Skk, H[idx][:, idx], (1j * sw.s * (P @ Skk @ P)).tocsr())
            batches, cur, mem = [], [], 0.0
            for k in todo:
                m = len(sw.idx[k]) ** 2 * 16 / 2 ** 30
                if cur and mem + m > batch_gb:
                    batches.append(cur)
                    cur, mem = [], 0.0
                cur.append(k)
                mem += m
            batches.append(cur)
            self.log(f"  window density matrices: {len(todo)} windows to build in {len(batches)} batches, {npts} points"
                     + (f", core block by Schur folding ({core})" if core else ""))
            for bi, ks in enumerate(batches):
                acc = {k: (SchurRho(np.where(sw.core[k])[0], len(sw.idx[k]), core["prec"], core["refine"],
                                    core.get("probes", 0), seed=k) if core else
                           xp.zeros((len(sw.idx[k]),) * 2, complex)) for k in ks}
                self.mem(f"window rho batch {bi + 1}/{len(batches)}: accumulators of {len(ks)} windows")
                lead = [k for k in ks if np.isin(sw.idx[k], dev.bL).any() or np.isin(sw.idx[k], dev.bR).any()]
                for z, dz in zip(zs, dzs):
                    SL, SR = dev.sigma(z) if lead else (None, None)
                    for k in ks:
                        Skk, Hkk, Wkk = caps[k]
                        M = (z * Skk - Hkk + Wkk).tocsr()                # A_kk(z) + i s P S P
                        if k in lead:                                   # - Sigma on the window's lead-face orbitals
                            idx = sw.idx[k]
                            for Sg, bb in ((SL, dev.bL), (SR, dev.bR)):
                                ii = np.where(np.isin(idx, bb))[0]
                                if len(ii):
                                    jj = np.searchsorted(bb, idx[ii])
                                    M = M - sp.csr_matrix((Sg[np.ix_(jj, jj)].ravel(), (np.repeat(ii, len(ii)),
                                                           np.tile(ii, len(ii)))), shape=M.shape)
                        Mk = gpu.csr_dense(M, np.complex128) if gpu.on else M.toarray()
                        if core:
                            acc[k].add(z, dz, Mk)
                        else:
                            acc[k] += xp.linalg.inv(Mk) * dz
                        del Mk
                    gpu.free()
                self.mem(f"window rho batch {bi + 1}/{len(batches)}: contour ({npts} inversions per window)")
                for k in ks:
                    if core:                                    # eigenpairs of rho_cc (V: core rows already)
                        ev, V = acc[k].result()
                        acc[k] = None
                        kp = gpu.asnumpy(ev) > keep
                        DM[k] = (gpu.asnumpy(ev)[kp], gpu.asnumpy(V[:, xp.asarray(kp)]))
                    else:
                        rho = rho_of(acc[k])
                        acc[k] = None
                        ev, V = xp.linalg.eigh(rho)
                        del rho
                        o = xp.argsort(-ev)
                        ev, V = ev[o], V[:, o]
                        kp = gpu.asnumpy(ev) > keep
                        DM[k] = (gpu.asnumpy(ev)[kp],
                                 gpu.asnumpy(V[xp.asarray(np.where(sw.core[k])[0])][:, xp.asarray(kp)]))
                    if d:
                        tmp = f(k) + f".tmp{os.getpid()}.npz"
                        np.savez(tmp, occ=DM[k][0], vec=DM[k][1], keep=keep)
                        os.replace(tmp, f(k))
                    del V, ev
                    gpu.free()
                self.mem(f"window rho batch {bi + 1}/{len(batches)}: eigh")
                self.log(f"    batch {bi + 1}/{len(batches)}: {len(ks)} windows ({time.time() - t:.0f} s)")
        for k in range(nw):
            if DM[k] is None:
                c = np.load(f(k))
                kp = c["occ"] > keep
                DM[k] = (c["occ"][kp], c["vec"][:, kp])
            assert DM[k][1].shape[0] == len(self.cores[k]), \
                f"window density matrix {k} in {d} belongs to other windows (core {DM[k][1].shape[0]} != {len(self.cores[k])})"
        self.log(f"  window density matrices" + (f" from {d}" if d else "") + ": trace "
                 + "/".join(f"{o.sum():.0f}" for o, _ in DM[:8]) + ("..." if nw > 8 else ""))
        return DM

    def basis1(self, DM, kind, cut):
        """Level-1 block bases from the window density matrices: dm = window modes above cut restricted to the core
        (SVD), dmc = eigenvectors of the core block rho_cc = Vc occ Vc^H above cut."""
        assert kind in ("dm", "dmc")
        xp = self.xp
        U = []
        for occ, Vc in DM:
            if kind == "dm":
                u, sv, _ = xp.linalg.svd(xp.asarray(Vc[:, occ > cut]), full_matrices=False)
                U.append(u[:, :int((sv > 1e-8 * sv[0]).sum())] if len(sv) else u[:, :0])
            else:
                Vg = xp.asarray(Vc)
                ev, V = xp.linalg.eigh((Vg * xp.asarray(occ)[None, :]) @ Vg.conj().T)
                U.append(xp.ascontiguousarray(V[:, ev > cut][:, ::-1]))
                del Vg, V
        return U

    def level1(self, U, E, SL, SR, A0=None):
        """Level 1 at the energy E from the block bases U (drops every level of the previous energy). With l1_free and
        the fine operator A0: no stored A_1 (GalerkinOp)."""
        t = time.time()
        xp, dev, nw, nbr = self.xp, self.dev, self.nw, self.nbr
        self.free()
        L = Lev()
        L.k, L.parent, L.kids, L.U = 1, None, None, U
        L.r = np.array([u.shape[1] for u in U])
        L.off = offs(L.r)
        L.N = int(L.off[-1])
        L.cores = [[k] for k in range(nw)]
        L.G, L.nbr = self.G, nbr
        L.grid = (np.arange(nw) // self.ny, np.arange(nw) % self.ny)
        self.mem("level 1: block bases U")
        L.Sb, L.Hb = self.core_blocks(dev.S, U), self.core_blocks(dev.H, U)
        self.mem("level 1: S, H blocks")
        L.E = E
        free = self.free_levels >= 1 and A0 is not None
        host = not free and self.streams(sum(int(L.r[i]) * int(L.r[j]) for i in range(nw) for j in nbr[i]) * 16 / 2 ** 30)
        if free:
            L.Ab, sig1 = None, self.sigma1(U, SL, SR)
        else:
            if host:                                                # A1 assembled block by block on the host
                L.Ab = [[E * gpu.asnumpy(s__) - gpu.asnumpy(h) for s__, h in zip(L.Sb[i], L.Hb[i])] for i in range(nw)]
            else:
                L.Ab = [[xp.asarray(E * s__ - h) for s__, h in zip(L.Sb[i], L.Hb[i])] for i in range(nw)]
            for (i, j), Bk in self.sigma1(U, SL, SR).items():
                L.Ab[i][self.pos[i][j]] = L.Ab[i][self.pos[i][j]] - (gpu.asnumpy(Bk) if host else Bk)
            L.A = BlockOp(L.off, list(range(nw)), nbr, L.Ab, host=host)
            self.mem("level 1: A blocks")
        cores_g = self.cores_g
        n = self.n

        def P(c):
            y = xp.zeros((n, c.shape[1]), complex)
            for k in range(nw):
                y[cores_g[k]] = U[k] @ c[L.off[k]:L.off[k + 1]]
            return y
        L.P = P
        L.PH = lambda r: xp.concatenate([U[k].conj().T @ xp.asarray(r)[cores_g[k]] for k in range(nw)])
        if free:
            L.A = GalerkinOp(self, L, A0, E, sig1)
            self.mem("level 1: A1 through the fine operator (no blocks)")
        self.LV = {1: L}
        gb = sum(int(L.r[i]) * int(L.r[j]) for i in range(nw) for j in nbr[i]) * 16 / 2 ** 30
        L.info = dict(level=1, blocks=nw, N=L.N, per_block=L.r.tolist(), A_gb=gb)
        self.log(f"  level 1: {nw} blocks, N1 {L.N} (n / N1 {n / max(L.N, 1):.1f}), per block {L.r.min()}.."
                 f"{int(np.median(L.r))}..{L.r.max()}, A1 {gb:.2f} GB; build {time.time() - t:.0f} s; {self.memstr()}")
        return L

    # ------------------------------------------------------------------ further levels
    def couplings(self, L):
        """Log the coupling blocks of A_k(E), S_k of level L by block distance (x distance in blocks, same / other y).
        Returns {name: (blocks, |A| median, |A| max, |S| median, |S| max)}."""
        ix, iy = L.grid
        grp = {}
        for i in range(len(L.r)):
            for q, j in enumerate(L.nbr[i]):
                key = (-1, False) if j == i else (int(abs(ix[i] - ix[j])), bool(iy[i] != iy[j]))
                grp.setdefault(key, []).append((nrm2(self.xp.asarray(L.Ab[i][q]) if L.Ab is not None else
                                                   L.E * self.xp.asarray(L.Sb[i][q]) - self.xp.asarray(L.Hb[i][q])),
                                              nrm2(self.xp.asarray(L.Sb[i][q]))))
        off = [np.median([v[0] for v in g]) for k, g in grp.items() if k[0] >= 0]
        ref = max(off) if off and max(off) > 0 else 1.0
        self.log(f"  level {L.k} coupling blocks by distance (2-norm, median / max over the blocks; A{L.k} relative to "
                 f"the strongest group):")
        out = {}
        for key in sorted(grp):
            g = np.array(grp[key])
            name = "diagonal blocks" if key[0] < 0 else f"dx {key[0]}, {'other' if key[1] else 'same'} y"
            out[name] = (len(g), float(np.median(g[:, 0])), float(g[:, 0].max()), float(np.median(g[:, 1])),
                         float(g[:, 1].max()))
            self.log(f"    {name:18s} {len(g):4d} blocks: |A| {np.median(g[:, 0]):.3f} / {g[:, 0].max():.3f} eV"
                     + (f" (rel {np.median(g[:, 0]) / ref:.2f})" if key[0] >= 0 else "")
                     + f", |S| {np.median(g[:, 1]):.3f} / {g[:, 1].max():.3f}")
        return out

    def ras(self, L, wins, writes):
        """RAS on L.A: window wins[c] (sorted blocks of L), local operator A[w, w] + projected CAP, the solution is
        kept on the blocks writes[c] (a subset of the window). Returns the smoother (level-L vectors) with .sizes, .gb.
        l1_solver rows: only the rows of the window inverse on writes[c] are kept (n_write x n_window instead of
        n_window^2 for the LU; the same correction, since RAS discards the other rows)."""
        xp, cdt = self.xp, self.cdt
        ops, sizes, gb = [], [], 0.0
        nb_ = lambda ks: int(sum(L.off[k + 1] - L.off[k] for k in ks))
        est = sum(nb_(wr) * nb_(w) for w, wr in zip(wins, writes)) * np.dtype(cdt).itemsize / 2 ** 30
        stream = self.l1_solver == "rows" and self.streams(est)    # core rows to pinned host memory window by window
        for c, w in enumerate(wins):
            Aw, o = L.A.sub(np.asarray(w))
            Cw = self.cap_of(L, w)
            if Cw is not None:
                Aw += Cw
                del Cw
            loc = {int(k): q for q, k in enumerate(w)}
            pos_ = xp.asarray(np.concatenate([np.arange(o[loc[int(k)]], o[loc[int(k)] + 1]) for k in writes[c]]))
            rw = xp.asarray(np.concatenate([np.arange(L.off[k], L.off[k + 1]) for k in w]))
            out = xp.asarray(np.concatenate([np.arange(L.off[k], L.off[k + 1]) for k in writes[c]]))
            Aw = Aw.astype(cdt)
            if self.l1_solver == "rows":                        # (A_w^-1)[write rows, :] = (A_w^-H e_rows)^H
                lu = gpu.lu_factor(Aw)
                del Aw
                Ek = xp.zeros((int(o[-1]), len(pos_)), cdt)
                Ek[pos_, xp.arange(len(pos_))] = 1
                Ri = xp.ascontiguousarray(gpu.lu_solve(lu, Ek, trans=2).conj().T)
                del lu, Ek
                gb += Ri.nbytes / 2 ** 30
                ops.append((rw, None, out, gpu.pinned(Ri) if stream else Ri))
                del Ri
            else:
                ops.append((rw, pos_, out, gpu.lu_factor(Aw)))
                gb += int(o[-1]) ** 2 * self.lub / 2 ** 30
                del Aw
            sizes.append(int(o[-1]))
            gpu.free()

        st = gpu.Streamer([(op,) for _, _, _, op in ops]) if stream else None

        def Sm(r):
            y = xp.zeros_like(r)
            if st is not None:
                def fn(c, v):
                    rw, _, out, _ = ops[c]
                    y[out] = v[0] @ r[rw].astype(cdt)
                st.run(fn)
                return y
            for rw, pos_, out, op in ops:
                y[out] = op @ r[rw].astype(cdt) if pos_ is None else gpu.lu_solve(op, r[rw].astype(cdt))[pos_]
            return y
        Sm.sizes, Sm.gb, Sm.streamed = sizes, gb, stream
        return Sm

    def _sketch_batch(self, ks, dat, isl, sf, zs, dzs, U1, rows_of, cut, kind, Nn, tr, bt, f=0.5, fill=0.75,
                      host_frac=0.6):
        """Bases of the windows ks from a probe block, without the window rho (as msbasis tile sketch): per contour
        point one LU of the window pencil M(z) and two solves, Y = rho Omega = -(1/pi) (acc - acc^H) Omega / 2i with
        acc Omega = sum dz M^-1 Omega, acc^H Omega = sum conj(dz) M^-H Omega; Nystrom rho ~ F F^H, F = Y (Omega^H Y)^-1/2
        (pseudo-inverse square root of the Hermitian core); X, lam = left singular vectors, squares of F. Probes:
        1.5 x the pair rows + 64 at first, more probe columns (another contour pass for them only) until the number of
        Nystrom occupations above f * cut is at most fill x the probes, or the probes span the window (exact).
        Probes above host_frac x n: the Nystrom eigh and SVD on the host (the GPU workspace of a 13k eigh is ~29 GB).
        self.sketch_prec "c64": pencil, LU, probes and solves in complex64; "lu64": pencil and LU in complex128, the LU
        factors down-cast to complex64, probes and solves in complex64 (accumulated and Nystrom always in complex128)."""
        xp = self.xp
        dt = complex if self.sketch_prec == "c128" else np.complex64
        n = {c: dat[c][0].shape[0] for c in ks}
        s_ = {c: min(n[c], int(1.5 * len(rows_of[c])) + 64) for c in ks}
        Om = {c: xp.zeros((n[c], 0), complex) for c in ks}
        Y = {c: xp.zeros((n[c], 0), complex) for c in ks}
        active, rnd, nhost = list(ks), 0, [0]
        while active:
            new = {}
            for c in active:
                m = s_[c] - Om[c].shape[1]
                rs = xp.random.RandomState(1000 * c + rnd)
                O = ((rs.standard_normal((n[c], m)) + 1j * rs.standard_normal((n[c], m))) / np.sqrt(2)).astype(dt)
                new[c] = [O, xp.zeros((n[c], m), complex), xp.zeros((n[c], m), complex)]
            lead = any(isl[c] for c in active)
            for z, dz in zip(zs, dzs):
                sg = self.sigma1(U1, *self.dev.sigma(z)) if lead else {}
                for c in active:
                    M = z * dat[c][0] + dat[c][1]
                    if isl[c]:
                        M -= self.sig_of(sf[c], sg)
                    lu = gpu.lu_factor(M if self.sketch_prec == "lu64" else M.astype(dt, copy=False))
                    del M
                    if self.sketch_prec == "lu64":
                        lu = (lu[0].astype(dt), lu[1])
                    new[c][1] += dz * gpu.lu_solve(lu, new[c][0])
                    new[c][2] += np.conj(dz) * gpu.lu_solve(lu, new[c][0], trans=2)
                    del lu
                del sg
                gpu.free()
            self.mem(f"{bt}: contour round {rnd + 1} ({len(zs)} LUs, 2 solves per window)")
            done = []
            for c in active:
                O, a, ah = new.pop(c)
                Om[c] = xp.concatenate([Om[c], O.astype(complex)], axis=1)
                Y[c] = xp.concatenate([Y[c], -((a - ah) / 2j) / np.pi], axis=1)
                del O, a, ah
                C = Om[c].conj().T @ Y[c]
                C = 0.5 * (C + C.conj().T)
                if gpu.on and s_[c] > host_frac * n[c]:          # the dense eigh / SVD workspace (~11 s^2) on the host
                    d, V = np.linalg.eigh(gpu.asnumpy(C))
                    ok = d > 1e-10 * d.max()
                    F = gpu.asnumpy(Y[c]) @ (V[:, ok] / np.sqrt(d[ok]))
                    X, sv, _ = np.linalg.svd(F, full_matrices=False)
                    F, X, sv = xp.asarray(F), xp.asarray(X), xp.asarray(sv)
                    nhost[0] += 1
                else:
                    d, V = xp.linalg.eigh(C)
                    ok = d > 1e-10 * d.max()
                    F = Y[c] @ (V[:, ok] / xp.sqrt(d[ok]))
                    X, sv, _ = xp.linalg.svd(F, full_matrices=False)
                lam = sv ** 2
                nb = int((lam > f * cut).sum())
                if nb <= fill * s_[c] or s_[c] >= n[c]:
                    rows = rows_of[c]
                    k = int((lam > cut).sum())
                    if kind == "dm":
                        u, sr, _ = xp.linalg.svd(X[rows][:, :k], full_matrices=False)
                        Nn.U[c] = xp.ascontiguousarray(u[:, :int((sr > 1e-8 * sr[0]).sum())] if k else u[:, :0])
                    else:                                            # dmc: rho_cc ~ F_c F_c^H
                        u, sr, _ = xp.linalg.svd(F[rows], full_matrices=False)
                        Nn.U[c] = xp.ascontiguousarray(u[:, sr ** 2 > cut])
                    tr[c] = float(xp.linalg.norm(F[rows]) ** 2)
                    EIG_STATS.append((n[c], s_[c], k, rnd + 1, s_[c] >= n[c]))
                    done.append(c)
                    Om[c] = Y[c] = dat[c] = None
                else:
                    s_[c] = min(n[c], max(2 * s_[c], int(nb / fill) + 64))
                del C, d, V, F, X, sv, lam
                gpu.free()
            active = [c for c in active if c not in done]
            rnd += 1
        self.mem(f"{bt}: block bases from the sketch" + (f" ({nhost[0]} Nystrom steps on the host)" if nhost[0] else ""))

    def merge(self, axis, cut, kind, mu1, mu2, npts, halo=None, bkey=None):
        """Append level k + 1 = pairs of blocks of the deepest level L along axis ("x" | "y"); also the smoother of
        L.A with these pairs as blocks. mu1, mu2: absolute energy window of the contour rho; halo: x halo of the
        windows in blocks of L (None: every coupled block); bkey: the bases of this level are taken from / stored in
        the basis store under bkey (energy independent: no contour rho when stored). Returns the new level."""
        assert axis in ("x", "y") and kind in ("dm", "dmc")
        xp, log = self.xp, self.log
        L = self.LV[len(self.LV)]
        t0 = time.time()
        ix, iy = L.grid
        key = (ix // 2) * 100000 + iy if axis == "x" else ix * 100000 + iy // 2
        Nn = Lev()
        Nn.k, Nn.parent = L.k + 1, L
        Nn.kids = [tuple(int(v) for v in np.where(key == v_)[0]) for v_ in np.unique(key)]
        nc = len(Nn.kids)
        k0 = np.array([ks[0] for ks in Nn.kids])
        Nn.grid = (ix[k0] // 2, iy[k0]) if axis == "x" else (ix[k0], iy[k0] // 2)
        Nn.cores = [self.cores_of(L, ks) for ks in Nn.kids]
        mem = np.zeros((nc, len(L.r)), np.int32)
        for c, ks in enumerate(Nn.kids):
            mem[c, list(ks)] = 1
        Nn.G = (mem @ L.G.astype(np.int32) @ mem.T) > 0
        Nn.nbr = [np.where(Nn.G[c])[0] for c in range(nc)]
        full = [np.unique(np.concatenate([L.nbr[kb] for kb in ks])) for ks in Nn.kids]    # C + coupled blocks of L
        xd = [np.abs(ix[w][:, None] - ix[list(ks)][None, :]).min(1) for w, ks in zip(full, Nn.kids)]   # x distance to C
        hmax = int(max(d.max() for d in xd))
        cpl = self.couplings(L)
        sizes = {h: np.array([int(L.r[w[d <= h]].sum()) for w, d in zip(full, xd)]) for h in range(hmax + 1)}
        cost = {h: float((v.astype(float) ** 3).sum()) for h, v in sizes.items()}
        hu = hmax if halo is None else min(int(halo), hmax)
        hs = hmax if self.halo_smooth == "full" else hu
        log(f"  level {Nn.k} windows by x halo (level-{L.k} blocks from the pair; rho cost = sum n^3 relative to all): "
            + "; ".join(f"{h}{' = all' if h == hmax else ''}: {int(np.median([(d <= h).sum() for d in xd]))} blocks, "
                        f"{v.min()}..{int(np.median(v))}..{v.max()} columns, cost {cost[h] / max(cost[hmax], 1):.3f}"
                        for h, v in sizes.items()) + f"; used: rho {hu}, smoother {hs}")
        win = [w[d <= hu] for w, d in zip(full, xd)]                                      # rho windows
        wsm = [w[d <= hs] for w, d in zip(full, xd)]                                      # windows of the smoother of L.A
        nwin = np.array([int(L.r[w].sum()) for w in win])
        lv = f"level {Nn.k}"
        stored = self.basis_get(bkey, Nn.k)
        if stored is not None:
            Nn.U, tr, batches = [xp.asarray(u) for u in stored[0]], np.asarray(stored[1]), []
            for c in range(nc):
                assert Nn.U[c].shape[0] == int(L.r[list(Nn.kids[c])].sum()), "stored bases belong to another hierarchy"
            self.mem(f"level {Nn.k}: bases from the store")
        else:
            zs, dzs = contour_points(mu1, mu2, npts)
            isl = [bool(set(self.cores_of(L, w)) & self.lead_cores) for w in win]
            order = sorted(range(nc), key=lambda c: (not isl[c], c))
            batches, cur, m_ = [], [], 0.0
            # per window of a batch: S_w, K_w, accumulator stay (3 n^2); the pencil and its inverse are transient (the
            # largest window once: + 3 n^2 with the LU workspace)
            lim = min(self.batch_gb, self.mem_frac * self.gpu_free_gb()) - 3 * nwin.max() ** 2 * 16 / 2 ** 30
            # sketch: S_w, K_w + probes, rho Omega and the two solve accumulators of a round (up to 4 n^2 when the
            # probes span the window): 6 n^2 per window
            per = 6 if self.merge_eig == "sketch" else 3
            for c in order:
                m = per * nwin[c] ** 2 * 16 / 2 ** 30
                if cur and m_ + m > lim:
                    batches.append(cur)
                    cur, m_ = [], 0.0
                cur.append(c)
                m_ += m
            batches.append(cur)
            Nn.U, tr = [None] * nc, np.zeros(nc)
            schur = kind == "dmc" and self.core is not None and self.merge_eig == "full"
            EIG_STATS.clear()
            U1 = self.LV[1].U
            self.mem(f"{lv}: start of the {axis} merge")
            for bi, ks in enumerate(batches):
                bt = f"{lv} rho batch {bi + 1}/{len(batches)}"
                dat = {}
                for c in ks:
                    w = win[c]
                    Kw = -self.dense(L, L.Hb, w, w)
                    Cw = self.cap_of(L, w)                           # not kept: the smoother below rebuilds it (cheap)
                    if Cw is not None:
                        Kw += Cw
                    del Cw
                    dat[c] = [self.dense(L, L.Sb, w, w), Kw,
                              None if self.merge_eig == "sketch" or schur else xp.zeros((nwin[c],) * 2, complex)]
                self.mem(f"{bt}: S, K, accumulator of {len(ks)} windows")
                lead = any(isl[c] for c in ks)
                sf = {c: self.sig_factor(L, win[c]) for c in ks if isl[c]}
                if sf:
                    self.mem(f"{bt}: lead Sigma factors of {len(sf)} windows")
                rows_of = {}
                for c in ks:
                    w = win[c]
                    o = offs(L.r[w])
                    loc = {int(k): q for q, k in enumerate(w)}
                    rows_of[c] = xp.asarray(np.concatenate([np.arange(o[loc[k]], o[loc[k] + 1]) for k in Nn.kids[c]]))
                if schur:
                    for c in ks:
                        dat[c][2] = SchurRho(rows_of[c], nwin[c], self.core["prec"], self.core["refine"],
                                             self.core.get("probes", 0), seed=1000 + c)
                if self.merge_eig == "sketch":
                    self._sketch_batch(ks, dat, isl, sf, zs, dzs, U1, rows_of, cut, kind, Nn, tr, bt)
                    del sf, dat
                    continue
                for z, dz in zip(zs, dzs):
                    sg = self.sigma1(U1, *self.dev.sigma(z)) if lead else {}
                    for c in ks:
                        M = z * dat[c][0] + dat[c][1]
                        if isl[c]:
                            M -= self.sig_of(sf[c], sg)
                        if schur:
                            dat[c][2].add(z, dz, M)
                        else:
                            M = xp.linalg.inv(M)
                            M *= dz
                            dat[c][2] += M
                        del M
                    del sg
                    gpu.free()
                del sf
                self.mem(f"{bt}: contour ({npts} inversions per window)")
                for c in ks:
                    rows = rows_of[c]
                    if schur:                                    # eigenpairs of rho_PP directly (dmc)
                        occ, V = dat[c][2].result()
                        dat[c] = None
                        Nn.U[c] = xp.ascontiguousarray(V[:, occ > cut])
                        tr[c] = float(xp.real(occ.sum()))
                        del occ, V
                        gpu.free()
                        continue
                    rho = rho_of(dat[c][2])
                    dat[c] = None
                    Nn.U[c] = select(rho, rows, cut, kind, self.merge_eig)
                    tr[c] = float(xp.real(xp.trace(rho[rows][:, rows])))
                    del rho
                    gpu.free()
                self.mem(f"{bt}: block bases from rho")
            self.basis_put(bkey, Nn.k, Nn.U, tr)
        t_rho = time.time() - t0
        eigs = ""
        if self.merge_eig != "full" and EIG_STATS:
            st = np.array(EIG_STATS, float)
            eigs = (f"; {self.merge_eig}{' ' + self.sketch_prec if self.merge_eig == 'sketch' and self.sketch_prec != 'c128' else ''} eig: window n {int(st[:, 0].min())}..{int(np.median(st[:, 0]))}..{int(st[:, 0].max())}, "
                    f"subspace l {int(st[:, 1].min())}..{int(np.median(st[:, 1]))}..{int(st[:, 1].max())}, "
                    f"k > cut {int(st[:, 2].min())}..{int(np.median(st[:, 2]))}..{int(st[:, 2].max())}, rounds "
                    f"{int(st[:, 3].max())} max, " + ("probes = window" if self.merge_eig == "sketch" else "dense fallback")
                    + f" {int(st[:, 4].sum())} of {len(st)}")
            EIG_STATS.clear()
        Nn.r = np.array([u.shape[1] for u in Nn.U])
        Nn.off = offs(Nn.r)
        Nn.N = int(Nn.off[-1])
        gal = lambda X, f=lambda B: B: [[f(Nn.U[c].conj().T @ (self.dense(L, X, Nn.kids[c], Nn.kids[c2]) @ Nn.U[c2]))
                                         for c2 in Nn.nbr[c]] for c in range(nc)]
        Nn.Sb, Nn.Hb = gal(L.Sb, self.keep), gal(L.Hb, self.keep)
        Nn.E = L.E
        nfree = Nn.k <= self.free_levels and isinstance(L.A, GalerkinOp)
        if nfree:                                            # no A_k blocks: Sigma of the lead blocks only
            Nn.Ab, Nn.A = None, None
            sigN = {}
            sg = L.A.sig1
            lb = {c for c in range(nc) for (i, j) in sg if i in Nn.kids[c] or j in Nn.kids[c]}
            for c in lb:
                for c2 in lb:
                    if any(i in Nn.kids[c] and j in Nn.kids[c2] for (i, j) in sg):
                        ro, co = offs(L.r[list(Nn.kids[c])]), offs(L.r[list(Nn.kids[c2])])
                        M = xp.zeros((ro[-1], co[-1]), complex)
                        for (i, j), B in sg.items():
                            if i in Nn.kids[c] and j in Nn.kids[c2]:
                                a_, b_ = Nn.kids[c].index(i), Nn.kids[c2].index(j)
                                M[ro[a_]:ro[a_ + 1], co[b_]:co[b_ + 1]] = B
                        sigN[(c, c2)] = Nn.U[c].conj().T @ (M @ Nn.U[c2])
        else:
            Nn.Ab = [[Nn.U[c].conj().T @ (L.A.take(Nn.kids[c], Nn.kids[c2]) @ Nn.U[c2]) for c2 in Nn.nbr[c]]   # = gal(L.Ab)
                     for c in range(nc)]
            Nn.A = BlockOp(Nn.off, list(range(nc)), Nn.nbr, Nn.Ab,
                           host=self.streams(sum(int(Nn.r[i]) * int(Nn.r[j]) for i in range(nc) for j in Nn.nbr[i]) * 16 / 2 ** 30))
        self.mem(f"{lv}: Galerkin S, H" + (" (A free)" if nfree else ", A blocks"))
        rows1 = [xp.asarray(np.concatenate([np.arange(L.off[k], L.off[k + 1]) for k in ks])) for ks in Nn.kids]

        def P(c2):
            y = xp.zeros((L.N, c2.shape[1]), complex)
            for c in range(nc):
                y[rows1[c]] = Nn.U[c] @ c2[Nn.off[c]:Nn.off[c + 1]]
            return y
        Nn.P = P
        Nn.PH = lambda r1: xp.concatenate([Nn.U[c].conj().T @ r1[rows1[c]] for c in range(nc)])
        if nfree:
            Nn.A = GalerkinOp(self, Nn, L.A, Nn.E, sigN)
        t = time.time()
        # the level below is complete: its S, H blocks and the block list of A are not needed again (L.A keeps the
        # block rows for the matvec, dense() and sub()). Freed before the smoother of L.A is built: 3 x the size of
        # A_L less at the memory peak (21600a y-cut: 33 GB, out of memory on a GH200 otherwise)
        if not isinstance(L.A, GalerkinOp):                  # a free A_1 keeps reading S1, H1 (smoother, SlabLU)
            L.Sb = L.Hb = None
        L.Ab = L.A.blocks = None
        gpu.free()
        self.mem(f"{lv}: S, H blocks and the A block list of level {L.k} freed")
        Nn.Sbelow = self.ras(L, wsm, Nn.kids)                # smoother of L.A formed with the blocks of this level
        gpu.free()
        self.mem(f"{lv}: smoother of A{L.k} ({len(wsm)} windows)")
        np_ = np.array([int(L.r[list(ks)].sum()) for ks in Nn.kids])
        nsm = np.array(Nn.Sbelow.sizes)
        gb = sum(int(Nn.r[i]) * int(Nn.r[j]) for i in range(nc) for j in Nn.nbr[i]) * 16 / 2 ** 30
        allh = lambda h: f"{h}{' = all' if h == hmax else ''}"
        log(f"  level {Nn.k} ({axis} merge, {kind} cut {cut:g}, E {mu1:+.4f}..{mu2:+.4f}): {nc} blocks, N{Nn.k} {Nn.N} "
            f"(N{L.k} / N{Nn.k} {L.N / max(Nn.N, 1):.2f}, n / N{Nn.k} {self.n / max(Nn.N, 1):.1f}), per block "
            f"{Nn.r.min()}..{int(np.median(Nn.r))}..{Nn.r.max()} of {np_.min()}..{int(np.median(np_))}..{np_.max()} "
            f"level-{L.k} columns; rho windows (halo {allh(hu)}): {nwin.min()}..{int(np.median(nwin))}..{nwin.max()} "
            f"columns; smoother of A{L.k} (halo {allh(hs)}): {nsm.min()}..{int(np.median(nsm))}..{nsm.max()} columns, "
            f"{self.l1_solver} {Nn.Sbelow.gb:.2f} GB (LUs {(nsm.astype(float) ** 2).sum() * self.lub / 2 ** 30:.2f} GB); "
            f"A{Nn.k} {gb:.2f} GB; trace per block {tr.min():.1f}..{np.median(tr):.1f}..{tr.max():.1f}; rho "
            + (f"{t_rho:.0f} s in {len(batches)} batches" if batches else f"from the store ({t_rho:.0f} s)") + eigs + f", smoother {time.time() - t:.0f} s; {self.memstr()}")
        Nn.info = dict(level=Nn.k, axis=axis, kind=kind, cut=cut, window=[mu1, mu2], npts=int(npts), blocks=nc,
                       N=Nn.N, per_block=Nn.r.tolist(), halo_max=hmax, halo_rho=hu, halo_smoother=hs,
                       rho_windows=nwin.tolist(), smoother_windows=nsm.tolist(), smoother_gb=Nn.Sbelow.gb, A_gb=gb,
                       halo_cost={str(h): cost[h] / max(cost[hmax], 1) for h in cost}, couplings_below=cpl,
                       trace=tr.tolist(), t_rho=t_rho, rho_batches=len(batches), stored=not batches)
        self.LV[Nn.k] = Nn
        return Nn

    # ------------------------------------------------------------------ cycles
    def exact(self, m):
        """Bottom solver: exact solve with A_m by the block-tridiagonal LU over x slabs (SlabLU), no dense LU."""
        return SlabLU(self.LV[m].A, self.LV[m].grid[0])

    def own_smoother(self, m=None):
        """RAS on A_m over its own blocks (block + coupled blocks, projected CAP): the smoother of the deepest level."""
        L = self.LV[m or len(self.LV)]
        return self.ras(L, [L.nbr[b_] for b_ in range(len(L.r))], [[b_] for b_ in range(len(L.r))])

    def coarse(self, m, bottom, gamma=1):
        """Coarse correction of the fine level with the levels 1 .. m (fine residual -> fine correction): cluster
        smoothers on A_1 .. A_{m-1}, `bottom` solves A_m."""
        LV = self.LV
        c = bottom
        for k in range(m, 1, -1):
            Lk, Lp = LV[k], LV[k - 1]
            c = gcycle(Lp.A, Lk.Sbelow, lambda r, c=c, Lk=Lk: Lk.P(c(Lk.PH(r))), gamma)
        L1 = LV[1]
        return lambda r, c=c: L1.P(c(L1.PH(r)))

    def cycle(self, m, bottom, A0, S0, gamma=1):
        """Fine preconditioner with the levels 1 .. m: S0 pre / post smoothing around the coarse correction. Returns
        (prec, timer of the levels >= 1)."""
        tm = Timed(self.coarse(m, bottom, gamma), "levels >= 1")
        return gcycle(A0, S0, tm, gamma), tm

    def free(self):
        """Drop every level. Lev objects are reference cycles (L.P / L.PH close over L): clear them and collect, else
        the previous energy's levels stay on the GPU until the cyclic GC runs (21600a y-cut sweep: ~35 GB held into
        the next energy, 0.4 GB free at its level-2 Galerkin step)."""
        for L in self.LV.values():
            L.__dict__.clear()
        self.LV = {}
        gc.collect()
        gpu.free()


# ---------------------------------------------------------------------- driver
def parse_up(up, halo, mu):
    """--up AXIS:CUT[:mu1:mu2[:npts]],... and --halo H,... -> [(axis, cut, mu1, mu2, npts | None, halo | None)]."""
    ups = [] if up in ("", "none") else [v.split(":") for v in up.split(",")]
    halos = [None if v in ("", "all") else int(v) for v in halo.split(",")] if halo else []
    assert len(halos) <= len(ups), "--halo has more entries than --up"
    halos += [None] * (len(ups) - len(halos))
    out = []
    for f_, h in zip(ups, halos):
        m1, m2 = (float(f_[2]), float(f_[3])) if len(f_) > 3 else mu
        out.append((f_[0], float(f_[1]), m1, m2, int(f_[4]) if len(f_) > 4 else None, h))
    return out


def load_psi(psi_dir, E, A, b, bL):
    """Exact psi for the columns b from psi_dir: psi_<E>.npz (qtx_ms: psi_m) or psi_<E>_<eta>.npy (release). The
    saved columns may be mixed / rephased: fitted by the column transform Q with b = (A psi_saved) Q. None if absent
    or if it does not solve A psi = b."""
    if not psi_dir:
        return None
    for f, get in ((os.path.join(psi_dir, f"psi_{E:+.8f}.npz"), lambda f: np.load(f)["psi_m"]),
                   (os.path.join(psi_dir, f"psi_{E:+.8f}_{0.0:.2e}.npy"), np.load)):
        if not os.path.exists(f):
            continue
        Xs = get(f)
        if Xs.shape[0] != A.shape[0]:
            continue
        Q = np.linalg.lstsq(np.asarray(A @ Xs)[bL], b[bL], rcond=None)[0]
        if np.linalg.norm(b - A @ (Xs @ Q)) < 1e-8 * np.linalg.norm(b):
            return Xs @ Q
    return None


def main(argv=None):
    from qtx_ms.prep.inputs import read_inputs
    from qtx_ms.schwarz import CapSchwarz, SlowCoarse, fgmres_block, operator
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True, help="qtx_ms input set or a release prep folder (prep.json)")
    p.add_argument("--dE", default="0.15", help="energies - Ec_lead [eV] (comma list)")
    p.add_argument("--Ec", type=float, help="reference edge for --dE, --mu (default Ec_lead of the inputs)")
    p.add_argument("--eta", type=float, default=1e-6, help="Im z of the lead Sigma on the real axis [eV]")
    p.add_argument("--obc", default="folded", choices=["auto", "qttools", "folded"])
    p.add_argument("--sigma-cache", help="default results_ms/sigma/<inputs name>")
    p.add_argument("--psi-dir", help="exact psi files (psi_<E>.npz of qtx_ms or psi_<E>_<eta>.npy of the release)")
    p.add_argument("--exact", action="store_true", help="no psi file: direct solve for the error in psi and T")
    p.add_argument("--gpu", action="store_true", help="cupy backend (also QTX_MODESPACE_GPU=1)")
    p.add_argument("--cap", default="16,2,2", help="fine windows: width [Å], absorbing strength [eV], ramp [Å]")
    p.add_argument("--cap-spacing", type=float, help="largest distance between window centres [Å]")
    p.add_argument("--cap-split", default="", help="transverse window split, e.g. 2,1")
    p.add_argument("--cap-tov", type=float, default=2.5)
    p.add_argument("--cap-solver", default="dense64", choices=["dense64", "dense", "rows64", "rows"])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--l1", default="dm:1e-2", help="level 1: dm:CUT (window selection) or dmc:CUT (core selection)")
    p.add_argument("--up", default="x:1e-2,x:1e-2", help="further levels AXIS:CUT[:mu1:mu2[:npts]] (comma list; none)")
    p.add_argument("--halo", default="", help="x halo of the windows of the further levels, in blocks of the level "
                                            "below (comma list like --up; all or empty entry: every coupled block)")
    p.add_argument("--halo-smooth", default="same", choices=["same", "full"],
                   help="windows of the smoother of A_k: same = the rho windows (halo), full = every coupled block")
    p.add_argument("--l1-cap", default="", help="s1,d1 of the projected CAP (default: the fine s,d)")
    p.add_argument("--l1-ramp", default="dist", choices=["dist", "plane"])
    p.add_argument("--l1-lu", default="c64", choices=["c64", "c128"])
    p.add_argument("--l1-solver", default="rows", choices=["rows", "lu"],
                   help="cluster smoothers of A_k: rows = only the rows of the window inverse that RAS writes, lu = LU "
                        "of the whole window")
    p.add_argument("--mem-frac", type=float, default=0.5, help="window accumulators of one rho batch: at most this "
                                                                "fraction of the free GPU memory (and --batch-gb)")
    p.add_argument("--batch-gb", type=float, default=24.0)
    p.add_argument("--stream-gb", type=float, default=None, help="operators larger than this [GB] (fine A, fine "
                   "smoother rows, level operators A_k, level smoothers) live in pinned host memory and are streamed "
                   "through the GPU at every application, copy overlapped with compute (0: all; default: none)")
    p.add_argument("--l1-free", action="store_true", help="no stored level-1 operator A1: A1 @ y = P^H A P y through "
                   "the fine operator, windows from the S1, H1 blocks (6 nm: ~33 GB of A1 blocks; implies --host-blocks)")
    p.add_argument("--free-levels", type=int, default=0, help="levels 1 .. K without stored A_k (A_k @ y = P^H A_{k-1} "
                   "P y, recursively; --l1-free = 1; implies --host-blocks)")
    p.add_argument("--host-blocks", action="store_true", help="S, H Galerkin blocks of every level in host memory "
                   "(GPU memory of the level-1 S, H blocks: 22 GB on the 21600a y-cut, ~44 GB on the 6 nm y-cut)")
    p.add_argument("--merge-eig", default="full", choices=["full", "rand", "sketch"], help="bases of the merges: "
                   "dense eigh of the window rho; rand: randomized range finder on the formed rho (n x l memory; dense "
                   "host eigh past 0.6 n); sketch: rho Omega of a probe block directly from the contour (one LU and two "
                   "solves per point instead of the inverse), Nystrom, probes grown until they resolve the cut")
    p.add_argument("--sketch-prec", default="c128", choices=["c128", "c64", "lu64"], help="--merge-eig sketch: "
                   "c64: pencil LU, probes and solves in complex64; lu64: LU in complex128, factors down-cast to "
                   "complex64 for the solves (accumulation and Nystrom in complex128)")
    p.add_argument("--schur", action="store_true", help="dmc: the core (level 1) / pair (merges) block of the window rho "
                   "by Schur folding of the window into the block (as the single-shot msbasis tile contour): no window "
                   "inverse or eigh; window DM cache in <tag>_core_...")
    p.add_argument("--schur-prec", default="single", choices=["single", "double"],
                   help="--schur: environment LU in complex64 + refinement (single) or complex128")
    p.add_argument("--schur-refine", type=int, default=1, help="--schur single: refinement steps of A_EE^-1 A_EP")
    p.add_argument("--schur-probes", type=int, default=0, help="--schur: r random probes, rho_PP Omega + Nystrom "
                   "(0: the full inverse of the Schur complement)")
    p.add_argument("--sweep", default="fine,exact,top")
    p.add_argument("--counts", default="32,128,256", help="top-level sizes (multiples of the slow m)")
    p.add_argument("--slow", default="32,3", help="slow modes per round m, subspace steps q")
    p.add_argument("--gamma", type=int, default=1, help="coarse corrections per level: 1 V-cycle, 2 W-cycle")
    p.add_argument("--exact-max", type=int, default=30000, help="largest level solved exactly")
    p.add_argument("--ncol", type=int, default=0, help="FGMRES on the k strongest injection columns (0: all)")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--maxit", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--fgmres-log", type=int, default=0)
    p.add_argument("--dm-cache", default="", help="window density-matrix cache dir (shared with the mlevel scripts)")
    p.add_argument("--basis-cache", default="", help="bases of the levels >= 2 (energy independent): dir, none, or "
                                                     "empty = <dm-cache>/<tag>_levels (in memory only without dm-cache)")
    p.add_argument("--mu", default="-0.15,0.6", help="contour window relative to Ec [eV]")
    p.add_argument("--npts", type=int, default=24)
    p.add_argument("--dm-keep", type=float, default=1e-4)
    p.add_argument("--dm-batch-gb", type=float, default=16.0)
    p.add_argument("--threads", type=int, default=8, help="CPU matvec threads")
    p.add_argument("--out", help="json with the level sizes and the iteration counts (merged across runs)")
    a = p.parse_args(argv)
    a.free_levels = max(a.free_levels, int(a.l1_free))
    a.host_blocks = a.host_blocks or a.free_levels > 0
    if a.gpu:
        gpu.enable()
    else:
        gpu.from_env()
    xp = _xp()
    log(f"backend: {gpu.describe()}")
    gpu.track_peak()
    t00 = time.time()
    name = os.path.basename(os.path.normpath(a.inputs))
    if os.path.exists(os.path.join(a.inputs, "prep.json")):  # release prep (python -m qtx_modespace.prep), e.g. the 6 / 12 nm Si sheets
        from qtx_modespace.quatrex import load_inputs                 # (needs PYTHONPATH=release/qtx_modespace/src)
        inp = load_inputs(a.inputs, log=log)
        if name == "prep":                                            # <dev>/prep -> <dev>
            name = os.path.basename(os.path.dirname(os.path.normpath(a.inputs)))
    else:
        inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc=a.obc)
    dev.sigma_cache = a.sigma_cache or os.path.join("results_ms", "sigma", name)
    n = dev.n
    Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
    w_, s_, d_ = (float(v) for v in a.cap.split(","))
    split = [int(v) for v in a.cap_split.split(",")] if a.cap_split else None
    log(f"{name}: device {n} orbitals, slabs {dev.ns}; Ec {Ec:.4f} eV")
    sw = CapSchwarz(dev, inp, w_, s_, d_, spacing=a.cap_spacing, solver=a.cap_solver, workers=a.workers, split=split,
                    tov=a.cap_tov, stream_gb=a.stream_gb)
    core = None
    if a.schur:
        assert a.l1.startswith("dmc:"), "--schur: dmc only (the core / pair block of rho; dm needs the window modes)"
        core = dict(prec=a.schur_prec, refine=a.schur_refine, probes=a.schur_probes)
    h = Hierarchy(dev, inp, sw, l1_cap=tuple(float(v) for v in a.l1_cap.split(",")) if a.l1_cap else None,
                  l1_ramp=a.l1_ramp, l1_lu=a.l1_lu, l1_solver=a.l1_solver, batch_gb=a.batch_gb, mem_frac=a.mem_frac,
                  halo_smooth=a.halo_smooth, host_blocks=a.host_blocks, stream_gb=a.stream_gb, l1_free=a.free_levels,
                  merge_eig=a.merge_eig, sketch_prec=a.sketch_prec, core=core)
    sweeps = a.sweep.split(",")
    counts = [int(v) for v in a.counts.split(",")]
    sm, sq = (int(v) for v in a.slow.split(","))
    assert all(N % sm == 0 for N in counts)
    kind, cut1 = a.l1.split(":")[0], float(a.l1.split(":")[1])
    mu = tuple(float(v) for v in a.mu.split(","))
    ups = parse_up(a.up, a.halo, mu)
    tag = (f"dm_w{a.cap.replace(',', '_')}_s{a.cap_split.replace(',', '_') or '1'}_t{a.cap_tov:g}"
           f"_mu{a.mu.replace(',', '_')}_n{a.npts}")
    if a.basis_cache != "none":
        h.basis_dir = a.basis_cache or (os.path.join(a.dm_cache, tag + "_levels") if a.dm_cache else None)
    s1_, d1_ = h.s1, h.d1
    keys, chain = [], f"{name}|{tag}|l1 {a.l1}|cap {s1_:g},{d1_:g},{a.l1_ramp}" + (
        f"|schur {a.schur_prec} {a.schur_refine} p{a.schur_probes} k{a.dm_keep:g}" if a.schur else "")
    for axis, cut, m1, m2, npts, halo in ups:                    # key of each level = everything its rho depends on
        chain += (f"|{axis}:{cut:g}:{m1:g}:{m2:g}:{npts or a.npts}:h{'all' if halo is None else halo}"
                  + ("" if a.merge_eig == "full" else f"|eig {a.merge_eig}")
                  + (f"|{a.sketch_prec}" if a.merge_eig == "sketch" and a.sketch_prec != "c128" else ""))
        keys.append(chain if a.basis_cache != "none" else None)
    DM = U1 = None
    for dE in (float(v) for v in a.dE.split(",")):
        E = Ec + dE
        SL, SR = dev.sigma(E + 1j * a.eta)
        B = dev.inject(SL)
        b = np.zeros((n, B.shape[1]), complex)
        b[dev.bL] = B
        nsel = min(a.ncol, b.shape[1]) if a.ncol else b.shape[1]
        A = dev.A(E, SL, SR).tocsr()
        # the fine operator also carries every A1 product with --l1-free (5 uses per outer iteration): kept resident
        Aop = (operator(A, stream_gb=None if a.free_levels else a.stream_gb) if gpu.on else operator(A, a.threads))
        GR = xp.asarray(1j * (SR - SR.conj().T))
        bR = xp.asarray(dev.bR)
        Tof = lambda y: float(xp.real(xp.einsum("ic,ij,jc->", y[bR].conj(), GR, y[bR])))
        X = load_psi(a.psi_dir, E, A, b, dev.bL)
        if X is None and a.exact:
            X, _ = mb.exact_solve(dev, A, b)
        if X is not None:
            X = xp.asarray(X[:, -nsel:])                     # injection columns ascend in Gamma eigenvalue: last = strongest
        bg = xp.asarray(b[:, -nsel:])
        Tex = Tof(X) if X is not None else np.nan
        t = time.time()
        sw.set_energy(E, A, log=quiet)
        log(f"Ec {dE:+.3f}: {b.shape[1]} columns, FGMRES on {nsel}, T_exact({nsel}) {Tex:.6f}; window factor "
            f"{time.time() - t:.0f} s" + (f", core rows {sw.rows_gb:.1f} GB" if sw.rows else "")
            + f"; gamma {a.gamma}; {h.memstr()}")
        h.mem("fine level: operator, right-hand sides, window factors of S0")
        S0, A0 = Timed(sw, "S0"), Timed(Aop, "A")
        rec = dict(E=E, channels=int(b.shape[1]), ncol=int(nsel), T_exact=Tex, runs=[], levels=[], args=vars(a))

        def run(tag_, prec, parts):
            for c in parts:
                c.n, c.t = 0, 0.0
            ch = a.chunk if a.chunk > 0 else nsel
            t = time.time()
            xs, its, res = [], [], []
            for k in range(0, nsel, ch):
                def cb(i, e, k=k):
                    if a.fgmres_log and ((i + 1) % a.fgmres_log == 0 or i == 0):
                        log(f"        {tag_[:28]} chunk {k // ch + 1}/{-(-nsel // ch)} it {i + 1}: residual est max "
                            f"{e.max():.2e}, {time.time() - t:.0f} s")
                xk, ik, rk = fgmres_block(Aop, bg[:, k:k + ch], xp.zeros_like(bg[:, k:k + ch]), prec, a.tol, a.maxit,
                                          cb=cb)
                xs.append(xk), its.append(np.asarray(ik)), res.append(np.asarray(rk))
                gpu.free()
            xs, its, res = xp.concatenate(xs, axis=1), np.concatenate(its), np.concatenate(res)
            dt = time.time() - t
            ep = float(xp.linalg.norm(xs - X) / xp.linalg.norm(X)) if X is not None else np.nan
            Tx = Tof(xs)
            work = ", ".join(f"{c.name} {c.t:.0f} s" for c in parts)
            log(f"    {tag_:38s} its {its.min():3d}..{its.max():3d}, res {res.max():.0e}, T err {(Tx - Tex) / Tex:+.1e}, "
                f"psi err {ep:.1e} ({dt:.0f} s: {work})")
            rec["runs"].append(dict(tag=tag_, its=its.tolist(), res=float(res.max()), T=Tx, T_err=(Tx - Tex) / Tex,
                                    psi_err=ep, t=dt, work={c.name: c.t for c in parts}))
            del xs
            gpu.free()
            h.mem(f"FGMRES {tag_}")

        def slow_sweep(tag_, A_, Sk, wrap, parts):
            """Slow modes of (A_, Sk) in rounds of sm; fine its with the first N as the top level."""
            t = time.time()
            C = SlowCoarse(A_, Sk, sm, sq, max(counts) // sm, log=quiet)
            log(f"  {tag_}: slow modes {len(C.info['rounds'])} x {sm} (q {sq}) on N {A_.shape[0]}: |theta|max per "
                f"round " + " / ".join(f"{d['theta_max']:.2f}" for d in C.info["rounds"]) + f" ({time.time() - t:.0f} s)")
            for N in counts:
                run(f"{tag_}, top N {N}", wrap(SubCoarse(A_, C.Q, N)), parts)
                gpu.free()
            C.free()

        if "fine" in sweeps:
            slow_sweep("two-level (slow modes on the fine level)", Aop, sw, lambda top: gcycle(A0, S0, top, a.gamma),
                       (S0, A0))
        if DM is None:
            DM = h.window_dm(a.dm_cache or None, tag, Ec + mu[0], Ec + mu[1], a.npts, a.dm_keep, a.dm_batch_gb,
                             core=core)
        if U1 is None:
            U1 = h.basis1(DM, kind, cut1)                         # energy independent: built once
        tb = time.time()
        h.level1(U1, E, SL, SR, A0=Aop)
        for (axis, cut, m1, m2, npts, halo), key in zip(ups, keys):
            h.merge(axis, cut, kind, Ec + m1, Ec + m2, npts or a.npts, halo, key)
        rec["t_levels"] = time.time() - tb
        log(f"  levels at Ec {dE:+.3f}: {rec['t_levels']:.0f} s (" + ", ".join(
            f"level {k} {'stored' if h.LV[k].info.get('stored') else 'rho'}" for k in range(2, len(h.LV) + 1)) + ")")
        nl = len(h.LV)
        rec["levels"] = [h.LV[k].info for k in range(1, nl + 1)]
        rec["mem"] = h.memlog
        h.inventory()

        def go(tag_, m, bottom):
            pr, tm = h.cycle(m, bottom, A0, S0, a.gamma)
            run(tag_, pr, (S0, A0, tm))

        if "exact" in sweeps:
            for m in range(1, nl + 1):
                if h.LV[m].N > a.exact_max:
                    continue
                bot = h.exact(m)
                h.mem(f"slab LU of A{m} (N {h.LV[m].N}): {len(bot.sizes)} slabs of {min(bot.sizes)}..{max(bot.sizes)}, "
                      f"{bot.gb:.2f} GB")
                c = bg                                           # plain mode space: psi = P A_m^-1 P^H b, no smoother
                for k in range(1, m + 1):
                    c = h.LV[k].PH(c)
                c = bot(c)
                for k in range(m, 0, -1):
                    c = h.LV[k].P(c)
                Tm = Tof(c)
                em = float(xp.linalg.norm(c - X) / xp.linalg.norm(X)) if X is not None else np.nan
                log(f"    mode space level {m} (N {h.LV[m].N}): T {Tm:.6f}, T err {(Tm - Tex) / Tex:+.1e}, psi err {em:.1e}")
                rec.setdefault("mode_space", []).append(dict(level=m, N=h.LV[m].N, T=Tm, T_err=(Tm - Tex) / Tex,
                                                             psi_err=em))
                del c
                go(f"exact level {m} (N {h.LV[m].N})", m, bot)
                del bot
                gpu.free()
        if "top" in sweeps:                                      # deepest level: RAS over its own blocks + slow modes
            Ln = h.LV[nl]
            t = time.time()
            own = h.own_smoother()
            h.mem(f"level {nl} own-block smoother")
            log(f"  level {nl} own-block smoother: windows {min(own.sizes)}..{int(np.median(own.sizes))}.."
                f"{max(own.sizes)}, {a.l1_solver} {own.gb:.2f} GB ({time.time() - t:.0f} s)")
            t = time.time()
            C = SlowCoarse(Ln.A, own, sm, sq, max(counts) // sm, log=quiet)
            log(f"  top: slow modes {max(counts) // sm} x {sm} (q {sq}) on N{nl} {Ln.N}: |theta|max per round "
                + " / ".join(f"{d['theta_max']:.2f}" for d in C.info["rounds"]) + f" ({time.time() - t:.0f} s)")
            h.mem(f"top: slow modes on N{nl}")
            for N in counts:
                go(f"{nl} levels + top N {N}", nl, gcycle(Ln.A, own, SubCoarse(Ln.A, C.Q, N), a.gamma))
                gpu.free()
            C.free()
            del own
        h.free()
        sw.free()
        del X, bg, GR, Aop, A, S0, A0
        gc.collect()
        gpu.free()
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            res = json.load(open(a.out)) if os.path.exists(a.out) else {"E": {}}
            res["Ec"] = Ec
            res["E"][f"{dE:.4f}"] = rec
            json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"done, {time.time() - t00:.0f} s")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        m_ = gpu.mem_mark()                                          # the step that failed: its peak before the error
        if m_:
            log("    mem | failed step: " + ", ".join(f"{k} {v / 2 ** 30:.2f}" for k, v in m_.items()) + " GB")
        raise
