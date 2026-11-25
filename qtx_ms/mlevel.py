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
import argparse, gc, hashlib, json, os, sys, time
import numpy as np
import scipy.sparse as sp

from qtx_ms import devstore, dist, gpu, msbasis as mb

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


def rand_top_eig(rho, cut, q=2, f=0.5, p=32, l0=256, lmax_frac=0.6, seed=0, dense_gpu_max=8192):
    """Eigenpairs of the Hermitian rho with eigenvalue > cut, by a randomized range finder (Halko, Martinsson, Tropp):
    Q = orth((rho)^(2q+1) Omega), grown in blocks until the Rayleigh-Ritz spectrum of Q^H rho Q has at most l - p
    values above f * cut (the subspace then reaches below the cut, so the kept vectors converge at the rate
    (f cut / lambda)^(2q+1)). Memory n x l instead of the ~11 n^2 workspace of the dense eigh. When l would pass
    lmax_frac * n: dense eigh (on the GPU up to n = dense_gpu_max, on the host above: no workspace limit). Returns
    (ev, V) ascending, ev > cut."""
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
            if gpu.on and n > dense_gpu_max:                   # the GPU eigh workspace (~11 n^2) only for small n
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

    def __init__(self, P, n, prec="single", refine=1, probes=0, seed=0, eig="full", cut=0.0):
        xp = _xp()
        self.xp, self.prec, self.refine = xp, prec, int(refine)
        self.eig, self.cut = eig, cut                    # eig "rand": rho_PP eigenpairs > cut by rand_top_eig
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

    def parts(self):
        """The contour sums (a subset of the points on every rank that shares the window), and back."""
        return [self.acc] if self.Om is None else [self.acc, self.acch]

    def set_parts(self, xs):
        self.acc = xs[0]
        if self.Om is not None:
            self.acch = xs[1]

    def result(self):
        xp = self.xp
        if self.Om is None:
            rho = rho_of(self.acc)
            self.acc = None
            ev, V = rand_top_eig(rho, self.cut) if self.eig == "rand" else xp.linalg.eigh(rho)
            del rho
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


def is_free(A):
    """Level operator without stored blocks (GalerkinOp, DistOp with A0)."""
    return isinstance(A, GalerkinOp) or getattr(A, "free", False)


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
        D = h.dense_A(L, self.E, wr, wc)
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


class DistOp:
    """Level operator of the distributed hierarchy (qtx_ms.dist). Products: the stored block rows of the owned blocks
    (R, on the GPU or streamed from the host), or with A0 (free: the operator of the level below) P_k^H A_{k-1} P_k as
    GalerkinOp. Dense pieces are E S - H - Sigma from the S, H block rows, the owned ones and, between
    Hierarchy.ghost_rows and drop_ghosts, the ghost ones (take: local); take_all: summed over the ranks (collective;
    owned rows only, from R once the S, H blocks are freed)."""

    def __init__(self, h, L, E, sig1, A0=None, host=False):
        self.h, self.L, self.A0, self.E, self.sig1 = h, L, A0, E, sig1
        self.off, self.N = L.off, int(L.off[-1])
        self.shape = (L.part.nloc,) * 2
        self.rows, self.cols = list(L.part.own), L.nbr
        self.free, self.blocks, self.host, self.st = A0 is not None, None, bool(host) and gpu.on, None
        self.R = {}
        self.gb = 0.0
        if self.free:
            return
        self.ex = dist.Exch(L.part, h.entries(L, set(int(j) for i in self.rows for j in L.nbr[i])))
        for i in self.rows:
            Ri = self._row(i)
            self.R[i] = gpu.pinned(Ri) if self.host else Ri
            del Ri
        pos = lambda b_: self.ex.pos[L.off[b_]:L.off[b_ + 1]]
        self.C = {i: gpu.asarray(np.concatenate([pos(int(j)) for j in L.nbr[i]])) for i in self.rows}
        self.O = {i: int(self.ex.pos[L.off[i]]) for i in self.rows}
        if self.host:
            self.st = gpu.Streamer([(self.R[i],) for i in self.rows])
        self.gb = sum(r.nbytes for r in self.R.values()) / 2 ** 30

    def _row(self, i):
        return self._take([i], [int(j) for j in self.L.nbr[i]])

    def __matmul__(self, Y):
        one = Y.ndim == 1
        Y = Y[:, None] if one else Y
        if self.free:
            out = self.L.PH(self.A0 @ self.L.P(Y))
            return out[:, 0] if one else out
        X = self.ex(Y)
        out = _xp().zeros((self.L.part.nloc, Y.shape[1]), complex)

        def put(i, Ri):
            out[self.O[i]:self.O[i] + Ri.shape[0]] = Ri @ X[self.C[i]]
        if self.st is not None:
            self.st.run(lambda q, v: put(self.rows[q], v[0]))
        else:
            for i in self.rows:
                put(i, self.R[i])
        return out[:, 0] if one else out

    def _take(self, wr, wc):
        h, L = self.h, self.L
        D = h.dense_A(L, self.E, wr, wc)
        orow, ocol = offs(L.r[wr]), offs(L.r[wc])
        lr, lc = {k: q for q, k in enumerate(wr)}, {k: q for q, k in enumerate(wc)}
        for (i, j), B in self.sig1.items():
            if i in lr and j in lc:
                D[orow[lr[i]]:orow[lr[i] + 1], ocol[lc[j]]:ocol[lc[j] + 1]] -= _xp().asarray(B)
        return D

    def take(self, wr, wc):
        return self._take([int(k) for k in wr], [int(k) for k in wc])

    def take_all(self, wr, wc):
        wr, wc = [int(k) for k in wr], [int(k) for k in wc]
        L, xp = self.L, _xp()
        orow, ocol = offs(L.r[wr]), offs(L.r[wc])
        D = xp.zeros((orow[-1], ocol[-1]), complex)
        mine, loc = set(self.rows), {k: q for q, k in enumerate(wc)}
        for q, i in enumerate(wr):
            if i not in mine:
                continue
            if L.Sb is not None:
                D[orow[q]:orow[q + 1]] = self._take([i], wc)
                continue
            co = offs(L.r[L.nbr[i]])
            for a_, j in enumerate(L.nbr[i]):
                if int(j) in loc:
                    D[orow[q]:orow[q + 1], ocol[loc[int(j)]]:ocol[loc[int(j)] + 1]] = xp.asarray(
                        self.R[i][:, co[a_]:co[a_ + 1]])
        return self.h.comm.sum(D)

    def sub(self, w):
        w = [int(k) for k in w]
        return self.take(w, w), offs([self.off[k + 1] - self.off[k] for k in w])

    def dense(self):
        return self.take_all(range(len(self.L.r)), range(len(self.L.r)))


class FineMat:
    """Level S or H that is never stored: dense(L, this, rows, cols) forms the pieces from the sparse fine blocks."""

    def __init__(self, name):
        self.name = name


class FineBlocks:
    """The fine S, H of the rows of the level-1 cores as sparse core-pair blocks M_ij = M[core_i, core_j] (i, j in the core
    graph), CSR on the device (user, 2026-11-25). The level S_k, H_k of --fine-blocks levels are formed from them, dense,
    only where a window, a Galerkin product or an LU needs them: P_i^H M_ij P_j, P = the prolongation of the blocks
    (level 1: U). Rows of the own cores are kept; those of other ranks are cut on first use (device store: from its files,
    without the store from the matrices) and dropped by drop_ghosts.
    "A": the sparse E S_ij - H_ij at the energy set by set_energy (without Sigma), formed once per pair: the smoother windows
    and the LUs need one multiplication instead of two. keep(L) .. release(): during the basis build of one level the dense
    S_k, H_k pairs (one triangle) stay on the device and the windows of its contour rho are assembled from them."""

    def __init__(self, h):
        self.h, self.own, self.B, self.gb = h, set(h.part0.own), {}, 0.0
        self.real = {nm: not np.iscomplexobj(np.zeros(0, self.mat(nm).dtype)) for nm in "SH"}
        self.real["A"] = self.real["S"] and self.real["H"]
        self.E, self.level, self.pairs = None, None, {}
        for nm in "SH":
            for i in h.part0.own:
                self.row(nm, i)
        self.gb = self.bytes() / 2 ** 30

    def mat(self, nm):
        return getattr(self.h.dev, nm)

    def bytes(self):
        return sum(sum(int(getattr(R, k).nbytes) for k in ("data", "indices", "indptr"))
                   for (_, i), bs in self.B.items() for R in bs if R is not None)

    def set_energy(self, E):
        """Energy of the "A" blocks (real: S, H Hermitian, so is A without Sigma)."""
        assert np.isreal(E)
        if self.E != E:
            self.E = float(np.real(E))
            for k in [k for k in self.B if k[0] == "A"]:
                del self.B[k]

    def row(self, nm, i):
        """[M_ij for j in nbr[i]], CSR on the device (None: no nonzero)."""
        bs = self.B.get((nm, i))
        if bs is None and nm == "A":
            bs = self.B[("A", i)] = []
            for s_, h_ in zip(self.row("S", i), self.row("H", i)):
                bs.append(None if s_ is None and h_ is None else -h_ if s_ is None else
                          self.E * s_ if h_ is None else (self.E * s_ - h_).tocsr())
        elif bs is None:
            h, M = self.h, self.mat(nm)
            R = (M.get(h.cores[i]) if hasattr(M, "hold") else M[h.cores[i]]).tocoo()
            bs = self.B[(nm, i)] = []
            for Rij in h.pair_blocks(R, i):
                bs.append(None if Rij is None else gpu.csr(Rij, None if self.real[nm] else np.complex128) if gpu.on else Rij)
        return bs

    def drop_ghosts(self):
        for k in [k for k in self.B if k[1] not in self.own]:
            del self.B[k]

    def free(self):
        self.B = {}
        self.release()

    # ---- dense pairs kept during the basis build of one level
    def need_gb(self, L, wins):
        """GB of the dense S, H pairs (one triangle) of the coupled blocks inside the windows `wins` of level L."""
        return 2 * 16 / 2 ** 30 * sum(int(L.r[b]) * int(L.r[b2]) for b, b2 in self.pair_set(L, wins))

    @staticmethod
    def pair_set(L, wins):
        out = set()
        for w in wins:
            w = [int(b) for b in w]
            s_ = set(w)
            for b in w:
                out.update((min(b, int(j)), max(b, int(j))) for j in L.nbr[b] if int(j) in s_)
        return out

    def keep(self, L):
        self.release()
        self.level = L.k

    def release(self):
        self.level, self.pairs = None, {}

    def mul(self, nm, R, X):
        """R @ X for complex X; a real R multiplies [Re X, Im X] (cusparse: one data type for both operands)."""
        xp = self.h.xp
        if not self.real[nm] or not gpu.on:
            return R @ X
        r = X.shape[1]
        W = xp.empty((X.shape[0], 2 * r), np.float64, order="F")
        W[:, :r], W[:, r:] = X.real, X.imag
        Y = R @ W
        return Y[:, :r] + 1j * Y[:, r:]

    def prol(self, L, b):
        """Prolongation of block b of level L to the fine orbitals: {core: n_core x r_b}."""
        xp = self.h.xp
        if L.k == 1:
            return {int(b): xp.asarray(L.U[int(b)])}
        Ub, out, o = xp.asarray(L.U[int(b)]), {}, 0
        for kb in L.kids[int(b)]:
            n = int(L.parent.r[kb])
            for c, Pk in self.prol(L.parent, kb).items():
                out[c] = Pk @ Ub[o:o + n]
            o += n
        return out

    def pair(self, L, nm, PH, P2, ids):
        """Dense S_k / H_k / A_k piece of a block pair: sum_ij PH_i M_ij P2_j (None: no nonzero)."""
        T = None
        for i, Pi in PH.items():
            Y = None
            for j, Pj in P2.items():
                c = self.h.pos[i].get(j)
                R = None if c is None else self.row(nm, i)[c]
                if R is not None:
                    Z = self.mul(nm, R, Pj)
                    Y = Z if Y is None else Y + Z
            if Y is not None:
                Y = Pi @ Y
                T = Y if T is None else T + Y
        return T

    def dense(self, L, nm, rows, cols):
        """Dense [rows, cols] (block lists) of S_k (nm "S") or H_k of level L. Hermitian when rows = cols: one triangle."""
        h, xp = self.h, self.h.xp
        orow, ocol = offs(L.r[rows]), offs(L.r[cols])
        D = xp.zeros((orow[-1], ocol[-1]), complex)
        herm = rows == cols
        if self.level == L.k and nm != "A":                          # from the kept pairs
            lr, lc = {b: a_ for a_, b in enumerate(rows)}, {b: a_ for a_, b in enumerate(cols)}
            todo = sorted({(min(b, int(b2)), max(b, int(b2))) for b in rows for b2 in L.nbr[b] if int(b2) in lc}
                          - {k[1:] for k in self.pairs if k[0] == nm})
            for b in sorted({b for b, _ in todo}):                 # pairs of one row at a time: P_b once
                tb = [b2 for b_, b2 in todo if b_ == b]
                Pb = {b: {c: V.conj().T for c, V in self.prol(L, b).items()}}
                for b2 in tb:
                    self.pairs[(nm, b, b2)] = self.pair(L, nm, Pb[b], self.prol(L, b2) if b2 != b else
                                                      {c: V.conj().T for c, V in Pb[b].items()}, (b, b2))
            for a_, b in enumerate(rows):
                for b2 in L.nbr[b]:
                    b2 = int(b2)
                    if b2 in lc:
                        T = self.pairs[(nm, min(b, b2), max(b, b2))]
                        if T is not None:
                            D[orow[a_]:orow[a_ + 1], ocol[lc[b2]]:ocol[lc[b2] + 1]] = T if b <= b2 else T.conj().T
            return D
        Pr = {b: {c: V.conj().T for c, V in self.prol(L, b).items()} for b in rows}
        for q, b2 in enumerate(cols):
            near = set(int(v) for v in L.nbr[b2])
            if not near & set(rows):
                continue
            P2 = {c: Pr[b2][c].conj().T for c in Pr[b2]} if herm else self.prol(L, b2)
            for a_, b in enumerate(rows):
                if b not in near or (herm and a_ > q):
                    continue
                T = self.pair(L, nm, Pr[b], P2, (b, b2))
                if T is None:
                    continue
                D[orow[a_]:orow[a_ + 1], ocol[q]:ocol[q + 1]] = T
                if herm and a_ < q:
                    D[ocol[q]:ocol[q + 1], orow[a_]:orow[a_ + 1]] = T.conj().T
        return D


class GatherSolve:
    """Bottom solver of the distributed hierarchy: the residual of the deepest level is gathered on every rank, the
    block-tridiagonal LU (SlabLU, the same on every rank) is applied, every rank keeps its rows."""

    def __init__(self, lu, part):
        self.lu, self.part = lu, part
        self.sizes, self.gb = lu.sizes, lu.gb

    def __call__(self, r):
        return self.lu(self.part.gather(r))[_xp().asarray(self.part.idx)]


class DistSlabLU:
    """Distributed bottom solver: the slabs of SlabLU (block-tridiagonal A_m) cut into p domains by p - 1 separator
    slabs, one domain per rank (dist.domains; p = min(ranks, (slabs + 1) // 2)). Every rank factors its interior slabs
    (SlabLU) and forms its Schur contributions on its two separators; the separator system (block tridiagonal) is
    summed over the ranks and solved on every rank. Apply: interior solve, sum of the separator right-hand sides,
    separator solve, interior solve again for the separator values (no coupling columns stored). Interiors without a
    lead are closed pieces (no Sigma): conditioning checked against SlabLU in the tests; the outer FGMRES absorbs the
    round-off. .sizes = slab sizes, .gb = storage of the largest rank."""

    def __init__(self, A, ix, part):
        xp, comm = _xp(), part.comm
        self.part, self.comm = part, comm
        bs = slab_blocks(A, ix)
        self.sizes = [sum(int(A.off[k + 1] - A.off[k]) for k in b_) for b_ in bs]
        ent = [np.concatenate([np.arange(A.off[k], A.off[k + 1]) for k in b_]) for b_ in bs]
        ns = np.array(self.sizes)
        self.dom, self.sep, ranks = dist.domains(ns.astype(float) ** 3, comm.size)
        p = len(self.dom)
        self.p = p
        mine = ranks.index(comm.rank) if comm.rank in ranks else None
        self.mine = mine
        # entry owner of the solution: interior -> the rank of its domain, separators -> the domain left of them
        own = np.zeros(len(bs), int)
        for d, (a, b) in enumerate(self.dom):
            own[a:b] = ranks[d]
        for d, s in enumerate(self.sep):
            own[s] = ranks[d]
        self.dpart = dist.Part(comm, ent, own, part.n)
        so = offs(ns[self.sep])                                  # separator system
        self.so, self.nsep = so, int(so[-1])
        rows = np.zeros(0, int)
        self.lu, self.inner = None, []
        if mine is not None:
            a, b = self.dom[mine]
            self.inner = list(range(a, b))
            self.L_, self.R_ = (mine - 1 if mine > 0 else None), (mine if mine < p - 1 else None)
            rows = np.concatenate([ent[s] for s in self.inner])
        self.exin = dist.Exch(part, rows)                        # residual rows of the domain + own separators
        self.ri = gpu.asarray(self.exin.pos[rows])
        sown = [d for d in range(p - 1) if ranks[d] == comm.rank]
        srow = np.concatenate([ent[self.sep[d]] for d in sown]) if sown else np.zeros(0, int)
        self.exsep = dist.Exch(part, srow)
        self.rs = gpu.asarray(self.exsep.pos[srow])
        self.srange = [(int(so[d]), int(so[d + 1])) for d in sown]
        self.exout = dist.Exch(self.dpart, part.idx)
        self.ro = gpu.asarray(self.exout.pos[part.idx])
        self.oi = gpu.asarray(self.dpart.gloc[rows]) if len(rows) else None
        self.os = gpu.asarray(self.dpart.gloc[srow]) if len(srow) else None
        # blocks of A: collective (every rank takes part), kept where they are needed
        def get(s, t, keep):
            B = A.take_all(bs[s], bs[t])
            return B if keep else None
        n_in = [0] + list(np.cumsum([ns[s] for s in self.inner])) if self.inner else [0]
        self.n_in = [int(v) for v in n_in]
        blocks = {}
        for s in range(len(bs)):
            for t in (s - 1, s, s + 1):
                if 0 <= t < len(bs):
                    B = get(s, t, s in self.inner and (t in self.inner or t in self.sep) or
                            s in self.sep and t in self.inner)
                    if B is not None:
                        blocks[(s, t)] = B
                    del B
        sepD = [A.take_all(bs[s], bs[s]) for s in self.sep]      # separator diagonals: on every rank
        # interior LU
        self.inner_lu = None
        if self.inner:
            T = [[blocks.get((s, t)) for t in self.inner] for s in self.inner]
            self.inner_lu = TriLU([T[i][i] for i in range(len(T))],
                                  [T[i][i + 1] for i in range(len(T) - 1)], [T[i + 1][i] for i in range(len(T) - 1)])
            del T
        f, l = (self.inner[0], self.inner[-1]) if self.inner else (None, None)
        self.AL = blocks.get((f, self.sep[self.L_])) if mine and self.L_ is not None else None       # first slab <- sep L
        self.AR = blocks.get((l, self.sep[self.R_])) if mine is not None and self.R_ is not None else None
        self.LA = blocks.get((self.sep[self.L_], f)) if mine and self.L_ is not None else None       # sep L <- first slab
        self.RA = blocks.get((self.sep[self.R_], l)) if mine is not None and self.R_ is not None else None
        del blocks
        # Schur complement on the separators: diagonal, upper, lower blocks, summed over the ranks
        Sd = [D.copy() if comm.rank == 0 else xp.zeros_like(D) for D in sepD]
        del sepD
        Su = [xp.zeros((int(ns[self.sep[d]]), int(ns[self.sep[d + 1]])), complex) for d in range(p - 2)]
        Sl = [xp.zeros((int(ns[self.sep[d + 1]]), int(ns[self.sep[d]])), complex) for d in range(p - 2)]
        if mine is not None and p > 1:
            k = len(self.inner)
            for side, Ac in (("L", self.AL), ("R", self.AR)):    # X = A_II^-1 (coupling columns), first / last slab
                if Ac is None:
                    continue
                X = self.inner_lu.solve_block(0 if side == "L" else k - 1, Ac)
                j = self.L_ if side == "L" else self.R_
                if self.LA is not None:
                    C = self.LA @ X[0]
                    if side == "L":
                        Sd[j] -= C
                    else:
                        Su[self.L_] -= C                         # sep L row, sep R column
                if self.RA is not None:
                    C = self.RA @ X[-1]
                    if side == "R":
                        Sd[j] -= C
                    else:
                        Sl[self.L_] -= C                         # sep R row, sep L column
                del X
        if p > 1:
            Sd, Su, Sl = [comm.sum(M) for M in Sd], [comm.sum(M) for M in Su], [comm.sum(M) for M in Sl]
            self.sep_lu = TriLU(Sd, Su, Sl)
        else:
            self.sep_lu = None
        local = (self.inner_lu.gb if self.inner_lu is not None else 0.0) + (self.sep_lu.gb if self.sep_lu else 0.0) + sum(
            M.nbytes for M in (self.AL, self.AR, self.LA, self.RA) if M is not None) / 2 ** 30
        self.gb = max(comm.allgather(float(local)))

    def __call__(self, r):
        xp = _xp()
        one = r.ndim == 1
        r = r[:, None] if one else r
        nc = r.shape[1]
        x = xp.zeros((self.dpart.nloc, nc), complex)
        xr = self.exin(r)
        xs = self.exsep(r)
        y = self.inner_lu.solve(xr[self.ri].astype(complex)) if self.inner_lu is not None else None
        if self.p > 1:
            g = xp.zeros((self.nsep, nc), complex)
            o = 0
            for a, b in self.srange:
                g[a:b] += xs[self.rs[o:o + b - a]]
                o += b - a
            k0, k1 = (0, self.n_in[1]) if self.inner else (0, 0)
            m0, m1 = (self.n_in[-2], self.n_in[-1]) if self.inner else (0, 0)
            if self.LA is not None:
                a, b = int(self.so[self.L_]), int(self.so[self.L_ + 1])
                g[a:b] -= self.LA @ y[k0:k1]
            if self.RA is not None:
                a, b = int(self.so[self.R_]), int(self.so[self.R_ + 1])
                g[a:b] -= self.RA @ y[m0:m1]
            g = self.comm.sum(g)
            xsep = self.sep_lu.solve(g)
            if y is not None:
                z = xp.zeros_like(y)
                if self.AL is not None:
                    z[k0:k1] += self.AL @ xsep[int(self.so[self.L_]):int(self.so[self.L_ + 1])]
                if self.AR is not None:
                    z[m0:m1] += self.AR @ xsep[int(self.so[self.R_]):int(self.so[self.R_ + 1])]
                y -= self.inner_lu.solve(z)
            if self.os is not None:
                x[self.os] = xp.concatenate([xsep[a:b] for a, b in self.srange])
        if y is not None:
            x[self.oi] = y
        x = self.exout(x)
        out = x[self.ro]
        return out[:, 0] if one else out


class TriLU:
    """Block-tridiagonal LU from its blocks (diagonals D, upper U[i] = (i, i + 1), lower L[i] = (i + 1, i)): consumed.
    solve(R) for a dense right-hand side over all rows; solve_block(i, B): the first and last slab of the solution of a
    right-hand side B on slab i only."""

    def __init__(self, D, U, L):
        xp = _xp()
        self.off = offs([d.shape[0] for d in D])
        self.lu, self.Lt, self.U = [], [], U
        Dt = D[0].copy()
        for s in range(len(D)):
            self.lu.append(gpu.lu_factor(Dt))
            del Dt
            if s < len(D) - 1:
                Lt = xp.ascontiguousarray(gpu.lu_solve(self.lu[s], xp.ascontiguousarray(L[s].T), trans=1).T)
                Dt = D[s + 1] - Lt @ U[s]
                self.Lt.append(Lt)
                D[s + 1] = L[s] = None
        self.gb = 3 * sum(float(b - a) ** 2 for a, b in zip(self.off[:-1], self.off[1:])) * 16 / 2 ** 30

    def solve(self, R):
        return _xp().concatenate(self._sweep([R[a:b] for a, b in zip(self.off[:-1], self.off[1:])]))

    def _sweep(self, Y, ends=False):
        """Forward / backward sweep over per-slab right-hand sides (None: zero)."""
        xp = _xp()
        n = len(self.lu)
        Y = list(Y)
        for s in range(1, n):
            if Y[s - 1] is not None:
                c = self.Lt[s - 1] @ Y[s - 1]
                Y[s] = -c if Y[s] is None else Y[s] - c
        X = [None] * n
        X[n - 1] = gpu.lu_solve(self.lu[n - 1], Y[n - 1])
        for s in range(n - 2, -1, -1):
            b = -(self.U[s] @ X[s + 1])
            if Y[s] is not None:
                b = b + Y[s]
            X[s] = gpu.lu_solve(self.lu[s], b)
            if ends and s + 1 < n - 1:
                X[s + 1] = None
            Y[s] = None
        return X

    def solve_block(self, i, B):
        Y = [None] * len(self.lu)
        Y[i] = xp_c(B)
        X = self._sweep(Y, ends=True)
        return [X[0], X[-1]]


def xp_c(B):
    return _xp().ascontiguousarray(B).astype(complex)


def slab_blocks(A, ix):
    """Blocks per slab: consecutive x positions grouped so that A is block tridiagonal over the slabs. The column lists
    are global."""
    ix = np.asarray(ix)
    rk = np.searchsorted(np.unique(ix), ix)                      # x position of every block: 0, 1, ...
    wd = max(1, max(int(abs(rk[i] - rk[int(j)])) for i in range(len(rk)) for j in A.cols[i]))
    sl = rk // wd
    return [np.where(sl == s)[0] for s in range(int(sl.max()) + 1)]


class SlabLU:
    """Exact solve with a level operator A_k without its dense LU: block-tridiagonal LU over x slabs (RGF-style
    left-to-right elimination). A slab = `width` consecutive x positions of the level's blocks, width = the largest x
    distance of two coupled blocks, so that A_k is block tridiagonal over the slabs. Storage 3 sum n_s^2 (slab LU,
    L_s = A_{s+1,s} D_s^-1, A_{s,s+1}) instead of N^2, work sum n_s^3 instead of N^3."""

    def __init__(self, A, ix):
        xp = _xp()
        take = getattr(A, "take_all", A.take)                    # distributed: the rows of every rank (collective)
        bs = slab_blocks(A, ix)
        self.idx = [xp.asarray(np.concatenate([np.arange(A.off[k], A.off[k + 1]) for k in b_])) for b_ in bs]
        self.sizes = [len(i_) for i_ in self.idx]
        self.lu, self.L, self.Up = [], [], []
        Dt = take(bs[0], bs[0])
        for s in range(len(bs)):
            self.lu.append(gpu.lu_factor(Dt))
            del Dt
            if s < len(bs) - 1:
                Lo = take(bs[s + 1], bs[s])
                Lt = xp.ascontiguousarray(gpu.lu_solve(self.lu[s], xp.ascontiguousarray(Lo.T), trans=1).T)   # Lo D^-1
                del Lo
                Up = take(bs[s], bs[s + 1])
                Dt = take(bs[s + 1], bs[s + 1])
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
                 sketch_prec="c128", core=None, rho_eig="full", log=log, comm=None, keep_max=np.inf,
                 bottom="dist", share_points=True, fine_blocks=0, stream_smoother_gb=None):
        assert l1_ramp in ("dist", "plane") and l1_lu in ("c64", "c128") and l1_solver in ("rows", "lu")
        assert halo_smooth in ("same", "full") and merge_eig in ("full", "rand", "sketch")
        self.dev, self.inp, self.sw, self.log = dev, inp, sw, log
        # domain decomposition (qtx_ms.dist): every rank builds and applies its own blocks; comm None / one rank: all
        self.comm = comm = comm if comm is not None else dist.Comm()
        self.dd = comm.size > 1
        # distributed: a free level larger than keep_max is never solved exactly (--exact-max): its S, H blocks are read
        # only by the merge above it and dropped after it (12 nm: ~84 GB of level 1, ~104 GB of level 2 on the host)
        self.keep_max = keep_max
        # bottom "dist": the deepest level solved by DistSlabLU (domains of slabs over the ranks), "repl": the same LU on
        # every rank (GatherSolve). share_points: contour points of the merge windows balanced over the ranks
        assert bottom in ("dist", "repl"), bottom
        self.bottom, self.share_points = bottom, bool(share_points)
        assert not self.dd or (sw.part is not None and sw.comm is comm), "sw.distribute(comm) first"
        # "rand": dm eigenvectors of the merge windows by the randomized range finder (rho formed); "sketch": rho
        # Omega from a probe block (one LU + 2 solves per contour point, no window rho), Nystrom (_sketch_batch)
        self.merge_eig = merge_eig
        self.sketch_prec = sketch_prec
        # core = dict(prec, refine, probes): dmc blocks by Schur folding of the window into the core / pair (SchurRho)
        self.core = core
        # rho_eig "rand": the eigenpairs of the level-1 window rho / Schur core block (> keep) and of the Schur pair
        # blocks (> cut) by the randomized range finder (rand_top_eig) instead of the dense eigh
        self.rho_eig = rho_eig
        # host_blocks: the S, H Galerkin blocks of every level live in host memory (read only to assemble A_k, the rho
        # windows and the Galerkin products of the next level); 6 nm y-cut level 1: ~44 GB off the GPU
        self.host_blocks = bool(host_blocks) and gpu.on
        # stream_gb: level operators A_k and level smoothers larger than this live in pinned host memory and are
        # streamed through the GPU at every application (copy overlapped with the GEMMs; None: everything resident)
        self.stream_gb = stream_gb if gpu.on else None
        # level smoothers (rows of the window inverses): their own threshold, 0 = always streamed from the host
        self.stream_sm = (stream_gb if stream_smoother_gb is None else stream_smoother_gb) if gpu.on else None
        # l1_free: no stored A_1; products through the fine operator, dense pieces from the S1, H1 blocks (GalerkinOp)
        # levels 1 .. fine_blocks: S_k, H_k are formed from the sparse fine core-pair blocks (FineBlocks) where they are
        # needed, never stored; their A_k is free (products through the level below)
        self.fine_blocks, self.fb = int(fine_blocks), None
        self.free_levels = max(int(l1_free), self.fine_blocks)   # levels 1 .. free_levels without stored A_k (bool: 0 / 1)
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
        self.owner1 = sw.owner if self.dd else np.zeros(nw, int)
        self.part0 = sw.part if self.dd else dist.Part(comm, self.cores, self.owner1, n)
        pos0 = np.full(n, -1, np.int64)                             # device orbital -> row of the local fine vector
        pos0[self.part0.idx] = np.arange(self.part0.nloc)
        self.cores_g = [xp.asarray(pos0[c]) if self.owner1[k] == comm.rank else None for k, c in enumerate(self.cores)]
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
        if isinstance(dev.H, devstore.SlabRows):                    # rows of the own cores only, summed over the ranks
            for k in self.part0.own:
                for M_ in (dev.H, dev.S):
                    G[k, np.unique(self.lab[M_[self.cores[k]].indices])] = True
            G = comm.sum(G.astype(np.int32)) > 0
        for M_ in (() if isinstance(dev.H, devstore.SlabRows) else (dev.H.tocsr(), self.S)):   # pattern only
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
        hm = devstore.hostmem()
        self.memlog.append(dict(step=step, host=hm, **{k: v / g for k, v in m.items()}))
        self.log(f"    mem | {step:58s} live {m['live'] / g:6.2f}  peak {m['peak'] / g:6.2f}  pool {m['pool'] / g:6.2f}  "
                 f"free {m['free'] / g:6.2f}  host {hm['anon']:.1f} + file {hm['file']:.1f}, CPU memory free {hm['cpu']:.0f}")

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
                     f"A list {nb(L.Ab) / g + nb(L.A.blocks) / g:.2f}, A operator (block rows) {nb(list(L.A.R.values()) if isinstance(L.A.R, dict) else L.A.R) / g:.2f}"
                     + (" (host, streamed)" if getattr(L.A, "host", False) else "")
                     + (f" (none: A{k} through the level below)" if is_free(L.A) else "")
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

    # ------------------------------------------------------------------ domain decomposition
    @staticmethod
    def entries(L, blocks):
        """Level-L vector entries of a set of blocks."""
        b = sorted(int(k) for k in blocks)
        return np.concatenate([np.arange(L.off[k], L.off[k + 1]) for k in b]) if b else np.zeros(0, int)

    def make_part(self, L, owner):
        """Vector distribution of level L (L.off set)."""
        L.owner = np.asarray(owner)
        L.part = dist.Part(self.comm, [np.arange(L.off[k], L.off[k + 1]) for k in range(len(L.r))], L.owner, L.N)
        L.own = L.part.own

    def share(self, U, owner, G):
        """Block bases on every rank: the owned blocks and those coupled to them on the device, the others on the host
        (windows further out copy them when they are used)."""
        if not self.dd:
            return U
        Uh = self.comm.share([None if u is None else gpu.asnumpy(u) for u in U], owner)
        mine = np.asarray(owner) == self.comm.rank
        near = mine | (np.asarray(G)[mine].any(0) if mine.any() else False)
        return [(u if u is not None else self.xp.asarray(Uh[b])) if near[b] else Uh[b] for b, u in enumerate(U)]

    def ghost_rows(self, L, need):
        """S, H block rows of the blocks `need` owned by other ranks (kept until drop_ghosts). Collective."""
        if not self.dd or isinstance(L.Sb, FineMat):
            return
        need = sorted(int(b) for b in need if L.owner[int(b)] != self.comm.rank and L.Sb[int(b)] is None)
        req = [[b for b in need if L.owner[b] == r] for r in range(self.comm.size)]
        give = self.comm.alltoall(req)
        new = lambda sh: gpu.pinned(np.zeros(sh, complex)) if self.host_blocks and gpu.on else np.empty(sh, complex)
        for X in (L.Sb, L.Hb):
            sends = {r: [np.ascontiguousarray(gpu.asnumpy(B)) for b in bs for B in X[b]]
                     for r, bs in enumerate(give) if bs and r != self.comm.rank}
            got = {r: [new((int(L.r[b]), int(L.r[j]))) for b in bs for j in L.nbr[b]] for r, bs in enumerate(req) if bs}
            self.comm.exchange_lists(sends, got)
            for r, bs in enumerate(req):
                it = iter(got.get(r, []))
                for b in bs:
                    X[b] = [next(it) for _ in L.nbr[b]]
            del sends, got

    def _loads(self, nwin, pts):
        w = np.zeros(self.comm.size)
        for c, d in enumerate(pts):
            for r, q in d.items():
                w[r] += float(nwin[c]) ** 3 * len(q)
        return "/".join(f"{v:.2f}" for v in w / max(w.mean(), 1e-300))

    def drop_ghosts(self, L):
        if isinstance(L.Sb, FineMat):
            self.fb.drop_ghosts()
        elif self.dd and L.Sb is not None:
            for X in (L.Sb, L.Hb):
                for b in range(len(L.r)):
                    if L.owner[b] != self.comm.rank:
                        X[b] = None

    def sigma_points(self, zs):
        """The lead Sigma of the contour points: computed by the ranks in turn, stored in the Sigma cache. Collective."""
        if not self.dd or not self.dev.sigma_cache:
            return
        for z in zs[self.comm.rank::self.comm.size]:
            self.dev.sigma(z)
        self.comm.barrier()

    def pair_blocks(self, R, i):
        """The rows of core i (COO, device columns) cut into [CSR R_ij for j in nbr[i]] (None: no nonzero): the nonzeros
        grouped by the core of their column (radix sort of int16 core labels), every nonzero touched once."""
        nw, cores = self.nw, self.cores
        lc = self.lab16[R.col]
        o = np.argsort(lc, kind="stable")
        lc, rr, cc, vv = lc[o], R.row[o], self.posin[R.col[o]], R.data[o]
        bnd = np.searchsorted(lc, np.arange(nw + 1))
        assert bnd[-1] == len(lc) and set(np.unique(lc)) <= set(self.nbr[i]), "M couples beyond the core graph"
        out = []
        for j in self.nbr[i]:
            s, e = bnd[j], bnd[j + 1]
            if e == s:
                out.append(None)
                continue
            # stable sort: inside the group the rows are still in CSR order -> indptr from row counts, no COO sort
            ip = np.concatenate([[0], np.cumsum(np.bincount(rr[s:e], minlength=len(cores[i])))])
            out.append(sp.csr_matrix((vv[s:e], cc[s:e], ip), shape=(len(cores[i]), len(cores[j]))))
        return out

    def core_blocks(self, M, U):
        """[[U_i^H M[core_i, core_j] U_j for j in nbr[i]] for i]: rows of core i only."""
        xp, nw, nbr, cores = self.xp, self.nw, self.nbr, self.cores
        blocks = [[None] * len(nbr[i]) if self.owner1[i] == self.comm.rank else None for i in range(nw)]
        M = M.tocsr()
        for i in self.part0.own:
            R = M[cores[i]].tocoo()                                     # rows of core i (row gather, cheap)
            for q, (j, Rij) in enumerate(zip(nbr[i], self.pair_blocks(R, i))):
                if Rij is None:
                    blocks[i][q] = self.keep(xp.zeros((U[i].shape[1], U[j].shape[1]), complex))
                    continue
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

    def sigma1(self, U, SL, SR, host=False):
        """{(i, j): U_i^H Sigma U_j} for the lead-face cores i, j (level-1 coordinates); host: the rows of the blocks of
        other ranks as numpy arrays."""
        xp, lab, posin = self.xp, self.lab, self.posin
        out = {}
        for Sg, b_ in ((SL, self.dev.bL), (SR, self.dev.bR)):
            Sg = xp.asarray(Sg)
            ks = [int(k) for k in np.unique(lab[b_])]
            sel = {k: np.where(lab[b_] == k)[0] for k in ks}
            Ub = {k: xp.asarray(U[k])[xp.asarray(posin[b_[sel[k]]])] for k in ks}
            for i in ks:
                for j in ks:
                    Sij = Sg[xp.asarray(sel[i])[:, None], xp.asarray(sel[j])[None, :]]
                    out[(i, j)] = out.get((i, j), 0) + Ub[i].conj().T @ (Sij @ Ub[j])
        if self.dd and host:                                     # blocks of the faces of other ranks: not on this GPU
            near = {k for k in self.lead_cores if self.owner1[k] == self.comm.rank}
            out = {ij: B if ij[0] in near else gpu.asnumpy(B) for ij, B in out.items()}
        return out

    def dense(self, L, blocks, rows, cols):
        """Dense [rows, cols] (block lists) of a block-sparse matrix of level L (blocks[i][q] <-> (i, L.nbr[i][q]))."""
        rows, cols = [int(v) for v in rows], [int(v) for v in cols]
        if isinstance(blocks, FineMat):                              # never stored: from the sparse fine blocks
            return self.fb.dense(L, blocks.name, rows, cols)
        orow, ocol = offs(L.r[rows]), offs(L.r[cols])
        D = self.xp.zeros((orow[-1], ocol[-1]), complex)
        loc = {k: q for q, k in enumerate(cols)}
        for i_, i in enumerate(rows):
            for j, Bk in zip(L.nbr[i], blocks[i]):
                if int(j) in loc:
                    D[orow[i_]:orow[i_ + 1], ocol[loc[int(j)]]:ocol[loc[int(j)] + 1]] = self.xp.asarray(Bk)
        return D

    def dense_A(self, L, E, rows, cols):
        """Dense [rows, cols] of E S - H (no Sigma) of level L; levels without stored blocks: from the sparse A_ij."""
        if isinstance(L.Sb, FineMat):
            self.fb.set_energy(E)
            return self.fb.dense(L, "A", [int(v) for v in rows], [int(v) for v in cols])
        D = E * self.dense(L, L.Sb, rows, cols)
        D -= self.dense(L, L.Hb, rows, cols)
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
            Z[:, o2[q]:o2[q + 1]] = Y[:, o1[q]:o1[q + 1]] @ self.xp.asarray(L.U[b])
        return Z

    def projop(self, L, w, M1):
        """M1 (dense, level-1 coordinates of the cores of the block list w, in order) -> level-L coordinates of w."""
        if L.k == 1:
            return M1
        T = self.projright(L, w, M1)
        del M1
        return self.projright(L, w, T.conj().T).conj().T

    @staticmethod
    def sub(M, idx):
        return M.sub(idx) if hasattr(M, "sub") else M[idx][:, idx]     # device store: rows of other ranks from the files

    def cap_of(self, L, w):
        """Projected CAP i s1 U^H (P S P) U of the window w (block list of level L) in level-L coordinates (None: the
        window has no artificial cut)."""
        xp, lab, posin = self.xp, self.lab, self.posin
        U1 = self.LV[1].U
        F = np.concatenate([self.cores[k] for k in self.cores_of(L, w)])
        rp = self.ramp(F)
        on_ = np.where(rp > 0)[0]
        if not len(on_):
            return None
        Fo = F[on_]
        Pr = sp.diags(np.sqrt(rp[on_]))
        Wf = (1j * self.s1 * (Pr @ self.sub(self.S, Fo) @ Pr)).tocsr()
        lf = lab[Fo]
        Wd = gpu.csr(Wf, np.complex128) if gpu.on else Wf
        # factored form: the CAP lives on the len(on_) ramp orbitals only: Y = their rows of the prolongation of the
        # window, one block of w at a time (block diagonal: a block reaches only the ramp orbitals of its own cores),
        # so no (ramp orbitals x all level-1 columns of the window) matrix is formed (12 nm, level 8: 5 GB and more)
        o2 = offs(L.r[[int(b) for b in w]])
        Y = xp.zeros((len(on_), o2[-1]), complex)
        for q, b in enumerate(w):
            cb = self.cores_of(L, [b])
            sel = {k: np.where(lf == k)[0] for k in cb}
            rows = np.concatenate([sel[k] for k in cb])
            if not len(rows):
                continue
            o, ro = offs([U1[k].shape[1] for k in cb]), offs([len(sel[k]) for k in cb])
            Uo = xp.zeros((len(rows), o[-1]), complex)
            for a_, k in enumerate(cb):
                if len(sel[k]):
                    Uo[ro[a_]:ro[a_ + 1], o[a_]:o[a_ + 1]] = xp.asarray(U1[k])[xp.asarray(posin[Fo[sel[k]]])]
            Y[xp.asarray(rows), o2[q]:o2[q + 1]] = self.projright(L, [b], Uo)
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
        ol = offs([o[q + 1] - o[q] for q in qs])
        lead_pos = {cw[q]: a_ for a_, q in enumerate(qs)}
        o2 = offs(L.r[[int(b) for b in w]])
        Y = xp.zeros((ol[-1], o2[-1]), complex)
        for c_, b in enumerate(w):                               # block diagonal: one block of w at a time
            cb = self.cores_of(L, [b])
            lb = [k for k in cb if k in lead_pos]
            if not lb:
                continue
            ob = offs([U1[k].shape[1] for k in cb])
            rows = np.concatenate([np.arange(ol[lead_pos[k]], ol[lead_pos[k] + 1]) for k in lb])
            cols = np.concatenate([np.arange(ob[q], ob[q + 1]) for q, k in enumerate(cb) if k in lead_pos])
            Ef = xp.zeros((len(rows), ob[-1]), complex)
            Ef[xp.arange(len(rows)), xp.asarray(cols)] = 1
            Y[xp.asarray(rows), o2[c_]:o2[c_ + 1]] = self.projright(L, [b], Ef)
        return [cw[q] for q in qs], offs([o[q + 1] - o[q] for q in qs]), Y

    def sig_of(self, fac, sg):
        """Sigma(z) of a window in its level coordinates from the level-1 blocks sg and the window's sig_factor."""
        ks, o, Y = fac
        M1 = self.xp.zeros((o[-1], o[-1]), complex)
        for a_, i in enumerate(ks):
            for b_, j in enumerate(ks):
                if (i, j) in sg:
                    M1[o[a_]:o[a_ + 1], o[b_]:o[b_ + 1]] = self.xp.asarray(sg[(i, j)])
        return Y.conj().T @ (M1 @ Y)

    # ------------------------------------------------------------------ basis store
    def basis_get(self, key, k):
        """(U blocks as host arrays, trace per block) of the level-k bases stored under key, or None."""
        if key is None:
            return None
        if key in self.ubank:
            return self.ubank[key]
        if self.basis_dir:
            self.comm.barrier()
            f = os.path.join(self.basis_dir, f"l{k}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz")
            if self.comm.bcast(os.path.exists(f)):                  # the same answer on every rank
                dist.wait_file(f)
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
        if self.basis_dir and self.comm.rank == 0:
            os.makedirs(self.basis_dir, exist_ok=True)
            f = os.path.join(self.basis_dir, f"l{k}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz")
            tmp = dist.tmpname(f)
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
        if self.rho_eig == "rand":
            tag += "_rr"
        EIG_STATS.clear()
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
        todo = [k for k in self.part0.own if kept(k) is None or kept(k) > keep * (1 + 1e-12)]
        on_lead = lambda k: np.isin(sw.idx[k], dev.bL).any() or np.isin(sw.idx[k], dev.bR).any()
        zs, dzs = contour_points(mu1, mu2, npts)
        if self.dd and any(self.comm.allgather(any(on_lead(k) for k in todo))):
            self.sigma_points(zs)
        if todo:
            t = time.time()
            H = dev.H.tocsr()
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
                caps = {}                                           # sparse S, H, CAP of the windows of this batch
                for k in ks:
                    idx = sw.idx[k]
                    P = sp.diags(np.sqrt(sw.ramp(k)))
                    Skk = self.S[idx][:, idx]
                    caps[k] = (Skk, H[idx][:, idx], (1j * sw.s * (P @ Skk @ P)).tocsr())
                acc = {k: (SchurRho(np.where(sw.core[k])[0], len(sw.idx[k]), core["prec"], core["refine"],
                                    core.get("probes", 0), seed=k, eig=self.rho_eig, cut=keep) if core else
                           xp.zeros((len(sw.idx[k]),) * 2, complex)) for k in ks}
                self.mem(f"window rho batch {bi + 1}/{len(batches)}: accumulators of {len(ks)} windows")
                lead = [k for k in ks if on_lead(k)]
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
                        ev, V = rand_top_eig(rho, keep) if self.rho_eig == "rand" else xp.linalg.eigh(rho)
                        del rho
                        o = xp.argsort(-ev)
                        ev, V = ev[o], V[:, o]
                        kp = gpu.asnumpy(ev) > keep
                        DM[k] = (gpu.asnumpy(ev)[kp],
                                 gpu.asnumpy(V[xp.asarray(np.where(sw.core[k])[0])][:, xp.asarray(kp)]))
                    if d:
                        tmp = dist.tmpname(f(k))
                        np.savez(tmp, occ=DM[k][0], vec=DM[k][1], keep=keep)
                        os.replace(tmp, f(k))
                    del V, ev
                    gpu.free()
                self.mem(f"window rho batch {bi + 1}/{len(batches)}: eigh")
                self.log(f"    batch {bi + 1}/{len(batches)}: {len(ks)} windows ({time.time() - t:.0f} s)")
            if EIG_STATS:
                st = np.array(EIG_STATS, float)
                self.log(f"    rand eig: n {int(st[:, 0].min())}..{int(np.median(st[:, 0]))}..{int(st[:, 0].max())}, "
                         f"subspace l {int(st[:, 1].min())}..{int(np.median(st[:, 1]))}..{int(st[:, 1].max())}, "
                         f"k > keep {int(st[:, 2].min())}..{int(np.median(st[:, 2]))}..{int(st[:, 2].max())}, "
                         f"dense fallback {int(st[:, 4].sum())} of {len(st)}")
                EIG_STATS.clear()
        for k in self.part0.own:
            if DM[k] is None:
                c = np.load(f(k))
                kp = c["occ"] > keep
                DM[k] = (c["occ"][kp], c["vec"][:, kp])
            assert DM[k][1].shape[0] == len(self.cores[k]), \
                f"window density matrix {k} in {d} belongs to other windows (core {DM[k][1].shape[0]} != {len(self.cores[k])})"
        self.log(f"  window density matrices" + (f" from {d}" if d else "") + ": trace "
                 + "/".join(f"{DM[k][0].sum():.0f}" for k in self.part0.own[:8]) + ("..." if nw > 8 else ""))
        return DM

    def basis1(self, DM, kind, cut):
        """Level-1 block bases from the window density matrices: dm = window modes above cut restricted to the core
        (SVD), dmc = eigenvectors of the core block rho_cc = Vc occ Vc^H above cut."""
        assert kind in ("dm", "dmc")
        xp = self.xp
        U = []
        for k, d_ in enumerate(DM):
            if d_ is None:                                          # window of another rank
                U.append(None)
                continue
            occ, Vc = d_
            if kind == "dm":
                u, sv, _ = xp.linalg.svd(xp.asarray(Vc[:, occ > cut]), full_matrices=False)
                U.append(u[:, :int((sv > 1e-8 * sv[0]).sum())] if len(sv) else u[:, :0])
            else:
                Vg = xp.asarray(Vc)
                ev, V = xp.linalg.eigh((Vg * xp.asarray(occ)[None, :]) @ Vg.conj().T)
                U.append(xp.ascontiguousarray(V[:, ev > cut][:, ::-1]))
                del Vg, V
        return self.share(U, self.owner1, self.G)

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
        self.make_part(L, self.owner1)
        self.mem("level 1: block bases U")
        if self.fine_blocks:
            assert A0 is not None, "fine_blocks: level 1 is free (A0 = the fine operator)"
            if self.fb is None:                                     # energy independent: once
                self.fb = FineBlocks(self)
                self.log(f"  sparse core-pair blocks of the own rows on the device: S + H {self.fb.gb:.2f} GB on this rank; "
                         f"levels 1..{self.fine_blocks} form S, H from them")
            self.fb.set_energy(E)
            L.Sb, L.Hb = FineMat("S"), FineMat("H")
            self.mem("level 1: sparse core-pair blocks")
        else:
            L.Sb, L.Hb = self.core_blocks(dev.S, U), self.core_blocks(dev.H, U)
            self.mem("level 1: S, H blocks")
        L.E = E
        free = self.free_levels >= 1 and A0 is not None
        host = not free and self.streams(sum(int(L.r[i]) * int(L.r[j]) for i in L.own for j in nbr[i]) * 16 / 2 ** 30)
        if self.dd:
            L.Ab, sig1 = None, self.sigma1(U, SL, SR, host=True)
        elif free:
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
        cores_g, own = self.cores_g, L.own
        n = self.part0.nloc
        lo = {k: int(v) for k, v in zip(own, offs(L.r[own]))}          # block -> first row of the local vector

        def P(c):
            y = xp.zeros((n, c.shape[1]), complex)
            for k in own:
                y[cores_g[k]] = U[k] @ c[lo[k]:lo[k] + L.r[k]]
            return y
        L.P = P
        L.PH = lambda r: xp.concatenate([U[k].conj().T @ xp.asarray(r)[cores_g[k]] for k in own]) if own else \
            xp.zeros((0, r.shape[1]), complex)
        if self.dd:
            L.A = DistOp(self, L, E, sig1, A0 if free else None, host)
            self.mem("level 1: A1 through the fine operator (no blocks)" if free else "level 1: A blocks")
        elif free:
            L.A = GalerkinOp(self, L, A0, E, sig1)
            self.mem("level 1: A1 through the fine operator (no blocks)")
        self.LV = {1: L}
        gb = sum(int(L.r[i]) * int(L.r[j]) for i in range(nw) for j in nbr[i]) * 16 / 2 ** 30
        L.info = dict(level=1, blocks=nw, N=L.N, per_block=L.r.tolist(), A_gb=gb)
        self.log(f"  level 1: {nw} blocks, N1 {L.N} (n / N1 {self.n / max(L.N, 1):.1f}), per block {L.r.min()}.."
                 f"{int(np.median(L.r))}..{L.r.max()}, A1 {gb:.2f} GB; build {time.time() - t:.0f} s; {self.memstr()}")
        return L

    # ------------------------------------------------------------------ further levels
    def couplings(self, L):
        """Log the coupling blocks of A_k(E), S_k of level L by block distance (x distance in blocks, same / other y).
        Returns {name: (blocks, |A| median, |A| max, |S| median, |S| max)}."""
        ix, iy = L.grid
        grp = {}
        for i in L.own:
            for q, j in enumerate(L.nbr[i]):
                key = (-1, False) if j == i else (int(abs(ix[i] - ix[j])), bool(iy[i] != iy[j]))
                grp.setdefault(key, []).append((nrm2(self.xp.asarray(L.Ab[i][q]) if L.Ab is not None else
                                                   L.E * self.xp.asarray(L.Sb[i][q]) - self.xp.asarray(L.Hb[i][q])),
                                              nrm2(self.xp.asarray(L.Sb[i][q]))))
        if self.dd:
            parts, grp = self.comm.allgather(grp), {}
            for d_ in parts:
                for k_, v_ in d_.items():
                    grp.setdefault(k_, []).extend(v_)
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
        mine = [c for c in range(len(wins)) if wins[c] is not None]   # windows of other ranks: None
        ex = dist.Exch(L.part, self.entries(L, {int(b) for c in mine for b in wins[c]}))
        ez = dist.Exch(L.part, self.entries(L, {int(b) for c in mine for b in writes[c]}))   # blocks written
        wins, writes = [wins[c] for c in mine], [writes[c] for c in mine]
        nb_ = lambda ks: int(sum(L.off[k + 1] - L.off[k] for k in ks))
        est = sum((nb_(wr) if self.l1_solver == "rows" else nb_(w)) * nb_(w) for w, wr in zip(wins, writes)) \
            * np.dtype(cdt).itemsize / 2 ** 30
        # rows of the inverse or the whole LU to pinned host memory window by window
        stream = self.stream_sm is not None and est > self.stream_sm
        for c, w in enumerate(wins):
            Aw, o = L.A.sub(np.asarray(w))
            Cw = self.cap_of(L, w)
            if Cw is not None:
                Aw += Cw
                del Cw
            loc = {int(k): q for q, k in enumerate(w)}
            pos_ = xp.asarray(np.concatenate([np.arange(o[loc[int(k)]], o[loc[int(k)] + 1]) for k in writes[c]]))
            rw = xp.asarray(np.concatenate([ex.pos[L.off[k]:L.off[k + 1]] for k in w]))
            out = xp.asarray(np.concatenate([ez.pos[L.off[k]:L.off[k + 1]] for k in writes[c]]))
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
                lu = gpu.lu_factor(Aw)
                ops.append((rw, pos_, out, gpu.lu_host(lu) if stream else lu))
                del lu
                gb += int(o[-1]) ** 2 * self.lub / 2 ** 30
                del Aw
            sizes.append(int(o[-1]))
            gpu.free()

        st = gpu.Streamer([op if isinstance(op, tuple) else (op,) for _, _, _, op in ops]) if stream else None

        def Sm(r):
            nloc = r.shape[0]
            r = ex(r)
            y = xp.zeros((nloc + ez.ng,) + r.shape[1:], r.dtype)
            if st is not None:
                def fn(c, v):
                    rw, pos_, out, _ = ops[c]
                    y[out] = (v[0] @ r[rw].astype(cdt) if pos_ is None else
                              gpu.lu_solve(gpu.lu_view(v), r[rw].astype(cdt))[pos_])
                st.run(fn)
            else:
                for rw, pos_, out, op in ops:
                    y[out] = op @ r[rw].astype(cdt) if pos_ is None else gpu.lu_solve(op, r[rw].astype(cdt))[pos_]
            return ez.rev(y, y[:nloc])
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
        dd, rank = self.dd, self.comm.rank
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
        cpl = {} if isinstance(L.Sb, FineMat) else self.couplings(L)
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
        owner = L.owner[k0]                                  # a merged block belongs to the owner of its first kid
        own = [c for c in range(nc) if owner[c] == rank]
        lv = f"level {Nn.k}"
        stored = self.basis_get(bkey, Nn.k)
        # contour points of the windows of overloaded ranks handed to the others (deep levels: few, unequal blocks)
        pts = [{int(owner[c]): list(range(int(npts)))} for c in range(nc)]
        if dd and stored is None and self.share_points and self.merge_eig != "sketch":
            pts = dist.share_points(nwin.astype(float) ** 3, owner, int(npts), self.comm.size)
        mine = [c for c in range(nc) if rank in pts[c]]                     # windows this rank integrates (part of)
        helped = [c for c in range(nc) if len(pts[c]) > 1]
        # S, H rows of the windows of own blocks (rho, smoother) and of the windows helped with (rho)
        self.ghost_rows(L, {int(b) for c in own for b in full[c]} | {int(b) for c in mine for b in win[c]})
        if stored is not None:
            near = (owner == rank) | (Nn.G[owner == rank].any(0) if own else False)
            Nn.U = [xp.asarray(u) if near[c] else u for c, u in enumerate(stored[0])]
            tr, batches = np.asarray(stored[1]), []
            for c in range(nc):
                assert Nn.U[c].shape[0] == int(L.r[list(Nn.kids[c])].sum()), "stored bases belong to another hierarchy"
            self.mem(f"level {Nn.k}: bases from the store")
        else:
            zs, dzs = contour_points(mu1, mu2, npts)
            isl = [bool(set(self.cores_of(L, w)) & self.lead_cores) for w in win]
            if dd and any(isl):
                self.sigma_points(zs)
            order = sorted(mine, key=lambda c: (not isl[c], c))
            part_acc = {}                                        # shared windows: contour sums of this rank (host)
            if helped:
                self.log(f"    {lv}: contour points shared over the ranks for {len(helped)} of {nc} windows; cost per "
                         "rank / mean " + self._loads(nwin, pts))
            batches, cur, m_ = [], [], 0.0
            est_keep = 0.0
            if isinstance(L.Sb, FineMat):                            # dense S, H pairs of the windows kept on the device
                est_keep = self.fb.need_gb(L, [win[c] for c in mine])
                free_gb = self.gpu_free_gb()
                if est_keep < 0.4 * free_gb:
                    self.fb.keep(L)
                    log(f"    level {Nn.k}: dense S, H pairs of level {L.k} kept on the GPU during the basis build: "
                        f"{est_keep:.1f} GB on this rank")
                else:
                    log(f"    level {Nn.k}: dense pairs of level {L.k} ({est_keep:.1f} GB) over 40% of the free GPU memory "
                        f"({free_gb:.1f} GB): formed per window")
                    est_keep = 0.0
            # per window of a batch: S_w, K_w, accumulator stay (3 n^2); the pencil and its inverse are transient (the
            # largest window once: + 3 n^2 with the LU workspace)
            lim = min(self.batch_gb, self.mem_frac * (self.gpu_free_gb() - est_keep)) - 3 * nwin.max() ** 2 * 16 / 2 ** 30
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
            shared_dat = {}

            def finish(c, acc, rows):
                """Basis of block c from its complete contour sum (SchurRho, or the window sum)."""
                if schur:                                        # eigenpairs of rho_PP directly (dmc)
                    occ, V = acc.result()
                    Nn.U[c] = xp.ascontiguousarray(V[:, occ > cut])
                    tr[c] = float(xp.real(occ.sum()))
                else:
                    rho = rho_of(acc)
                    Nn.U[c] = select(rho, rows, cut, kind, self.merge_eig)
                    tr[c] = float(xp.real(xp.trace(rho[rows][:, rows])))
                gpu.free()
            for bi, ks in enumerate(batches):
                if not ks:                                       # a rank without windows on this level
                    continue
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
                                             self.core.get("probes", 0), seed=1000 + c, eig=self.rho_eig, cut=cut)
                if self.merge_eig == "sketch":
                    self._sketch_batch(ks, dat, isl, sf, zs, dzs, U1, rows_of, cut, kind, Nn, tr, bt)
                    del sf, dat
                    continue
                for q, (z, dz) in enumerate(zip(zs, dzs)):
                    kz = [c for c in ks if q in pts[c][rank]]
                    if not kz:
                        continue
                    sg = self.sigma1(U1, *self.dev.sigma(z)) if any(isl[c] for c in kz) else {}
                    for c in kz:
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
                    if len(pts[c]) > 1:                          # finished below, after the sums of the other ranks
                        acc = dat[c][2]
                        part_acc[c] = [np.ascontiguousarray(gpu.asnumpy(v)) for v in (acc.parts() if schur else [acc])]
                        if schur:
                            acc.set_parts([None, None])
                        dat[c] = (acc, rows_of[c]) if schur else (None, rows_of[c])
                        del acc
                        if owner[c] == rank:
                            shared_dat[c] = dat[c]
                        dat[c] = None
                        gpu.free()
                        continue
                    finish(c, dat[c][2], rows_of[c])
                    dat[c] = None
                self.mem(f"{bt}: block bases from rho")
            for c in helped:                                     # contour sums of the helpers -> the owner (host)
                o_ = int(owner[c])
                src = [r for r in pts[c] if r != o_]
                if rank == o_:
                    got = {r: [np.empty_like(v) for v in part_acc[c]] for r in src}
                    self.comm.exchange_lists({}, got)
                    for r in src:
                        for v, g_ in zip(part_acc[c], got[r]):
                            v += g_
                    del got
                else:
                    self.comm.exchange_lists({o_: part_acc.pop(c)} if rank in src else {}, {})
            for c in helped:
                if owner[c] == rank:
                    xs = [xp.asarray(v) for v in part_acc.pop(c)]
                    acc, rows = shared_dat.pop(c)
                    if schur:
                        acc.set_parts(xs)
                    else:
                        acc = xs[0]
                    del xs
                    finish(c, acc, rows)
                    del acc
            Nn.U = self.share(Nn.U, owner, Nn.G)
            tr = self.comm.sum(tr)
            self.basis_put(bkey, Nn.k, Nn.U, tr)
        t_rho = time.time() - t0
        eigs = ""
        if (self.merge_eig != "full" or self.rho_eig == "rand") and EIG_STATS:
            st = np.array(EIG_STATS, float)
            eigs = (f"; {self.merge_eig if self.merge_eig != 'full' else 'rho rand'}{' ' + self.sketch_prec if self.merge_eig == 'sketch' and self.sketch_prec != 'c128' else ''} eig: window n {int(st[:, 0].min())}..{int(np.median(st[:, 0]))}..{int(st[:, 0].max())}, "
                    f"subspace l {int(st[:, 1].min())}..{int(np.median(st[:, 1]))}..{int(st[:, 1].max())}, "
                    f"k > cut {int(st[:, 2].min())}..{int(np.median(st[:, 2]))}..{int(st[:, 2].max())}, rounds "
                    f"{int(st[:, 3].max())} max, " + ("probes = window" if self.merge_eig == "sketch" else "dense fallback")
                    + f" {int(st[:, 4].sum())} of {len(st)}")
            EIG_STATS.clear()
        Nn.r = np.array([u.shape[1] for u in Nn.U])
        Nn.off = offs(Nn.r)
        Nn.N = int(Nn.off[-1])
        self.make_part(Nn, owner)
        gal = lambda X, f=lambda B: B: [[f(Nn.U[c].conj().T @ (self.dense(L, X, Nn.kids[c], Nn.kids[c2]) @
                                                               xp.asarray(Nn.U[c2]))) for c2 in Nn.nbr[c]]
                                        if owner[c] == rank else None for c in range(nc)]
        if Nn.k <= self.fine_blocks:
            Nn.Sb, Nn.Hb = FineMat("S"), FineMat("H")
        else:
            Nn.Sb, Nn.Hb = gal(L.Sb, self.keep), gal(L.Hb, self.keep)
        Nn.E = L.E
        nfree = Nn.k <= self.free_levels and is_free(L.A)
        if nfree or dd:                                      # no A_k blocks (or built below): Sigma of the lead blocks
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
                                M[ro[a_]:ro[a_ + 1], co[b_]:co[b_ + 1]] = xp.asarray(B)
                        Bn = xp.asarray(Nn.U[c]).conj().T @ (M @ xp.asarray(Nn.U[c2]))
                        sigN[(c, c2)] = Bn if owner[c] == rank else gpu.asnumpy(Bn)
        else:                                                # one rank
            Nn.Ab = [[Nn.U[c].conj().T @ (L.A.take(Nn.kids[c], Nn.kids[c2]) @ Nn.U[c2]) for c2 in Nn.nbr[c]]   # = gal(L.Ab)
                     for c in range(nc)]
            Nn.A = BlockOp(Nn.off, list(range(nc)), Nn.nbr, Nn.Ab,
                           host=self.streams(sum(int(Nn.r[i]) * int(Nn.r[j]) for i in range(nc) for j in Nn.nbr[i]) * 16 / 2 ** 30))
        if self.fb is not None:
            self.fb.release()                                       # basis (and the Galerkin blocks) of this level formed
        self.mem(f"{lv}: Galerkin S, H" + (" (A free)" if nfree else ", A blocks"))
        # P: the owned rows of level L from the blocks over them (a kid may belong to another rank than its block);
        # PH: the owned blocks of the new level from the rows of their kids
        below = [c for c in range(nc) if any(L.owner[k] == rank for k in Nn.kids[c])]
        eP = dist.Exch(Nn.part, self.entries(Nn, below))
        eH = dist.Exch(L.part, self.entries(L, {k for c in own for k in Nn.kids[c]}))
        lpos = np.full(L.N, -1, np.int64)
        lpos[L.part.idx] = np.arange(L.part.nloc)
        kent = lambda c: np.concatenate([np.arange(L.off[k], L.off[k + 1]) for k in Nn.kids[c]])
        rows1 = {c: xp.asarray(eH.pos[kent(c)]) for c in own}
        up = []                                              # (rows of y, U block restricted to them, rows of the coefficients)
        for c in below:
            pk, Uc = lpos[kent(c)], xp.asarray(Nn.U[c])
            part_ = pk >= 0
            up.append((xp.asarray(pk[part_]), Uc if part_.all() else Uc[xp.asarray(np.where(part_)[0])],
                       int(eP.pos[Nn.off[c]])))

        def P(c2):
            y = xp.zeros((L.part.nloc, c2.shape[1]), complex)
            c2 = eP(c2)
            for ry, Uc, a_ in up:
                y[ry] = Uc @ c2[a_:a_ + Uc.shape[1]]
            return y
        Nn.P = P

        def PH(r1):
            r1 = eH(r1)
            return xp.concatenate([Nn.U[c].conj().T @ r1[rows1[c]] for c in own]) if own else \
                xp.zeros((0, r1.shape[1]), complex)
        Nn.PH = PH
        if dd:
            Nn.A = DistOp(self, Nn, Nn.E, sigN, L.A if nfree else None, host=not nfree and self.streams(
                sum(int(Nn.r[i]) * int(Nn.r[j]) for i in own for j in Nn.nbr[i]) * 16 / 2 ** 30))
        elif nfree:
            Nn.A = GalerkinOp(self, Nn, L.A, Nn.E, sigN)
        t = time.time()
        # the level below is complete: its S, H blocks and the block list of A are not needed again (L.A keeps the
        # block rows for the matvec, dense() and sub()). Freed before the smoother of L.A is built: 3 x the size of
        # A_L less at the memory peak (21600a y-cut: 33 GB, out of memory on a GH200 otherwise)
        if dd:                                               # the windows below are assembled from S, H
            Nn.Sbelow = self.ras(L, [w if owner[c] == rank else None for c, w in enumerate(wsm)], Nn.kids)
            self.drop_ghosts(L)
            if L.N > self.keep_max and not isinstance(L.Sb, FineMat):
                L.Sb = L.Hb = None
        if not is_free(L.A):                  # a free A_1 keeps reading S1, H1 (smoother, SlabLU)
            L.Sb = L.Hb = None
        L.Ab = L.A.blocks = None
        gpu.free()
        self.mem(f"{lv}: S, H blocks and the A block list of level {L.k} freed")
        if not dd:
            Nn.Sbelow = self.ras(L, wsm, Nn.kids)            # smoother of L.A formed with the blocks of this level
        gpu.free()
        self.mem(f"{lv}: smoother of A{L.k} ({len(wsm)} windows)")
        np_ = np.array([int(L.r[list(ks)].sum()) for ks in Nn.kids])
        nsm = np.array([v for d_ in self.comm.allgather(Nn.Sbelow.sizes) for v in d_])      # all windows (rank by rank)
        smgb = sum(self.comm.allgather(Nn.Sbelow.gb))
        gb = sum(int(Nn.r[i]) * int(Nn.r[j]) for i in range(nc) for j in Nn.nbr[i]) * 16 / 2 ** 30
        allh = lambda h: f"{h}{' = all' if h == hmax else ''}"
        log(f"  level {Nn.k} ({axis} merge, {kind} cut {cut:g}, E {mu1:+.4f}..{mu2:+.4f}): {nc} blocks, N{Nn.k} {Nn.N} "
            f"(N{L.k} / N{Nn.k} {L.N / max(Nn.N, 1):.2f}, n / N{Nn.k} {self.n / max(Nn.N, 1):.1f}), per block "
            f"{Nn.r.min()}..{int(np.median(Nn.r))}..{Nn.r.max()} of {np_.min()}..{int(np.median(np_))}..{np_.max()} "
            f"level-{L.k} columns; rho windows (halo {allh(hu)}): {nwin.min()}..{int(np.median(nwin))}..{nwin.max()} "
            f"columns; smoother of A{L.k} (halo {allh(hs)}): {nsm.min()}..{int(np.median(nsm))}..{nsm.max()} columns, "
            f"{self.l1_solver} {smgb:.2f} GB (LUs {(nsm.astype(float) ** 2).sum() * self.lub / 2 ** 30:.2f} GB); "
            f"A{Nn.k} {gb:.2f} GB; trace per block {tr.min():.1f}..{np.median(tr):.1f}..{tr.max():.1f}; rho "
            + (f"{t_rho:.0f} s in {len(batches)} batches" if batches else f"from the store ({t_rho:.0f} s)") + eigs + f", smoother {time.time() - t:.0f} s; {self.memstr()}")
        Nn.info = dict(level=Nn.k, axis=axis, kind=kind, cut=cut, window=[mu1, mu2], npts=int(npts), blocks=nc,
                       N=Nn.N, per_block=Nn.r.tolist(), halo_max=hmax, halo_rho=hu, halo_smoother=hs,
                       rho_windows=nwin.tolist(), smoother_windows=nsm.tolist(), smoother_gb=smgb, A_gb=gb,
                       halo_cost={str(h): cost[h] / max(cost[hmax], 1) for h in cost}, couplings_below=cpl,
                       trace=tr.tolist(), t_rho=t_rho, rho_batches=len(batches), stored=not batches)
        self.LV[Nn.k] = Nn
        return Nn

    # ------------------------------------------------------------------ cycles
    def exact(self, m):
        """Bottom solver: exact solve with A_m by the block-tridiagonal LU over x slabs (SlabLU), no dense LU."""
        L = self.LV[m]
        if self.dd and self.bottom == "dist":
            return DistSlabLU(L.A, L.grid[0], L.part)
        lu = SlabLU(L.A, L.grid[0])
        return GatherSolve(lu, L.part) if self.dd else lu

    def own_smoother(self, m=None):
        """RAS on A_m over its own blocks (block + coupled blocks, projected CAP): the smoother of the deepest level."""
        L = self.LV[m or len(self.LV)]
        self.ghost_rows(L, {int(j) for b_ in L.own for j in L.nbr[b_]})
        return self.ras(L, [L.nbr[b_] if L.owner[b_] == self.comm.rank else None for b_ in range(len(L.r))],
                        [[b_] for b_ in range(len(L.r))])

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
        if self.fb is not None:
            self.fb.drop_ghosts()
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
    p.add_argument("--stream-smoother-gb", type=float, default=None, help="threshold of the smoothers (fine window rows and "
                   "level window rows) alone; 0: always streamed from pinned host memory (default: --stream-gb)")
    p.add_argument("--l1-free", action="store_true", help="no stored level-1 operator A1: A1 @ y = P^H A P y through "
                   "the fine operator, windows from the S1, H1 blocks (6 nm: ~33 GB of A1 blocks; implies --host-blocks)")
    p.add_argument("--free-levels", type=int, default=0, help="levels 1 .. K without stored A_k (A_k @ y = P^H A_{k-1} "
                   "P y, recursively; --l1-free = 1; implies --host-blocks)")
    p.add_argument("--fine-blocks", type=int, default=0, help="levels 1 .. K without stored S_k, H_k: the fine S, H are kept "
                   "as sparse core-pair blocks (CSR) on the GPU and S_k, H_k are formed, dense, only for the windows, Galerkin "
                   "products and LUs that need them (implies --free-levels K)")
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
    p.add_argument("--rho-eig", default="full", choices=["full", "rand"], help="eigenpairs of the level-1 window rho / "
                   "Schur core block (> --dm-keep) and the Schur pair blocks (> cut): dense eigh or the randomized range "
                   "finder (rand_top_eig; dense past 0.6 n). Window-DM cache in <tag>_rr...")
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
    p.add_argument("--bottom", default="dist", choices=["dist", "repl"], help="--dist: exact solve of the deepest level "
                   "by domains of slabs over the ranks (dist) or by the same LU on every rank (repl)")
    p.add_argument("--no-share-points", action="store_true", help="--dist: every merge window integrated by its owner only "
                   "(default: the contour points of overloaded ranks are shared)")
    p.add_argument("--dev-store", default="", help="device store of --inputs (python -m qtx_ms.devstore): with --dist every "
                   "rank reads only the rows of H, S of its own windows; folded-lead OBC")
    p.add_argument("--dist", default="", choices=["", "mpi", "nccl", "auto"], help="domain decomposition over the MPI "
                   "ranks (one per GPU; mpiexec / srun): window rho, pair rho, level blocks and Schwarz solves on their "
                   "owner, distributed vectors; GPU halos through NCCL or host-staged MPI (qtx_ms.dist). --sweep exact only")
    p.add_argument("--out", help="json with the level sizes and the iteration counts (merged across runs)")
    a = p.parse_args(argv)
    a.free_levels = max(a.free_levels, int(a.l1_free), a.fine_blocks)
    a.host_blocks = a.host_blocks or a.free_levels > 0
    comm = dist.launch(a.dist)
    dd = comm.size > 1
    if dd and comm.rank:
        f_ = os.environ.get("QTX_RANK_LOG")                      # log of the other ranks: <QTX_RANK_LOG>.<rank>
        sys.stdout = open(f"{f_}.{comm.rank}", "w", buffering=1) if f_ else open(os.devnull, "w")
    if a.gpu:
        ndev = 0
        if dd:
            import cupy
            ndev = cupy.cuda.runtime.getDeviceCount()
        gpu.enable(device=comm.local_rank % ndev if dd else None)
    else:
        gpu.from_env()
    if dd:
        assert a.sweep == "exact", "--dist: --sweep exact (the slow modes of top / fine are not distributed)"
        if gpu.on:
            comm.start_nccl()
        log(f"domain decomposition: {comm.size} ranks ({comm.local_size} on this node), GPU halos through "
            f"{'NCCL' if comm.nccl is not None else 'MPI (host)'}")
    xp = _xp()
    log(f"backend: {gpu.describe()}")
    gpu.track_peak()
    t00 = time.time()
    name = os.path.basename(os.path.normpath(a.inputs))
    if a.dev_store:                                          # sliced H, S (qtx_ms.devstore): rows read after the windows
        inp, dev = devstore.load(a.dev_store, source=a.inputs)
    else:
        inp = devstore.read_any(a.inputs)                         # or a release prep (needs PYTHONPATH=release/qtx_modespace/src)
        dev = mb.Device(inp, obc=a.obc)
    if name == "prep" and os.path.exists(os.path.join(a.inputs, "prep.json")):                                            # <dev>/prep -> <dev>
        name = os.path.basename(os.path.dirname(os.path.normpath(a.inputs)))
    dev.sigma_cache = a.sigma_cache or os.path.join("results_ms", "sigma", name)
    n = dev.n
    Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
    w_, s_, d_ = (float(v) for v in a.cap.split(","))
    split = [int(v) for v in a.cap_split.split(",")] if a.cap_split else None
    log(f"{name}: device {n} orbitals, slabs {dev.ns}; Ec {Ec:.4f} eV")
    sw = CapSchwarz(dev, inp, w_, s_, d_, spacing=a.cap_spacing, solver=a.cap_solver, workers=a.workers, split=split,
                    tov=a.cap_tov, stream_gb=a.stream_gb if a.stream_smoother_gb is None else a.stream_smoother_gb)
    if dd:
        sw.distribute(comm)
        log("  windows per rank " + "/".join(str(int(v)) for v in np.bincount(sw.owner, minlength=comm.size))
            + ", halo orbitals of the windows per rank " + "/".join(str(v) for v in comm.allgather(sw.ex.ng)))
    if a.dev_store:
        t = time.time()
        if dd:
            comm.barrier()
            dev.hold(np.concatenate([sw.idx[k] for k in sw.own]))
            q = comm.allgather((len(dev.H.M.indptr) - 1, dev.H.gb + dev.S.gb, devstore.hostmem()["anon"]))
            log(f"  device store {a.dev_store}: rows held per rank " + "/".join(f"{v[0] / n:.3f}" for v in q) + " n, H + S "
                + "/".join(f"{v[1]:.2f}" for v in q) + " GiB, private host memory " + "/".join(f"{v[2]:.1f}" for v in q)
                + f" GiB ({time.time() - t:.0f} s)")
        else:
            dev.hold(None)
    core = None
    if a.schur:
        assert a.l1.startswith("dmc:"), "--schur: dmc only (the core / pair block of rho; dm needs the window modes)"
        core = dict(prec=a.schur_prec, refine=a.schur_refine, probes=a.schur_probes)
    h = Hierarchy(dev, inp, sw, l1_cap=tuple(float(v) for v in a.l1_cap.split(",")) if a.l1_cap else None,
                  l1_ramp=a.l1_ramp, l1_lu=a.l1_lu, l1_solver=a.l1_solver, batch_gb=a.batch_gb, mem_frac=a.mem_frac,
                  halo_smooth=a.halo_smooth, host_blocks=a.host_blocks, stream_gb=a.stream_gb, l1_free=a.free_levels,
                  merge_eig=a.merge_eig, sketch_prec=a.sketch_prec, core=core,
                  rho_eig=a.rho_eig, comm=comm, keep_max=a.exact_max,
                  bottom=a.bottom, share_points=not a.no_share_points, fine_blocks=a.fine_blocks, stream_smoother_gb=a.stream_smoother_gb)
    part0 = h.part0
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
        f"|schur {a.schur_prec} {a.schur_refine} p{a.schur_probes} k{a.dm_keep:g}" if a.schur else "") + (
        "|rho rand" if a.rho_eig == "rand" else "")
    for axis, cut, m1, m2, npts, halo in ups:                    # key of each level = everything its rho depends on
        chain += (f"|{axis}:{cut:g}:{m1:g}:{m2:g}:{npts or a.npts}:h{'all' if halo is None else halo}"
                  + ("" if a.merge_eig == "full" else f"|eig {a.merge_eig}")
                  + (f"|{a.sketch_prec}" if a.merge_eig == "sketch" and a.sketch_prec != "c128" else ""))
        keys.append(chain if a.basis_cache != "none" else None)
    DM = U1 = None
    for dE in (float(v) for v in a.dE.split(",")):
        E = Ec + dE
        if dd and dev.sigma_cache:                               # rank 0 first, the others read the cache
            if comm.rank == 0:
                dev.sigma(E + 1j * a.eta)
            comm.barrier()
        SL, SR = dev.sigma(E + 1j * a.eta)
        B = dev.inject(SL)
        b = np.zeros((n, B.shape[1]), complex)
        b[dev.bL] = B
        nsel = min(a.ncol, b.shape[1]) if a.ncol else b.shape[1]
        if dd:                                                   # Sigma of rank 0 for all (decimation round-off)
            SL, SR = comm.bcast(SL), comm.bcast(SR)
            b = comm.bcast(b)
        need_A = not dd or (comm.rank == 0 and (a.exact or a.psi_dir))
        A = dev.A(E, SL, SR).tocsr() if need_A else None
        X = None
        if dd:                                                   # the exact solution on rank 0, before its GPU is filled
            if need_A:
                X = load_psi(a.psi_dir, E, A, b, dev.bL)
                if X is None and a.exact:
                    X, _ = mb.exact_solve(dev, A, b)
                A = None
                gc.collect()
                gpu.free()
            if a.exact or a.psi_dir:
                X = comm.bcast(X) if comm.bcast(X is not None) else None
            A = dist.FineA(dev, E, SL, SR)
        # the fine operator also carries every A1 product with --l1-free (5 uses per outer iteration): kept resident
        if dd:
            Aop = dist.FineOp(A, part0, None if a.free_levels else a.stream_gb, a.threads)
        else:
            Aop = (operator(A, stream_gb=None if a.free_levels else a.stream_gb) if gpu.on else operator(A, a.threads))
        GR = xp.asarray(1j * (SR - SR.conj().T))
        bR = xp.asarray(dev.bR)
        rR = np.where(part0.gown[dev.bR] == comm.rank)[0]        # last-slab rows of this rank
        lR, rR = xp.asarray(part0.gloc[dev.bR[rR]]), xp.asarray(rR)

        def Tof(y):
            yR = xp.zeros((len(dev.bR), y.shape[1]), complex)
            yR[rR] = y[lR]
            yR = comm.sum(yR)
            return float(xp.real(xp.einsum("ic,ij,jc->", yR.conj(), GR, yR)))

        def nrm(y):
            return float(np.sqrt(comm.sum(np.asarray(float(xp.linalg.norm(y)) ** 2))))
        if not dd:
            X = load_psi(a.psi_dir, E, A, b, dev.bL)
            if X is None and a.exact:
                X, _ = mb.exact_solve(dev, A, b)
        if X is not None:
            X = part0.local(X)
            X = xp.asarray(X[:, -nsel:])                     # injection columns ascend in Gamma eigenvalue: last = strongest
        bg = xp.asarray(part0.local(b)[:, -nsel:])
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
                                          cb=cb, comm=comm)
                xs.append(xk), its.append(np.asarray(ik)), res.append(np.asarray(rk))
                gpu.free()
            xs, its, res = xp.concatenate(xs, axis=1), np.concatenate(its), np.concatenate(res)
            dt = time.time() - t
            ep = nrm(xs - X) / nrm(X) if X is not None else np.nan
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
        if a.dev_store and dd and a.fine_blocks and dE == float(a.dE.split(",")[-1]):   # the fine blocks are on the device
            dev.drop()
            gc.collect()
            h.mem("device store: rows of H, S dropped (sparse core-pair blocks on the GPU)")
        for (axis, cut, m1, m2, npts, halo), key in zip(ups, keys):
            h.merge(axis, cut, kind, Ec + m1, Ec + m2, npts or a.npts, halo, key)
        rec["t_levels"] = time.time() - tb
        if a.dev_store and dd and not a.fine_blocks and dE == float(a.dE.split(",")[-1]):   # last energy: CAP rows still come from the files
            h.log(f"    device store: {dev.S.nread} S rows read in total on this rank")
            dev.drop()
            gc.collect()
            h.mem("device store: rows of H, S dropped")
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
                if h.fb is not None and dE == float(a.dE.split(",")[-1]) and not any(
                        h.LV[k].N <= a.exact_max for k in range(m + 1, nl + 1)) and "top" not in sweeps:
                    h.fb.free()                              # last LU factored: the solve needs no S_k, H_k
                h.mem(f"slab LU of A{m} (N {h.LV[m].N}): {len(bot.sizes)} slabs of {min(bot.sizes)}..{max(bot.sizes)}, "
                      f"{bot.gb:.2f} GB")
                c = bg                                           # plain mode space: psi = P A_m^-1 P^H b, no smoother
                for k in range(1, m + 1):
                    c = h.LV[k].PH(c)
                c = bot(c)
                for k in range(m, 0, -1):
                    c = h.LV[k].P(c)
                Tm = Tof(c)
                em = nrm(c - X) / nrm(X) if X is not None else np.nan
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
        if dd and gpu.on:
            pk = comm.allgather((max((m_["peak"] for m_ in h.memlog), default=0.0),
                                 max((m_["pool"] for m_ in h.memlog), default=0.0)))
            rec["peak_gb_per_rank"] = [float(v) for v, _ in pk]
            log("  GPU peak per rank (GB, live / pool): " + ", ".join(f"{u:.1f} / {v:.1f}" for u, v in pk))
        h.free()
        sw.free()
        del X, bg, GR, Aop, A, S0, A0
        gc.collect()
        gpu.free()
        if a.out and comm.rank == 0:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            res = json.load(open(a.out)) if os.path.exists(a.out) else {"E": {}}
            res["Ec"] = Ec
            res["E"][f"{dE:.4f}"] = rec
            json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"done, {time.time() - t00:.0f} s")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e_:
        m_ = gpu.mem_mark()                                          # the step that failed: its peak before the error
        if m_:
            log("    mem | failed step: " + ", ".join(f"{k} {v / 2 ** 30:.2f}" for k, v in m_.items()) + " GB")
        if "mpi4py.MPI" in sys.modules and sys.modules["mpi4py.MPI"].COMM_WORLD.Get_size() > 1 and not (
                isinstance(e_, SystemExit) and not e_.code):
            import traceback
            traceback.print_exc()                                # the other ranks wait in a collective: stop them all
            sys.stdout.flush(), sys.stderr.flush()
            sys.modules["mpi4py.MPI"].COMM_WORLD.Abort(1)
        raise
