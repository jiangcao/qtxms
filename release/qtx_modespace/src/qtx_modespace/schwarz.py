"""Iterative QTBM solve: flexible GMRES with a CAP-Schwarz smoother and, optionally, the mode-space coarse level.

CAP-Schwarz smoother (restricted additive Schwarz along transport): windows of `width` A centred on the slab
interfaces and between them (centres "all": one centre between neighbouring interfaces, or as many as needed for a
distance <= spacing; "ends": the interfaces + the two lead-face windows), every device orbital belongs to the core of
its nearest centre. Local solve on window j: LU of
    A_jj + i s P S_jj P,   P^2 = quadratic ramp 0 -> 1 over the outer d A towards each artificial cut,
i.e. an absorbing layer on the cuts (the lead faces are real boundaries and keep Sigma, which is in A). The
correction of window j is kept on its core only. split = pieces per periodic transverse axis: every x-window is also
cut into boxes of period / n (+ tov A on each side, minimum image) with the same ramp on the transverse cuts.
Solver dense (complex128 LU), dense64 (complex64 LU and solves; same convergence, half the factor time) or pardiso.

Two-level preconditioner (one multiplicative cycle from z = 0):
    pre x [z += S(r - A z)];  z += P_c (r - A z);  post x [z += S(r - A z)],   P_c = U A_c^-1 U^H
(A_c = Galerkin mode-space matrix, optionally at E + i coarse_eta: damps narrow spurious mode-space resonances).
Outer: right-preconditioned flexible GMRES on all injected channels at once (one Krylov space per column, the
smoother and coarse solve act on the block of active columns), x0 = P_c b (the mode-space solution) or 0 without
coarse level (one-level: preconditioner = the smoother alone).
"""
import time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor

from .core import CSolver, log
from .level1 import periodic_axes


class CapSchwarz:
    def __init__(self, dev, inp, width=16.0, s=2.0, d=2.0, spacing=None, centres="all", split=None, tov=2.5,
                 solver="dense64", workers=1, log=log):
        assert centres in ("all", "ends") and solver in ("dense", "dense64", "pardiso")
        ax = inp["meta"]["axis"]
        atom = np.repeat(np.arange(len(inp["norb"])), inp["norb"])[dev.dev]
        self.x = x = inp["xyz"][atom, ax]
        sl = [np.arange(dev.sb[k], dev.sb[k + 1]) for k in range(len(dev.ns))]
        xi = [0.5 * (x[sl[k]].max() + x[sl[k + 1]].min()) for k in range(len(sl) - 1)]   # slab interfaces
        anchors = [x.min()] + xi + [x.max()]
        fill = []
        for lo, hi in zip(anchors[:-1], anchors[1:]):
            m = max(1, int(np.ceil((hi - lo) / spacing)) - 1) if spacing else 1
            fill += [lo + (hi - lo) * (k + 1) / (m + 1) for k in range(m)]
        self.centres = np.sort(np.array(xi + (fill if centres == "all" else [fill[0], fill[-1]])))
        near = np.argmin(np.abs(x[:, None] - self.centres[None, :]), axis=1)
        self.idx = [np.where((np.abs(x - c) < width / 2) | (near == k))[0] for k, c in enumerate(self.centres)]
        self.core = [near[i] == k for k, i in enumerate(self.idx)]
        self.tcut = [[] for _ in self.idx]                       # per window: (|u - box centre|, half width) per cut
        for (axt, L), n in zip(periodic_axes(inp["meta"].get("periodic")), split or []):
            if n <= 1:
                continue
            u = inp["xyz"][atom, axt]
            idx, core, tcut = [], [], []
            for i, c, tc in zip(self.idx, self.core, self.tcut):
                for j in range(n):
                    du = np.abs((u[i] - (j + 0.5) * L / n + L / 2) % L - L / 2)
                    w = du < L / (2 * n) + tov
                    idx.append(i[w])
                    core.append(c[w] & ((du[w] < L / (2 * n)) | ((du[w] == L / (2 * n)) & (j == 0))))
                    tcut.append([(dv[w], h) for dv, h in tc] + [(du[w], L / (2 * n) + tov)])
            self.idx, self.core, self.tcut = idx, core, tcut
        cov = np.zeros(len(x), int)
        for i, c in zip(self.idx, self.core):
            cov[i[c]] += 1
        assert (cov == 1).all(), "window cores must partition the device"
        self.s, self.d, self.solver, self.dev = s, d, solver, dev
        if workers > 1:                  # concurrent LAPACK calls crash a multi-threaded MKL: 1 BLAS thread per window
            try:
                from threadpoolctl import threadpool_limits
                self.limit = lambda: threadpool_limits(1)
            except ImportError:
                log("  CAP-Schwarz: threadpoolctl not installed, windows run one after another (--workers 1)")
                workers = 1
        self.pool = ThreadPoolExecutor(workers) if workers > 1 else None
        self.ops, self.twall = None, 0.0
        log(f"CAP-Schwarz: {len(self.idx)} windows ({len(self.centres)} along x at "
            f"{', '.join(f'{c:.1f}' for c in self.centres)} A, width {width} A"
            + (f", transverse split {split} +{tov} A" if split and max(split) > 1 else "")
            + f"), {min(map(len, self.idx))}..{max(map(len, self.idx))} orbitals, absorbing {s} eV over {d} A, "
            f"{solver} LU, {workers} workers")

    def set_energy(self, A, dE0=0.0, log=log):
        """Factor the windows of A (at E + dE0 if dE0: A_jj + dE0 S_jj, for reuse over an energy interval)."""
        S, x = self.dev.S.tocsr(), self.x
        A = A.tocsr()
        xmin, xmax = x.min(), x.max()

        def factor(k):
            idx = self.idx[k]
            xi = x[idx]
            lo, hi = xi.min(), xi.max()
            ramp = np.zeros(len(idx))
            if lo > xmin + 1e-6:                                 # artificial cut on the left
                ramp = np.maximum(ramp, np.clip((lo + self.d - xi) / self.d, 0, 1) ** 2)
            if hi < xmax - 1e-6:
                ramp = np.maximum(ramp, np.clip((xi - (hi - self.d)) / self.d, 0, 1) ** 2)
            for du, h in self.tcut[k]:
                ramp = np.maximum(ramp, np.clip((du - (h - self.d)) / self.d, 0, 1) ** 2)
            P = sp.diags(np.sqrt(ramp))
            Sjj = S[idx][:, idx]
            Ajj = A[idx][:, idx] + dE0 * Sjj + 1j * self.s * (P @ Sjj @ P)
            if self.solver == "pardiso":
                return CSolver(Ajj, n_ref=0).factor_solve
            dt = np.complex64 if self.solver == "dense64" else complex
            lu = sla.lu_factor(Ajj.toarray().astype(dt), check_finite=False, overwrite_a=True)
            return lambda rj: sla.lu_solve(lu, rj.astype(dt), check_finite=False).astype(complex)
        t = time.time()
        ks = range(len(self.idx))
        if self.pool:
            with self.limit():
                self.ops = list(self.pool.map(factor, ks))
        else:
            self.ops = list(map(factor, ks))
        self.twall = time.time() - t
        log(f"  CAP-Schwarz factor: {self.twall:.1f} s")

    def __call__(self, r):
        z = np.zeros_like(r)
        solve = lambda k: self.ops[k](r[self.idx[k]])[self.core[k]]
        ks = range(len(self.idx))
        if self.pool:
            with self.limit():
                out = list(self.pool.map(solve, ks))
        else:
            out = list(map(solve, ks))
        for k, zk in zip(ks, out):
            z[self.idx[k][self.core[k]]] = zk
        return z

    def free(self):
        self.ops = None


class FastOp:
    """A @ X for a sparse complex A and a block X, faster than scipy's complex CSR product: A = Ar + i Ai with real
    Ar, Ai (Ai holds only the complex entries, e.g. the lead self-energy blocks), real products on [Re X, Im X], rows
    split over `threads` threads (scipy releases the GIL). MoS2/HfO2, 32M nonzeros, 64 columns: 2.8 s -> 0.3 s (8)."""

    def __init__(self, A, threads=4):
        A = sp.csr_matrix(A)
        self.shape, self.n = A.shape, A.shape[0]
        Ar = sp.csr_matrix(A.real, copy=True)            # copies: A.real / A.imag share A's index arrays, which
        Ai = (sp.csr_matrix(A.imag, copy=True) if np.iscomplexobj(A.data)   # eliminate_zeros changes in place
              else sp.csr_matrix(A.shape))
        Ar.eliminate_zeros(); Ai.eliminate_zeros()
        k = max(1, int(threads))
        rows = np.linspace(0, self.n, k + 1).astype(int)
        self.parts = [(rows[i], rows[i + 1], Ar[rows[i]:rows[i + 1]], Ai[rows[i]:rows[i + 1]] if Ai.nnz else None)
                      for i in range(k)]
        self.pool = ThreadPoolExecutor(k) if k > 1 else None

    def __matmul__(self, X):
        X = np.asarray(X)
        one = X.ndim == 1
        X = X.reshape(self.n, -1)
        nc = X.shape[1]
        Xs = np.hstack([X.real, X.imag]) if np.iscomplexobj(X) else np.hstack([X, np.zeros_like(X)])
        Y = np.empty((self.n, nc), complex)

        def part(p):
            a, b, Ar, Ai = p
            P = Ar @ Xs
            yr, yi = P[:, :nc], P[:, nc:]
            if Ai is not None:
                Q = Ai @ Xs
                yr, yi = yr - Q[:, nc:], yi + Q[:, :nc]
            Y[a:b].real, Y[a:b].imag = yr, yi
        list(self.pool.map(part, self.parts)) if self.pool else [part(p) for p in self.parts]
        return Y[:, 0] if one else Y


def fgmres_block(A, b, x0, prec, tol=1e-10, m=150, cb=None):
    """Flexible GMRES on all columns of b: one Krylov space and Hessenberg matrix per column, A and prec applied to the
    block of the active columns. A column stops when its GMRES residual estimate is <= tol |b_j|. Returns x,
    iterations per column, true relative residuals. cb(i, est) after iteration i (estimates of all columns).
    Storage per column (V[j], Z[j]: (m + 1) x n, rows = basis vectors): the Gram-Schmidt step is one GEMV pair per
    column on contiguous memory (no gather of the whole Krylov basis)."""
    n, nc = b.shape
    r0 = b - A @ x0
    beta, bn = np.linalg.norm(r0, axis=0), np.linalg.norm(b, axis=0)
    V = np.zeros((nc, m + 1, n), complex)
    Z = np.zeros((nc, m, n), complex)
    Hh = np.zeros((nc, m + 1, m), complex)
    V[:, 0] = (r0 / np.where(beta > 0, beta, 1)).T
    its = np.zeros(nc, int)
    est = beta / bn
    act = np.where(beta > tol * bn)[0]
    Y = [np.zeros(0, complex) for _ in range(nc)]
    for i in range(m):
        if len(act) == 0:
            break
        zb = prec(np.ascontiguousarray(V[act, i].T))
        Z[act, i] = zb.T
        w = np.asarray(A @ zb)
        keep = []
        for jj, j in enumerate(act):
            wj = w[:, jj].copy()
            Vj = V[j, :i + 1]
            for _ in range(2):                                  # classical Gram-Schmidt, twice
                h = Vj.conj() @ wj
                wj -= h @ Vj
                Hh[j, :i + 1, i] += h
            hn = np.linalg.norm(wj)
            Hh[j, i + 1, i] = hn
            if hn > 0:
                V[j, i + 1] = wj / hn
            e1 = np.zeros(i + 2, complex)
            e1[0] = beta[j]
            Y[j] = np.linalg.lstsq(Hh[j, :i + 2, :i + 1], e1, rcond=None)[0]
            its[j] = i + 1
            est[j] = np.linalg.norm(Hh[j, :i + 2, :i + 1] @ Y[j] - e1) / bn[j]
            if est[j] > tol and hn > 0:
                keep.append(j)
        act = np.array(keep, int)
        if cb is not None:
            cb(i, est)
    x = x0.copy()
    for j in range(nc):
        if its[j]:
            x[:, j] += Y[j] @ Z[j, :its[j]]
    return x, its, np.linalg.norm(b - A @ x, axis=0) / bn


def two_level(A, smoother, coarse=None, pre=1, post=1):
    """Preconditioner r -> z: pre smoothing steps, coarse correction, post smoothing steps (multiplicative); without
    coarse: the smoother alone."""
    if coarse is None:
        return smoother

    def prec(r):
        z = smoother(r) if pre else np.zeros_like(r)
        for _ in range(pre - 1):
            z = z + smoother(r - A @ z)
        z = z + coarse(r - A @ z if pre else r)
        for _ in range(post):
            z = z + smoother(r - A @ z)
        return z
    return prec
