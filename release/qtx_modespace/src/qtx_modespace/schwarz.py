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
Coarse level from the slow error modes instead of mode space: SlowCoarse (adaptive setup per energy, global vectors).
"""
import time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor

from . import gpu
from .core import CSolver, log
from .level1 import periodic_axes


class CapSchwarz:
    def __init__(self, dev, inp, width=16.0, s=2.0, d=2.0, spacing=None, centres="all", split=None, tov=2.5,
                 solver="dense64", workers=1, lu_host=False, pad=0.85, log=log):
        assert centres in ("all", "ends") and solver in ("dense", "dense64", "pardiso", "inv", "inv64", "rows",
                                                           "rows64")
        assert not (lu_host and solver[:3] in ("inv", "row")), "inv / rows keep the window inverses resident"
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
                strip = np.minimum(np.floor((u[i] % L) / (L / n)).astype(int), n - 1)
                for j in range(n):
                    du = np.abs((u[i] - (j + 0.5) * L / n + L / 2) % L - L / 2)
                    w = du < L / (2 * n) + tov
                    idx.append(i[w])
                    core.append(c[w] & (strip[w] == j))             # each orbital in exactly one strip
                    tcut.append([(dv[w], h) for dv, h in tc] + [(du[w], L / (2 * n) + tov)])
            self.idx, self.core, self.tcut = idx, core, tcut
        cov = np.zeros(len(x), int)
        for i, c in zip(self.idx, self.core):
            cov[i[c]] += 1
        assert (cov == 1).all(), "window cores must partition the device"
        self.s, self.d, self.solver, self.dev = s, d, solver, dev
        self.gpu = gpu.on and solver != "pardiso"          # GPU: window LUs and solves on the device, serially
        self.lu_host = lu_host and self.gpu                # GPU: LU factors in pinned host memory, streamed per solve
        if self.gpu:
            workers = 1
            self.didx = [gpu.cp.asarray(i) for i in self.idx]
            self.dcore = [gpu.cp.asarray(np.where(c)[0]) for c in self.core]
            self.dout = [gpu.cp.asarray(i[c]) for i, c in zip(self.idx, self.core)]
        if workers > 1:                  # concurrent LAPACK calls crash a multi-threaded MKL: 1 BLAS thread per window
            try:
                from threadpoolctl import threadpool_limits
                self.limit = lambda: threadpool_limits(1)
            except ImportError:
                log("  CAP-Schwarz: threadpoolctl not installed, windows run one after another (--workers 1)")
                workers = 1
        self.pool = ThreadPoolExecutor(workers) if workers > 1 else None
        self.ops, self.twall = None, 0.0
        self.inv = solver.startswith("inv")
        # rows / rows64: only the core rows of each window inverse (RAS keeps nothing else): n_core x n_window per
        # window instead of the n_window^2 LU, one GEMM per window and pass; the same correction as dense / dense64
        self.rows = solver.startswith("rows")
        if self.inv:
            self._groups(pad)
        log(f"CAP-Schwarz: {len(self.idx)} windows ({len(self.centres)} along x at "
            f"{', '.join(f'{c:.1f}' for c in self.centres)} A, width {width} A"
            + (f", transverse split {split} +{tov} A" if split and max(split) > 1 else "")
            + f"), {min(map(len, self.idx))}..{max(map(len, self.idx))} orbitals, absorbing {s} eV over {d} A, "
            + ("core rows of the window inverses" if self.rows else f"{solver} LU") + ((" on the GPU" + (", host-resident, streamed" if self.lu_host else "")) if self.gpu
                              else f", {workers} workers"))

    def set_energy(self, A, dE0=0.0, log=log):
        """Factor the windows of A (at E + dE0 if dE0: A_jj + dE0 S_jj, for reuse over an energy interval)."""
        S, x = self.dev.S.tocsr(), self.x
        A = A.tocsr()
        xmin, xmax = x.min(), x.max()

        def factor_matrix(k):
            """A_jj (+ dE0 S_jj) + i s P S_jj P of window k (sparse)."""
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
            return A[idx][:, idx] + dE0 * Sjj + 1j * self.s * (P @ Sjj @ P)

        def factor(k):
            Ajj = factor_matrix(k)
            if self.solver == "pardiso":
                return CSolver(Ajj, n_ref=0).factor_solve
            dt = np.complex64 if self.solver in ("dense64", "rows64") else complex
            if self.rows:                                # (A_jj^-1)[core, :] = (A_jj^-H e_core)^H, LU dropped
                xq = gpu.cp if self.gpu else np
                c = np.where(self.core[k])[0]
                lu = gpu.lu_factor(gpu.csr_dense(Ajj, dt) if self.gpu else Ajj.toarray().astype(dt))
                Ek = xq.zeros((Ajj.shape[0], len(c)), dt)
                Ek[xq.asarray(c), xq.arange(len(c))] = 1
                Ri = xq.ascontiguousarray(gpu.lu_solve(lu, Ek, trans=2).conj().T)
                del lu, Ek
                self.rows_gb += Ri.nbytes / 2 ** 30
                return lambda rj: (Ri @ rj.astype(dt)).astype(complex)
            if self.gpu:
                lu = gpu.lu_factor(gpu.csr_dense(Ajj, dt))
                if self.lu_host:                         # (pinned host LU in Fortran order, device pivots)
                    import cupyx
                    h = cupyx.empty_pinned(lu[0].shape, dt, order="F")
                    lu[0].get(out=h)
                    gpu.sync()
                    return (h, lu[1])
                return lambda rj: gpu.lu_solve(lu, rj.astype(dt)).astype(complex)
            lu = sla.lu_factor(Ajj.toarray().astype(dt), check_finite=False, overwrite_a=True)
            return lambda rj: sla.lu_solve(lu, rj.astype(dt), check_finite=False).astype(complex)
        t = time.time()
        ks = range(len(self.idx))
        self.rows_gb = 0.0
        if self.rows:
            self.ops = None
            gpu.free()
        if self.inv:
            self._factor_inv(factor_matrix, dE0, A, S)
            self.twall = time.time() - t
            log(f"  CAP-Schwarz factor: {self.twall:.1f} s (window inverses, {len(self.grp)} batches, "
                f"{sum(g['M'].nbytes for g in self.grp) / 2**30:.1f} GB)")
            return
        if self.pool:
            with self.limit():
                self.ops = list(self.pool.map(factor, ks))
        else:
            self.ops = list(map(factor, ks))
        if self.lu_host:
            gpu.free()
            nmax = max(h.shape[0] for h, _ in self.ops)
            dt = self.ops[0][0].dtype
            cp = gpu.cp
            self.buf = [cp.empty(nmax * nmax, dt) for _ in range(2)]   # double buffer for the streamed LUs
            self.cstream = cp.cuda.Stream(non_blocking=True)
        self.twall = time.time() - t
        log(f"  CAP-Schwarz factor: {self.twall:.1f} s"
            + (f" (LUs on the host, {sum(h.nbytes for h, _ in self.ops) / 2**30:.1f} GB pinned, streamed)"
               if self.lu_host else "")
            + (f" (core rows of the window inverses, {self.rows_gb:.1f} GB; LUs would be "
               f"{sum(len(i) ** 2 for i in self.idx) * (8 if self.solver == 'rows64' else 16) / 2**30:.1f} GB)"
               if self.rows else ""))

    def _streamed(self, r, z):
        """Window solves with the LUs copied host -> device, window k + 1 prefetched while window k is solved."""
        cp = gpu.cp
        import cupyx.scipy.linalg as cla
        cur = cp.cuda.get_current_stream()
        nw = len(self.ops)
        loaded = [None] * nw                               # event: LU k is in its buffer
        freed = [cp.cuda.Event(), cp.cuda.Event()]         # event: the solve that used buffer i is done
        freed[0].record(cur), freed[1].record(cur)

        def load(k):
            h, _ = self.ops[k]
            n = h.shape[0]
            d = cp.ndarray((n, n), h.dtype, memptr=self.buf[k % 2].data, order="F")
            self.cstream.wait_event(freed[k % 2])
            d.set(h, stream=self.cstream)
            ev = cp.cuda.Event()
            ev.record(self.cstream)
            loaded[k] = ev
            return d
        views = [None] * nw
        views[0] = load(0)
        for k in range(nw):
            cur.wait_event(loaded[k])
            dt = views[k].dtype
            zk = cla.lu_solve((views[k], self.ops[k][1]), r[self.didx[k]].astype(dt), check_finite=False)
            freed[k % 2] = cp.cuda.Event()
            freed[k % 2].record(cur)
            if k + 1 < nw:
                views[k + 1] = load(k + 1)                # buffer (k + 1) % 2 is free once solve k - 1 finished
            z[self.dout[k]] = zk.astype(complex)[self.dcore[k]]
            views[k] = None
        cur.synchronize()
        return z

    def _groups(self, pad):
        """Batches of windows for the inverse solver: windows sorted by size, a batch takes windows down to pad x the
        largest one (zero padding: <= 1 / pad^2 - 1 extra memory), so one gather, one batched matmul and one scatter
        per batch replace the loop over windows."""
        sizes = np.array([len(i) for i in self.idx])
        order = list(np.argsort(-sizes, kind="stable"))
        xq = gpu.cp if self.gpu else np
        nrow = len(self.x)
        self.grp = []
        while order:
            m = int(sizes[order[0]])
            ks = [k for k in order if sizes[k] >= pad * m]
            order = [k for k in order if k not in ks]
            gidx = np.full((len(ks), m), nrow, np.int64)           # row nrow = the appended zero row of r
            out, src = [], []
            for q, k in enumerate(ks):
                gidx[q, :sizes[k]] = self.idx[k]
                c = np.where(self.core[k])[0]
                out.append(self.idx[k][c])
                src.append(q * m + c)
            self.grp.append(dict(ks=ks, m=m, gidx=xq.asarray(gidx.ravel()), out=xq.asarray(np.concatenate(out)),
                                 src=xq.asarray(np.concatenate(src)), M=None))

    def _factor_inv(self, factor_matrix, dE0, A, S):
        """Window inverses (complex64 for inv64) written into the zero-padded batch stacks, one window at a time."""
        dt = np.complex64 if self.solver == "inv64" else complex
        xq = gpu.cp if self.gpu else np
        for g in self.grp:
            g["M"] = None
        gpu.free()
        for g in self.grp:
            M = xq.zeros((len(g["ks"]), g["m"], g["m"]), dt)
            for q, k in enumerate(g["ks"]):
                Ajj = factor_matrix(k)
                D = gpu.csr_dense(Ajj, dt) if self.gpu else Ajj.toarray().astype(dt)
                n = D.shape[0]
                M[q, :n, :n] = xq.linalg.inv(D)
                del D
            g["M"] = M
            gpu.free()
        self.ops = True

    def _batched(self, r):
        xq = gpu.cp if self.gpu else np
        r = xq.asarray(r)
        c = r.shape[1]
        dt = self.grp[0]["M"].dtype
        rext = xq.concatenate([r, xq.zeros((1, c), r.dtype)]).astype(dt)
        z = xq.zeros(r.shape, complex)
        for g in self.grp:
            R = rext[g["gidx"]].reshape(len(g["ks"]), g["m"], c)
            Z = xq.matmul(g["M"], R).reshape(-1, c)
            z[g["out"]] = Z[g["src"]].astype(complex)
        return z

    def __call__(self, r):
        if self.inv:
            return self._batched(r)
        if self.gpu:                                     # r: cupy (host arrays are moved, the result is cupy)
            r = gpu.cp.asarray(r)
            z = gpu.cp.zeros_like(r)
            if self.lu_host:
                return self._streamed(r, z)
            for k in range(len(self.idx)):
                zk = self.ops[k](r[self.didx[k]])
                z[self.dout[k]] = zk if self.rows else zk[self.dcore[k]]
            return z
        z = np.zeros_like(r)
        solve = lambda k: self.ops[k](r[self.idx[k]]) if self.rows else self.ops[k](r[self.idx[k]])[self.core[k]]
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
        self.ops, self.buf = None, None
        for g in getattr(self, "grp", []):
            g["M"] = None
        gpu.free()


class GpuOp:
    """A @ X on the GPU: the sparse device matrix as a cupyx CSR (complex128); X numpy or cupy, result cupy."""

    def __init__(self, A):
        self.A = gpu.csr(A, np.complex128)
        self.shape = self.A.shape

    def __matmul__(self, X):
        return self.A @ gpu.cp.asarray(X, dtype=complex)


def operator(A, threads=4):
    """FastOp on the CPU, GpuOp in GPU mode."""
    return GpuOp(A) if gpu.on else FastOp(A, threads)


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
    column on contiguous memory (no gather of the whole Krylov basis). b, x0, A @ and prec: numpy or cupy arrays
    (Hessenberg and least-squares on the host); x, the residuals: arrays of b's kind."""
    xp = gpu.xp_of(b)
    n, nc = b.shape
    r0 = b - A @ x0
    beta, bn = gpu.asnumpy(xp.linalg.norm(r0, axis=0)), gpu.asnumpy(xp.linalg.norm(b, axis=0))
    V = xp.zeros((nc, m + 1, n), complex)
    Z = xp.zeros((nc, m, n), complex)
    Hh = np.zeros((nc, m + 1, m), complex)
    V[:, 0] = (r0 / xp.asarray(np.where(beta > 0, beta, 1))).T
    its = np.zeros(nc, int)
    est = beta / bn
    act = np.where(beta > tol * bn)[0]
    Y = [np.zeros(0, complex) for _ in range(nc)]
    for i in range(m):
        if len(act) == 0:
            break
        zb = prec(xp.ascontiguousarray(V[act, i].T))
        Z[act, i] = zb.T
        w = xp.asarray(A @ zb)
        keep = []
        for jj, j in enumerate(act):
            wj = w[:, jj].copy()
            Vj = V[j, :i + 1]
            for _ in range(2):                                  # classical Gram-Schmidt, twice
                h = Vj.conj() @ wj
                wj -= h @ Vj
                Hh[j, :i + 1, i] += gpu.asnumpy(h)
            hn = float(xp.linalg.norm(wj))
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
            x[:, j] += xp.asarray(Y[j]) @ Z[j, :its[j]]
    del V, Z
    return x, its, gpu.asnumpy(xp.linalg.norm(b - A @ x, axis=0)) / bn


def two_level(A, smoother, coarse=None, pre=1, post=1):
    """Preconditioner r -> z: pre smoothing steps, coarse correction, post smoothing steps (multiplicative); without
    coarse: the smoother alone."""
    if coarse is None:
        return smoother

    def prec(r):
        z = smoother(r) if pre else gpu.xp_of(r).zeros_like(r)
        for _ in range(pre - 1):
            z = z + smoother(r - A @ z)
        z = z + coarse(r - A @ z if pre else r)
        for _ in range(post):
            z = z + smoother(r - A @ z)
        return z
    return prec


class SlowCoarse:
    """Coarse level from the slow error modes of the smoother and of the two-level cycle (adaptive AMG-type setup, as
    in Luescher's inexact deflation / DD-alphaAMG, but with global vectors: no aggregation).

    One application of a preconditioner B changes the error by E_B = I - B A. Round 1 (no coarse level yet): B = pre +
    post multiplicative smoothing steps, E_B = T^(pre + post), T = I - M A. Round r > 1: B = the two-level cycle with
    the coarse space of round r - 1, E_B = T^post (I - P_c A) T^pre. Per round, block subspace iteration on E_B from
    m random vectors (q steps, QR after each) gives W ~ the dominant m-dimensional invariant subspace of E_B (the
    error components the current method damps worst); W is orthogonalized against Q (twice) and appended, Q = [Q, W].
    Coarse level: Galerkin A_c = Q^H A Q (dense, LU), P_c r = Q A_c^-1 Q^H r (call the object).
    info["rounds"][r]: N after the round, Ritz |theta| (damping factors) of the operator iterated on in that round (the
    smoother, then the cycle with the previous coarse space), sigma_min(A_c), time.
    Cost per round: m q applications of E_B, each = (pre + post) smoother passes (+ coarse solve) + matvecs, on m
    columns at once. The modes depend on the energy (A and the smoother): build them at every energy (2808a: ~1 s,
    6 nm: ~20-25 s per round of 128 vectors on a GH200); carried across energies they lose half the gain within
    0.05 eV. Q is dense (n x m rounds), numpy or cupy (GPU mode).
    Measured (two-level block FGMRES, CAP-Schwarz 16 A, tol 1e-10, its = max over columns): 2808a Ec+0.05 / 0.2 / 0.3
    with m 64, q 8, 3 rounds 13 / 15 / 18 its vs mode space (slab level 1 1e-8, N 2373) 29 / 47 / 150; 6 nm Ec+0.15
    (y split 4) m 128, q 3, 4 rounds (N 512): 24-30 its, where every mode-space coarse level stalls."""

    def __init__(self, A, smoother, m=64, q=3, rounds=4, pre=1, post=1, seed=0, log=log):
        t0 = time.time()
        self.A, self.S, self.pre, self.post = A, smoother, int(pre), int(post)
        assert self.pre + self.post >= 1 and q >= 1 and m >= 1
        n = A.shape[0]
        xp = gpu.cp if gpu.on else np
        rng = np.random.default_rng(seed)
        self.Q, self.lu = xp.zeros((n, 0), complex), None
        self.info = dict(m=int(m), q=int(q), pre=self.pre, post=self.post, rounds=[])
        for r in range(rounds):
            t = time.time()
            B = self.cycle if self.lu is not None else self._smooth
            Y = xp.linalg.qr(xp.asarray(rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))))[0]
            for k in range(q):
                Z = Y - B(A @ Y)
                if k == q - 1:                                 # Ritz values of E_B on span(Y): damping factors
                    th = np.abs(np.linalg.eigvals(gpu.asnumpy(Y.conj().T @ Z)))
                Y = xp.linalg.qr(Z)[0]
            del Z
            for _ in range(2):
                Y = Y - self.Q @ (self.Q.conj().T @ Y)
            self.Q = xp.hstack([self.Q, xp.linalg.qr(Y)[0]])
            del Y
            smin = self._galerkin()
            d = dict(N=int(self.Q.shape[1]), theta_max=float(th.max()), n_theta_gt1=int((th > 1).sum()),
                     sigma_min=smin, t=time.time() - t)
            self.info["rounds"].append(d)
            log(f"  slow modes round {r + 1}: Ritz |theta| of the " + (f"cycle with N {d['N'] - m}" if r else
                "smoother") + f": max {d['theta_max']:.2f}, {d['n_theta_gt1']} of {m} > 1 -> N {d['N']}, "
                f"sigma_min(A_c) {smin:.1e}; {d['t']:.0f} s")
        self.N = int(self.Q.shape[1])
        self.info.update(N=self.N, t_build=time.time() - t0)

    def _galerkin(self):
        Ac = self.Q.conj().T @ gpu.xp_of(self.Q).asarray(self.A @ self.Q)
        smin = float(gpu.asnumpy(gpu.xp_of(Ac).linalg.svd(Ac, compute_uv=False))[-1]) if Ac.shape[0] <= 4096 else np.nan
        self.lu = gpu.lu_factor(Ac)
        return smin

    def _smooth(self, r):
        """pre + post multiplicative smoothing steps from z = 0 (round 1)."""
        z = self.S(r)
        for _ in range(self.pre + self.post - 1):
            z = z + self.S(r - self.A @ z)
        return z

    def __call__(self, r):
        """Coarse correction P_c r = Q A_c^-1 Q^H r."""
        Q = self.Q
        return Q @ gpu.lu_solve(self.lu, Q.conj().T @ gpu.xp_of(Q).asarray(r))

    def cycle(self, r):
        """One two-level cycle from z = 0 (pre smoothing, coarse correction, post smoothing)."""
        if self.pre:
            z = self.S(r)
            for _ in range(self.pre - 1):
                z = z + self.S(r - self.A @ z)
            z = z + self(r - self.A @ z)
        else:
            z = self(r)
        for _ in range(self.post):
            z = z + self.S(r - self.A @ z)
        return z

    def free(self):
        self.Q, self.lu = None, None
        gpu.free()
