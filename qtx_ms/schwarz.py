"""Overlapping Schwarz smoother along the transport axis + two-level cycle with the Galerkin mode-space coarse level.

Subdomains: device orbitals in windows of `width` Å along x, consecutive windows overlapping by `overlap` Å
(restricted additive Schwarz: each orbital takes the correction of the window whose core holds it). The windows cross
the slab interfaces, where the Galerkin residual concentrates.
Local solve on window j (closed, no Sigma):  z_j = V_j f(E - lambda) V_j^H r_j  from the generalized eigenpairs
(H_jj, S_jj) V = S V Lambda, V^H S V = I, computed once (energy independent), with
    f = 1 / (E - lambda) for |E - lambda| > delta, 0 otherwise:
the local states within delta of E (the local resonances that make a closed local solve singular) are left to the
coarse level, the rest (high-energy, valence, oxide components) is inverted exactly. Windows that touch the lead
faces (bL, bR) use a per-energy dense LU of A_jj with Sigma instead (open, no resonance).

Two-level cycle (multiplicative, on a block of right-hand sides):  nu1 x [x += w S r] ; x += P_c r ; nu2 x [x += w S r]
with P_c = U A_c^-1 U^H (Galerkin, the mode-space solve), x0 = P_c b = mode space. Each smoothing step costs one fine
matvec (residual update). Error, residual, T and the charge per atom are compared with the exact psi of a psi file
(qtx_ms.prep.compare_qtbm or exact_T --save-psi) after every cycle.

    python -m qtx_ms.schwarz --inputs inputs/mos2_hfo2 --psi results_ms/qtbm_cmp/mos2_hfo2_eta0/psi_+0.55434897.npz \\
        --sigma-cache results_ms/sigma_qtx_eta0/mos2_hfo2 --modes results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead
"""
import argparse, json, os, time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import gpu, msbasis as mb
from qtx_ms.plot_residual import ms_solution
from qtx_ms.twolevel import gmres_fixed

log = mb.log


class Schwarz:
    def __init__(self, dev, inp, width=6.0, overlap=2.0, open_lu=True):
        ax = inp["meta"]["axis"]
        atom = np.repeat(np.arange(len(inp["norb"])), inp["norb"])
        x = inp["xyz"][atom[dev.dev], ax]                       # device orbital positions along transport
        lo, hi, step = x.min(), x.max(), width - overlap
        nw = max(int(np.ceil((hi - lo - overlap) / step)), 1)   # windows [a_k, a_k + width), last one to hi
        starts = lo + step * np.arange(nw)
        cuts = np.concatenate([[-np.inf], starts[1:] + 0.5 * overlap, [np.inf]])   # RAS cores: midpoints of overlaps
        self.idx, self.core, self.eig, self.lu = [], [], [], []
        ends = set(dev.bL) | set(dev.bR)
        H, S = dev.H.tocsr(), dev.S.tocsr()
        t = time.time()
        for k, a in enumerate(starts):
            idx = np.where((x >= a) & ((x < a + width) | (k == nw - 1)))[0]
            core = (x[idx] >= cuts[k]) & (x[idx] < cuts[k + 1])
            self.idx.append(idx)
            self.core.append(core)
            if open_lu and ends & set(idx):
                self.eig.append(None)                           # open window: per-energy LU with Sigma
            else:
                Hj, Sj = H[idx][:, idx].toarray(), S[idx][:, idx].toarray()
                self.eig.append(sla.eigh(Hj, Sj, check_finite=False))
        cover = np.zeros(dev.n, int)
        for i, c in zip(self.idx, self.core):
            cover[i[c]] += 1
        assert (cover == 1).all(), f"RAS cores do not partition the device ({cover.min()}..{cover.max()})"
        log(f"Schwarz: {len(starts)} windows of {width} Å (overlap {overlap} Å), "
            f"{min(map(len, self.idx))}..{max(map(len, self.idx))} orbitals, "
            f"{sum(e is None for e in self.eig)} open (lead) windows; eigh {time.time() - t:.0f} s")

    def set_energy(self, E, A, delta):
        self.E, self.delta = E, delta
        self.f, self.lu = [], []
        for idx, e in zip(self.idx, self.eig):
            if e is None:
                self.f.append(None)
                self.lu.append(sla.lu_factor(A[idx][:, idx].toarray(), check_finite=False))
            else:
                d = E - e[0]
                self.f.append(np.where(np.abs(d) > delta, 1 / np.where(d == 0, 1, d), 0.0))
                self.lu.append(None)
        self.ndefl = [int((f == 0).sum()) for f in self.f if f is not None]

    def __call__(self, r):
        z = np.zeros_like(r)
        for idx, core, e, f, lu in zip(self.idx, self.core, self.eig, self.f, self.lu):
            rj = r[idx]
            zj = sla.lu_solve(lu, rj, check_finite=False) if lu is not None else e[1] @ (f[:, None] * (e[1].conj().T @ rj))
            z[idx[core]] += zj[core]
        return z


class CapSchwarz:
    """Restricted additive Schwarz with windows of `width` Å centred on the slab interfaces and on points between them
    (cores: nearest centre), local solves = LU of A_jj + i s P S_jj P, P a quadratic absorbing ramp over the outer d Å
    towards each artificial cut (smoother_diag cap<s>/<d>; lead faces keep Sigma).
    Cost options: spacing [Å] = largest distance between neighbouring centres (default: one centre between interfaces,
    i.e. the slab middles); solver dense (complex128 LU) | dense64 (complex64 LU, solves in single precision) |
    pardiso (sparse, real-equivalent, msbasis.CSolver without refinement) | rows / rows64 (only the core rows of each
    window inverse, complex128 / complex64: n_core x n_window per window instead of the n_window^2 LU, one GEMM per
    window and pass; the same correction, since RAS keeps nothing else). GPU mode (gpu.on, every solver but pardiso):
    factors and solves on the device, r and the result are cupy arrays (twin of the release CapSchwarz).
    dE0 = factor at E + dE0 instead of E (one
    factorization reused over an energy interval: A_jj(E + dE0) = A_jj(E) + dE0 S_jj, Sigma kept);
    centres "iface": windows on the slab interfaces only, no correction outside them (the slab interiors are left to
    the coarse level); "ends": interfaces + the two windows on the lead faces.
    split = pieces per periodic transverse axis (meta["periodic"] order): every x-window is cut into boxes of
    period / n (core, disjoint) + tov Å on each side (minimum image), with the same absorbing ramp towards the
    transverse cuts.
    centres "tile": one window per tile of mb.partition(tile_orbs, xsplit) instead of x-windows: core = the tile's
    orbitals, window = every device orbital within tov Å (3D, minimum image) of a tile atom, absorbing ramp over the
    outer d Å of that shell (the lead faces are real boundaries and keep Sigma)."""

    def __init__(self, dev, inp, width=16.0, s=2.0, d=3.0, spacing=None, solver="dense", dE0=0.0, centres="all",
                 workers=1, split=None, tov=2.5, tile_orbs=2500, xsplit=1, tile_cut=""):
        ax = inp["meta"]["axis"]
        atom = np.repeat(np.arange(len(inp["norb"])), inp["norb"])
        self.x = x = inp["xyz"][atom[dev.dev], ax]
        if centres == "tile":
            self._tiles(dev, inp, atom[dev.dev], tov, tile_orbs, xsplit, ax, tile_cut)
            width = f"tile {tile_orbs}/x{xsplit} +{tov}"
            split = None
        else:
            self._xwin(dev, inp, atom, x, width, spacing, centres, split, tov)
        cov = np.zeros(len(x), int)
        for i, c in zip(self.idx, self.core):
            cov[i[c]] += 1
        assert (cov == 1).all(), "window cores must partition the device"
        assert solver in ("dense", "dense64", "pardiso", "rows", "rows64"), solver
        self.s, self.d, self.solver, self.dE0, self.dev = s, d, solver, dE0, dev
        self.rows = solver.startswith("rows")
        self.gpu = gpu.on and solver != "pardiso"          # GPU: window LUs / rows and solves on the device, serially
        self.ops, self.twall, self.rows_gb = None, 0.0, 0.0
        if self.gpu:
            workers = 1
            self.didx = [gpu.cp.asarray(i) for i in self.idx]
            self.dcore = [gpu.cp.asarray(np.where(c)[0]) for c in self.core]
            self.dout = [gpu.cp.asarray(i[c]) for i, c in zip(self.idx, self.core)]
        self.pool = ThreadPoolExecutor(workers) if workers > 1 else None   # windows are independent (additive)
        log(f"CapSchwarz: {len(self.centres)} {centres} windows of {width} Å at x = {', '.join(f'{c:.1f}' for c in self.centres)}, "
            f"{min(map(len, self.idx))}..{max(map(len, self.idx))} orbitals (sum {sum(map(len, self.idx))}), "
            f"absorbing {s} eV over {d} Å, " + ("core rows of the window inverses" if self.rows else f"{solver} LU")
            + (" on the GPU" if self.gpu else "") + (f" at E {dE0:+g} eV" if dE0 else "")
            + (f", transverse split {split} (+{tov} Å)" if split and max(split) > 1 else ""))

    def _tiles(self, dev, inp, atom_d, tov, tile_orbs, xsplit, ax, tile_cut=""):
        blocks, _, _ = mb.partition(dev, cores=False, tile_orbs=tile_orbs, xsplit=xsplit, axis=ax,
                                    cut=mb.parse_cut(tile_cut))
        geo = mb.Geometry(inp["xyz"], inp["meta"].get("periodic"))
        dat = np.unique(atom_d)
        tree_d = geo.tree(dat)
        self.idx, self.core, self.dist = [], [], []
        for b in blocks:
            near = dat[np.unique(np.concatenate([np.asarray(q, int) for q in
                                                 tree_d.query_ball_point(geo.points(b["atoms"]), tov + 1e-9)]))]
            wa = np.union1d(near, b["atoms"])
            i = np.where(np.isin(atom_d, wa))[0]
            dt = geo.tree(b["atoms"]).query(geo.points(atom_d[i]))[0]
            self.idx.append(i)
            self.core.append(np.isin(atom_d[i], b["atoms"]))
            self.dist.append((dt, tov))
        self.centres = np.array([np.mean(self.x[i[c]]) for i, c in zip(self.idx, self.core)])
        self.tcut = [[] for _ in self.idx]

    def _xwin(self, dev, inp, atom, x, width, spacing, centres, split, tov):
        self.dist = None
        sl = [np.arange(dev.sb[k], dev.sb[k + 1]) for k in range(len(dev.ns))]
        xi = [0.5 * (x[sl[k]].max() + x[sl[k + 1]].min()) for k in range(len(sl) - 1)]
        anchors = [x.min()] + xi + [x.max()]
        fill = []
        for lo, hi in zip(anchors[:-1], anchors[1:]):
            m = max(1, int(np.ceil((hi - lo) / spacing)) - 1) if spacing else 1
            fill += [lo + (hi - lo) * (k + 1) / (m + 1) for k in range(m)]
        keep = {"all": fill, "iface": [], "ends": [fill[0], fill[-1]]}[centres]
        self.centres = np.sort(np.array(xi + keep))
        near = np.argmin(np.abs(x[:, None] - self.centres[None, :]), axis=1)
        self.idx = [np.where((np.abs(x - c) < width / 2) | ((near == k) & (centres == "all")))[0]
                    for k, c in enumerate(self.centres)]
        self.core = [near[i] == k for k, i in enumerate(self.idx)]
        self.tcut = [[] for _ in self.idx]                      # per window: (u, half width) of each transverse cut
        per = mb.periodic_axes(inp["meta"].get("periodic"))
        for (axt, L), n in zip(per, split or []):
            if n <= 1:
                continue
            u = inp["xyz"][atom[dev.dev], axt]
            box = np.minimum((u % L * (n / L)).astype(int), n - 1)   # owner box: one per orbital, also on a box face
            idx, core, tcut = [], [], []
            for i, c, tc in zip(self.idx, self.core, self.tcut):
                for j in range(n):
                    cj = (j + 0.5) * L / n
                    du = np.abs((u[i] - cj + L / 2) % L - L / 2)       # minimum-image distance to the box centre
                    w = du < L / (2 * n) + tov
                    idx.append(i[w])
                    core.append(c[w] & (box[i[w]] == j))
                    tcut.append([(dv[w], h) for dv, h in tc] + [(du[w], L / (2 * n) + tov)])
            self.idx, self.core, self.tcut = idx, core, tcut

    def ramp(self, k):
        """Absorbing ramp (0 .. 1) of window k towards its artificial cuts (the lead faces are real boundaries)."""
        x, idx = self.x, self.idx[k]
        xmin, xmax = x.min(), x.max()
        xi = x[idx]
        lo, hi = xi.min(), xi.max()
        ramp = np.zeros(len(idx))
        if self.dist is not None:                              # tile window: ramp over the outer d of the shell
            dt, h = self.dist[k]
            ramp = np.clip((dt - (h - self.d)) / self.d, 0, 1) ** 2
        elif lo > xmin + 1e-6:                                 # a cut on the left (not the lead face)
            ramp = np.maximum(ramp, np.clip((lo + self.d - xi) / self.d, 0, 1) ** 2)
        if self.dist is None and hi < xmax - 1e-6:
            ramp = np.maximum(ramp, np.clip((xi - (hi - self.d)) / self.d, 0, 1) ** 2)
        for du, h in self.tcut[k]:                             # transverse cuts at |du| = h
            ramp = np.maximum(ramp, np.clip((du - (h - self.d)) / self.d, 0, 1) ** 2)
        return ramp

    def set_energy(self, E, A, delta=None, log=log):
        S = self.dev.S.tocsr()
        A = A.tocsr()
        def factor(k):
            idx = self.idx[k]
            t = time.time()
            P = sp.diags(np.sqrt(self.ramp(k)))
            Sjj = S[idx][:, idx]
            Ajj = A[idx][:, idx] + self.dE0 * Sjj + 1j * self.s * (P @ Sjj @ P)
            if self.solver == "pardiso":
                cs = mb.CSolver(Ajj, n_ref=0)
                op = cs.factor_solve
            else:
                dt = np.complex64 if self.solver in ("dense64", "rows64") else complex
                D = gpu.csr_dense(Ajj, dt) if self.gpu else Ajj.toarray().astype(dt)
                lu = gpu.lu_factor(D) if self.gpu else sla.lu_factor(D, check_finite=False, overwrite_a=True)
                del D
                if self.rows:                                      # (A_jj^-1)[core, :] = (A_jj^-H e_core)^H, LU dropped
                    xq = gpu.cp if self.gpu else np
                    c = np.where(self.core[k])[0]
                    Ek = xq.zeros((Ajj.shape[0], len(c)), dt)
                    Ek[xq.asarray(c), xq.arange(len(c))] = 1
                    Ri = xq.ascontiguousarray((gpu.lu_solve(lu, Ek, trans=2) if self.gpu else
                                               sla.lu_solve(lu, Ek, trans=2, check_finite=False)).conj().T)
                    del lu, Ek
                    self.rows_gb += Ri.nbytes / 2 ** 30
                    op = (lambda Ri, dt: lambda rj: (Ri @ rj.astype(dt)).astype(complex))(Ri, dt)
                elif self.gpu:
                    op = (lambda lu, dt: lambda rj: gpu.lu_solve(lu, rj.astype(dt)).astype(complex))(lu, dt)
                else:
                    op = (lambda lu, dt: lambda rj: sla.lu_solve(lu, rj.astype(dt), check_finite=False).astype(complex))(lu, dt)
            return op, time.time() - t
        t = time.time()
        ks = range(len(self.idx))
        self.ops, self.rows_gb = None, 0.0                         # drop the factors of the previous energy first
        if self.gpu:
            gpu.free()
        out = list(self.pool.map(factor, ks) if self.pool else map(factor, ks))
        self.ops, self.tfac = [o for o, _ in out], [t for _, t in out]
        self.twall = time.time() - t
        log(f"  factor {self.solver}: wall {self.twall:.1f} s, per window (" +
            ", ".join(f"{len(i)}: {t:.1f}" for i, t in zip(self.idx, self.tfac)) + ")"
            + (f"; core rows {self.rows_gb:.1f} GB (LUs would be "
               f"{sum(len(i) ** 2 for i in self.idx) * (8 if self.solver == 'rows64' else 16) / 2**30:.1f} GB)"
               if self.rows else ""))
        self.ndefl = [0]

    def __call__(self, r):
        if self.gpu:                                              # r: cupy (host arrays are moved), result cupy
            r = gpu.cp.asarray(r)
            z = gpu.cp.zeros_like(r)
            for k in range(len(self.idx)):
                zk = self.ops[k](r[self.didx[k]])
                z[self.dout[k]] = zk if self.rows else zk[self.dcore[k]]
            return z
        z = np.zeros_like(r)
        solve = lambda k: self.ops[k](r[self.idx[k]]) if self.rows else self.ops[k](r[self.idx[k]])[self.core[k]]
        ks = range(len(self.idx))
        for k, zk in zip(ks, self.pool.map(solve, ks) if self.pool else map(solve, ks)):
            z[self.idx[k][self.core[k]]] = zk                  # cores are disjoint
        return z

    def free(self):
        self.ops = None
        gpu.free()


class GpuOp:
    """A @ X on the GPU: the sparse device matrix as a cupyx CSR (complex128); X numpy or cupy, result cupy."""

    def __init__(self, A):
        self.A = gpu.csr(A, np.complex128)
        self.shape = self.A.shape

    def __matmul__(self, X):
        return self.A @ gpu.cp.asarray(X, dtype=complex)


def operator(A, threads=4):
    """FastOp on the CPU, GpuOp in GPU mode (twin of the release schwarz.operator)."""
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


def fgmres_block(A, b, x0, prec, tol=1e-8, m=60, clock=None, cb=None):
    """Flexible GMRES for all columns of b at once without diagnostics: every column has its own Krylov space and
    Hessenberg matrix, but A and the preconditioner act on the block of the active columns' newest basis vectors
    (one multi-RHS local solve per window). A column stops when its GMRES residual estimate is below tol |b_j|.
    Returns x, iterations per column, true relative residuals; clock[key] accumulates seconds (matvec, prec, orth).
    cb(i, est): called after iteration i with the GMRES residual estimates |r_j| / |b_j| of all columns.
    Storage per column (V[j], Z[j]: (m + 1) x n, rows = basis vectors): the Gram-Schmidt step is one GEMV pair per
    column on contiguous memory. b, x0, A @ and prec: numpy or cupy arrays (Hessenberg and least squares on the host);
    x: an array of b's kind (twin of the release fgmres_block)."""
    clock = {} if clock is None else clock
    def tick(key, t0):
        clock[key] = clock.get(key, 0.0) + time.time() - t0
    xp = gpu.xp_of(b)
    n, nc = b.shape
    t0 = time.time(); r0 = b - A @ x0; tick("matvec", t0)
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
        t0 = time.time(); zb = xp.asarray(prec(xp.ascontiguousarray(V[act, i].T))); tick("prec", t0)
        Z[act, i] = zb.T
        t0 = time.time(); w = xp.asarray(A @ zb); tick("matvec", t0)
        t0 = time.time()
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
        tick("orth", t0)
        if cb is not None:
            cb(i, est)
    x = x0.copy()
    for j in range(nc):
        if its[j]:
            x[:, j] += xp.asarray(Y[j]) @ Z[j, :its[j]]
    del V, Z
    t0 = time.time(); res = gpu.asnumpy(xp.linalg.norm(b - A @ x, axis=0)) / bn; tick("matvec", t0)
    return x, its, res


def fgmres_restart(A, b, x0, prec, tol=1e-8, m=60, maxit=1000, clock=None, cb=None, log=None):
    """Restarted block FGMRES(m) with frozen columns: cycles of fgmres_block on the still active columns only. After
    each cycle x is updated and the true residual b - A x is computed; columns with |r_j| <= tol |b_j| are frozen
    (dropped from all further cycles: no matvec, preconditioner or orthogonalization work). Stops after maxit
    iterations in total. Returns x, iterations per column, true relative residuals, number of cycles.
    cb(i, est): after global iteration i with the residual estimates of all columns (frozen: their true residual)."""
    n, nc = b.shape
    x = x0.copy()
    bn = np.linalg.norm(b, axis=0)
    res = np.full(nc, np.inf)
    its = np.zeros(nc, int)
    act = np.arange(nc)
    done, cyc = 0, 0
    while len(act) and done < maxit:
        mm = min(m, maxit - done)
        def cbc(i, est, act=act, done=done):
            if cb is not None:
                e = res.copy(); e[act] = est
                cb(done + i, e)
        xa, ia, ra = fgmres_block(A, b[:, act], x[:, act], prec, tol, mm, clock, cbc)
        x[:, act] = xa
        its[act] += ia
        res[act] = ra
        if ia.max() == 0:
            break
        done += int(ia.max())
        cyc += 1
        new = act[ra > tol]
        if log is not None:
            log(f"  restart cycle {cyc}: {int(ia.max())} its (total {done}), true residual max {ra.max():.2e}, "
                f"frozen {nc - len(new)} / {nc} columns")
        act = new
    return x, its, res, cyc


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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--psi", required=True)
    p.add_argument("--sigma-cache", required=True)
    p.add_argument("--modes", required=True)
    p.add_argument("--width", type=float, default=6.0, help="Schwarz window width along x [Å]")
    p.add_argument("--overlap", type=float, default=2.0, help="window overlap [Å]")
    p.add_argument("--delta", default="0,0.1,0.3", help="comma list: local eigenvalues within delta of E left to the "
                   "coarse level [eV]")
    p.add_argument("--smoothers", default="none,schwarz,gmres", help="comma list of none | schwarz | gmres | cap")
    p.add_argument("--cap", default="16,2,3", help="cap smoother: window width [Å], absorbing strength [eV], ramp [Å]")
    p.add_argument("--cap-spacing", type=float, help="cap smoother: largest distance between window centres [Å] "
                   "(default: interfaces + slab middles)")
    p.add_argument("--cap-solver", default="dense", choices=["dense", "dense64", "pardiso"], help="cap local LU")
    p.add_argument("--cap-tile-orbs", type=int, default=2500, help="cap smoother with --cap-centres tile: tile size")
    p.add_argument("--cap-tile-cut", default="", help="cap smoother with --cap-centres tile: tiles by msbasis.parse_cut "
                   "(as run_ms --tile-cut), e.g. '0:n2@1,2'")
    p.add_argument("--cap-xsplit", type=int, default=1, help="cap smoother with --cap-centres tile: tile pieces along x")
    p.add_argument("--cap-centres", default="all", choices=["all", "iface", "ends", "tile"], help="cap smoother: windows on the "
                   "interfaces and between them (all), on the slab interfaces only (iface), interfaces + lead-face windows (ends)")
    p.add_argument("--cap-split", default="1,1", help="cap smoother: pieces per periodic transverse axis (meta order)")
    p.add_argument("--cap-tov", type=float, default=2.5, help="cap smoother: transverse overlap on each side of a box [Å]")
    p.add_argument("--cap-dE0", type=float, default=0.0, help="cap smoother: factor the windows at E + dE0 [eV]")
    p.add_argument("--basis", help="tile levels from run_ms --save-basis (<prefix>_U1/_V_slab<s>.npy) instead of --modes "
                   "as coarse space; --modes is then only used to set up the system")
    p.add_argument("--l1", default="schwarz", choices=["coarse", "exact", "schwarz"], help="with --basis: level 1 "
                   "treatment: coarse (two-level, coarse = U1 V), exact (two-level, coarse = U1), schwarz (V-cycle)")
    p.add_argument("--l1-cap", default="32,2,2", help="level-1 Schwarz: window width [Å], absorbing strength [eV], ramp [Å]")
    p.add_argument("--l1-centres", default="iface", choices=["iface", "all"], help="level-1 windows on the slab "
                   "interfaces (iface) or also on the slab middles (all)")
    p.add_argument("--l1-split", default="none", choices=["none", "tile"], help="level-1 windows: full cross-section "
                   "(none) or also cut transversely along the tile cuts (tile)")
    p.add_argument("--l1-tov", type=float, default=2.5, help="level-1 tile split: transverse overlap [Å]")
    p.add_argument("--l1-wmin", type=float, default=0.1, help="level-1 tile split: least column weight in a region + overlap")
    p.add_argument("--workers", type=int, default=1, help="cap smoother: windows factored / solved concurrently")
    p.add_argument("--pre", type=int, help="pre-smoothing steps (default --nu); post-smoothing = --nu")
    p.add_argument("--k", type=int, default=10, help="GMRES(k) smoother steps")
    p.add_argument("--nu", type=int, default=1, help="pre- and post-smoothing steps")
    p.add_argument("--omega", type=float, default=1.0, help="smoother damping")
    p.add_argument("--cycles", type=int, default=12, help="stationary cycles, or FGMRES iterations with --accel")
    p.add_argument("--accel", default="none", choices=["none", "fgmres", "block"], help="none: stationary two-level cycle; "
                   "fgmres: the cycle as right preconditioner of flexible GMRES (x0 = mode space), per column, error after every "
                   "iteration; block: the same for all columns at once, no diagnostics, timing per part (fgmres_block)")
    p.add_argument("--tol", type=float, default=1e-8, help="block: relative residual per column")
    p.add_argument("--open", default="lu", choices=["lu", "closed"], help="windows at the lead faces: lu (A_jj with "
                   "Sigma, per energy) or closed (deflated eigensolve like the others, Sigma ignored)")
    p.add_argument("--coarse-eta", type=float, default=0.0, help="coarse solve at E + i eta_c (A_c(z) = z S_c - H_c - "
                   "U^H Sigma(E) U; the fine A keeps E): damps narrow spurious mode-space resonances; x0 = P_c b with it")
    p.add_argument("--out", help="json (default next to --psi: <psi>_mg.json)")
    a = p.parse_args()
    inp = read_inputs(a.inputs)
    c = dict(np.load(a.psi))                                   # dict: ms_solution may re-phase psi_m
    dev, E, b, A, x0, cs, T = ms_solution(inp, c, a.modes, a.sigma_cache)
    psi = c["psi_m"]
    Ec = inp["meta"]["edges"]["Ec_lead"]
    natoms = len(inp["xyz"])
    atom = np.repeat(np.arange(natoms), inp["norb"])[dev.dev]
    Sd = inp["S"].tocsr()[dev.dev][:, dev.dev]
    dens = lambda y: np.bincount(atom, np.real(y.conj() * (Sd @ y)).sum(1), minlength=natoms)
    n_ex = dens(psi)
    big = n_ex > 1e-3 * n_ex.max()
    SL, SR = dev.sigma(E + 1j * (float(c["eta"]) if "eta" in c else 0.0))
    Ac = cs.Ac(E + 1j * a.coarse_eta, SL, SR)
    Pc = lambda r: cs.prolong(Ac.solve(cs.restrict(r)))
    if a.coarse_eta:
        x0 = Pc(b)
        log(f"coarse solve at E + {a.coarse_eta:g} i")
    if a.basis:                                                # tile-built levels instead of the direct basis
        from qtx_ms.vcycle import Levels
        lev = Levels(dev, inp, a.basis)
        lev.set_energy(E, SL, SR, A)
        x0 = lev.ms(b)
        if a.l1 == "schwarz":
            xs = lev.x
            sl = [np.arange(dev.sb[k], dev.sb[k + 1]) for k in range(len(dev.ns))]
            xi = [0.5 * (xs[sl[k]].max() + xs[sl[k + 1]].min()) for k in range(len(sl) - 1)]
            mids = [0.5 * (p + q) for p, q in zip([xs.min()] + xi, xi + [xs.max()])]
            cen = np.sort(np.array(xi + (mids if a.l1_centres == "all" else [])))
            w1, s1, d1 = (float(v) for v in a.l1_cap.split(","))
            S1 = lev.schwarz1(cen, w1, s1, d1, split=a.l1_split == "tile", tov=a.l1_tov, wmin=a.l1_wmin)
        else:
            S1 = {"coarse": None, "exact": "exact"}[a.l1]
        Pc = lev.cycle1(S1)
        log(f"tile levels ({a.basis}), level 1: {a.l1}; tile mode space x0: err "
            f"{np.linalg.norm(c['psi_m'] - x0) / np.linalg.norm(c['psi_m']):.2e}, T {(T(x0) - T(c['psi_m'])) / T(c['psi_m']):+.1e}")
    nb, npsi, Tex = np.linalg.norm(b), np.linalg.norm(psi), T(psi)
    log(f"E = Ec {E - Ec:+.3f}: {b.shape[1]} columns, N {cs.N}, T exact {Tex:.6f}; mode space x0: err "
        f"{np.linalg.norm(psi - x0) / npsi:.2e}, T {(T(x0) - Tex) / Tex:+.1e}, residual {np.linalg.norm(b - A @ x0) / nb:.1e}")

    def stats(x, nmv):
        n = dens(x)
        da = np.abs(n - n_ex)[big] / n_ex[big]
        return dict(matvecs=nmv, res=float(np.linalg.norm(b - A @ x) / nb), err=float(np.linalg.norm(psi - x) / npsi),
                    T_err=float((T(x) - Tex) / Tex), atom_med=float(np.median(da)), atom_max=float(da.max()),
                    Q_err=float((n.sum() - n_ex.sum()) / n_ex.sum()))

    sw = Schwarz(dev, inp, a.width, a.overlap, a.open == "lu") if "schwarz" in a.smoothers else None

    def cycle(r, S, nmv_s):
        """One multiplicative two-level cycle on A z = r from z = 0: returns z and the fine matvecs used."""
        z, nmv, res = np.zeros_like(r), 0, r
        for _ in range(a.nu if S is not None else 0):
            z = z + a.omega * S(res); res = r - A @ z; nmv += 1 + nmv_s
        z = z + Pc(res)
        if S is not None:
            res = r - A @ z; nmv += 1
            for i in range(a.nu):
                z = z + a.omega * S(res); nmv += nmv_s
                if i < a.nu - 1:
                    res = r - A @ z; nmv += 1
        return z, nmv

    def fgmres(S, nmv_s, m):
        """Flexible GMRES per column, right preconditioner = cycle; returns stats after every iteration."""
        X = [[x0[:, j].copy()] for j in range(b.shape[1])]
        NM = [[0] for _ in range(b.shape[1])]
        for j in range(b.shape[1]):
            r0 = b[:, j] - A @ x0[:, j]
            be = np.linalg.norm(r0)
            V = np.zeros((dev.n, m + 1), complex); Z = np.zeros((dev.n, m), complex)
            Hh = np.zeros((m + 1, m), complex); V[:, 0] = r0 / be
            nmv = 1
            for i in range(m):
                Z[:, i], k = (lambda zz: (zz[0][:, 0], zz[1]))(cycle(V[:, i:i + 1], S, nmv_s))
                w = A @ Z[:, i]; nmv += k + 1
                for _ in range(2):
                    h = V[:, :i + 1].conj().T @ w; Hh[:i + 1, i] += h; w -= V[:, :i + 1] @ h
                Hh[i + 1, i] = np.linalg.norm(w); V[:, i + 1] = w / Hh[i + 1, i]
                e1 = np.zeros(i + 2, complex); e1[0] = be
                y = np.linalg.lstsq(Hh[:i + 2, :i + 1], e1, rcond=None)[0]
                X[j].append(x0[:, j] + Z[:, :i + 1] @ y); NM[j].append(nmv)
                if np.linalg.norm(Hh[:i + 2, :i + 1] @ y - e1) < 1e-12 * np.linalg.norm(b[:, j]):
                    break
        L = max(map(len, X))
        return [stats(np.column_stack([X[j][min(i, len(X[j]) - 1)] for j in range(len(X))]),
                      max(NM[j][min(i, len(NM[j]) - 1)] for j in range(len(X)))) for i in range(L)]
    runs = []
    for sm in a.smoothers.split(","):
        for delta in ([float(d) for d in a.delta.split(",")] if sm == "schwarz" else [None]):
            if sm == "cap":
                t = time.time()
                S = CapSchwarz(dev, inp, *(float(v) for v in a.cap.split(",")), spacing=a.cap_spacing,
                               solver=a.cap_solver, dE0=a.cap_dE0, centres=a.cap_centres,
                               workers=a.workers, split=[int(v) for v in a.cap_split.split(",")], tov=a.cap_tov,
                               tile_orbs=a.cap_tile_orbs, xsplit=a.cap_xsplit, tile_cut=a.cap_tile_cut)
                S.set_energy(E, A)
                tag = (f"cap-schwarz {a.cap} {a.cap_centres} h {a.cap_spacing} {a.cap_solver} dE0 {a.cap_dE0:g} "
                       f"split {a.cap_split} tov {a.cap_tov:g} " + (f"tiles {a.cap_tile_orbs}/x{a.cap_xsplit} " if a.cap_centres == "tile" else "") +
                       f"(factor {time.time() - t:.0f} s)")
            elif sm == "schwarz":
                sw.set_energy(E, A, delta)
                S = sw
                tag = f"schwarz w{a.width:g}/o{a.overlap:g} delta {delta:g} (deflated {min(sw.ndefl)}..{max(sw.ndefl)} per window)"
            elif sm == "gmres":
                S = lambda r: np.column_stack([gmres_fixed(lambda v: A @ v, r[:, j], a.k) for j in range(r.shape[1])])
                tag = f"gmres({a.k})"
            else:
                S, tag = None, "none"
            nmv_s = 0 if S is None else (a.k if sm == "gmres" else 0)   # matvecs inside the smoother
            t = time.time()
            if a.accel == "block":
                clock = {}
                def timed(key, f):
                    def g(r):
                        t0 = time.time(); out = f(r); clock[key] = clock.get(key, 0.0) + time.time() - t0
                        return out
                    return g
                Afine = A
                class Timed:                                     # A @ v with timing
                    def __matmul__(self, v):
                        t0 = time.time(); out = Afine @ v; clock["mv_cycle"] = clock.get("mv_cycle", 0.0) + time.time() - t0
                        return out
                At, St, Pt = Timed(), timed("smoother", S) if S is not None else None, timed("coarse", Pc)
                def prec(r):
                    z = np.zeros_like(r)
                    res = r
                    for _ in range((a.nu if a.pre is None else a.pre) if St is not None else 0):
                        z = z + a.omega * St(res); res = r - At @ z
                    z = z + Pt(res)
                    if St is not None and a.nu > 0:
                        res = r - At @ z
                        for i in range(a.nu):
                            z = z + a.omega * St(res)
                            if i < a.nu - 1:
                                res = r - At @ z
                    return z
                x, its, rres = fgmres_block(A, b, x0, prec, a.tol, a.cycles, clock)
                tit = time.time() - t
                s_ = stats(x, 0)
                tfac = getattr(S, "twall", 0.0)
                log(f"  block {tag}: {b.shape[1]} columns, its {its.min()}..{its.max()}, residual {rres.max():.1e}, "
                    f"err {s_['err']:.1e}, T {s_['T_err']:+.0e}, atom max {s_['atom_max']:.0e}, Q {s_['Q_err']:+.0e}; "
                    f"factor {tfac:.1f} s + iterations {tit:.1f} s (" + ", ".join(f"{k} {v:.1f}" for k, v in clock.items())
                    + f"), total {tfac + tit:.1f} s")
                runs.append(dict(smoother=tag, accel="block", its=its.tolist(), res=rres.tolist(), final=s_,
                                 t_factor=tfac, t_iter=tit, clock=clock))
                continue
            if a.accel == "fgmres":
                h = fgmres(S, nmv_s, a.cycles)
                tag = "fgmres + " + tag
                log(f"  {tag}: " + " | ".join(f"{i}: err {s['err']:.1e} res {s['res']:.1e} T {s['T_err']:+.0e} "
                                             f"atom {s['atom_max']:.0e} mv {s['matvecs']}"
                                             for i, s in enumerate(h) if i in (0, 1, 2, 4, 8, 16, len(h) - 1))
                    + f"; {time.time() - t:.0f} s")
                runs.append(dict(smoother=tag, nu=a.nu, omega=a.omega, hist=h, t_iter=time.time() - t,
                                 t_factor=sum(getattr(S, "tfac", [0])), sizes=[len(i) for i in getattr(S, "idx", [])]))
                continue
            x, nmv = x0.copy(), 0
            hist = [stats(x, nmv)]
            t = time.time()
            for cyc in range(a.cycles):
                r = b - A @ x; nmv += 1
                for _ in range(a.nu if S is not None else 0):
                    x = x + a.omega * S(r); r = b - A @ x; nmv += 1 + nmv_s
                x = x + Pc(r); r = b - A @ x; nmv += 1
                for _ in range(a.nu if S is not None else 0):
                    x = x + a.omega * S(r); nmv += nmv_s
                    if _ < a.nu - 1:
                        r = b - A @ x; nmv += 1
                hist.append(stats(x, nmv))
                if hist[-1]["err"] > 1e3 or hist[-1]["err"] < 1e-10:
                    break
            h = hist
            log(f"  {tag}: " + " | ".join(f"{i}: err {s['err']:.1e} res {s['res']:.1e} T {s['T_err']:+.0e} atom {s['atom_max']:.0e}"
                                         for i, s in enumerate(h) if i in (0, 1, 2, 4, 8, len(h) - 1))
                + f"; {time.time() - t:.0f} s")
            runs.append(dict(smoother=tag, nu=a.nu, omega=a.omega, hist=h))
    out = a.out or os.path.splitext(a.psi)[0] + "_mg.json"
    json.dump(dict(E=E, dE=E - Ec, N=cs.N, width=a.width, overlap=a.overlap, coarse_eta=a.coarse_eta, cap_dE0=a.cap_dE0,
                   x0=stats(x0, 0), runs=runs), open(out, "w"), indent=1)
    log(f"-> {out}")


if __name__ == "__main__":
    main()
