"""Fine level: the rows of S, H of the own cores and of the own windows (one CSR pattern, two real value arrays) on the device;
dense blocks of them, the fine operator and the CAP-Schwarz smoother S_0."""
import numpy as np
import scipy.sparse as sp

from .. import dist, gpu


def _xp():
    return gpu.cp if gpu.on else np


def r2(x):
    """Real view of a C-ordered complex block (n x c -> n x 2c, re / im interleaved) and back (c2): real matrix @ complex block."""
    x = _xp().ascontiguousarray(x)
    return x.view(x.real.dtype)


def c2(y):
    y = _xp().ascontiguousarray(y)
    return y.view({4: np.complex64, 8: np.complex128}[y.dtype.itemsize])


def inverse_rows(A, wr):
    """(A^-1)[wr, :] of a dense A (F-ordered, overwritten)."""
    xp = _xp()
    lu = gpu.lu_factor(A)
    E = xp.zeros((A.shape[0], len(wr)), A.dtype, order="F")
    E[wr, xp.arange(len(wr))] = 1
    return xp.ascontiguousarray(gpu.lu_solve(lu, E, trans=2).conj().T)


class Fine:
    def __init__(self, dev, sw, owner, comm):
        """sw: CapSchwarz geometry (idx, core, ramp per window); owner of the windows = cores."""
        xp = _xp()
        self.dev, self.sw, self.comm, self.n = dev, sw, comm, dev.n
        self.cores = [i[c] for i, c in zip(sw.idx, sw.core)]
        self.own = [int(k) for k in np.where(np.asarray(owner) == comm.rank)[0]]
        self.part = dist.Part(comm, self.cores, owner, dev.n)
        self.lab, self.posin = np.empty(dev.n, int), np.empty(dev.n, int)
        for k, c in enumerate(self.cores):
            self.lab[c], self.posin[c] = k, np.arange(len(c))
        wr = np.unique(np.concatenate([sw.idx[k] for k in self.own]))
        rows = np.concatenate([self.part.idx, np.setdiff1d(wr, self.part.idx)])    # own rows first
        if hasattr(dev.S, "hold"):
            dev.hold(rows)
        S, H = dev.S[rows].tocsr(), dev.H[rows].tocsr()
        assert not np.iscomplexobj(S.data) and not np.iscomplexobj(H.data), "qtx_ms.hier: real H, S"
        M = (S + 1j * H).tocsr()                                       # one pattern for both
        del S, H
        if hasattr(dev.S, "hold"):
            dev.drop()
        M.sort_indices()
        nl = self.part.nloc
        # core coupling graph (own rows, summed over the ranks; the lead Sigma couples its whole face)
        G = np.zeros((len(self.cores),) * 2, np.int32)
        rl = self.lab[rows[:nl]]
        for k in self.own:
            for i in np.where(rl == k)[0]:
                G[k, self.lab[M.indices[M.indptr[i]:M.indptr[i + 1]]]] = 1
        G = np.asarray(comm.sum(G)) > 0
        for b in (dev.bL, dev.bR):
            u = np.unique(self.lab[b])
            G[np.ix_(u, u)] = True
        self.G = G | G.T
        self.row_of = np.full(dev.n, -1, np.int64)
        self.row_of[rows] = np.arange(len(rows))
        self.ip_h = M.indptr.astype(np.int64)
        self.ip, self.ix = xp.asarray(self.ip_h), xp.asarray(M.indices.astype(np.int32))
        self.s, self.h = xp.asarray(M.data.real.copy()), xp.asarray(M.data.imag.copy())
        self.nnz_own = int(self.ip_h[nl])
        need = np.zeros(dev.n, bool)
        need[M.indices[:self.nnz_own]] = True
        self.lead = []                                                 # per side: own rows of the face, all its columns
        for b in (dev.bL, dev.bR):
            ii = np.where(self.part.gown[b] == comm.rank)[0]
            if len(ii):
                need[b] = True
            self.lead.append((ii, xp.asarray(self.part.gloc[b[ii]]), xp.asarray(b)))
        del M
        self.ex = dist.Exch(self.part, np.where(need)[0])              # halo of the operator
        g = np.full(self.part.nloc + self.ex.ng, -1, np.int64)
        j = np.where(self.ex.pos >= 0)[0]
        g[self.ex.pos[j]] = j
        self.gidx = xp.asarray(g)
        self.exw = dist.Exch(self.part, wr)                            # halo of the windows

    @property
    def gb(self):
        return sum(a.nbytes for a in (self.ip, self.ix, self.s, self.h)) / 2 ** 30

    def dense(self, rows, cols):
        """Dense (2, rows, cols): S and H."""
        xp = _xp()
        r = self.row_of[rows]
        assert (r >= 0).all(), "rows of another rank"
        st, cnt = self.ip_h[r], self.ip_h[r + 1] - self.ip_h[r]
        o = np.concatenate([[0], np.cumsum(cnt)])
        og, st = xp.asarray(o), xp.asarray(st)
        q = xp.arange(int(o[-1]))
        rid = xp.searchsorted(og, q, side="right") - 1
        q += st[rid] - og[rid]                                         # nonzeros of the rows
        cp_ = xp.full(self.n, -1, np.int64)
        cp_[xp.asarray(cols)] = xp.arange(len(cols))
        c = cp_[self.ix[q]]
        m = c >= 0
        f, q = rid[m] * len(cols) + c[m], q[m]
        D = xp.zeros((2, len(rows) * len(cols)))
        D[0, f], D[1, f] = self.s[q], self.h[q]
        return D.reshape(2, len(rows), len(cols))

    # ---- fine operator at one energy
    def set_energy(self, E, Sig):
        """E S - H - Sigma of the own rows; Sig = (Sigma_L, Sigma_R) numpy."""
        xp = _xp()
        a, nl = E * self.s[:self.nnz_own] - self.h[:self.nnz_own], self.part.nloc
        ip, ix = self.ip[:nl + 1], self.ix[:self.nnz_own]
        if gpu.on:
            import cupyx.scipy.sparse as csp
            self.A = csp.csr_matrix((a, ix, ip.astype(np.int32)), shape=(nl, self.n), copy=False)
        else:
            self.A = sp.csr_matrix((a, ix, ip), shape=(nl, self.n))
        self.sig = [(r, b, xp.asarray(S[ii], complex)) for S, (ii, r, b) in zip(Sig, self.lead) if len(ii)]

    def __matmul__(self, x):
        xp = _xp()
        g = xp.zeros((self.n, x.shape[1]), complex)
        g[self.gidx] = self.ex(x)
        y = c2(self.A @ r2(g))
        for r, b, S in self.sig:
            y[r] -= S @ g[b]
        return y

    # ---- fine windows
    def window(self, k, perm=None):
        """S_w, H_w, i s P S P of window k (device), and its lead part [(side, window positions, positions on the face)]."""
        xp, sw, idx = _xp(), self.sw, self.sw.idx[k]
        ramp = sw.ramp(k)
        if perm is not None:
            idx, ramp = idx[perm], ramp[perm]
        D = self.dense(idx, idx)
        p = xp.asarray(np.sqrt(ramp))
        lead = []
        for side, b in enumerate((self.dev.bL, self.dev.bR)):
            w = np.where(np.isin(idx, b))[0]
            if len(w):
                lead.append((side, xp.asarray(w), np.searchsorted(b, idx[w])))
        return D[0], D[1], sw.s * (p[:, None] * D[0] * p[None, :]), lead

    def smoother(self, E, Sig, dt=np.complex64):
        """S_0: CAP-Schwarz (RAS) with the window inverses' rows on the cores."""
        xp, ops = _xp(), []
        for k in self.own:
            S, H, C, lead = self.window(k)
            A = xp.asfortranarray((E * S - H).astype(dt) + 1j * C.astype(dt))
            del S, H, C
            for side, w, j in lead:
                A[w[:, None], w] -= xp.asarray(Sig[side][np.ix_(j, j)], dt)
            R = inverse_rows(A, xp.asarray(np.where(self.sw.core[k])[0]))
            del A
            ops.append((xp.asarray(self.exw.pos[self.sw.idx[k]]), xp.asarray(self.part.gloc[self.cores[k]]), R))

        def S0(r):
            x = self.exw(r).astype(dt)
            z = xp.zeros(r.shape, r.dtype)
            for rw, out, R in ops:
                z[out] = R @ x[rw]
            return z
        S0.gb = sum(R.nbytes for _, _, R in ops) / 2 ** 30
        return S0
