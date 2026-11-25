"""Numerical part of the hierarchy. Once for all energies (setup): bases U_k, S_k and H_k of the block pairs, projected CAPs and lead
faces. Per energy (set_energy): window factors of every level, A_k rows, LU of the deepest level. cycle(): V-cycle, single precision
below the fine operator."""
import hashlib, json, os, time
import numpy as np
import scipy.sparse as sp

from .. import dist, gpu
from .. import msbasis as mb
from .fine import Fine, _xp, c2, inverse_rows, r2
from .obc import contours
from .plan import Plan, swap
from .slow import Slow

F4, C8 = np.float32, np.complex64


def offs(r):
    return np.concatenate([[0], np.cumsum(r)]).astype(int)


def sync():
    if gpu.on:
        gpu.sync()


def eig_above(rho, cut):
    """Eigenvectors of the real symmetric rho with eigenvalue > cut, largest first."""
    xp = _xp()
    ev, V = xp.linalg.eigh(0.5 * (rho + rho.T))
    return xp.ascontiguousarray(V[:, ev > cut][:, ::-1])


def fold(S, K, add, ne, zs, dzs, qs):
    """Contour sum over the points qs of G_PP(z) of M(z) = z S + K - Sigma(z), P = the rows after the first ne: the rest is folded
    in (one LU and one solve per point). add(M, q): subtracts Sigma."""
    xp = _xp()
    acc = xp.zeros((S.shape[0] - ne,) * 2, complex)
    for q in qs:
        M = zs[q] * S + K
        add(M, q)
        Sp = M[ne:, ne:]
        if ne:
            Sp = Sp - M[ne:, :ne] @ gpu.lu_solve(gpu.lu_factor(xp.asfortranarray(M[:ne, :ne])), xp.asfortranarray(M[:ne, ne:]))
        acc += dzs[q] * xp.linalg.inv(Sp)
        del M, Sp
    return acc


class Hier:
    def __init__(self, dev, inp, sw, comm, obc, ups, cut1, sl, cache=None, name="dev", log=mb.log):
        """sw: CapSchwarz (geometry of the fine windows, x windows x y strips); ups [(axis, cut, halo)] merges; cut1: cut of level 1;
        sl: contours [(mu1, mu2, npts, weight)] in absolute energies; cache: folder of the bases."""
        self.dev, self.sw, self.comm, self.obc, self.log = dev, sw, comm, obc, log
        self.slow, self.slow_par = None, None                          # slow-mode coarse level: (m, q, rounds), with schwarz
        self.bottom_kind, self.bottom_res = "bt", None              # solver of the deepest level: dense LU or block-tridiagonal LU
        self.geo = mb.Geometry(inp["xyz"], inp["meta"].get("periodic"))
        self.atom = np.repeat(np.arange(len(inp["norb"])), inp["norb"])[dev.dev]
        self.cuts, self.t = [None, cut1] + [c for _, c, _ in ups], {}
        self.zs, self.dzs = contours(sl)
        nw, nx = len(sw.idx), len(sw.centres)
        ny = nw // nx
        cost = [float(sum(len(sw.idx[ix * ny + j]) ** 3 for j in range(ny))) for ix in range(nx)]
        owner = np.repeat(dist.chunks(cost, comm.size), ny)            # whole x positions: the strips of one window stay together
        t = time.time()
        self.fine = f = Fine(dev, sw, owner, comm)
        self.tick("fine rows", t)
        lead = [np.unique(f.lab[b]) for b in (dev.bL, dev.bR)]
        self.plan = p = Plan(f.G, (np.arange(nw) // ny, np.arange(nw) % ny), owner, [(a, h) for a, _, h in ups], lead, comm.size)
        self.K, self.lv = p.K, p.levels
        t = time.time()
        self.cap = {}                                                  # CAPs of the own windows: (ramp orbitals, P S P)
        for k in range(1, self.K):
            for g, w in enumerate(self.lv[k].wins):
                if w.owner == comm.rank:
                    self.cap[k, g] = self._cap(k, w)
        p.add_chains(comm, [(np.unique(f.lab[c[0]]), k) for (k, _), c in self.cap.items() if c is not None])
        self.tick("cap", t)
        p.report(log)
        self.dir = None
        if cache:
            tag = json.dumps([name, dev.n, sw.s, sw.d, [len(i) for i in sw.idx], ups, cut1]
                             + ([list(sl[0][:2]), sl[0][2]] if len(sl) == 1 and sl[0][3] == 1 else [[list(c) for c in sl]]) + ["hier1"])
            self.dir = os.path.join(cache, "hier_" + hashlib.sha1(tag.encode()).hexdigest()[:12])
            os.makedirs(self.dir, exist_ok=True)
        self.cached = True
        self.U, self.r, self.off, self.N = {}, {}, {}, {}
        self.SH, self.Y, self.part, self.order, self.capw, self.trans, self.halo = {}, {}, {}, {}, {}, {}, {}

    def tick(self, key, t0):
        sync()
        self.t[key] = self.t.get(key, 0.0) + time.time() - t0

    # ------------------------------------------------------------------ setup
    def setup(self):
        """Bases and pair blocks of every level."""
        self.sigc = {}
        for k in range(1, self.K + 1):
            t = time.time()
            self._bases(k)
            self.tick("bases", t)
            t = time.time()
            self._galerkin(k)
            self._lead(k)
            self.tick("galerkin", t)
            self._vectors(k)
            self.log(f"  level {k}: N {self.N[k]}, {self.r[k].min()}..{int(np.median(self.r[k]))}..{self.r[k].max()} per block, pairs on this "
                     f"rank {sum(v.nbytes for v in self.SH[k].values()) / 2**30:.2f} GB; {self.t['bases']:.0f} s bases, {self.t['galerkin']:.0f} s Galerkin so far")
            if k < self.K:
                self._windows(k)
            gpu.free()
        self.sigc = {}

    def sigma_c(self, q, side):
        if (q, side) not in self.sigc:
            self.sigc[q, side] = np.array(self.obc.sigma(self.zs[q], side))
        return self.sigc[q, side]

    def _bases(self, k):
        L, me, xp, U = self.lv[k], self.comm.rank, _xp(), {}
        own = [b for b in range(L.n) if L.owner[b] == me]
        f = lambda b: os.path.join(self.dir, f"U{k}_{b}.npy")
        have = bool(self.dir) and self.cached and all(os.path.exists(f(b)) for b in own)
        self.cached = all(self.comm.allgather(have))                   # a new level: the files above it belong to other bases
        if self.cached:
            U = {b: xp.asarray(np.load(f(b))) for b in own}
        else:
            U = self._bases_fine(own) if k == 1 else self._bases_level(k - 1)
            if self.dir:
                for b, u in U.items():
                    tmp = dist.tmpname(f(b))[:-4] + ".npy"
                    np.save(tmp, gpu.asnumpy(u))
                    os.replace(tmp, f(b))
        r = np.zeros(L.n, int)
        for d in self.comm.allgather({b: u.shape[1] for b, u in U.items()}):
            for b, v in d.items():
                r[b] = v
        self.r[k], self.off[k] = r, offs(r)
        self.N[k] = int(r.sum())
        nrow = (lambda b: len(self.fine.cores[b])) if k == 1 else (lambda b: int(self.r[k - 1][L.kids[b]].sum()))
        self.U[k] = {**U, **swap(self.comm, U, self.plan.need_U[k], lambda b: L.owner[b], lambda b: (nrow(b), int(r[b])), float, xp)}

    def _bases_fine(self, own):
        """Level 1: eigenvectors of the core block of the rho of every fine window."""
        xp, U = _xp(), {}
        for k in own:
            core = self.sw.core[k]
            ne = int((~core).sum())
            S, H, C, lead = self.fine.window(k, np.concatenate([np.where(~core)[0], np.where(core)[0]]))

            def add(M, q):
                for side, w, j in lead:
                    M[w[:, None], w] -= xp.asarray(self.sigma_c(q, side)[np.ix_(j, j)])
            acc = fold(S, 1j * C - H, add, ne, self.zs, self.dzs, range(len(self.zs)))
            del S, H, C
            U[k] = eig_above(-acc.imag / np.pi, self.cuts[1])
        return U

    def _bases_level(self, k):
        """Bases of level k + 1 from the windows of level k. Few windows: every window is shared by all ranks over the points."""
        xp, me, size = _xp(), self.comm.rank, self.comm.size
        L, N, U = self.lv[k], self.lv[k + 1], {}
        shared = size > 1 and len(L.wins) <= 3 * size
        for g, w in enumerate(L.wins):
            if not shared and w.owner != me:
                continue
            order, o, npr = self.order[k][g]
            n, S, Kw = int(o[-1]), None, None
            if w.owner == me:
                D = self.dense(k, order, order)
                S, Kw = D[0], (-D[1]).astype(complex)
                del D
                for a, b, C in self.capw[k, g]:
                    Kw[a, b] += 1j * C.astype(float)
            if shared:
                S = self.comm.sum(S if w.owner == me else xp.zeros((n, n)))
                Kw = self.comm.sum(Kw if w.owner == me else xp.zeros((n, n), complex))
            lead = self._window_lead(k, order, o)
            Yw = [(side, p, self.Y[k][side][0][:, c]) for side, p, c in lead]

            def add(M, q):
                for side, p, Y in Yw:
                    M[p[:, None], p] -= Y.T @ (xp.asarray(self.sigma_c(q, side)) @ Y)
            nz = len(self.zs)
            acc = fold(S, Kw, add, n - npr, self.zs, self.dzs, range(me, nz, size) if shared else range(nz))
            del S, Kw, Yw
            if shared:
                acc = self.comm.sum(acc)
            if w.owner == me:
                rho, a = -acc.imag / np.pi, 0
                for I in w.parents:
                    m = int(self.r[k][N.kids[I]].sum())
                    U[I] = eig_above(rho[a:a + m, a:a + m], self.cuts[k + 1])
                    a += m
            del acc
            gpu.free()
        return U

    def dense(self, k, rows, cols):
        """Dense (2, rows, cols) of S_k, H_k for two block lists, from the pairs."""
        xp, L, r = _xp(), self.lv[k], self.r[k]
        a, b = offs(r[rows]), offs(r[cols])
        D = xp.zeros((2, a[-1], b[-1]))
        loc = {int(j): q for q, j in enumerate(cols)}
        for p, i in enumerate(rows):
            i = int(i)
            for j in L.nbr[i]:
                q = loc.get(int(j))
                if q is not None:
                    D[:, a[p]:a[p + 1], b[q]:b[q + 1]] = self.SH[k][i, int(j)] if i <= j else self.SH[k][int(j), i].transpose(0, 2, 1)
        return D

    def _galerkin(self, k):
        """S_k, H_k of the pairs (I, J), I <= J, of the own I: the rows of I are projected once for all J."""
        xp, L, me, f = _xp(), self.lv[k], self.comm.rank, self.fine
        X = {}
        for I in (b for b in range(L.n) if L.owner[b] == me):
            Js = [int(J) for J in L.nbr[I] if J >= I]
            if k == 1:
                sz = [len(f.cores[J]) for J in Js]
                D = f.dense(f.cores[I], np.concatenate([f.cores[J] for J in Js]))
            else:
                sz = [int(self.r[k - 1][L.kids[J]].sum()) for J in Js]
                D = self.dense(k - 1, L.kids[I], np.concatenate([L.kids[J] for J in Js]))
            W = self.U[k][I].T @ D
            del D
            o = offs(sz)
            for q, J in enumerate(Js):
                X[I, J] = W[:, :, o[q]:o[q + 1]] @ self.U[k][J]
            del W
        need = self.plan.need_pairs[k]
        X.update(swap(self.comm, X, need, lambda p: L.owner[p[0]], lambda p: (2, int(self.r[k][p[0]]), int(self.r[k][p[1]])), float, xp))
        self.SH[k] = X

    def _lead(self, k):
        """Y_k per side: the rows of the prolongation at the lead face (face x columns of the lead blocks), and the lead blocks."""
        xp, L, out = _xp(), self.lv[k], []
        for side, b in enumerate((self.dev.bL, self.dev.bR)):
            lb = np.unique(L.of_core[self.plan.lead_cores[side]])
            o = offs(self.r[k][lb])
            Y = xp.zeros((len(b), o[-1]))
            for q, B in enumerate(lb):
                if k == 1:
                    i = np.where(self.fine.lab[b] == B)[0]
                    Y[xp.asarray(i), o[q]:o[q + 1]] = self.U[1][B][xp.asarray(self.fine.posin[b[i]])]
                    continue
                Yp, lp, op = self.Y[k - 1][side]
                ok = offs(self.r[k - 1][L.kids[B]])
                for a, kid in enumerate(L.kids[B]):
                    j = np.where(lp == kid)[0]
                    if len(j):
                        j = int(j[0])
                        Y[:, o[q]:o[q + 1]] += Yp[:, op[j]:op[j + 1]] @ self.U[k][int(B)][ok[a]:ok[a + 1]]
            out.append((Y, lb, o))
        self.Y[k] = out

    # ------------------------------------------------------------------ windows and CAPs
    def _cap(self, k, w):
        """Ramp orbitals of the window (distance to the nearest atom outside it, minimum image) and s P S P on them."""
        f, sw = self.fine, self.sw
        inside = np.zeros(self.dev.n, bool)
        inside[np.concatenate([f.cores[c] for c in np.where(np.isin(self.lv[k].of_core, w.blocks))[0]])] = True
        F = np.where(inside)[0]
        aw, ao = np.unique(self.atom[F]), np.unique(self.atom[~inside])
        if not len(ao):
            return None
        d, _ = self.geo.tree(ao).query(self.geo.points(aw), distance_upper_bound=sw.d + 6.0)
        d = np.where(np.isfinite(d), d, np.inf)
        ra = np.clip((sw.d - (d - d.min())) / sw.d, 0, 1) ** 2          # the outermost atoms: 1
        ra = ra[np.searchsorted(aw, self.atom[F])]
        Fo, p = F[ra > 0], sp.diags(np.sqrt(ra[ra > 0]))
        S = self.dev.S
        S = S.sub(Fo) if hasattr(S, "sub") else S[Fo][:, Fo]
        return Fo, (sw.s * (p @ S @ p)).tocsr()

    def rows(self, k, b, F):
        """Rows of the prolongation of block b of level k at the device orbitals F (inside b)."""
        xp, f, L = _xp(), self.fine, self.lv[k]
        if k == 1:
            return self.U[1][b][xp.asarray(f.posin[F])]
        out, o, kid = xp.zeros((len(F), int(self.r[k][b]))), offs(self.r[k - 1][L.kids[b]]), self.lv[k - 1].of_core[f.lab[F]]
        for a, c in enumerate(L.kids[b]):
            i = np.where(kid == c)[0]
            if len(i):
                out[xp.asarray(i)] = self.rows(k - 1, int(c), F[i]) @ self.U[k][b][o[a]:o[a + 1]]
        return out

    def _windows(self, k):
        """Per window: block order (the parents' blocks last), offsets, size of the parents' part; projected CAP blocks (single)."""
        L, N, xp = self.lv[k], self.lv[k + 1], _xp()
        self.order[k] = []
        for g, w in enumerate(L.wins):
            P = np.concatenate([N.kids[I] for I in w.parents])
            order = np.concatenate([np.setdiff1d(w.blocks, P), P]).astype(int)
            o = offs(self.r[k][order])
            self.order[k].append((order, o, int(self.r[k][P].sum())))
            if w.owner != self.comm.rank:
                continue
            self.capw[k, g] = []
            if self.cap[k, g] is None:
                continue
            Fo, W = self.cap[k, g]
            blk = L.of_core[self.fine.lab[Fo]]
            hit = [(q, np.where(blk == b)[0]) for q, b in enumerate(order) if (blk == b).any()]
            ro = offs([len(i) for _, i in hit])
            perm = np.concatenate([i for _, i in hit])
            Wd = W[perm][:, perm]
            Wd = gpu.csr(Wd, np.float64) if gpu.on else Wd
            Ys = [self.rows(k, int(order[q]), Fo[i]) for q, i in hit]
            for b, (q2, _) in enumerate(hit):
                X = xp.zeros((len(perm), Ys[b].shape[1]))
                X[ro[b]:ro[b + 1]] = Ys[b]
                X = Wd @ X
                for a, (q1, _) in enumerate(hit):
                    C = Ys[a].T @ X[ro[a]:ro[a + 1]]
                    self.capw[k, g].append((slice(o[q1], o[q1 + 1]), slice(o[q2], o[q2 + 1]), C.astype(F4)))

    def _window_lead(self, k, order, o):
        """[(side, window positions, columns of Y_k)] of the lead blocks of a window."""
        xp, out = _xp(), []
        for side, (_, lb, ol) in enumerate(self.Y[k]):
            p, c = [], []
            for q, b in enumerate(order):
                j = np.where(lb == b)[0]
                if len(j):
                    p.append(np.arange(o[q], o[q + 1])), c.append(np.arange(ol[j[0]], ol[j[0] + 1]))
            if p:
                out.append((side, xp.asarray(np.concatenate(p)), xp.asarray(np.concatenate(c))))
        return out

    # ------------------------------------------------------------------ vectors and transfers
    def entries(self, k, blocks):
        return np.concatenate([np.arange(self.off[k][b], self.off[k][b + 1]) for b in blocks] or [np.zeros(0, int)])

    def _vectors(self, k):
        """Partition of the level-k vectors, halos of A_k, prolongation and restriction to the level below (single precision)."""
        xp, L, me, off = _xp(), self.lv[k], self.comm.rank, self.off[k]
        self.part[k] = part = dist.Part(self.comm, [np.arange(off[b], off[b + 1]) for b in range(L.n)], L.owner, self.N[k])
        own = part.own
        if k < self.K:
            lead = [b for s, (_, lb, _) in enumerate(self.Y[k]) if np.isin(lb, own).any() for b in lb]
            self.halo[k] = dist.Exch(part, self.entries(k, sorted({int(j) for b in own for j in L.nbr[b]} | set(map(int, lead)))))
        if k == 1:
            f = self.fine
            up = [(xp.asarray(f.part.gloc[f.cores[b]]), self.U[1][b].astype(F4), int(part.gloc[off[b]])) for b in own]
            nl = f.part.nloc

            def P(c):
                y = xp.zeros((nl, c.shape[1]), C8)
                for r, U, a in up:
                    y[r] = c2(U @ r2(c[a:a + U.shape[1]]))
                return y

            def PT(x):
                y = xp.zeros((part.nloc, x.shape[1]), C8)
                for r, U, a in up:
                    y[a:a + U.shape[1]] = c2(U.T @ r2(x[r]))
                return y
        else:
            lo, ol = self.part[k - 1], self.off[k - 1]
            ent = lambda b: self.entries(k - 1, L.kids[b])
            below = [b for b in range(L.n) if (self.lv[k - 1].owner[L.kids[b]] == me).any()]      # over own kids
            eP, eT = dist.Exch(part, self.entries(k, below)), dist.Exch(lo, np.concatenate([ent(b) for b in own] or [np.zeros(0, int)]))
            up = []
            for b in below:
                i = ent(b)
                m = lo.gown[i] == me
                U = self.U[k][b].astype(F4)
                up.append((xp.asarray(lo.gloc[i[m]]), U if m.all() else U[xp.asarray(np.where(m)[0])], int(eP.pos[off[b]])))
            down = [(xp.asarray(eT.pos[ent(b)]), self.U[k][b].astype(F4), int(part.gloc[off[b]])) for b in own]

            def P(c):
                c, y = eP(c), xp.zeros((lo.nloc, c.shape[1]), C8)
                for r, U, a in up:
                    y[r] = c2(U @ r2(c[a:a + U.shape[1]]))
                return y

            def PT(x):
                x, y = eT(x), xp.zeros((part.nloc, x.shape[1]), C8)
                for r, U, a in down:
                    y[a:a + U.shape[1]] = c2(U.T @ r2(x[r]))
                return y
        self.trans[k] = P, PT

    # ------------------------------------------------------------------ one energy
    def set_energy(self, E, Sig, schwarz=False):
        """Factors at the energy E (schwarz: of S_0 only, no level is used); Sig = (Sigma_L, Sigma_R), numpy, on every rank."""
        xp = _xp()
        self.S0 = self.A = self.S = self.lu = self.bottom = self.slow = None
        gpu.free()
        t = time.time()
        self.fine.set_energy(E, Sig)
        self.tick("fine operator", t)
        t = time.time()
        self.S0 = self.fine.smoother(E, Sig)
        self.tick("S0", t)
        if self.slow_par:
            t = time.time()
            self.slow = Slow(self.fine, self.S0, self.fine.part, self.comm, *self.slow_par, log=self.log)
            self.tick("slow modes", t)
        self.A, self.S, self.gb, self.schwarz = {}, {}, {}, schwarz
        for k in range(1, 0 if schwarz else self.K + 1):
            t = time.time()
            sg = [Y.T @ (xp.asarray(Sig[s]) @ Y) for s, (Y, _, _) in enumerate(self.Y[k])]
            if k == self.K:
                (self._bottom_bt if self.bottom_kind == "bt" else self._bottom)(E, sg)
                self.tick("bottom LU", t)
                break
            sg = [x.astype(C8) for x in sg]
            A = {p: (E * X[0] - X[1]).astype(F4) for p, X in self.SH[k].items()}
            self.A[k] = self._operator(k, A, sg)
            self.tick("A_k", t)
            t = time.time()
            self.S[k] = self._smoother(k, A, sg)
            self.tick("level smoothers", t)
            del A, sg
            gpu.free()

    def _operator(self, k, A, sg):
        xp, L, part, ex, off = _xp(), self.lv[k], self.part[k], self.halo[k], self.off[k]
        ops = []
        for b in part.own:
            nb = [int(j) for j in L.nbr[b]]
            R = xp.concatenate([A[b, j] if b <= j else A[j, b].T for j in nb], axis=1)
            ops.append((int(part.gloc[off[b]]), R, xp.asarray(ex.pos[self.entries(k, nb)])))
        sig = []
        for side, (_, lb, o) in enumerate(self.Y[k]):
            q = [a for a, b in enumerate(lb) if L.owner[b] == self.comm.rank]
            if q:
                rows = np.concatenate([np.arange(o[a], o[a + 1]) for a in q])
                sig.append((xp.asarray(part.gloc[self.entries(k, lb[q])]), xp.asarray(ex.pos[self.entries(k, lb)]), sg[side][xp.asarray(rows)]))

        def mul(x):
            x = ex(x)
            y = xp.zeros((part.nloc, x.shape[1]), C8)
            for a, R, c in ops:
                y[a:a + R.shape[0]] = c2(R @ r2(x[c]))
            for r, c, Sg in sig:
                y[r] -= Sg @ x[c]
            return y
        return mul

    def _smoother(self, k, A, sg):
        """RAS on A_k: one window inverse (rows of the parents' blocks) per window."""
        xp, L, part, me = _xp(), self.lv[k], self.part[k], self.comm.rank
        mine = [g for g, w in enumerate(L.wins) if w.owner == me]
        ent = lambda g, P: self.entries(k, P)
        wr = {}
        for g in mine:
            order, o, npr = self.order[k][g]
            wr[g] = order[np.searchsorted(o, o[-1] - npr):]
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, int)
        ex = dist.Exch(part, cat([self.entries(k, self.order[k][g][0]) for g in mine]))
        ez = dist.Exch(part, cat([self.entries(k, wr[g]) for g in mine]))
        ops = []
        for g in mine:
            order, o, npr = self.order[k][g]
            n, pos = int(o[-1]), {int(b): q for q, b in enumerate(order)}
            M = xp.zeros((n, n), C8, order="F")
            for i, j in L.inside(order):
                a, b = pos[i], pos[j]
                M[o[a]:o[a + 1], o[b]:o[b + 1]] = A[i, j]
                if i != j:
                    M[o[b]:o[b + 1], o[a]:o[a + 1]] = A[i, j].T
            for a, b, C in self.capw[k, g]:
                M[a, b] += 1j * C
            for side, p, c in self._window_lead(k, order, o):
                M[p[:, None], p] -= sg[side][c[:, None], c]
            R = inverse_rows(M, xp.arange(n - npr, n))
            del M
            ops.append((xp.asarray(ex.pos[self.entries(k, order)]), xp.asarray(ez.pos[self.entries(k, wr[g])]), R))
        self.gb[k] = sum(R.nbytes for _, _, R in ops) / 2 ** 30

        def Sm(r):
            x = ex(r)
            z = xp.zeros((part.nloc + ez.ng, r.shape[1]), C8)
            for c, out, R in ops:
                z[out] = R @ x[c]
            return ez.rev(z, z[:part.nloc])
        return Sm

    def _bottom(self, E, sg):
        """Dense LU of A_K on every rank."""
        xp, k, L, o, part = _xp(), self.K, self.lv[self.K], self.off[self.K], self.part[self.K]
        D = xp.zeros((self.N[k],) * 2)
        for (i, j), X in self.SH[k].items():
            if L.owner[i] == self.comm.rank:
                B = E * X[0] - X[1]
                D[o[i]:o[i + 1], o[j]:o[j + 1]] = B
                if i != j:
                    D[o[j]:o[j + 1], o[i]:o[i + 1]] = B.T
        D = xp.asfortranarray(self.comm.sum(D).astype(complex))
        for side, (_, lb, _) in enumerate(self.Y[k]):
            e = xp.asarray(self.entries(k, lb))
            D[e[:, None], e] -= sg[side]
        self.lu = gpu.lu_factor(D)
        idx = xp.asarray(part.idx)
        self.bottom = lambda r: gpu.lu_solve(self.lu, part.gather(r).astype(complex))[idx].astype(C8)

    def _bottom_bt(self, E, sg):
        """Exact block-tridiagonal LU of A_K over its x positions (slabs) on every rank, double precision: Schur complements
        D_i - L_i-1 D_i-1^-1 U_i-1 from the left; their inverses replace D, D^-1 U replaces U (three blocks per slab in all).
        self.bottom_res: relative residual of a random solve."""
        xp, k, L, part, me = _xp(), self.K, self.lv[self.K], self.part[self.K], self.comm.rank
        ix = np.unique(L.grid[0], return_inverse=True)[1]
        lb = [l for _, l, _ in self.Y[k]]
        reach = lambda i, j: abs(int(ix[i]) - int(ix[j]))
        far = max([reach(i, j) for i, j in L.pairs] + [int(np.ptp(ix[l])) for l in lb])
        ix = ix // max(far, 1)                                         # wider coupling: join neighbouring slabs
        ns = int(ix.max()) + 1
        sl = [np.where(ix == a)[0] for a in range(ns)]
        o, at = [offs(self.r[k][b]) for b in sl], {int(b): (a, q) for a, bs in enumerate(sl) for q, b in enumerate(bs)}
        n = [int(v[-1]) for v in o]
        M = {(a, b): xp.zeros((n[a], n[b]), complex) for a in range(ns) for b in (a - 1, a, a + 1) if 0 <= b < ns}

        def put(i, j, B):
            (a, p), (b, q) = at[int(i)], at[int(j)]
            M[a, b][o[a][p]:o[a][p + 1], o[b][q]:o[b][q + 1]] += B
        for (i, j), X in self.SH[k].items():
            if L.owner[i] == me:
                B = E * X[0] - X[1]
                put(i, j, B)
                if i != j:
                    put(j, i, B.T)
        M = {ab: self.comm.sum(M[ab]) for ab in sorted(M)}
        for side, (_, l, ol) in enumerate(self.Y[k]):
            for p, i in enumerate(l):
                for q, j in enumerate(l):
                    put(i, j, -sg[side][ol[p]:ol[p + 1], ol[q]:ol[q + 1]])
        inv, X = [], []
        for a in range(ns):
            D = M.pop((a, a))
            if a:
                D -= M[a, a - 1] @ X[-1]
            inv.append(xp.linalg.inv(D))
            del D
            if a + 1 < ns:
                X.append(inv[a] @ M.pop((a, a + 1)))
        gpu.free()
        ent = [xp.asarray(self.entries(k, b)) for b in sl]
        Lo = [None] + [M[a, a - 1] for a in range(1, ns)]

        def solve(r):
            t, x = [], xp.empty(r.shape, complex)
            for a in range(ns):
                y = r[ent[a]]
                t.append(inv[a] @ (y - Lo[a] @ t[-1] if a else y))
            z = None
            for a in range(ns - 1, -1, -1):
                z = t[a] - X[a] @ z if a + 1 < ns else t[a]
                x[ent[a]] = z
            return x
        rng = np.random.default_rng(0)
        b = xp.asarray(rng.standard_normal((self.N[k], 2)) + 1j * rng.standard_normal((self.N[k], 2)))
        x = solve(b)
        y, e = xp.zeros_like(b), lambda i: slice(int(self.off[k][i]), int(self.off[k][i + 1]))
        for (i, j), P in self.SH[k].items():                           # A_K x from the block pairs of the owners
            if L.owner[i] == me:
                B = E * P[0] - P[1]
                y[e(i)] += B @ x[e(j)]
                if i != j:
                    y[e(j)] += B.T @ x[e(i)]
        y = self.comm.sum(y)
        for side, l in enumerate(lb):
            q = xp.asarray(self.entries(k, l))
            y[q] -= sg[side] @ x[q]
        d = b - y
        self.bottom_res = float(xp.linalg.norm(d) / xp.linalg.norm(b))
        self.bottom_n = n
        idx = xp.asarray(part.idx)
        self.bottom = lambda r: solve(part.gather(r).astype(complex))[idx].astype(C8)

    # ------------------------------------------------------------------ preconditioner
    def coarse(self, k, r):
        """V-cycle on A_k."""
        if k == self.K:
            return self.bottom(r)
        A, S, (P, PT) = self.A[k], self.S[k], self.trans[k + 1]
        z = S(r)
        z = z + P(self.coarse(k + 1, PT(r - A(z))))
        return z + S(r - A(z))

    def prec(self, r):
        if self.slow is not None:
            return self.slow.cycle(r)
        z = self.S0(r)
        if self.schwarz:
            return z
        P, PT = self.trans[1]
        z = z + P(self.coarse(1, PT((r - self.fine @ z).astype(C8))))
        return z + self.S0(r - self.fine @ z)
