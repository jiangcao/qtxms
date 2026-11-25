"""Symbolic part: from the coupling graph of the cores to the levels, their windows and the block pairs every rank needs.
Nothing here depends on the energy or on the bases; every rank computes the same plan.

Level 1 blocks = cores of the fine windows, level k + 1 blocks = pairs of level-k blocks along x or y. A block belongs to one rank
(level 1: contiguous x positions; merged: the owner of its first kid). A block pair (i, j), i <= j, of coupled blocks is computed
once, on the owner of i, and sent to the ranks that need it: for a window, for a pair of the next level, for the rows of A_k."""
import numpy as np


class Win:
    """Window of level-k blocks around one or more parents (merged blocks): smoother of A_k on it, basis of the parents.
    Parents with the same blocks (y pairs over the full width) share it."""

    def __init__(self, blocks, parents, owner):
        self.blocks, self.parents, self.owner = blocks, parents, owner


class Level:
    def __init__(self, k, kids, grid, G, owner, of_core):
        self.k, self.kids, self.grid, self.G, self.owner, self.of_core = k, kids, grid, G, np.asarray(owner), of_core
        self.n = len(self.owner)
        self.nbr = [np.where(G[i])[0] for i in range(self.n)]
        self.pairs = [(i, int(j)) for i in range(self.n) for j in self.nbr[i] if i <= j]
        self.wins = []

    def inside(self, blocks):
        s = set(int(b) for b in blocks)
        return [(i, int(j)) for i in sorted(s) for j in self.nbr[i] if i <= j and int(j) in s]


def merge(L, axis, halo):
    """The level of pairs of L along axis; fills L.wins with the windows around the pairs (x halo in blocks, None: all coupled)."""
    ix, iy = L.grid
    key = (ix // 2) * 100000 + iy if axis == "x" else ix * 100000 + iy // 2
    kids = [np.where(key == v)[0] for v in np.unique(key)]
    first = np.array([ks[0] for ks in kids])
    par = np.empty(L.n, int)
    for c, ks in enumerate(kids):
        par[ks] = c
    M = np.zeros((len(kids), L.n), np.int32)
    M[par, np.arange(L.n)] = 1
    N = Level(L.k + 1, kids, (ix[first] // 2, iy[first]) if axis == "x" else (ix[first], iy[first] // 2),
              (M @ L.G.astype(np.int32) @ M.T) > 0, L.owner[first], par[L.of_core])
    grp = {}
    for c, ks in enumerate(kids):
        w = np.unique(np.concatenate([L.nbr[b] for b in ks]))
        if halo is not None:
            w = w[np.abs(ix[w][:, None] - ix[ks][None, :]).min(1) <= halo]
        grp.setdefault((int(N.owner[c]), tuple(int(b) for b in w)), []).append(c)
    L.wins = [Win(np.array(w), ps, o) for (o, w), ps in grp.items()]
    return N


class Plan:
    """levels [None, L_1, .., L_K]; need_pairs[k][r], need_U[k][r]: the pairs / bases of level k rank r works with."""

    def __init__(self, G, grid, owner, ups, lead_cores, size):
        self.size, self.levels = size, [None, Level(1, None, grid, G, owner, np.arange(len(owner)))]
        for axis, halo in ups:
            self.levels.append(merge(self.levels[-1], axis, halo))
        self.K = len(self.levels) - 1
        self.lead_cores = lead_cores                                   # [cores of the left face, of the right face]
        R = range(size)
        self.need_pairs = [None] + [[set() for _ in R] for _ in range(self.K)]
        self.need_U = [None] + [[set() for _ in R] for _ in range(self.K)]
        self.uses = [0] * (self.K + 1)                                 # pair uses in windows and Galerkin products (statistics)
        for k in range(1, self.K + 1):
            L, N = self.levels[k], self.levels[k + 1] if k < self.K else None
            for i, j in L.pairs:
                self.need_pairs[k][L.owner[i]].add((i, j))             # computed here
                self.need_U[k][L.owner[i]].update((i, j))
                if N is not None:                                      # rows of A_k
                    self.need_pairs[k][L.owner[j]].add((i, j))
            if N is None:
                break
            for w in L.wins:
                p = L.inside(w.blocks)
                self.need_pairs[k][w.owner].update(p)
                self.uses[k] += len(p)
            for I, J in N.pairs:
                p = {(min(a, b), max(a, b)) for a in N.kids[I] for b in N.kids[J] if L.G[a, b]}
                self.need_pairs[k][N.owner[I]].update(p)
                self.uses[k] += len(p)
            for b in range(N.n):                                       # prolongation: on the owners of the kids
                for a in N.kids[b]:
                    self.need_U[k + 1][L.owner[a]].add(b)
        for cs in lead_cores:                                          # the projected Sigma: every rank
            for r in R:
                self.chain(cs, self.K, r)

    def chain(self, cores, k, r):
        """Rank r needs the bases of levels 1 .. k over the cores."""
        for m in range(1, k + 1):
            self.need_U[m][r].update(int(b) for b in np.unique(self.levels[m].of_core[np.asarray(cores, int)]))

    def add_chains(self, comm, chains):
        """chains [(cores, k)] this rank needs (its CAPs): known to all ranks afterwards."""
        for r, cs in enumerate(comm.allgather(chains)):
            for c, k in cs:
                self.chain(c, k, r)

    def report(self, log):
        L1 = self.levels[1]
        log(f"plan: {L1.n} cores, {L1.grid[0].max() + 1} x {L1.grid[1].max() + 1}, {int(L1.G.sum())} coupled "
            f"({L1.G.sum(1).min()}..{L1.G.sum(1).max()} per core); leads couple {len(self.lead_cores[0])} / {len(self.lead_cores[1])} cores")
        for k in range(1, self.K + 1):
            L = self.levels[k]
            own = [int((L.owner == r).sum()) for r in range(self.size)]
            mv = sum(len(self.need_pairs[k][r]) for r in range(self.size)) - len(L.pairs)
            s = f"  level {k}: {L.n} blocks (per rank {own}), {len(L.pairs)} block pairs formed once, {mv} copies sent"
            if L.wins:
                s += (f"; used {self.uses[k]} times by {len(L.wins)} windows ({sum(len(w.parents) for w in L.wins)} parents, "
                      f"{min(len(w.blocks) for w in L.wins)}..{max(len(w.blocks) for w in L.wins)} blocks) and level {k + 1}")
            log(s)


def swap(comm, have, needs, src, shape, dtype, xp):
    """Move keyed arrays between ranks: needs[r] = keys rank r wants, src(key) = the rank that has it. One message per peer;
    returns {key: array} received. Collective."""
    me, out = comm.rank, {}
    if comm.size == 1:
        return out
    sz = lambda q: int(np.prod(shape(q)))
    send = {r: sorted(q for q in needs[r] if src(q) == me) for r in range(comm.size) if r != me}
    recv = {r: sorted(q for q in needs[me] if src(q) == r) for r in range(comm.size) if r != me}
    sb = {r: xp.concatenate([xp.ascontiguousarray(have[q]).ravel() for q in qs]) for r, qs in send.items() if sum(map(sz, qs))}
    rb = {r: xp.empty(sum(map(sz, qs)), dtype) for r, qs in recv.items() if sum(map(sz, qs))}
    comm.exchange(sb, rb)
    for r, qs in recv.items():
        o = 0
        for q in qs:
            out[q] = rb[r][o:o + sz(q)].reshape(shape(q)).copy() if r in rb else xp.zeros(shape(q), dtype)
            o += sz(q)
    return out
