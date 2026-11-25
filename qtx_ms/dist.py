"""Domain decomposition of the multilevel solver (qtx_ms.mlevel --dist; user, 2026-11-24): communicators, the block
partition of every level, halo exchange and the distributed fine operator.

Ownership: the cores of the CAP-Schwarz windows (= level-1 blocks, ordered k = ix * ny + iy) are split into contiguous
chunks of equal factorization cost, one per rank (one rank per GPU). A merged block belongs to the owner of its first
kid. The owner of a block
  * builds it (window rho, pair rho, rows of the Galerkin S / H / A, its Schwarz window)
  * applies it (rows of A_k, the Schwarz solve of its window)
and every vector of every level (fine residual, Krylov basis, level vectors) holds only the rows of the owned blocks.
Communication: halo exchange of vector rows (GPU arrays: NCCL send / recv, or MPI staged through the host), sums of dot
products (MPI, host: reduce + broadcast, so every rank takes the same branch), host blocks (MPI). The deepest level is
solved by domains of slabs (mlevel.DistSlabLU; --bottom repl: the same LU on every rank, GatherSolve). Contour points of
the merge windows move from overloaded to idle ranks (share_points): deep levels have few, unequal blocks.

Communicators: Comm (one rank), MPIComm (mpi4py; nccl = True: GPU arrays through NCCL), ThreadComm (ranks = threads of
one process, numpy only: the unit tests). NCCL is bound with ctypes: the cupy 14 wheels have no compiled NCCL module
(pip: nvidia-nccl-cu13; $QTX_NCCL = path of libnccl.so.2). One rank per GPU: NCCL refuses two ranks on one device (use
--dist mpi there)."""
import ctypes
import os
import socket
import threading
import time

import numpy as np
import scipy.sparse as sp

from . import gpu


def tmpname(f):
    """Name of a temporary file for an atomic write (tmp + os.replace): unique over nodes, processes and threads."""
    return f"{f}.tmp{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}.npz"


def wait_file(f, seconds=60):
    """Wait until a file another rank has written is visible on this node (shared file systems cache lookups)."""
    t = time.time()
    while not os.path.exists(f):
        if time.time() - t > seconds:
            raise FileNotFoundError(f"{f}: written by another rank, not visible here after {seconds} s")
        try:
            os.listdir(os.path.dirname(f) or ".")                # refresh the directory cache
        except OSError:
            pass
        time.sleep(0.5)


# ------------------------------------------------------------------- NCCL
class Nccl:
    """Minimal NCCL binding: point-to-point, sum, broadcast on cupy arrays, on the current cupy stream. Arrays are
    sent as bytes (p2p, bcast) or as reals of their precision (sum; complex = 2 reals)."""
    INT8, F32, F64, SUM = 0, 7, 8, 0

    def __init__(self, mpi):
        self.lib = lib = self._load()
        self.n = mpi.Get_size()
        uid = (ctypes.c_char * 128)()
        if mpi.Get_rank() == 0:
            self._ck(lib.ncclGetUniqueId(ctypes.byref(uid)))
        raw = mpi.bcast(bytes(uid), root=0)

        class UID(ctypes.Structure):                             # passed by value
            _fields_ = [("b", ctypes.c_char * 128)]
        u = UID()
        ctypes.memmove(ctypes.byref(u), raw, 128)
        self.h = ctypes.c_void_p()
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UID, ctypes.c_int]
        self._ck(lib.ncclCommInitRank(ctypes.byref(self.h), self.n, u, mpi.Get_rank()))
        v, sz, i = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        lib.ncclSend.argtypes = lib.ncclRecv.argtypes = [v, sz, i, i, v, v]
        lib.ncclAllReduce.argtypes = [v, v, sz, i, i, v, v]
        lib.ncclBroadcast.argtypes = [v, v, sz, i, i, v, v]

    @staticmethod
    def _load():
        cand = [os.environ.get("QTX_NCCL")]
        try:
            import nvidia
            for p in nvidia.__path__:
                cand.append(os.path.join(p, "nccl", "lib", "libnccl.so.2"))
        except ImportError:
            pass
        for c in [x for x in cand if x and os.path.exists(x)] + ["libnccl.so.2"]:
            try:
                return ctypes.CDLL(c)
            except OSError:
                continue
        raise RuntimeError("libnccl.so.2 not found (pip install nvidia-nccl-cu13, or $QTX_NCCL); --dist mpi needs none")

    def _ck(self, rc):
        if rc != 0:
            self.lib.ncclGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(f"NCCL: {self.lib.ncclGetErrorString(rc).decode()}")

    def _real(self, x):
        one = x.dtype.itemsize // (2 if x.dtype.kind == "c" else 1)
        return (x.size * (2 if x.dtype.kind == "c" else 1)), {4: self.F32, 8: self.F64}[one]

    def exchange(self, sends, recvs):
        """sends / recvs {peer: C-contiguous cupy array}; enqueued on the current stream."""
        st = gpu.cp.cuda.get_current_stream().ptr
        self._ck(self.lib.ncclGroupStart())
        try:
            for p, x in recvs.items():
                self._ck(self.lib.ncclRecv(x.data.ptr, x.nbytes, self.INT8, p, self.h, st))
            for p, x in sends.items():
                self._ck(self.lib.ncclSend(x.data.ptr, x.nbytes, self.INT8, p, self.h, st))
        finally:
            rc = self.lib.ncclGroupEnd()
        self._ck(rc)

    def sum(self, x):
        n, t = self._real(x)
        y = gpu.cp.empty_like(x)
        self._ck(self.lib.ncclAllReduce(x.data.ptr, y.data.ptr, n, t, self.SUM, self.h,
                                        gpu.cp.cuda.get_current_stream().ptr))
        return y


# ------------------------------------------------------------------- communicators
class Comm:
    """One rank; also the base: every operation in terms of allgather (python objects / numpy arrays)."""
    rank, size, nccl = 0, 1, None

    def allgather(self, x):
        return [x]

    def barrier(self):
        self.allgather(None)

    def bcast(self, x, root=0):
        return self.allgather(x if self.rank == root else None)[root]

    def alltoall(self, xs):
        return [row[self.rank] for row in self.allgather(xs)]

    def sum(self, x):
        """Sum over the ranks, the same bits on every rank; numpy (scalars too) or cupy array."""
        if self.size == 1:
            return x
        if gpu.on and isinstance(x, gpu.cp.ndarray):
            if self.nccl is not None:
                return self.nccl.sum(gpu.cp.ascontiguousarray(x))
            return gpu.cp.asarray(self.sum(gpu.asnumpy(x)))
        return self._sum(np.asarray(x))

    def _sum(self, x):
        out = None
        for y in self.allgather(x):
            out = y.copy() if out is None else out + y
        return out

    def exchange(self, sends, recvs):
        """Point to point: sends {rank: array}, recvs {rank: array to fill}; C-contiguous numpy or cupy arrays."""
        dev = gpu.on and any(isinstance(x, gpu.cp.ndarray) for x in list(sends.values()) + list(recvs.values()))
        if dev and self.nccl is not None:
            return self.nccl.exchange(sends, recvs)
        if dev:
            h = {p: np.empty(x.shape, x.dtype) for p, x in recvs.items()}
            self._exchange({p: gpu.asnumpy(x) for p, x in sends.items()}, h)
            for p, x in recvs.items():
                x[...] = gpu.cp.asarray(h[p])
            return
        self._exchange(sends, recvs)

    def _exchange(self, sends, recvs):
        for p, row in enumerate(self.allgather(sends)):
            if self.rank in row and p != self.rank:
                recvs[p][...] = row[self.rank]

    def exchange_lists(self, sends, recvs):
        """sends / recvs {rank: [numpy arrays]} (same order on both sides)."""
        for p, row in enumerate(self.allgather(sends)):
            if p != self.rank:
                for a, b in zip(recvs.get(p, []), row.get(self.rank, [])):
                    a[...] = b

    def share(self, U, owner):
        """Complete a list of blocks: U[b] given on rank owner[b] (numpy), returned on every rank (numpy)."""
        if self.size == 1:
            return U
        rows = self.allgather({b: U[b] for b in range(len(U)) if owner[b] == self.rank})
        return [rows[owner[b]][b] for b in range(len(U))]


class MPIComm(Comm):
    def __init__(self, nccl=False):
        from mpi4py import MPI
        self.MPI, self.c = MPI, MPI.COMM_WORLD
        self.rank, self.size = self.c.Get_rank(), self.c.Get_size()
        loc = self.c.Split_type(MPI.COMM_TYPE_SHARED)
        self.local_rank, self.local_size = loc.Get_rank(), loc.Get_size()
        self.want_nccl = nccl

    def start_nccl(self):
        """After the GPU is selected (gpu.enable)."""
        if self.want_nccl and self.size > 1:
            self.nccl = Nccl(self.c)

    def abort(self):
        self.c.Abort(1)

    def allgather(self, x):
        return self.c.allgather(x)

    def barrier(self):
        self.c.Barrier()

    def bcast(self, x, root=0):
        if self.c.bcast(isinstance(x, np.ndarray), root):
            meta = self.c.bcast((x.shape, x.dtype.str) if self.rank == root else None, root)
            y = np.ascontiguousarray(x) if self.rank == root else np.empty(meta[0], meta[1])
            self._big(lambda v: self.c.Bcast(v, root), y)
            return y
        return self.c.bcast(x, root)

    @staticmethod
    def _big(f, y, chunk=2 ** 30):
        """f on byte chunks of y: MPI-3 counts are 32 bit."""
        v = y.reshape(-1).view(np.uint8) if y.size else np.empty(0, np.uint8)
        for a in range(0, len(v), chunk):
            f(v[a:a + chunk])

    def alltoall(self, xs):
        return self.c.alltoall(xs)

    def _sum(self, x):                                           # reduce + broadcast: the same bits on every rank
        nd, x = np.ndim(x), np.ascontiguousarray(x)
        y = np.empty_like(x)
        fx, fy = x.reshape(-1), y.reshape(-1)
        for a in range(0, max(fx.size, 1), 2 ** 26):
            self.c.Reduce(fx[a:a + 2 ** 26], fy[a:a + 2 ** 26] if self.rank == 0 else None, self.MPI.SUM, 0)
            self.c.Bcast(fy[a:a + 2 ** 26], 0)
        return y if nd else y[0]

    def _exchange(self, sends, recvs):
        self.exchange_lists({p: [x] for p, x in sends.items()}, {p: [x] for p, x in recvs.items()})

    def exchange_lists(self, sends, recvs):
        rq = []
        for p, xs in recvs.items():
            for x in xs:
                self._big(lambda v: rq.append(self.c.Irecv(v, p)), x)
        for p, xs in sends.items():
            for x in xs:
                self._big(lambda v: rq.append(self.c.Isend(v, p)), np.ascontiguousarray(x))
        self.MPI.Request.Waitall(rq)

    def share(self, U, owner):
        meta = {}
        for d in self.allgather({b: (U[b].shape, U[b].dtype.str) for b in range(len(U)) if owner[b] == self.rank}):
            meta.update(d)
        out = []
        for b in range(len(U)):
            y = np.ascontiguousarray(U[b]) if owner[b] == self.rank else np.empty(*meta[b])
            self._big(lambda v: self.c.Bcast(v, int(owner[b])), y)
            out.append(y)
        return out


class ThreadComm(Comm):
    """Ranks as threads of one process (tests). run(size, f) -> [f(comm) per rank]."""

    def __init__(self, rank, size, shared):
        self.rank, self.size, self.sh = rank, size, shared

    def allgather(self, x):
        self.sh["slot"][self.rank] = x
        self.sh["bar"].wait(300)
        out = list(self.sh["slot"])
        self.sh["bar"].wait(300)
        return out

    @staticmethod
    def run(size, f):
        sh = dict(slot=[None] * size, bar=threading.Barrier(size))
        out, err = [None] * size, []

        def go(r):
            try:
                out[r] = f(ThreadComm(r, size, sh))
            except BaseException as e:                            # noqa: BLE001
                err.append(e)
                sh["bar"].abort()
        old = threading.stack_size(256 << 20)                     # the threaded complex64 LU of OpenBLAS overflows the default thread stack
        try:
            th = [threading.Thread(target=go, args=(r,)) for r in range(size)]
            [t.start() for t in th]
        finally:
            threading.stack_size(old)
        [t.join() for t in th]
        if err:
            raise [e for e in err if not isinstance(e, threading.BrokenBarrierError)][0] \
                if any(not isinstance(e, threading.BrokenBarrierError) for e in err) else err[0]
        return out


def launch(kind):
    """The communicator of the driver: kind "" (one rank) | "mpi" | "nccl" | "auto" (nccl when the GPU backend and
    libnccl are there and every rank has its own device)."""
    if not kind:
        return Comm()
    assert kind in ("mpi", "nccl", "auto"), kind
    c = MPIComm(nccl=kind != "mpi")
    want = int(os.environ.get("SLURM_STEP_NUM_TASKS", 1)) if os.environ.get("QTX_LAUNCH") == "srun" else 1
    assert c.size > 1 or want == 1, f"srun started {want} tasks, MPI sees one: the MPI library under mpi4py does not " \
                                    "belong to this launcher"
    return c


# ------------------------------------------------------------------- partition and halo
def chunks(cost, size):
    """Owner per item: contiguous chunks of about equal cost, every rank at least one item when there are enough."""
    cost = np.asarray(cost, float)
    n = len(cost)
    assert n >= size, f"{size} ranks for {n} blocks"
    cum = np.concatenate([[0.0], np.cumsum(cost)])
    own, lo = np.zeros(n, int), 0
    for r in range(size):
        hi = n if r == size - 1 else int(np.searchsorted(cum, cum[-1] * (r + 1) / size - 1e-9 * cum[-1]))
        if r < size - 1 and hi > lo + 1 and cum[hi] - cum[-1] * (r + 1) / size > cum[-1] * (r + 1) / size - cum[hi - 1]:
            hi -= 1                                              # the nearer of the two cuts
        hi = min(max(hi, lo + 1), n - (size - 1 - r))
        own[lo:hi] = r
        lo = hi
    return own


def share_points(cost, owner, npts, size, tol=0.1, penalty=1.0):
    """Contour points of the windows per rank: [{rank: [point indices]} per window]. Every window starts on its owner
    (all npts points); while the most loaded rank is more than tol above the mean, points move from its largest window
    to the least loaded rank. A helper pays `penalty` points of the window once (assembly, transfer). Pure function of
    its arguments: the same answer on every rank."""
    cost, owner = np.asarray(cost, float), np.asarray(owner)
    nc = len(cost)
    n = [{int(owner[c]): npts} for c in range(nc)]               # points per rank
    load = np.zeros(size)
    for c in range(nc):
        load[owner[c]] += cost[c] * npts
    mean = load.sum() / size
    for _ in range(4 * nc * size * max(npts, 1)):
        r, t = int(np.argmax(load)), int(np.argmin(load))
        if load[r] <= (1 + tol) * mean or r == t:
            break
        best = None
        for c in range(nc):
            k = n[c].get(r, 0)
            if k < 1 or (owner[c] == r and k < 2):               # the owner keeps at least one point
                continue
            extra = 0.0 if t in n[c] else penalty
            if best is None or (t in n[c], cost[c]) > best[0]:
                if load[t] + cost[c] * (1 + extra) < load[r] - 1e-12 * mean:
                    best = ((t in n[c], cost[c]), c, extra)
        if best is None:
            break
        _, c, extra = best
        n[c][r] -= 1
        if n[c][r] == 0:
            del n[c][r]
        n[c][t] = n[c].get(t, 0) + 1
        load[r] -= cost[c]
        load[t] += cost[c] * (1 + extra)
    out = []
    for c in range(nc):
        # points are dealt out in blocks of consecutive indices, owner first
        q, d = 0, {}
        for r in sorted(n[c], key=lambda r: (r != owner[c], r)):
            d[r] = list(range(q, q + n[c][r]))
            q += n[c][r]
        assert q == npts
        out.append(d)
    return out


def domains(cost, size):
    """Slabs -> p = min(size, (S + 1) // 2) interior ranges [a, b) separated by single slabs: (domains, separators,
    ranks of the domains)."""
    cost = np.asarray(cost, float)
    S = len(cost)
    p = min(size, (S + 1) // 2)
    cum = np.concatenate([[0.0], np.cumsum(cost)])
    sep, lo = [], 1
    for j in range(p - 1):
        hi = S - 2 - 2 * (p - 2 - j)                             # room for the domains behind it
        want = int(np.argmin(np.abs(cum[1:S + 1] - cum[-1] * (j + 1) / p - 0.5 * cost)))
        sep.append(min(max(want, lo), hi))
        lo = sep[-1] + 2
    e = [-1] + sep + [S]
    dom = [(e[d] + 1, e[d + 1]) for d in range(p)]
    ranks = [int(round(v)) for v in np.linspace(0, size - 1, p)]
    return dom, sep, ranks


class Part:
    """Distribution of the n entries of a vector: entry sets blocks[b] (a partition of range(n)) belong to rank
    owner[b]; the local vector holds the owned entries in ascending global order (one rank: the whole vector)."""

    def __init__(self, comm, blocks, owner, n):
        self.comm, self.owner, self.n = comm, np.asarray(owner), n
        self.own = [int(b) for b in np.where(self.owner == comm.rank)[0]]
        self.gown = np.zeros(n, np.int32)
        for b, i in enumerate(blocks):
            self.gown[i] = self.owner[b]
        self.gloc = np.empty(n, np.int64)
        for r in range(comm.size):
            w = np.where(self.gown == r)[0]
            self.gloc[w] = np.arange(len(w))
            if r == comm.rank:
                self.idx = w
        self.nloc = len(self.idx)

    def local(self, x):
        """Owned rows of a global array (host: numpy in, numpy out)."""
        return x if self.comm.size == 1 else x[self.idx]

    def gather(self, r):
        """Global array from the owned rows, on every rank."""
        if self.comm.size == 1:
            return r
        xp = gpu.xp_of(r)
        g = xp.zeros((self.n,) + r.shape[1:], r.dtype)
        g[xp.asarray(self.idx)] = r
        return self.comm.sum(g)


class Exch:
    """Halo exchange for the global entries `need` of a Part: ex(r) = owned rows + ghost rows; pos[g] = row of the
    global entry g in it (-1: not there). rev(zx, z): the ghost rows of zx are sent back to their owners and
    overwrite the rows of z (used when a window writes cores of another rank). Collective (all ranks, same order)."""

    def __init__(self, part, need):
        self.part, comm = part, part.comm
        self.pos = np.full(part.n, -1, np.int64)
        self.pos[part.idx] = np.arange(part.nloc)
        self.ng, self.sends, self.recvs = 0, {}, {}
        if comm.size == 1:
            return
        need = np.unique(np.asarray(need, np.int64))
        g = need[part.gown[need] != comm.rank]
        g = g[np.argsort(part.gown[g], kind="stable")]
        self.ng = len(g)
        self.pos[g] = part.nloc + np.arange(len(g))
        req = [None] * comm.size
        for r in range(comm.size):
            w = np.where(part.gown[g] == r)[0]
            req[r] = part.gloc[g[w]]
            if len(w):
                self.recvs[r] = (part.nloc + int(w[0]), part.nloc + int(w[-1]) + 1)
        for r, i in enumerate(comm.alltoall(req)):
            if r != comm.rank and len(i):
                self.sends[r] = gpu.asarray(i)

    def __call__(self, r):
        if self.part.comm.size == 1:
            return r
        xp = gpu.xp_of(r)
        x = xp.empty((self.part.nloc + self.ng,) + r.shape[1:], r.dtype)
        x[:self.part.nloc] = r
        self.part.comm.exchange({p: xp.ascontiguousarray(r[i]) for p, i in self.sends.items()},
                                {p: x[a:b] for p, (a, b) in self.recvs.items()})
        return x

    def rev(self, zx, z):
        if self.part.comm.size == 1:
            return zx
        xp = gpu.xp_of(z)
        got = {p: xp.empty((len(i),) + z.shape[1:], z.dtype) for p, i in self.sends.items()}
        self.part.comm.exchange({p: xp.ascontiguousarray(zx[a:b]) for p, (a, b) in self.recvs.items()}, got)
        for p, i in self.sends.items():
            z[i] = got[p]
        return z


# ------------------------------------------------------------------- fine operator
class FineA:
    """Rows and windows of E S - H - Sigma_L - Sigma_R, without the global matrix."""

    def __init__(self, dev, E, SL, SR):
        self.dev, self.E, self.sg = dev, E, ((SL, dev.bL), (SR, dev.bR))
        self.S, self.H = dev.S.tocsr(), dev.H.tocsr()

    def rows(self, r):
        r = np.asarray(r)
        M = self.E * self.S[r] - self.H[r]
        for Sg, b in self.sg:
            ii = np.where(np.isin(r, b))[0]
            if len(ii):
                jb = np.searchsorted(b, r[ii])
                M = M - sp.csr_matrix((Sg[jb].ravel(), (np.repeat(ii, len(b)), np.tile(b, len(ii)))), shape=M.shape)
        return M.tocsr()

    def window(self, idx):
        return self.rows(idx)[:, idx]


class FineOp:
    """A @ x on the owned rows: halo exchange + the local rows (columns renumbered to owned + ghost)."""

    def __init__(self, fa, part, stream_gb=None, threads=1):
        from .schwarz import operator
        M = fa.rows(part.idx)
        M.sum_duplicates()
        self.ex = Exch(part, np.unique(M.indices))
        M = sp.csr_matrix((M.data, self.ex.pos[M.indices], M.indptr), shape=(part.nloc, part.nloc + self.ex.ng))
        M.sort_indices()
        self.op = operator(M, threads, stream_gb)
        self.shape = (part.nloc, part.nloc)

    def __matmul__(self, x):
        return self.op @ self.ex(x)
