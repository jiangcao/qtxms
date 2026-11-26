"""Device store: the fine H and S cut into slab row blocks on disk, so that a rank of a distributed run holds only the
rows it needs (its own windows) instead of two copies of the whole matrices (inp and dev; 12 nm: 53 GiB per rank).

    python -m qtx_ms.devstore --inputs <dir> --out <store> [--obc folded]        (one process, no GPU)
    python -m qtx_ms.mlevel --inputs <dir> --dev-store <store> ...

<store>/store.json                     format, meta, sizes, size and mtime of the input files (stale store: refused)
<store>/small.npz                      geometry (labels, xyz, norb, blocks), dev, ns, bL, bR, lead matrices (csr parts)
<store>/c{L,R}{0..3}_{data,indices,indptr}.npy   couplings device boundary - lead face of Device.cL, cR (CSR, read at the first
                                       use). Format 1 stores: dense c{L,R}{0..3}.npy; `--sparse-couplings STORE` adds the CSR files
<store>/s<k>_{H,S}_{data,indices,indptr}.npy   rows of device slab k, columns in device order, sorted (memory mapped)

The slabs are those of the block-tridiagonal partition: the store does not depend on the window settings. Folded-lead
OBC only (no qttools modes)."""
import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np
import scipy.sparse as sp

from . import msbasis as mb
from .msbasis import log

FORMAT = 2
CSR = ("data", "indices", "indptr")
LEAD = ("H00", "H01", "S00", "S01")
SRC = ("H.npz", "S.npz", "lead.npz", "arrays.npz", "prep.json", "meta.json", "blocks.txt", "norb.txt", "device.xyz",
       "perm.npy") + tuple(f"{p}_{k}.npz" for p in ("lead", "leadR") for k in LEAD)


def source_files(folder):
    """{file: [size, mtime]} of the input files that define the device."""
    out = {}
    for f in SRC:
        p = os.path.join(folder, f)
        if os.path.exists(p):
            st = os.stat(p)
            out[f] = [int(st.st_size), int(st.st_mtime)]
    return out


def mapped(folder, pre, shape):
    """The CSR matrix <folder>/<pre>_{data,indices,indptr}.npy, memory mapped read-only (only touched pages are read)."""
    d, i, p = (np.load(os.path.join(folder, f"{pre}_{k}.npy"), mmap_mode="r") for k in CSR)
    M = sp.csr_matrix((d, i, p), shape=shape, copy=False)
    M.has_sorted_indices = True
    M.has_canonical_format = True
    return M


def write_coupling(folder, s, q, M):
    M = sp.csr_matrix(M)
    M.sum_duplicates()
    for p in CSR:
        np.save(os.path.join(folder, f"c{s}{q}_{p}.npy"), M.data if p == "data" else getattr(M, p).astype(np.int32))


def coupling(folder, s, q, nb):
    """Coupling q (H_ld, S_ld, H_dl, S_dl) of side s (CSR; nb: device boundary orbitals): its CSR files, read whole (no mapping), or the
    dense file of a format 1 store."""
    f = lambda p: os.path.join(folder, f"c{s}{q}_{p}.npy")
    if not os.path.exists(f("data")):
        return sp.csr_matrix(np.load(os.path.join(folder, f"c{s}{q}.npy")))
    d, i, p = (np.load(f(k)) for k in CSR)
    nr = len(p) - 1
    # lead x device (q < 2): the columns are the boundary; device x lead: the columns are the lead face, as many as rows of c{s}0
    nc = nb if q < 2 else len(np.load(os.path.join(folder, f"c{s}0_indptr.npy"))) - 1
    return sp.csr_matrix((d, i, p), shape=(nr, nc))


def sparse_couplings(folder):
    """Adds the CSR files of the couplings to a format 1 store (the dense files stay, for older code)."""
    for s in "LR":
        for q in range(4):
            if not os.path.exists(os.path.join(folder, f"c{s}{q}_data.npy")):
                t = time.time()
                M = sp.csr_matrix(np.load(os.path.join(folder, f"c{s}{q}.npy")))
                write_coupling(folder, s, q, M)
                log(f"c{s}{q}: {M.shape}, {M.nnz} nonzeros ({100 * M.nnz / np.prod(M.shape):.1f} %), {time.time() - t:.1f} s")


class SlabRows:
    """Rows of one matrix (H or S) of the store. M[rows] = CSR rows x all columns (as csr_matrix: any order, repeated
    rows) from the held rows only, KeyError for a row that is not held (hold / drop); sub(idx) = M[idx][:, idx], rows
    that are not held are read from the slab files (the CAP layers of merged windows lie on other ranks)."""

    def __init__(self, folder, name, sb):
        self.folder, self.name, self.sb = folder, name, np.asarray(sb)
        n = int(self.sb[-1])
        self.shape, self.n = (n, n), n
        self.dtype = np.load(os.path.join(folder, f"s000_{name}_data.npy"), mmap_mode="r").dtype
        self.pos, self.M, self.nread = np.full(n, -1, np.int32), None, 0

    def tocsr(self):
        return self

    def slab(self, k):
        return mapped(self.folder, f"s{k:03d}_{self.name}", (int(self.sb[k + 1] - self.sb[k]), self.n))

    def read(self, rows, cols=None):
        """CSR of the rows (any order) from the files, slab by slab; cols: only these columns (renumbered)."""
        rows = np.asarray(rows)
        o = np.argsort(rows, kind="stable")
        r = rows[o]
        cut = np.searchsorted(r, self.sb)
        parts = []
        for k in range(len(self.sb) - 1):
            if cut[k + 1] > cut[k]:
                q = self.slab(k)[r[cut[k]:cut[k + 1]] - self.sb[k]]
                parts.append(q if cols is None else q[:, cols])
        self.nread += len(rows)
        if not parts:
            return sp.csr_matrix((0, self.n if cols is None else len(cols)))
        M = sp.vstack(parts, format="csr")
        if np.array_equal(o, np.arange(len(o))):
            return M
        inv = np.empty(len(o), int)
        inv[o] = np.arange(len(o))
        return M[inv]

    def get(self, rows):
        """CSR of the rows: the held ones, else from the files."""
        rows = np.asarray(rows)
        return self[rows] if self.M is not None and (self.pos[rows] >= 0).all() else self.read(rows)

    def hold(self, rows):
        rows = np.unique(rows)
        self.drop()
        self.M = self.read(rows)
        self.pos[rows] = np.arange(len(rows))

    def drop(self):
        self.M = None
        self.pos[:] = -1

    @property
    def gb(self):
        return 0.0 if self.M is None else sum(getattr(self.M, k).nbytes for k in CSR) / 2 ** 30

    def __getitem__(self, rows):
        rows = np.asarray(rows)
        p = self.pos[rows]
        if self.M is None or (p < 0).any():
            raise KeyError(f"{self.name}: {int((p < 0).sum())} of {len(rows)} rows are not held by this rank")
        return self.M[p]

    def sub(self, idx):
        idx = np.asarray(idx)
        return self[idx][:, idx] if self.M is not None and (self.pos[idx] >= 0).all() else self.read(idx, idx)

    def full(self):
        return sp.vstack([self.slab(k) for k in range(len(self.sb) - 1)], format="csr")


class StoreDevice(mb.Device):
    """Device (folded-lead OBC) of a store: the same attributes as mb.Device, H and S as SlabRows; the leads are built at
    the first Sigma that is not in the cache. hold(rows): the rows of H, S this rank works on; hold(None): the matrices."""

    def __init__(self, folder, inp, z, obc="folded"):
        self.folder, self.inp, self.meta, self.obc = folder, inp, inp["meta"], obc
        tr = self.meta["transport"]
        self.off = np.concatenate([[0], np.cumsum(inp["norb"])])
        self.dev_blocks = tr["device_blocks"]
        self.dev, self.ns = z["dev"], [int(v) for v in z["ns"]]
        self.sb = np.concatenate([[0], np.cumsum(self.ns)])
        self.slab_orbs = [self.dev[a:b] for a, b in zip(self.sb[:-1], self.sb[1:])]
        self.n = len(self.dev)
        self.loc = -np.ones(int(self.off[-1]), int)
        for o in self.slab_orbs:
            self.loc[o] = np.arange(len(o))
        self.bL, self.bR = z["bL"], z["bR"]
        self._cpl = {}
        self.qtt, self._inj, self._leads = None, {}, None
        self.H, self.S = (SlabRows(folder, nm, self.sb) for nm in "HS")

    def _lead(self, side):
        if self._leads is None:
            inp = self.inp
            f = lambda d: mb.FoldedLead(*(d[k] for k in LEAD))
            l_ = f(inp["lead"])
            self._leads = l_, (l_ if inp.get("lead_right") is None else f(inp["lead_right"]))
        return self._leads[side]

    def _couplings(self, s):
        if s not in self._cpl:
            self._cpl[s] = [coupling(self.folder, s, q, len(self.bL if s == "L" else self.bR)) for q in range(4)]
        return self._cpl[s]

    cL = property(lambda self: self._couplings("L"))
    cR = property(lambda self: self._couplings("R"))
    lead = property(lambda self: self._lead(0))
    leadR = property(lambda self: self._lead(1))

    def hold(self, rows):
        for nm in "HS":
            M = getattr(self, nm)
            if rows is None:
                setattr(self, nm, M.full() if isinstance(M, SlabRows) else M)
            else:
                M.hold(rows)

    def drop(self):
        for M in (self.H, self.S):
            if isinstance(M, SlabRows):
                M.drop()

    def A(self, E, SL, SR):
        """The whole fine operator (exact reference): H, S read from the files, not kept."""
        H, S = self.H, self.S
        try:
            self.H, self.S = (M.full() if isinstance(M, SlabRows) else M for M in (H, S))
            return super().A(E, SL, SR)
        finally:
            self.H, self.S = H, S


def save(dev, inp, out, source=None):
    """Write the store of a Device (atomic: a temporary folder is renamed at the end)."""
    assert dev.qtt is None, "device store: folded-lead OBC only (Device(..., obc='folded'))"
    out = os.path.abspath(out)
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists: remove it first")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=os.path.basename(out) + ".tmp", dir=os.path.dirname(out))
    os.chmod(tmp, 0o755)
    try:
        nnz = dict(H=0, S=0)
        for k in range(len(dev.ns)):
            for nm in "HS":
                R = getattr(dev, nm)[dev.sb[k]:dev.sb[k + 1]]
                R.sum_duplicates()                                 # canonical: sorted columns (the maps are read-only)
                assert R.nnz < 2 ** 31, "slab block beyond int32"
                for p in CSR:
                    np.save(os.path.join(tmp, f"s{k:03d}_{nm}_{p}.npy"),
                            R.data if p == "data" else getattr(R, p).astype(np.int32))
                nnz[nm] += int(R.nnz)
        for s, c in (("L", dev.cL), ("R", dev.cR)):
            for q, M in enumerate(c):
                write_coupling(tmp, s, q, M)
        z = dict(dev=dev.dev, ns=np.asarray(dev.ns), bL=dev.bL, bR=dev.bR,
                 **{k: np.asarray(inp[k]) for k in ("labels", "xyz", "norb", "blocks")})
        for pre in ("lead", "lead_right"):
            if inp.get(pre) is not None:
                for k in LEAD:
                    M = sp.csr_matrix(inp[pre][k])
                    z.update({f"{pre}_{k}_{p}": getattr(M, p) for p in CSR}, **{f"{pre}_{k}_shape": np.asarray(M.shape)})
        np.savez(os.path.join(tmp, "small.npz"), **z)
        json.dump(dict(format=FORMAT, n=int(dev.n), slabs=len(dev.ns), nnz=nnz, dtype=str(dev.H.dtype), meta=inp["meta"],
                       source=None if source is None else os.path.abspath(source),
                       source_files=None if source is None else source_files(source)),
                  open(os.path.join(tmp, "store.json"), "w"), indent=1)
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def load(folder, source=None):
    """(inp, dev) of a store; inp has no H, S. source: the --inputs folder of the run, refused when its files are not
    the ones the store was built from."""
    info = json.load(open(os.path.join(folder, "store.json")))
    if info.get("format") not in (1, FORMAT):
        raise ValueError(f"{folder}: store format {info.get('format')}, this version reads 1, {FORMAT}; build it again")
    if source is not None and source_files(source) != info["source_files"]:
        raise ValueError(f"{folder} was not built from the present files of {source} (built from {info['source']}); "
                         f"build it again: python -m qtx_ms.devstore")
    z = np.load(os.path.join(folder, "small.npz"))
    inp = {k: z[k] for k in ("labels", "xyz", "norb", "blocks")}
    for pre in ("lead", "lead_right"):
        inp[pre] = ({k: sp.csr_matrix(tuple(z[f"{pre}_{k}_{p}"] for p in CSR), shape=tuple(z[f"{pre}_{k}_shape"]))
                     for k in LEAD} if f"{pre}_H00_data" in z else None)
    inp["meta"], inp["perm"] = info["meta"], None
    return inp, StoreDevice(folder, inp, z)


def read_any(folder):
    """Input dict of an input folder or of a release prep folder (prep.json)."""
    if os.path.exists(os.path.join(folder, "prep.json")):
        from qtx_modespace.quatrex import load_inputs              # needs release/qtx_modespace/src on PYTHONPATH
        return load_inputs(folder, log=log)
    from qtx_ms.prep.inputs import read_inputs
    return read_inputs(folder)


def hostmem():
    """Memory of this process and of the node in GiB: anon = private, file = mapped file pages (shared, reclaimable), cpu =
    free + reclaimable file memory of the NUMA nodes with CPUs (GH200: without the HBM of the GPUs, which MemAvailable
    counts)."""
    g = 2 ** 20
    st = {ln.split(":")[0]: ln.split()[1] for ln in open("/proc/self/status") if ln.startswith("Rss")}
    cpu, base = 0, "/sys/devices/system/node"
    for d in (sorted(os.listdir(base)) if os.path.isdir(base) else ()):
        if not (d.startswith("node") and d[4:].isdigit() and open(f"{base}/{d}/cpulist").read().strip()):
            continue
        m = {ln.split()[2].rstrip(":"): int(ln.split()[3]) for ln in open(f"{base}/{d}/meminfo")}
        cpu += m["MemFree"] + max(m.get("FilePages", 0) - m.get("Shmem", 0), 0)
    return dict(anon=int(st.get("RssAnon", 0)) / g, file=int(st.get("RssFile", 0)) / g, cpu=cpu / g)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs")
    p.add_argument("--out")
    p.add_argument("--sparse-couplings", default="", metavar="STORE", help="only: add the CSR coupling files to a format 1 store")
    a = p.parse_args(argv)
    if a.sparse_couplings:
        return sparse_couplings(a.sparse_couplings)
    if not (a.inputs and a.out):
        p.error("--inputs and --out, or --sparse-couplings")
    t = time.time()
    src = source_files(a.inputs)                                   # before reading: a change during the build is refused
    inp = read_any(a.inputs)
    log(f"inputs read ({time.time() - t:.0f} s), host {hostmem()['anon']:.1f} GiB")
    dev = mb.Device(inp, obc="folded")
    inp.pop("H"), inp.pop("S")
    log(f"device {dev.n} orbitals, {len(dev.ns)} slabs ({time.time() - t:.0f} s), host {hostmem()['anon']:.1f} GiB")
    save(dev, inp, a.out, a.inputs)
    assert source_files(a.inputs) == src, "the inputs changed during the build"
    gb = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)) / 2 ** 30
    import resource
    log(f"{a.out}: {gb:.1f} GiB ({time.time() - t:.0f} s), peak host {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.1f} GiB")


if __name__ == "__main__":
    main()
