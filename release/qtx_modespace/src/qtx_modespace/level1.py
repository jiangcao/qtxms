"""Level 1: local density-matrix basis per block (slab or tile).

Partition (independent of the level): every device slab is either one block ("slab") or cut into tiles ("tile") by
recursive transverse bisection into ~tile_orbs orbitals, each tile optionally cut into xsplit pieces of equal orbital
weight along transport. A basis is block diagonal over the blocks, so per slab U_s = diag(blocks of slab s).

Local density matrix of block P (the "local rho"): the block embedded in every device atom within d_emb (minimum
image across the periodic transverse faces), the embedding closed by a complex absorbing potential
    W = eta_cap ((d - (d_emb - d_cap)) / d_cap)^2  on its outer d_cap shell (d = distance to the nearest block atom),
and, with boundary "obc" (CAP + OBC), the lead-block atoms dropped and the exact lead self-energy -Sigma_L/R(z)
added on the lead-face orbitals inside the embedding (only blocks whose embedding reaches a lead face are affected).
    rho_PP = -(1/pi) Im-part of int_arc G_PP(z) dz,  G = (z S - H + i W - Sigma_emb(z))^-1,
over the upper half circle from Ec + mu1 to Ec + mu2 (Gauss-Legendre in the angle; the same points as the level-2
contour, so the lead self-energies are shared through the Sigma cache). G_PP by the Schur complement of the
environment E: G_PP = (A_PP - A_PE A_EE^-1 A_EP)^-1 (one LU of A_EE per point, m solves; dense LU when the embedding
is > dense_fill filled). The basis of the block = eigenvectors of rho_PP with occupation > cut, largest first.
"""
import hashlib, json, os, time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.spatial import cKDTree

from . import gpu
from .core import CSolver, eigh, log, savez_atomic, write_text_atomic


# ------------------------------------------------------------------ geometry and partition
def periodic_axes(periodic):
    """[(axis, length)] from meta["periodic"] (None, a dict or a list of dicts {axis, length})."""
    if not periodic:
        return []
    return [(int(p["axis"]), float(p["length"])) for p in ([periodic] if isinstance(periodic, dict) else periodic)]


class Geometry:
    """Distance queries on atom positions, minimum image along the periodic transverse axes."""

    def __init__(self, xyz, periodic=None):
        self.xyz = np.asarray(xyz, float)
        self.box, self.per = None, periodic_axes(periodic)
        if self.per:
            self.lo = self.xyz.min(0) - 1.0
            self.box = np.full(3, 2 * np.ptp(self.xyz, axis=0).max() + 1e3)
            for ax, L in self.per:
                self.box[ax] = L

    def _map(self, r):
        if self.box is None:
            return r
        q = r - self.lo
        for ax, L in self.per:
            q[:, ax] %= L
        return q

    def tree(self, idx=None):
        return cKDTree(self._map(self.xyz if idx is None else self.xyz[idx]), boxsize=self.box)

    def points(self, idx):
        return self._map(self.xyz[idx])


def bisect(atoms, pos, w, nt):
    """atoms in nt groups of ~equal weight w by recursive bisection along the longest extent of pos."""
    if nt <= 1 or len(atoms) <= 1:
        return [atoms]
    ax = int(np.argmax(np.ptp(pos[atoms], axis=0)))
    o = atoms[np.argsort(pos[atoms, ax], kind="stable")]
    cum = np.cumsum(w[o])
    k1 = nt // 2
    cut = int(np.searchsorted(cum, cum[-1] * k1 / nt)) + 1
    return bisect(o[:cut], pos, w, k1) + bisect(o[cut:], pos, w, nt - k1)


def strips(atoms, coord, w, n):
    """atoms in n groups of ~equal weight w, cut along coord only (sorted strips)."""
    if n <= 1 or len(atoms) <= 1:
        return [atoms]
    o = atoms[np.argsort(coord[atoms], kind="stable")]
    cum = np.cumsum(w[o])
    cuts = [int(np.searchsorted(cum, cum[-1] * k / n)) + 1 for k in range(1, n)]
    return [g for g in np.split(o, cuts) if len(g)]


def partition(inp, dev, kind="slab", tile_orbs=2500, xsplit=1, tsplit=None, xlen=None):
    """Blocks: [dict(slab, atoms (sorted atom ids), orbs (slab-local orbitals))], slab by slab.
    kind "slab": one block per device slab (then xsplit still cuts it along transport); "tile": the slab cut
    transversely into round(n_slab / tile_orbs) tiles (at least 1) by bisection along the longest extent, or, with
    tsplit = (n1, n2), into n1 x n2 strips of equal orbital count along the two transverse axes (in axis order, e.g.
    y then z for transport along x). Each tile is cut into xsplit pieces along transport; xlen [A] sets xsplit per
    slab to round(slab length / xlen) (at least 1) instead."""
    labels, xyz, norb, ax = inp["labels"], inp["xyz"], inp["norb"], inp["meta"]["axis"]
    off = np.concatenate([[0], np.cumsum(norb)])
    tr = [i for i in range(3) if i != ax]
    blocks = []
    for s, b in enumerate(inp["meta"]["transport"]["device_blocks"]):
        a = np.where(inp["blocks"] == b)[0]
        if kind == "slab":
            tiles = [a]
        elif tsplit:
            tiles = [t2 for t1 in strips(a, xyz[:, tr[0]], norb, int(tsplit[0]))
                     for t2 in strips(t1, xyz[:, tr[1]], norb, int(tsplit[1]))]
        else:
            tiles = bisect(a, xyz[:, tr], norb, max(1, int(round(norb[a].sum() / tile_orbs))))
        nx = max(1, int(round(inp["meta"]["block_length"] / xlen))) if xlen else xsplit
        for t in tiles:
            for tx in bisect(t, xyz[:, [ax]], norb, nx):
                tx = np.sort(tx)
                g = np.concatenate([np.arange(off[i], off[i + 1]) for i in tx])
                blocks.append(dict(slab=s, atoms=tx, orbs=dev.loc[g]))
    assert len(labels) == len(norb)
    return blocks


# ------------------------------------------------------------------ embedding and local rho
def device_atoms(inp, dev):
    atom_of = np.repeat(np.arange(len(inp["norb"])), inp["norb"])
    return np.unique(atom_of[dev.dev])


def lead_faces(dev, oe):
    """For read_quatrex orbitals oe: [(positions in oe on the left lead face, their rows in Sigma_L), (right)]."""
    g = getattr(dev, "_gdev", None)
    if g is None:
        g = dev._gdev = -np.ones(dev.inp["H"].shape[0], int)
        g[dev.dev] = np.arange(dev.n)
    dl = g[oe]
    out = []
    for b in (dev.bL, dev.bR):
        pos = np.where(np.isin(dl, b))[0]
        out.append((pos, np.searchsorted(b, dl[pos])))
    return out


def embedding(inp, geo, tree, atoms, d_emb, d_cap, eta_cap, obc=None):
    """CAP-terminated embedding of a block: atoms within d_emb of any block atom (obc = Device: device atoms only, and
    the lead faces inside the embedding). pidx = block orbitals in the embedding, in block atom order."""
    norb = inp["norb"]
    off = np.concatenate([[0], np.cumsum(norb)])
    near = tree.query_ball_point(geo.points(atoms), d_emb)
    emb = np.union1d(np.unique(np.concatenate([np.asarray(q, int) for q in near])), atoms)
    if obc is not None:
        emb = emb[np.isin(emb, device_atoms(inp, obc))]
    d, _ = geo.tree(atoms).query(geo.points(emb), k=1)
    w = eta_cap * np.clip((d - (d_emb - d_cap)) / d_cap, 0, None) ** 2
    oe = np.concatenate([np.arange(off[i], off[i + 1]) for i in emb])
    ne = norb[emb]
    eoff = np.concatenate([[0], np.cumsum(ne)])
    pidx = np.concatenate([np.arange(eoff[j], eoff[j + 1]) for j in np.searchsorted(emb, atoms)])
    faces = [] if obc is None else lead_faces(obc, oe)
    return dict(emb=emb, oe=oe, pidx=pidx, He=inp["H"][oe][:, oe].tocsr(), Se=inp["S"][oe][:, oe].tocsr(),
                W=sp.diags(np.repeat(w, ne)), faces=faces, obc=obc,
                nface=[int(len(p)) for p, _ in faces], ncap=int((w > 0).sum()))


def sigma_emb(e, z):
    """-Sigma_L/R(z) on the embedded lead-face orbitals (sparse, embedding size); zero without faces."""
    n = e["He"].shape[0]
    M = sp.csr_matrix((n, n), dtype=complex)
    if not e["faces"] or not any(len(p) for p, _ in e["faces"]):
        return M
    for (pos, fi), Sg in zip(e["faces"], e["obc"].sigma(z)):
        if len(pos):
            ii, jj = np.meshgrid(pos, pos, indexing="ij")
            M = M - sp.csr_matrix((Sg[np.ix_(fi, fi)].ravel(), (ii.ravel(), jj.ravel())), shape=(n, n))
    return M


def contour_points(mu1, mu2, npts):
    """Upper half circle over [mu1, mu2]: points z and weights dz (Gauss-Legendre in the angle, as rho_contour)."""
    c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    x, w = np.polynomial.legendre.leggauss(int(npts))
    th, w = 0.5 * np.pi * (x + 1), 0.5 * np.pi * w
    return c + r * np.exp(1j * th), -1j * r * np.exp(1j * th) * w


def local_rho(e, contour, gpp="schur", env_prec="double", refine=1, workers=1, threads=None, dense_fill=0.02,
              dense_max=30000):
    """rho_PP of the embedding e over contour = (mu1, mu2, npts) [absolute eV]; returns (rho, seconds).
    gpp "schur": G_PP = (A_PP - A_PE A_EE^-1 A_EP)^-1 (env_prec "single": A_EE LU in complex64 + `refine` steps of
    iterative refinement); "full": one LU of the whole embedding with m right-hand sides.
    workers > 1: the contour points over forked processes with `threads` BLAS threads each (needs threadpoolctl).
    GPU mode (gpu.on, dense embeddings): every point on the GPU (blocks built there from the CSR matrix, LU, solves,
    Schur complement and its inverse), rho accumulated on the device; points one after another (workers ignored)."""
    zs, dzs = contour_points(*contour)
    He, Se, W, pidx = e["He"], e["Se"], e["W"], e["pidx"]
    n, m = He.shape[0], len(pidx)
    dense = (He + Se).nnz > dense_fill * n * n and n <= dense_max
    cdt = np.complex64 if env_prec == "single" else np.complex128
    eidx = np.setdiff1d(np.arange(n), pidx)
    on_gpu = gpu.on and dense

    if on_gpu:
        # device copies, made once: the E / P blocks of S and K = H - i W as CSR, and where the lead-face orbitals sit
        # in E and P. Per point only A = z S - K is formed (on the device) and Sigma(z) is scattered into the blocks;
        # assembling and slicing the sparse A on the host cost more than the factorization.
        K = (He - 1j * W).tocsr()
        Sc = sp.csr_matrix(Se)
        gS = {a + b: gpu.csr(Sc[i][:, j], np.complex128) for a, i in (("E", eidx), ("P", pidx))
              for b, j in (("E", eidx), ("P", pidx))}
        gK = {a + b: gpu.csr(K[i][:, j], np.complex128) for a, i in (("E", eidx), ("P", pidx))
              for b, j in (("E", eidx), ("P", pidx))}
        del K, Sc
        loc = {}
        for a, i in (("E", eidx), ("P", pidx)):
            loc[a] = np.full(n, -1)
            loc[a][i] = np.arange(len(i))
        faces = []                                                  # per lead: {E, P: (block rows, Sigma rows)}
        for pos, fi in (e["faces"] or []):
            pos, fi = np.asarray(pos, int), np.asarray(fi, int)
            faces.append({a: (gpu.cp.asarray(loc[a][pos][loc[a][pos] >= 0]), gpu.cp.asarray(fi[loc[a][pos] >= 0]))
                          for a in "EP"})
        has_sigma = any(len(f[a][0]) for f in faces for a in "EP")

    def point_gpu(k):
        cp, z = gpu.cp, zs[k]
        if gpp == "full":
            A = (z * Se - He + 1j * W + sigma_emb(e, z)).tocsr()
            Ad = gpu.csr_dense(A, np.complex128)
            R = cp.zeros((n, m), complex)
            R[cp.asarray(pidx), cp.arange(m)] = 1.0
            X = gpu.lu_solve(gpu.lu_factor(Ad), R)
            return X[cp.asarray(pidx)]
        Sg = [cp.asarray(M) for M in e["obc"].sigma(z)] if has_sigma else []

        def block(ab):                                              # dense A_ab = z S - K - Sigma on the device
            M = (z * gS[ab] - gK[ab]).toarray()
            for f, G in zip(faces, Sg):
                (r, fr), (c, fc) = f[ab[0]], f[ab[1]]
                if len(r) and len(c):
                    ix = cp.ix_(r, c)
                    M[ix] = M[ix] - G[cp.ix_(fr, fc)]
            return M

        AEE, AEP = block("EE"), block("EP")
        if env_prec == "single":
            lu = gpu.lu_factor(AEE.astype(cdt))
            X = gpu.lu_solve(lu, AEP.astype(cdt)).astype(complex)
            for _ in range(refine):
                X += gpu.lu_solve(lu, (AEP - AEE @ X).astype(cdt))
        else:
            lu = gpu.lu_factor(AEE)                                 # overwrites AEE
            X = gpu.lu_solve(lu, AEP)
        del lu, AEE, AEP
        Sp = block("PP") - (z * gS["PE"] - gK["PE"]) @ X            # A_PE X: sparse part, then its Sigma part
        for f, G in zip(faces, Sg):
            (r, fr), (c, fc) = f["P"], f["E"]
            if len(r) and len(c):
                Sp[r] = Sp[r] + G[cp.ix_(fr, fc)] @ X[c]
        del X
        return cp.linalg.inv(Sp)

    def point(k):
        if on_gpu:
            return point_gpu(k)
        z = zs[k]
        A = (z * Se - He + 1j * W + sigma_emb(e, z)).tocsr()
        if gpp == "full":
            R = np.zeros((n, m), complex)
            R[pidx, np.arange(m)] = 1.0
            if dense:
                lu = sla.lu_factor(A.toarray(), check_finite=False, overwrite_a=True)
                return sla.lu_solve(lu, R, check_finite=False)[pidx]
            cs = CSolver(A)
            G = cs.solve(R)[pidx]
            cs.free()
            return G
        Ar, Ap = A[eidx], A[pidx]
        AEE, AEP = Ar[:, eidx].tocsr(), Ar[:, pidx].toarray()
        APP, APE = Ap[:, pidx].toarray(), Ap[:, eidx].tocsr()
        if dense:
            AEEd = AEE.toarray()
            lu = sla.lu_factor(AEEd.astype(cdt), check_finite=False)
            X = sla.lu_solve(lu, AEP.astype(cdt), check_finite=False).astype(complex)
            if env_prec == "single":
                for _ in range(refine):
                    X += sla.lu_solve(lu, (AEP - AEEd @ X).astype(cdt), check_finite=False)
            del lu, AEEd
        else:
            cs = CSolver(AEE)
            X = cs.solve(AEP)
            cs.free()
        Sp = APP - APE @ X
        del X
        return np.linalg.inv(Sp)

    def run(ks):
        acc = (gpu.cp if on_gpu else np).zeros((m, m), complex)
        for k in ks:
            acc += point(k) * dzs[k]
        return gpu.asnumpy(acc)

    def limited(f, nthr):
        if not nthr:
            return f()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(nthr):
            return f()

    t0 = time.time()
    if gpu.on:                                       # no fork after CUDA initialization: one process
        acc = limited(lambda: run(range(len(zs))), threads)
        gpu.free()
    elif workers <= 1:
        acc = limited(lambda: run(range(len(zs))), threads)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        parts = [list(range(j, len(zs), workers)) for j in range(workers)]

        def child(j):
            try:
                q.put((j, limited(lambda: run(parts[j]), threads or 1), None))
            except Exception as ex:                                         # noqa: BLE001
                q.put((j, None, repr(ex)))
        ps = [ctx.Process(target=child, args=(j,)) for j in range(workers) if parts[j]]
        for p_ in ps:
            p_.start()
        acc = np.zeros((m, m), complex)
        for _ in ps:
            j, a_, err = q.get()
            if err:
                raise RuntimeError(f"local_rho worker {j}: {err}")
            acc += a_
        for p_ in ps:
            p_.join()
    rho = -((acc - acc.conj().T) / 2j) / np.pi
    return 0.5 * (rho + rho.conj().T), time.time() - t0


# ------------------------------------------------------------------ level-1 basis
def build_level1(inp, dev, contour, kind="slab", tile_orbs=2500, xsplit=1, boundary="obc", d_emb=12.0, d_cap=3.0,
                 eta_cap=0.3, cut=1e-6, store_cut=1e-10, cache=None, gpp="schur", env_prec="double", workers=1,
                 threads=None, only=None, tsplit=None, xlen=None, dense_max=30000, log=log):
    """Level-1 basis per slab, V_s (slab-local orbitals x N1_s), block diagonal over the blocks of the partition, each
    block = eigenvectors of its local rho with occupation > cut (largest first). Returns (V, occupations per slab,
    info). cache: directory of per-block files block<i>.npz (vectors with occupation > min(cut, store_cut), all
    occupations, atoms) under a subdirectory named by a hash of the settings; a later run with a cut >= the stored one
    reuses them. only = set of block indices: build those (if missing) and return None (parallel builds)."""
    assert boundary in ("cap", "obc")
    blocks = partition(inp, dev, kind, tile_orbs, xsplit, tsplit, xlen)
    keep_cut = min(cut, store_cut)
    par = dict(kind=kind, tile_orbs=tile_orbs if kind == "tile" else None, xsplit=xsplit, boundary=boundary,
               d_emb=d_emb, d_cap=d_cap, eta_cap=eta_cap, contour=[round(float(c), 8) for c in contour[:2]] +
               [int(contour[2])], ns=list(map(int, dev.ns)))
    if kind == "tile" and tsplit:                   # only when used: older caches keep their hash
        par.update(tsplit=[int(x) for x in tsplit], tile_orbs=None)
    if xlen:
        par.update(xlen=float(xlen), xsplit=None)
    cdir = None
    if cache:
        h = hashlib.sha1(json.dumps(par, sort_keys=True).encode()).hexdigest()[:10]
        cdir = os.path.join(cache, f"{kind}_{boundary}_{h}")
        os.makedirs(cdir, exist_ok=True)
        write_text_atomic(os.path.join(cdir, "settings.json"), json.dumps(par, indent=1))
    sizes = [len(b["orbs"]) for b in blocks]
    log(f"level 1: {kind} partition, {len(blocks)} blocks of {min(sizes)}..{max(sizes)} orbitals "
        f"(per slab {[sum(b['slab'] == s for b in blocks) for s in range(len(dev.ns))]}); local rho with "
        f"{'CAP + lead Sigma' if boundary == 'obc' else 'CAP'}, embedding {d_emb} A, CAP {eta_cap} eV over {d_cap} A, "
        f"window [{contour[0]:.4f}, {contour[1]:.4f}] eV, {contour[2]} points; cut {cut:g}" +
        (f"; cache {cdir}" if cdir else ""))
    geo = Geometry(inp["xyz"], inp["meta"].get("periodic"))
    tree = geo.tree()
    res = [None] * len(blocks)
    order = np.argsort([-s for s in sizes])
    for i in order:
        b = blocks[i]
        if only is not None and i not in only:
            continue
        f = cdir and os.path.join(cdir, f"block{i:03d}.npz")
        if f and os.path.exists(f):
            z = np.load(f)
            assert np.array_equal(z["atoms"], b["atoms"]), f"{f}: another partition"
            if float(z["cut"]) <= cut:
                res[i] = (z["U"], z["s"])
                continue
        t = time.time()
        e = embedding(inp, geo, tree, b["atoms"], d_emb, d_cap, eta_cap, dev if boundary == "obc" else None)
        rho, trho = local_rho(e, contour, gpp, env_prec, workers=workers, threads=threads, dense_max=dense_max)
        lam, X = eigh(rho)
        k = int((lam > keep_cut).sum())
        res[i] = (X[:, :k], lam)
        if f:
            savez_atomic(f, U=X[:, :k], s=lam, atoms=b["atoms"], cut=keep_cut)
        log(f"  block {i} (slab {b['slab']}): {len(b['orbs'])} orbitals, embedding {len(e['oe'])} "
            f"(CAP {e['ncap']}, lead faces {e['nface']}), trace {lam.sum():.2f}, occupations > {cut:g}: "
            f"{int((lam > cut).sum())}; {trho:.0f} s")
    if only is not None:
        return None
    V, occ, nblk = [], [], []
    for s in range(len(dev.ns)):
        bl = [i for i, b in enumerate(blocks) if b["slab"] == s]
        cols = [res[i][0][:, :int((res[i][1][:res[i][0].shape[1]] > cut).sum())] for i in bl]
        X = np.zeros((dev.ns[s], sum(c.shape[1] for c in cols)), np.result_type(*cols))
        c0 = 0
        for i, c in zip(bl, cols):
            X[blocks[i]["orbs"], c0:c0 + c.shape[1]] = c
            c0 += c.shape[1]
        o = np.concatenate([res[i][1][:c.shape[1]] for i, c in zip(bl, cols)])
        V.append(X)
        occ.append(o)
        nblk.append([int(c.shape[1]) for c in cols])
    info = dict(level=1, partition=kind, tile_orbs=par["tile_orbs"], xsplit=par["xsplit"], tsplit=par.get("tsplit"),
                xlen=par.get("xlen"), boundary=boundary, d_emb=d_emb,
                d_cap=d_cap, eta_cap=eta_cap, contour=par["contour"], cut=cut, blocks_per_slab=nblk,
                block_sizes=sizes)
    log(f"level 1: N1 {[v.shape[1] for v in V]}, total {sum(v.shape[1] for v in V)}; columns per block {nblk}")
    return V, occ, info
