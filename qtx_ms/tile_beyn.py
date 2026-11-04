"""Level-1 tile basis from the local nonlinear OBC eigenproblem (Beyn), compared with the tile density-matrix basis.

Per tile t, the same embedding as msbasis.tile_modes(obc=dev): device atoms within d_emb of the tile, CAP -iW on the
outer d_cap shell, the exact lead self-energy on the embedded lead-face orbitals. The local problem
    T_t(z) v = 0,   T_t(z) = z S_e - H_e + i W - Sigma_face(z)
is nonlinear through Sigma only. Below the real axis Sigma is the continued retarded sheet (obc_eig.ContinuedSigma,
fixed channel count per contour), analytic between lead subband edges only, so the window is covered by several
ellipses, each inside one band interval (gap part with nprop = 0 included). Beyn (one moment pair, r probes) per
ellipse gives the eigenpairs (lambda, v) inside; v is S_e-normalized over the embedding, its tile rows v_t carry the
tile weight |v_t|^2. Tile basis: SVD of [v_t over all eigenpairs with a small residual], keep s^2 > thr -- the analogue
of the occupation cut of the tile rho (rho_tt ~ sum_n v_t v_t^H for the states in the window).

Steps (workers share the Sigma / moment caches, so each step can be split and resumed):
  sigma    K,N   Sigma at all contour nodes (+ the tile-rho arc nodes of --rho-win), split over N workers
  tiles    K,N   Beyn per tile and contour (tiles assigned largest first), -> <out>/tiles/tile_<i>.npz
  assemble       per-slab U1 from the Beyn tiles and from a tile-rho cache (--rho-dir) at the cuts --thr
  compare        per tile: occupation of the rho basis outside span(Beyn) and Beyn weight outside span(rho)

  python -m qtx_ms.tile_beyn --inputs inputs/mos2_hfo2 --out results_ms/tile_beyn sigma 0,4
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from . import msbasis as mb
from . import obc_eig as oe
from .obc_eig_l1 import SigmaInterp

log = mb.log

# name: centre (absolute eV), a, b, npts. c1 / c2 / c3 are the full-size obc_eig contours (Sigma cached).
CONTOURS = "gap:0.42835,0.074,0.03,32;c1:0.52135,0.015,0.05,24;c2:0.58935,0.045,0.03,32;c3:0.68835,0.040,0.03,32"


def parse_contours(spec):
    out = []
    for item in spec.split(";"):
        name, v = item.split(":")
        c, a, b, n = v.split(",")
        out.append(dict(name=name, c=float(c), a=float(a), b=float(b), n=int(n)))
    return out


def arc_nodes(mu1, mu2, npts):
    """Nodes of the tile-rho arc exactly as msbasis.tile_modes(contour=...) builds them (Sigma cache keys)."""
    c, rr = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    x, _ = np.polynomial.legendre.leggauss(int(npts))
    th = 0.5 * np.pi * (x + 1)
    return [c + rr * np.exp(1j * t_) for t_ in th]


class Setup:
    def __init__(self, a):
        from .prep.inputs import read_inputs
        self.a = a
        self.inp = read_inputs(a.inputs)
        self.Ec = self.inp["meta"]["edges"]["Ec_lead"]
        self.dev = mb.Device(self.inp, obc="folded")
        self.cache = os.path.join("results_ms", "sigma", os.path.basename(os.path.normpath(a.inputs)))
        self.dev.sigma_cache = self.cache
        self.cons = parse_contours(a.contours)
        os.makedirs(a.out, exist_ok=True)

    def sig(self, con):
        """ContinuedSigma of a contour (channel count at its centre, cached in <out>/channels_<c>.json)."""
        f = os.path.join(self.a.out, f"channels_{con['c']:+.8f}.json")
        if os.path.exists(f):
            nL, nR = (tuple(x) for x in json.load(open(f)))
        else:
            nL = oe.channel_counts(self.dev.lead, con["c"])
            nR = nL if self.dev.leadR is self.dev.lead else oe.channel_counts(self.dev.leadR, con["c"])
            json.dump([nL, nR], open(f, "w"))
        return oe.ContinuedSigma(self.dev, cache=self.cache, nprop=nL, nprop_R=nR)

    def nodes(self, con):
        return oe.ellipse(con["c"], con["a"], con["b"], con["n"])

    def blocks(self):
        blocks, _, _ = mb.partition(self.dev, cores=False, tile_orbs=self.a.tile_orbs)
        return blocks


# ------------------------------------------------------------------ tile embedding (as tile_modes(obc=dev))
class Embedding:
    def __init__(self, inp, geo, tree, atoms, dev, d_emb=12.0, d_cap=3.0, eta_cap=0.3):
        norb = inp["norb"]
        off = np.concatenate([[0], np.cumsum(norb)])
        near = tree.query_ball_point(geo.points(atoms), d_emb)
        emb = np.union1d(np.unique(np.concatenate([np.asarray(q, int) for q in near])), atoms)
        emb = emb[np.isin(emb, mb.obc_atoms(dev))]
        d, _ = geo.tree(atoms).query(geo.points(emb), k=1)
        w = eta_cap * np.clip((d - (d_emb - d_cap)) / d_cap, 0, None) ** 2
        oe_ = np.concatenate([np.arange(off[i], off[i + 1]) for i in emb])
        ne = norb[emb]
        eoff = np.concatenate([[0], np.cumsum(ne)])
        pe = np.searchsorted(emb, atoms)
        self.pidx = np.concatenate([np.arange(eoff[j], eoff[j + 1]) for j in pe])
        self.He = inp["H"][oe_][:, oe_].tocsr()
        self.Se = inp["S"][oe_][:, oe_].tocsr()
        self.w = np.repeat(w, ne)
        self.n, self.m, self.emb = self.He.shape[0], len(self.pidx), emb
        self.faces = mb.obc_faces(dev, oe_)

    def dense(self, z, SL, SR):
        A = (z * self.Se - self.He).toarray()
        A[np.diag_indices(self.n)] += 1j * self.w
        for (pos, fi), Sg in zip(self.faces, (SL, SR)):
            if len(pos):
                A[np.ix_(pos, pos)] -= Sg[np.ix_(fi, fi)]
        return A

    def apply(self, lam, SL, SR, V):
        R = lam * (self.Se @ V) - self.He @ V + 1j * self.w[:, None] * V
        for (pos, fi), Sg in zip(self.faces, (SL, SR)):
            if len(pos):
                R[pos] -= Sg[np.ix_(fi, fi)] @ V[pos]
        return R


def beyn_moments(A0, A1, center, scale, rank_tol):
    W, s, Vh = np.linalg.svd(A0, full_matrices=False)
    k = int((s > rank_tol * s[0]).sum())
    W, sk, Vk = W[:, :k], s[:k], Vh[:k].conj().T
    lam, Y = np.linalg.eig((W.conj().T @ A1 @ Vk) / sk)
    return center + scale * lam, W @ Y, s, k


# ------------------------------------------------------------------ steps
def step_sigma(st, k, nw):
    jobs = []
    for con in st.cons:
        zs, _ = st.nodes(con)
        jobs += [("beyn", con, z) for z in zs]
    if st.a.rho_win:
        lo, hi = (float(x) for x in st.a.rho_win.split(","))
        jobs += [("arc", None, z) for z in arc_nodes(st.Ec + lo, st.Ec + hi, st.a.rho_npts)]
    sigs = {}
    for j, (kind, con, z) in enumerate(jobs):
        if j % nw != k:
            continue
        t = time.time()
        if kind == "arc":
            st.dev.sigma(z)
        else:
            sigs.setdefault(con["name"], st.sig(con))(z)
        log(f"  Sigma {kind} {con['name'] if con else ''} z = {z.real:.5f}{z.imag:+.5f}i: {time.time() - t:.0f} s")
    log(f"sigma worker {k},{nw} done")


def step_tiles(st, k, nw):
    a = st.a
    blocks = st.blocks()
    geo = mb.Geometry(st.inp["xyz"], st.inp["meta"].get("periodic"))
    tree = geo.tree()
    tiles = sorted([i for i, b in enumerate(blocks) if b["kind"] == "tile"], key=lambda i: -len(blocks[i]["orbs"]))
    if a.only:
        tiles = [i for i in tiles if i in {int(x) for x in a.only.split(",")}]
    mine = [i for j, i in enumerate(tiles) if j % nw == k]
    tdir = os.path.join(a.out, "tiles")
    os.makedirs(tdir, exist_ok=True)
    for i in mine:
        fout = os.path.join(tdir, f"tile_{i:03d}.npz")
        if os.path.exists(fout):
            continue
        t0 = time.time()
        E = Embedding(st.inp, geo, tree, blocks[i]["atoms"], st.dev)
        log(f"tile {i} (slab {blocks[i]['slab']}): {E.m} tile orb, embedding {len(E.emb)} atoms / {E.n} orb, "
            f"faces {[len(p) for p, _ in E.faces]}")
        V = np.random.default_rng(1000 + i).standard_normal((E.n, a.probes)) + 0j
        res = dict(lam=[], Vt=[], wt=[], res=[], con=[], rank={}, sv={})
        for ci, con in enumerate(st.cons):
            sig = st.sig(con)
            zs, ws = st.nodes(con)
            scale = max(con["a"], con["b"])
            fm = os.path.join(tdir, f"mom_{i:03d}_{con['name']}_p{a.probes}.npz")
            if os.path.exists(fm):
                z_ = np.load(fm)
                A0, A1, done = z_["A0"], z_["A1"], set(z_["done"].tolist())
            else:
                A0, A1, done = np.zeros_like(V), np.zeros_like(V), set()
            SLs, SRs = [], []
            for j, (z, w) in enumerate(zip(zs, ws)):
                SL, SR = sig(z)
                SLs.append(SL); SRs.append(SR)
                if j in done:
                    continue
                t = time.time()
                lu = sla.lu_factor(E.dense(z, SL, SR), check_finite=False, overwrite_a=True)
                X = sla.lu_solve(lu, V, check_finite=False)
                del lu
                A0 += w * X
                A1 += w * ((z - con["c"]) / scale) * X
                done.add(j)
                np.savez(fm, A0=A0, A1=A1, done=np.array(sorted(done)))
                log(f"  tile {i} {con['name']} node {j}: {time.time() - t:.0f} s")
            lam, vec, s, rk = beyn_moments(A0, A1, con["c"], scale, a.rank_tol)
            ins = ((lam.real - con["c"]) / con["a"]) ** 2 + (lam.imag / con["b"]) ** 2 < 1
            lam, vec = lam[ins], vec[:, ins]
            nS = np.sqrt(np.abs(np.einsum("ij,ij->j", vec.conj(), E.Se @ vec)))
            vec = vec / nS
            si = SigmaInterp(zs, ws, SLs, SRs)
            r = np.array([np.linalg.norm(E.apply(l, *si(l), v[:, None])) / np.linalg.norm(v)
                          for l, v in zip(lam, vec.T)])
            Vt = vec[E.pidx]
            wt = np.linalg.norm(Vt, axis=0) ** 2 / np.linalg.norm(vec, axis=0) ** 2
            res["rank"][con["name"]], res["sv"][con["name"]] = rk, s
            log(f"  tile {i} {con['name']}: rank {rk} of {len(s)} probes (s[k-1]/s0 {s[rk - 1] / s[0]:.1e}), "
                f"{ins.sum()} inside, residual < 1e-6 / 1e-3 / 1e-1 eV: "
                f"{[(r < x).sum() for x in (1e-6, 1e-3, 1e-1)]}, tile weight > 0.5 / 1e-2: {(wt > 0.5).sum()} / "
                f"{(wt > 1e-2).sum()}" + ("  WARNING: rank close to the probe count" if rk > 0.8 * a.probes else ""))
            res["lam"].append(lam); res["Vt"].append(Vt); res["wt"].append(wt); res["res"].append(r)
            res["con"].append(np.full(len(lam), ci))
        np.savez(fout, atoms=blocks[i]["atoms"], lam=np.concatenate(res["lam"]), Vt=np.hstack(res["Vt"]),
                 wt=np.concatenate(res["wt"]), res=np.concatenate(res["res"]), con=np.concatenate(res["con"]),
                 rank=json.dumps(res["rank"]), **{f"sv_{c}": v for c, v in res["sv"].items()})
        log(f"tile {i} done: {sum(len(x) for x in res['lam'])} eigenpairs, {time.time() - t0:.0f} s -> {fout}")
    log(f"tiles worker {k},{nw} done")


def beyn_basis(z, thr, rmax):
    ok = z["res"] < rmax
    Y = z["Vt"][:, ok]
    if Y.shape[1] == 0:
        return Y, np.zeros(0)
    U, s, _ = np.linalg.svd(Y, full_matrices=False)
    return U[:, s ** 2 > thr], s ** 2


def rho_basis(f, thr):
    z = np.load(f)
    return z["U"][:, :int(np.sum(z["s"][:z["U"].shape[1]] > thr))], z["s"]


def step_assemble(st):
    a = st.a
    blocks = st.blocks()
    tiles = [i for i, b in enumerate(blocks) if b["kind"] == "tile"]
    summ = {}
    for thr in (float(x) for x in a.thr.split(",")):
        for kind in ("beyn", "rho"):
            if kind == "rho" and not a.rho_dir:
                continue
            bases = [None] * len(blocks)
            for i in tiles:
                if kind == "beyn":
                    z = np.load(os.path.join(a.out, "tiles", f"tile_{i:03d}.npz"))
                    assert np.array_equal(z["atoms"], blocks[i]["atoms"])
                    bases[i] = beyn_basis(z, thr, a.res_max)[0]
                else:
                    bases[i] = rho_basis(os.path.join(a.rho_dir, f"tile_obc_{i:03d}.npz"), thr)[0]
            Us, _ = mb.slab_U(st.dev, blocks, bases)
            pre = os.path.join(a.out, f"{kind}_t{thr:g}")
            for s_, U in enumerate(Us):
                np.save(f"{pre}_U1_slab{s_}.npy", U.toarray())
            N = [u.shape[1] for u in Us]
            summ[f"{kind}_t{thr:g}"] = dict(N_slab=N, N=sum(N), N_tile=[bases[i].shape[1] for i in tiles])
            log(f"{kind} cut {thr:g}: N1 {sum(N)} {N}, per tile {[bases[i].shape[1] for i in tiles]} -> {pre}_U1_slab<s>")
    json.dump(summ, open(os.path.join(a.out, "assemble.json"), "w"), indent=1)


def step_compare(st):
    """Per tile: (1) rho-weighted occupation of the rho tile modes outside span(Beyn basis), (2) tile weight of the
    Beyn eigenvectors outside span(rho basis), both at each cut."""
    a = st.a
    blocks = st.blocks()
    tiles = [i for i, b in enumerate(blocks) if b["kind"] == "tile"]
    out = {}
    for thr in (float(x) for x in a.thr.split(",")):
        rows = []
        for i in tiles:
            z = np.load(os.path.join(a.out, "tiles", f"tile_{i:03d}.npz"))
            B, _ = beyn_basis(z, thr, a.res_max)
            zr = np.load(os.path.join(a.rho_dir, f"tile_obc_{i:03d}.npz"))
            Ur, occ = zr["U"], zr["s"][:zr["U"].shape[1]]
            R = Ur[:, occ > thr]
            # (1) rho modes (occupation-weighted) not in span(B)
            miss = np.linalg.norm(Ur - B @ (B.conj().T @ Ur), axis=0) ** 2
            # (2) Beyn eigenvectors (tile rows, residual ok) not in span(R)
            ok = z["res"] < a.res_max
            Vt = z["Vt"][:, ok]
            mb_ = np.linalg.norm(Vt - R @ (R.conj().T @ Vt), axis=0) ** 2
            ang = sla.subspace_angles(B, R) if B.shape[1] and R.shape[1] else np.array([np.pi / 2])
            row = dict(tile=i, N_beyn=B.shape[1], N_rho=R.shape[1],
                       rho_occ_missing=float(np.sum(occ * miss)), rho_occ_total=float(occ.sum()),
                       rho_occ_missing_max=float(np.max(occ * miss)),
                       beyn_wt_missing=float(np.sum(mb_)), beyn_wt_total=float(np.sum(np.linalg.norm(Vt, axis=0) ** 2)),
                       beyn_wt_missing_max=float(np.max(mb_, initial=0)),
                       sin_max=float(np.sin(ang.max())), n_sin_gt_0p1=int((np.sin(ang) > 0.1).sum()))
            rows.append(row)
            log(f"cut {thr:g} tile {i}: N beyn {row['N_beyn']} rho {row['N_rho']}; rho occupation outside span(beyn) "
                f"{row['rho_occ_missing']:.2e} of {row['rho_occ_total']:.1f} (max one mode {row['rho_occ_missing_max']:.1e}); "
                f"beyn weight outside span(rho) {row['beyn_wt_missing']:.2e} of {row['beyn_wt_total']:.1f} "
                f"(max {row['beyn_wt_missing_max']:.1e}); angles sin > 0.1: {row['n_sin_gt_0p1']}")
        out[f"{thr:g}"] = rows
    json.dump(out, open(os.path.join(a.out, "compare.json"), "w"), indent=1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["sigma", "tiles", "assemble", "compare"])
    p.add_argument("worker", nargs="?", default="0,1", help="K,N (sigma, tiles)")
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--contours", default=CONTOURS, help="name:centre,a,b,npts;... (centre absolute, eV)")
    p.add_argument("--rho-win", help="sigma step: also the tile-rho arc nodes of this window (- Ec, eV)")
    p.add_argument("--rho-npts", type=int, default=24)
    p.add_argument("--rho-dir", help="tile-rho cache (tile_obc_<i>.npz) for assemble / compare")
    p.add_argument("--tile-orbs", type=int, default=2500)
    p.add_argument("--probes", type=int, default=400)
    p.add_argument("--rank-tol", type=float, default=1e-8)
    p.add_argument("--only", help="tiles step: only these tile indices")
    p.add_argument("--thr", default="1e-4,1e-5,1e-6,1e-7")
    p.add_argument("--res-max", type=float, default=1e-3, help="keep Beyn eigenpairs with |T v|/|v| below this (eV)")
    a = p.parse_args(argv)
    st = Setup(a)
    k, nw = (int(x) for x in a.worker.split(","))
    if a.step == "sigma":
        step_sigma(st, k, nw)
    elif a.step == "tiles":
        step_tiles(st, k, nw)
    elif a.step == "assemble":
        step_assemble(st)
    else:
        step_compare(st)


if __name__ == "__main__":
    main()
