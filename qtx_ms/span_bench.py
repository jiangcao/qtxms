"""Span benchmark of local bases against exact device states (step 1 of the fresh tile-basis design).

Input: exact psi on an energy grid from `density_scan run --save-psi` (<psi dir>/E_<dE>.npz, device order, L + R
injection columns). Per region R (slabs, the old tiles, x-sub-bins of every slab, x-sub-bins x old tiles):
  * old bases: |(I - U U^H) psi|_R / |psi|_R with U orthonormal per slab (tile U1, tile U1 V, direct level 2);
  * ideal local basis: SVD of the exact psi restricted to R at the training energies (every other grid point),
    error on the held-out energies vs the number of vectors k -> k needed for a tolerance.
The ideal curve bounds what any local basis on R can do with k vectors (for energies inside the training window);
sum_R k_R is the coarse dimension such a basis needs.

    python -m qtx_ms.span_bench --inputs inputs/mos2_hfo2 --psi results_ms/span_bench/mos2_hfo2_psi
"""
import argparse, glob, json, os, time
import numpy as np

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb

log = mb.log
TOLS = (1e-2, 3e-3, 1e-3, 3e-4)


def load_psi(d):
    fs = sorted(glob.glob(os.path.join(d, "E_*.npz")))
    out = []
    for f in fs:
        c = np.load(f)
        if "psi" in c:
            out.append((float(c["dE"]), c["psi"]))
    return out


def regions(dev, inp, tile_dir, nbins=(2, 4)):
    """dict name -> list of (label, device-order orbital indices)."""
    norb = inp["norb"]
    off = np.concatenate([[0], np.cumsum(norb)])
    atom = np.repeat(np.arange(len(norb)), norb)
    ga = atom[dev.dev]                                       # atom of every device orbital
    x = inp["xyz"][ga, inp["meta"]["axis"]]
    inv = np.full(off[-1], -1)
    inv[dev.dev] = np.arange(len(dev.dev))
    sb = dev.sb
    R = {"slab": [(f"s{s}", np.arange(sb[s], sb[s + 1])) for s in range(len(dev.ns))]}
    tiles = []
    for f in sorted(glob.glob(os.path.join(tile_dir, "tile_*.npz"))):
        a = np.load(f)["atoms"]
        o = inv[np.concatenate([np.arange(off[i], off[i + 1]) for i in a])]
        assert (o >= 0).all()
        tiles.append((os.path.basename(f)[:-4], np.sort(o)))
    cov = np.zeros(len(dev.dev), bool)
    for _, o in tiles:
        cov[o] = True
    for s in range(len(dev.ns)):                             # orbitals in no tile (cores): one region per slab
        o = np.where(~cov[sb[s]:sb[s + 1]])[0] + sb[s]
        if len(o):
            tiles.append((f"rest_s{s}", o))
    R["tile"] = tiles
    for nb in nbins:
        xb, xt = [], []
        for s in range(len(dev.ns)):
            idx = np.arange(sb[s], sb[s + 1])
            lo, hi = x[idx].min(), x[idx].max() + 1e-9
            e = np.linspace(lo, hi, nb + 1)
            k = np.clip(np.searchsorted(e, x[idx], side="right") - 1, 0, nb - 1)
            for j in range(nb):
                xb.append((f"s{s}x{j}", idx[k == j]))
        R[f"x{nb}"] = xb
        for tl, to in tiles:
            for bl, bo in xb:
                o = np.intersect1d(to, bo)
                if len(o):
                    xt.append((f"{tl}.{bl}", o))
        R[f"x{nb}tile"] = xt
    return R


def slab_projector_residual(dev, Us, psi):
    q = np.empty_like(psi)
    for s, U in enumerate(Us):
        a = slice(dev.sb[s], dev.sb[s + 1])
        q[a] = psi[a] - U @ (U.conj().T @ psi[a])
    return q


def ideal_curve(Ptr, Pte, groups):
    """SVD of training columns; held-out error vs k: total and max over test energies (groups = column bounds)."""
    W, sv, _ = np.linalg.svd(Ptr, full_matrices=False)
    c = np.abs(W.conj().T @ Pte) ** 2                        # (rank, ncols_test)
    tot = np.sum(np.abs(Pte) ** 2)
    cum = np.concatenate([np.zeros((1, c.shape[1])), np.cumsum(c, 0)])
    err_tot = np.sqrt(np.maximum(tot - cum.sum(1), 0) / tot)
    per = []
    for a, b in groups:
        t = np.sum(np.abs(Pte[:, a:b]) ** 2)
        per.append(np.sqrt(np.maximum(t - cum[:, a:b].sum(1), 0) / t))
    return sv, err_tot, np.max(per, 0)


def kneed(err, tol):
    i = np.where(err <= tol)[0]
    return int(i[0]) if len(i) else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--psi", required=True, help="density_scan --save-psi output folder")
    p.add_argument("--tiles", default="results_ms/mos2_hfo2_nocore_tc6_sk600/tiles")
    p.add_argument("--bases", default="tile_t1e-5_U1=results_ms/mos2_hfo2_vcycle/tile_t1e-5_U1,"
                   "tile_t1e-6_U1=results_ms/mos2_hfo2_vcycle/tile_t1e-6_U1,"
                   "direct_n400=results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead",
                   help="name=prefix (<prefix>_slab<s>.npy, orthonormal per slab); <name>_U1 also tries U1 V")
    p.add_argument("--train-step", type=float, default=0.02, help="training energies: multiples of this")
    p.add_argument("--out", help="default <psi>/span_bench.json")
    a = p.parse_args()
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc="folded")
    P = load_psi(a.psi)
    log(f"{len(P)} energies with psi: {[round(d, 3) for d, _ in P]}")
    istr = [abs(d / a.train_step - round(d / a.train_step)) < 1e-6 for d, _ in P]
    tr = [ps for (d, ps), t in zip(P, istr) if t]
    te = [ps for (d, ps), t in zip(P, istr) if not t]
    dte = [d for (d, _), t in zip(P, istr) if not t]
    Ptr, Pte = np.hstack(tr), np.hstack(te)
    gb = np.concatenate([[0], np.cumsum([x.shape[1] for x in te])])
    groups = list(zip(gb[:-1], gb[1:]))
    log(f"train {len(tr)} energies / {Ptr.shape[1]} columns, test {len(te)} / {Pte.shape[1]}")
    Pall = np.hstack([ps for _, ps in P])
    gall = np.concatenate([[0], np.cumsum([ps.shape[1] for _, ps in P])])
    R = regions(dev, inp, a.tiles)

    # old bases: residual of the per-slab projection, all energies
    old = {}
    nsl = len(dev.ns)
    for spec in a.bases.split(","):
        nm, pre = spec.split("=")
        Us = [np.load(f"{pre}_slab{s}.npy") for s in range(nsl)]
        cands = {nm: Us}
        if pre.endswith("_U1") and os.path.exists(f"{pre[:-3]}_V_slab0.npy"):
            Vs = [np.load(f"{pre[:-3]}_V_slab{s}.npy") for s in range(nsl)]
            cands[nm[:-3] + "_U1V"] = [U @ V for U, V in zip(Us, Vs)]
        for k, U in cands.items():
            q = slab_projector_residual(dev, U, Pall)
            per_E = [np.linalg.norm(q[:, i:j]) / np.linalg.norm(Pall[:, i:j]) for i, j in zip(gall[:-1], gall[1:])]
            old[k] = dict(N=int(sum(u.shape[1] for u in U)), err=float(np.linalg.norm(q) / np.linalg.norm(Pall)),
                          err_E=per_E,
                          region={rn: {lab: float(np.linalg.norm(q[o]) / np.linalg.norm(Pall[o])) for lab, o in rl}
                                  for rn, rl in R.items()})
            log(f"old {k}: N {old[k]['N']}, |(I-P)psi|/|psi| {old[k]['err']:.2e} (per E {min(per_E):.1e}..{max(per_E):.1e}); "
                f"per tile " + " ".join(f"{v:.1e}" for v in old[k]['region']['tile'].values()))

    # ideal local bases
    ideal = {}
    for rn, rl in R.items():
        t0 = time.time()
        rows, kt = [], {tol: 0 for tol in TOLS}
        ok = {tol: True for tol in TOLS}
        err2 = {tol: 0.0 for tol in TOLS}
        nrm2 = float(np.sum(np.abs(Pte) ** 2))
        for lab, o in rl:
            sv, et, em = ideal_curve(Ptr[o], Pte[o], groups)
            w = float(np.sum(np.abs(Pte[o]) ** 2))
            r = dict(label=lab, norb=int(len(o)), weight=w / nrm2,
                     k={f"{tol:g}": kneed(et, tol) for tol in TOLS},
                     kmax={f"{tol:g}": kneed(em, tol) for tol in TOLS},
                     err_k={str(k): float(et[k]) for k in (25, 50, 100, 200, 400) if k < len(et)},
                     sv=(sv[:: max(1, len(sv) // 50)] / sv[0]).tolist())
            for tol in TOLS:
                k = r["k"][f"{tol:g}"]
                if k is None:
                    ok[tol] = False
                    k = len(et) - 1
                kt[tol] += k
                err2[tol] += et[k] ** 2 * w
            rows.append(r)
        ideal[rn] = dict(regions=rows, N={f"{tol:g}": kt[tol] for tol in TOLS},
                         reached={f"{tol:g}": ok[tol] for tol in TOLS},
                         err={f"{tol:g}": float(np.sqrt(err2[tol] / nrm2)) for tol in TOLS})
        log(f"ideal {rn:8s} ({len(rl):2d} regions, {time.time() - t0:.0f} s): N for tol "
            + ", ".join(f"{tol:g}: {kt[tol]}{'' if ok[tol] else '+'}" for tol in TOLS)
            + " | per region k(1e-3) " + " ".join(str(r["k"]["0.001"]) for r in rows[:16]))
    out = a.out or os.path.join(a.psi, "span_bench.json")
    json.dump(dict(train=[d for (d, _), t in zip(P, istr) if t], test=dte, ncols_train=int(Ptr.shape[1]),
                   old=old, ideal=ideal), open(out, "w"), indent=1)
    log(f"-> {out}")


if __name__ == "__main__":
    main()
