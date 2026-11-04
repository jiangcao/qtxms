"""Beyn on the level-1 OBC eigenproblem and residual enrichment of U1.

Reduced problem  T1(z) c = 0,  T1(z) = U1^H T(z) U1 = z S1 - H1 - U1^H Sigma(z) U1  (Sigma on the continued retarded
sheet, obc_eig.ContinuedSigma, read from the cache of the fine contour when it is the same contour). U1 is block
diagonal over the slabs, so T1 is block tridiagonal (msbasis.slab_blocks / BlockTridiag) and the solves are cheap.

For each reduced eigenpair (lambda1, c), v1 = U1 c:
    fine residual r = T(lambda1) v1  (Galerkin: U1^H r = 0),  Sigma(lambda1) by barycentric Cauchy interpolation
    from the contour nodes (analytic inside the contour, no new Sigma).
Enrichment candidates per slab s (orthogonalized against U1_s, SVD, kept above --tol):
    raw     r_s
    newton  t = T(lambda1)^-1 T'(lambda1) v1 (one step of nonlinear inverse iteration; T' = S - Sigma'(lambda1)),
            one fine block-LU solve per eigenvalue
With the fine eigenpairs of obc_eig (eigpairs.npz) the reduced ones are matched: |lambda1 - lambda|, sin of the angle
between v1 and v, and span(U1) missing of v, before / after the enrichment.

  python -m qtx_ms.obc_eig_l1 --inputs inputs/mos2_hfo2 --center 0.52135 --a 0.015 --b 0.01 --npts 24 \
      --U1 results_ms/mos2_hfo2_vcycle/tobc_t1e-6_lc --fine results_ms/obc_eig/c1_Ec+0.017 --enrich newton \
      --out results_ms/obc_eig/c1_Ec+0.017/l1
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.linalg as sla

from . import msbasis as mb
from . import obc_eig as oe

log = mb.log


class SigmaInterp:
    """Sigma_L, Sigma_R inside a contour from their values at the nodes (barycentric Cauchy formula) and Sigma'."""

    def __init__(self, zs, ws, SLs, SRs):
        self.zs, self.ws, self.S = np.asarray(zs), np.asarray(ws), (np.array(SLs), np.array(SRs))

    def __call__(self, lam, deriv=False):
        d = self.zs - lam
        c = self.ws / d
        out = [np.tensordot(c, S, 1) / c.sum() for S in self.S]
        if not deriv:
            return tuple(out)
        c2 = self.ws / d ** 2                 # d/dlam of sum w S/(z - lam) / sum w/(z - lam)
        D = [(np.tensordot(c2, S, 1) - o * c2.sum()) / c.sum() for S, o in zip(self.S, out)]
        return tuple(out), tuple(D)


def project_sigma(dev, Us, SL, SR):
    U0 = Us[0][dev.bL]
    U3 = Us[-1][dev.bR - dev.sb[-2]]
    return U0.conj().T @ SL @ U0, U3.conj().T @ SR @ U3


def fine_vec(Us, c):
    off = np.concatenate([[0], np.cumsum([u.shape[1] for u in Us])])
    return np.concatenate([Us[s] @ c[off[s]:off[s + 1]] for s in range(len(Us))])


def apply_T(dev, lam, SL, SR, v):
    r = lam * (dev.S @ v) - dev.H @ v
    r[dev.bL] -= SL @ v[dev.bL]
    r[dev.bR] -= SR @ v[dev.bR]
    return r


def apply_dT(dev, dSL, dSR, v):
    r = dev.S @ v
    r[dev.bL] -= dSL @ v[dev.bL]
    r[dev.bR] -= dSR @ v[dev.bR]
    return r


def slab_norms(dev, x):
    return np.array([np.linalg.norm(x[dev.sb[s]:dev.sb[s + 1]]) for s in range(len(dev.ns))])


def enrich_slabs(dev, Us, cands, tol, nmax):
    """Per slab: candidates (columns, fine vectors) orthogonalized against U1_s (twice), SVD, keep sigma > tol*sigma_0
    of the normalized candidates (at most nmax). Returns new Us and the number added per slab."""
    new, added = [], []
    for s, U in enumerate(Us):
        C = cands[dev.sb[s]:dev.sb[s + 1]]
        nrm = np.linalg.norm(C, axis=0)
        C = C[:, nrm > 1e-14 * max(nrm.max(initial=0), 1e-300)]
        if C.shape[1] == 0:
            new.append(U); added.append(0)
            continue
        C = C / np.linalg.norm(C, axis=0)
        for _ in range(2):
            C = C - U @ (U.conj().T @ C)
        W, sv, _ = np.linalg.svd(C, full_matrices=False)
        k = int(min((sv > tol).sum(), nmax))
        new.append(np.hstack([U, W[:, :k]])); added.append(k)
        log(f"  slab {s}: candidate sv {np.array2string(sv[:8], precision=2)} ... -> +{k}")
    return new, added


def match(lam1, lam):
    """Index of the nearest fine eigenvalue for each reduced one."""
    return np.array([int(np.argmin(np.abs(lam - l))) for l in lam1]) if len(lam) else np.zeros(0, int)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--center", type=float, required=True)
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--b", type=float, required=True)
    p.add_argument("--npts", type=int, default=24)
    p.add_argument("--probes", type=int, default=300)
    p.add_argument("--rank-tol", type=float, default=1e-8)
    p.add_argument("--U1", required=True, help="prefix of the level-1 slab bases (<prefix>_U1_slab<s>.npy)")
    p.add_argument("--fine", help="obc_eig output dir (channels json, eigpairs.npz) of the same contour")
    p.add_argument("--enrich", choices=["none", "raw", "newton"], default="newton")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--rthr", type=float, default=0.0, help="enrich only eigenpairs with |r|/|v| above this [eV]")
    p.add_argument("--tol", type=float, default=1e-2, help="SVD keep threshold of the normalized candidates")
    p.add_argument("--nmax", type=int, default=200, help="max vectors added per slab")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sigma-cache", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from .prep.inputs import read_inputs
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc="folded")
    os.makedirs(a.out, exist_ok=True)
    cache = a.sigma_cache or os.path.join("results_ms", "sigma", os.path.basename(os.path.normpath(a.inputs)))
    fch = a.fine and os.path.join(a.fine, f"channels_{a.center:+.8f}.json")
    if fch and os.path.exists(fch):
        nprop, nR = (tuple(x) for x in json.load(open(fch)))
    else:
        nprop = oe.channel_counts(dev.lead, a.center); nR = nprop
    sig = oe.ContinuedSigma(dev, cache=cache, nprop=nprop, nprop_R=nR)
    Us = oe.load_U1(a.U1, len(dev.ns))
    N1 = [u.shape[1] for u in Us]
    log(f"{a.inputs}: level 1 {a.U1} N1 {N1} = {sum(N1)}; contour c {a.center} a {a.a} b {a.b}, {a.npts} pts, "
        f"channels {nprop}")
    zs, ws = oe.ellipse(a.center, a.a, a.b, a.npts)

    # ---- Sigma at the nodes (cache of the fine contour), reduced Beyn
    t0 = time.time()
    SLs, SRs = [], []
    for z in zs:
        SL, SR = sig(z)
        SLs.append(SL); SRs.append(SR)
    sint = SigmaInterp(zs, ws, SLs, SRs)
    log(f"Sigma at {a.npts} nodes: {time.time() - t0:.0f} s")

    def reduced_beyn(Us, tag):
        t0 = time.time()
        blk1 = mb.slab_blocks(dev, Us)
        n1 = sum(u.shape[1] for u in Us)
        V = np.random.default_rng(a.seed).normal(size=(n1, a.probes)) + 0j
        Xs = []
        for z, SL, SR in zip(zs, SLs, SRs):
            S1L, S1R = project_sigma(dev, Us, SL, SR)
            Xs.append(mb.BlockTridiag(blk1, z, S1L, S1R).solve(V))
        lam, C, s, k = oe.beyn(Xs, zs, ws, 1, a.rank_tol, a.center, max(a.a, a.b))
        ins = ((lam.real - a.center) / a.a) ** 2 + (lam.imag / a.b) ** 2 < 1
        lam, C = lam[ins], C[:, ins]
        o = np.argsort(lam.real)
        lam, C = lam[o], C[:, o]
        log(f"[{tag}] reduced Beyn N1 {n1}: rank {k} (s[k-1]/s0 {s[k - 1] / s[0]:.1e}), {len(lam)} eigenvalues "
            f"inside, {time.time() - t0:.0f} s")
        if k >= 0.8 * a.probes:
            log("  WARNING: rank close to the number of probes")
        return lam, C

    fine = None
    if a.fine and os.path.exists(os.path.join(a.fine, "eigpairs.npz")):
        f = np.load(os.path.join(a.fine, "eigpairs.npz"))
        fine = dict(lam=f["lam"], vec=f["vec"])
        log(f"fine eigenpairs: {len(fine['lam'])} from {a.fine}")

    def report(Us, lam1, C, tag):
        rows = []
        for i, (l, c) in enumerate(zip(lam1, C.T)):
            v = fine_vec(Us, c)
            v /= np.linalg.norm(v)
            SL, SR = sint(l)
            r = apply_T(dev, l, SL, SR, v)
            row = dict(lam=[l.real, l.imag], res=float(np.linalg.norm(r)), res_slab=slab_norms(dev, r).tolist())
            if fine is not None and len(fine["lam"]):
                j = int(np.argmin(np.abs(fine["lam"] - l)))
                w = fine["vec"][:, j] / np.linalg.norm(fine["vec"][:, j])
                row.update(fine=j, dlam=float(abs(fine["lam"][j] - l)),
                           sin=float(np.sqrt(max(0.0, 1 - abs(np.vdot(w, v)) ** 2))),
                           miss=float(oe.span_missing(dev, Us, w[:, None])[0][0]))
            rows.append(row)
        log(f"[{tag}] reduced eigenpairs: lambda, |T v| (eV), per slab" +
            ("; nearest fine: |dlambda|, sin(v1, v), missing of v in span(U1)" if fine is not None else ""))
        for r in rows:
            x = f"  {r['lam'][0]:.5f} {r['lam'][1]:+.2e}i  |r| {r['res']:.2e}  [" + \
                " ".join(f"{y:.1e}" for y in r["res_slab"]) + "]"
            if "fine" in r:
                x += f"  fine {r['fine']:3d}: dlam {r['dlam']:.1e} sin {r['sin']:.1e} miss {r['miss']:.1e}"
            log(x)
        if fine is not None and len(fine["lam"]):
            got = {r["fine"] for r in rows if r["dlam"] < 1e-3}
            ms = oe.span_missing(dev, Us, fine["vec"])[0]
            log(f"[{tag}] fine eigenvalues with a reduced one within 1 meV: {len(got)} of {len(fine['lam'])}; "
                f"missing of the fine eigenvectors in span(U1): median {np.median(ms):.2e} max {ms.max():.2e}")
            for j, (l, m) in enumerate(zip(fine["lam"], ms)):
                log(f"    fine {j:3d} {l.real:.5f} {l.imag:+.2e}i  miss {m:.2e}  {'matched' if j in got else '-'}")
        return rows

    lam1, C = reduced_beyn(Us, "U1")
    rows = report(Us, lam1, C, "U1")
    out = dict(U1=a.U1, N1=N1, before=rows)
    rounds = []
    for rnd in range(a.rounds if a.enrich != "none" else 0):
        sel = [i for i, r in enumerate(rows) if r["res"] > a.rthr]
        t0 = time.time()
        cands = []
        blk = oe.dense_blocks(dev) if a.enrich == "newton" else None
        for i in sel:
            l = lam1[i]
            v = fine_vec(Us, C[:, i]); v /= np.linalg.norm(v)
            (SL, SR), (dSL, dSR) = sint(l, deriv=True)
            if a.enrich == "raw":
                cands.append(apply_T(dev, l, SL, SR, v))
            else:
                t = oe.block_solver(dev, blk, l, SL, SR).solve(apply_dT(dev, dSL, dSR, v)[:, None])[:, 0]
                cands.append(t / np.linalg.norm(t))
        log(f"round {rnd}: enrichment ({a.enrich}) from {len(sel)} eigenpairs: {time.time() - t0:.0f} s")
        if not cands:
            break
        Us, added = enrich_slabs(dev, Us, np.array(cands).T, a.tol, a.nmax)
        N = [u.shape[1] for u in Us]
        log(f"round {rnd}: enriched N1 {N} = {sum(N)} (+{added})")
        pre = os.path.join(a.out, f"enr_{a.enrich}_r{rnd}")
        for s_, u in enumerate(Us):
            np.save(f"{pre}_U1_slab{s_}.npy", u)
        lam1, C = reduced_beyn(Us, f"round {rnd}")
        rows = report(Us, lam1, C, f"round {rnd}")
        rounds.append(dict(added=added, N1=N, rows=rows, prefix=pre))
        if sum(added) == 0:
            break
    out.update(enrich=a.enrich, rounds=rounds)
    with open(os.path.join(a.out, f"l1_{a.enrich}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    log(f"-> {a.out}")


if __name__ == "__main__":
    main()
