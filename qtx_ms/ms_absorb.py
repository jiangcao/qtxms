"""Mode-space T on a fine grid inside one Beyn contour, with an absorber on the spurious level-1 OBC eigenstates.

Spurious states of the level-1 open system (reduced Beyn, obc_eig_l1: |T(lambda1) U1 c| >> physical, Im lambda1 ~ 1e-6)
are nearly decoupled resonances that the MS solve hits whenever E comes within ~0.1 meV. Absorber:
    A_c'(E) = A_c(E) + i eta S1 Q (Q^H S1 Q)^-1 Q^H S1,   Q = flagged reduced eigenvectors (|r| > --rthr)
i.e. E -> E + i eta on the spurious subspace only. Sigma on the real grid by barycentric interpolation from the
contour nodes (analytic inside the contour; checked against cached Sigma(E + 1e-6 i) at the --check energies).
Exact T on the same grid by the dense block-tridiagonal LU of the fine operator (optional, cached).

  python -m qtx_ms.ms_absorb --inputs inputs/mos2_hfo2 --center 0.52135 --a 0.015 --b 0.01 --npts 24 \
      --U1 results_ms/mos2_hfo2_vcycle/tobc_t1e-6_lc --fine results_ms/obc_eig/c1_Ec+0.017 --grid 0.003:0.031:0.001 \
      --eta 0,0.001,0.005,0.02 --out results_ms/obc_eig/absorb/c1_base
"""
import argparse
import json
import os
import time

import numpy as np

from . import msbasis as mb
from . import obc_eig as oe
from .obc_eig_l1 import SigmaInterp, project_sigma, fine_vec, apply_T

log = mb.log


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--center", type=float, required=True)
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--b", type=float, required=True)
    p.add_argument("--npts", type=int, default=24)
    p.add_argument("--probes", type=int, default=300)
    p.add_argument("--U1", required=True)
    p.add_argument("--fine", help="obc_eig dir of the same contour (channel counts)")
    p.add_argument("--grid", default="0.003:0.031:0.001", help="lo:hi:step, E - Ec [eV]")
    p.add_argument("--eta", default="0,0.001,0.005,0.02")
    p.add_argument("--rthr", type=float, default=2.0, help="spurious: |T v1| above this [eV]")
    p.add_argument("--exact", action="store_true", help="exact T on the grid (block LU, cached in --exact-cache)")
    p.add_argument("--exact-cache", default="results_ms/obc_eig/absorb/exact_grid.json")
    p.add_argument("--check", default="0.01,0.02,0.03", help="E - Ec with cached real-axis Sigma: interpolation check")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from .prep.inputs import read_inputs
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc="folded")
    Ec = inp["meta"]["edges"]["Ec_lead"]
    name = os.path.basename(os.path.normpath(a.inputs))
    cache = os.path.join("results_ms", "sigma", name)
    fch = a.fine and os.path.join(a.fine, f"channels_{a.center:+.8f}.json")
    nprop, nR = (tuple(x) for x in json.load(open(fch))) if fch and os.path.exists(fch) else ((None,) * 2)
    sig = oe.ContinuedSigma(dev, cache=cache, nprop=nprop, nprop_R=nR)
    zs, ws = oe.ellipse(a.center, a.a, a.b, a.npts)
    SLs, SRs = zip(*[sig(z) for z in zs])
    sint = SigmaInterp(zs, ws, SLs, SRs)
    for x in (float(v) for v in a.check.split(",")) if a.check else ():
        E = Ec + x
        f = os.path.join(cache, f"sig_{E:+.8f}_{1e-6:.8e}.npz")
        if os.path.exists(f):
            c = np.load(f)
            SL, SR = sint(E)
            log(f"Sigma interpolation check Ec{x:+.3f}: rel err L {np.abs(SL - c['SL']).max() / np.abs(c['SL']).max():.1e} "
                f"R {np.abs(SR - c['SR']).max() / np.abs(c['SR']).max():.1e}")

    Us = oe.load_U1(a.U1, len(dev.ns))
    N1 = [u.shape[1] for u in Us]
    blk1 = mb.slab_blocks(dev, Us)
    off = np.concatenate([[0], np.cumsum(N1)])
    S1 = np.zeros((off[-1], off[-1]), complex)               # dense S1 (block tridiagonal) for the absorber
    for s in range(len(Us)):
        S1[off[s]:off[s + 1], off[s]:off[s + 1]] = blk1["SD"][s]
        if s < len(Us) - 1:
            S1[off[s]:off[s + 1], off[s + 1]:off[s + 2]] = blk1["SU"][s]
            S1[off[s + 1]:off[s + 2], off[s]:off[s + 1]] = blk1["SU"][s].conj().T

    # ---- reduced Beyn: spurious eigenvectors
    t0 = time.time()
    V = np.random.default_rng(1).normal(size=(off[-1], a.probes)) + 0j
    Xs = [mb.BlockTridiag(blk1, z, *project_sigma(dev, Us, SL, SR)).solve(V) for z, SL, SR in zip(zs, SLs, SRs)]
    lam, C, s, k = oe.beyn(Xs, zs, ws, 1, 1e-8, a.center, max(a.a, a.b))
    ins = ((lam.real - a.center) / a.a) ** 2 + (lam.imag / a.b) ** 2 < 1
    lam, C = lam[ins], C[:, ins]
    res = []
    for l, c in zip(lam, C.T):
        v = fine_vec(Us, c); n = np.linalg.norm(v)
        res.append(np.linalg.norm(apply_T(dev, l, *sint(l), v / n)))
    res = np.array(res)
    sp_ = res > a.rthr
    log(f"reduced Beyn N1 {sum(N1)}: {len(lam)} eigenvalues inside ({time.time() - t0:.0f} s), {sp_.sum()} spurious:")
    for l, r, f in zip(lam, res, sp_):
        log(f"  {l.real:.5f} (Ec{l.real - Ec:+.4f}) {l.imag:+.2e}i  |r| {r:.2e}  {'spurious' if f else 'physical'}")
    Q = C[:, sp_]
    if Q.shape[1]:
        B = S1 @ Q
        Pabs = B @ np.linalg.solve(Q.conj().T @ B, B.conj().T)
    else:
        Pabs = np.zeros_like(S1)

    # ---- fine grid
    lo, hi, st = (float(x) for x in a.grid.split(":"))
    grid = np.round(np.arange(lo, hi + 0.5 * st, st), 6)
    etas = [float(x) for x in a.eta.split(",")]
    exact = json.load(open(a.exact_cache)) if os.path.exists(a.exact_cache) else {}
    blk = oe.dense_blocks(dev) if a.exact else None
    rows = []
    for x in grid:
        E = Ec + x
        SL, SR = sint(E)
        B0 = mb.injection(SL)[0]
        b = np.zeros((dev.n, B0.shape[1]), complex); b[dev.bL] = B0
        GR = 1j * (SR - SR.conj().T)
        Tof = lambda X: float(np.real(np.einsum("ic,ij,jc->", X[dev.bR].conj(), GR, X[dev.bR])))
        row = dict(dE=float(x), channels=int(B0.shape[1]))
        key = f"{name}:{a.center:g}:{x:.6f}"
        if a.exact and key not in exact:
            if True:
                t = time.time()
                X = oe.block_solver(dev, blk, E, SL, SR).solve(b)
                exact[key] = Tof(X)
                json.dump(exact, open(a.exact_cache, "w"))
                log(f"  exact Ec{x:+.3f}: T {exact[key]:.6f} ({time.time() - t:.0f} s)")
        if key in exact:
            row["T_exact"] = exact[key]
        S1L, S1R = project_sigma(dev, Us, SL, SR)
        Ac = E * S1 - _dense_H1(blk1, off)
        Ac[:N1[0], :N1[0]] -= S1L
        Ac[off[-2]:, off[-2]:] -= S1R
        Ub = np.vstack([Us[s_].conj().T @ b[dev.sb[s_]:dev.sb[s_ + 1]] for s_ in range(len(Us))])
        for eta in etas:
            y = np.linalg.solve(Ac + 1j * eta * Pabs, Ub)
            row[f"T_ms_{eta:g}"] = Tof(fine_vec_block(Us, y, off))
        rows.append(row)
        line = f"Ec{x:+.3f} ({row['channels']} ch)" + (f" exact {row['T_exact']:.6f}" if "T_exact" in row else "")
        for eta in etas:
            t = row[f"T_ms_{eta:g}"]
            line += f" | eta {eta:g}: {t:.6f}" + (f" ({(t - row['T_exact']) / row['T_exact']:+.1e})" if "T_exact" in row else "")
        log(line)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(a), N1=N1, lam=[[l.real, l.imag] for l in lam], res=res.tolist(),
                   spurious=sp_.tolist(), rows=rows), open(f"{a.out}.json", "w"), indent=1)
    if any("T_exact" in r for r in rows):
        for eta in etas:
            e = np.array([abs(r[f"T_ms_{eta:g}"] - r["T_exact"]) / r["T_exact"] for r in rows])
            log(f"eta {eta:g}: MS T error median {np.median(e):.1e} max {e.max():.1e} at Ec{grid[np.argmax(e)]:+.3f}")
    log(f"-> {a.out}.json")


def _dense_H1(blk1, off):
    n = len(blk1["HD"])
    H = np.zeros((off[-1], off[-1]), complex)
    for s in range(n):
        H[off[s]:off[s + 1], off[s]:off[s + 1]] = blk1["HD"][s]
        if s < n - 1:
            H[off[s]:off[s + 1], off[s + 1]:off[s + 2]] = blk1["HU"][s]
            H[off[s + 1]:off[s + 2], off[s]:off[s + 1]] = blk1["HU"][s].conj().T
    return H


def fine_vec_block(Us, Y, off):
    return np.vstack([Us[s] @ Y[off[s]:off[s + 1]] for s in range(len(Us))])


if __name__ == "__main__":
    main()
