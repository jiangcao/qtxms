"""Remove the spurious level-1 OBC eigenstates by deflating them out of the coarse space (reduced Beyn on the open
level-1 system), then MS T on a fine grid inside the contour vs exact.

Level-1 open system T1(z) = z S1 - H1 - U1^H Sigma(z) U1 (Sigma on the continued retarded sheet at the contour nodes,
obc_eig.ContinuedSigma cache). Round k, coarse space span(U1 Z_k) (Z_0 = I):
  1. reduced Beyn on Z^H T1(z) Z  ->  (lambda, c);  fine vector v = U1 Z c, residual |T(lambda) v| / |v| with Sigma(lambda)
     interpolated from the nodes (no full-size solve)
  2. spurious = |r| > --rthr (physical resonances: 0.1-0.4 eV, spurious ~7 eV on mos2_hfo2)
  3. Q <- [Q, Z c_spurious],  Z_{k+1} = orthonormal complement of span(Q) in the level-1 coordinates
until no spurious eigenvalue is left inside the contour (or --rounds). The MS solve on the final space is the Galerkin
x = U1 Z (Z^H A_c Z)^-1 Z^H U1^H b (dense here; for large N1 the same as a bordered block-tridiagonal solve with
the k constraints Q^H y = 0).

  python -m qtx_ms.ms_deflate --inputs inputs/mos2_hfo2 --center 0.52135 --a 0.015 --b 0.01 --npts 24 \
      --U1 results_ms/mos2_hfo2_vcycle/tobc_t1e-6_lc --fine results_ms/obc_eig/c1_Ec+0.017 --out results_ms/obc_eig/deflate/c1_base
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.linalg as sla

from . import msbasis as mb
from . import obc_eig as oe
from .obc_eig_l1 import SigmaInterp, project_sigma, fine_vec, apply_T
from .ms_absorb import _dense_H1, fine_vec_block

log = mb.log


def dense_S1(blk1, off):
    n = len(blk1["SD"])
    S = np.zeros((off[-1], off[-1]), complex)
    for s in range(n):
        S[off[s]:off[s + 1], off[s]:off[s + 1]] = blk1["SD"][s]
        if s < n - 1:
            S[off[s]:off[s + 1], off[s + 1]:off[s + 2]] = blk1["SU"][s]
            S[off[s + 1]:off[s + 2], off[s]:off[s + 1]] = blk1["SU"][s].conj().T
    return S


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--center", type=float, required=True)
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--b", type=float, required=True)
    p.add_argument("--npts", type=int, default=24)
    p.add_argument("--probes", type=int, default=300)
    p.add_argument("--rank-tol", type=float, default=1e-8)
    p.add_argument("--U1", required=True)
    p.add_argument("--fine", help="obc_eig dir of the same contour (channel counts, full-size eigenpairs)")
    p.add_argument("--rthr", type=float, default=2.0, help="spurious: |T v| / |v| above this [eV]")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--grid", default="0.003:0.031:0.001", help="lo:hi:step, E - Ec [eV]")
    p.add_argument("--exact-cache", default="results_ms/obc_eig/absorb/exact_grid.json")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from .prep.inputs import read_inputs
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc="folded")
    Ec = inp["meta"]["edges"]["Ec_lead"]
    name = os.path.basename(os.path.normpath(a.inputs))
    cache = os.path.join("results_ms", "sigma", name)
    fch = a.fine and os.path.join(a.fine, f"channels_{a.center:+.8f}.json")
    if fch and os.path.exists(fch):
        nprop, nR = (tuple(x) for x in json.load(open(fch)))
    else:
        nprop = oe.channel_counts(dev.lead, a.center); nR = nprop
    sig = oe.ContinuedSigma(dev, cache=cache, nprop=nprop, nprop_R=nR)
    zs, ws = oe.ellipse(a.center, a.a, a.b, a.npts)
    SLs, SRs = zip(*[sig(z) for z in zs])
    sint = SigmaInterp(zs, ws, SLs, SRs)
    fine = None
    if a.fine and os.path.exists(os.path.join(a.fine, "eigpairs.npz")):
        f = np.load(os.path.join(a.fine, "eigpairs.npz"))
        fine = dict(lam=f["lam"], vec=f["vec"] / np.linalg.norm(f["vec"], axis=0))

    Us = oe.load_U1(a.U1, len(dev.ns))
    N1 = [u.shape[1] for u in Us]
    blk1 = mb.slab_blocks(dev, Us)
    off = np.concatenate([[0], np.cumsum(N1)])
    n1 = off[-1]
    S1, H1 = dense_S1(blk1, off), _dense_H1(blk1, off)

    def Ac(z, SL, SR):
        S1L, S1R = project_sigma(dev, Us, SL, SR)
        A = z * S1 - H1
        A[:N1[0], :N1[0]] -= S1L
        A[off[-2]:, off[-2]:] -= S1R
        return A

    Anodes = [Ac(z, SL, SR) for z, SL, SR in zip(zs, SLs, SRs)]
    log(f"{a.inputs}: level 1 {a.U1} N1 {N1} = {n1}; contour c {a.center} a {a.a} b {a.b}, {a.npts} pts, channels {nprop}")

    def reduced_beyn(Z, tag):
        t0 = time.time()
        V = np.random.default_rng(1).normal(size=(Z.shape[1], a.probes)) + 0j
        Xs = [sla.lu_solve(sla.lu_factor(Z.conj().T @ A @ Z, check_finite=False), V, check_finite=False) for A in Anodes]
        lam, C, s, k = oe.beyn(Xs, zs, ws, 1, a.rank_tol, a.center, max(a.a, a.b))
        ins = ((lam.real - a.center) / a.a) ** 2 + (lam.imag / a.b) ** 2 < 1
        lam, C = lam[ins], C[:, ins]
        o = np.argsort(lam.real)
        lam, C = lam[o], Z @ C[:, o]                                   # level-1 coordinates
        rows = []
        for l, c in zip(lam, C.T):
            v = fine_vec(Us, c)
            v /= np.linalg.norm(v)
            row = dict(lam=[float(l.real), float(l.imag)], dE=float(l.real - Ec),
                       res=float(np.linalg.norm(apply_T(dev, l, *sint(l), v))))
            if fine is not None and len(fine["lam"]):
                j = int(np.argmin(np.abs(fine["lam"] - l)))
                row.update(fine=j, dlam=float(abs(fine["lam"][j] - l)),
                           sin=float(np.sqrt(max(0.0, 1 - abs(np.vdot(fine["vec"][:, j], v)) ** 2))))
            rows.append(row)
        sp_ = np.array([r["res"] > a.rthr for r in rows], bool)
        log(f"[{tag}] reduced Beyn dim {Z.shape[1]}: rank {k} (s[k-1]/s0 {s[k - 1] / s[0]:.1e}), {len(lam)} inside, "
            f"{sp_.sum()} spurious, {time.time() - t0:.0f} s")
        for r, f_ in zip(rows, sp_):
            log(f"    Ec{r['dE']:+.5f} {r['lam'][1]:+.2e}i  |r| {r['res']:.2e}  {'spurious' if f_ else 'physical'}" +
                (f"  (fine {r['fine']}: dlam {r['dlam']:.1e} sin {r['sin']:.1e})" if "fine" in r else ""))
        return rows, C[:, sp_]

    Z = np.eye(n1, dtype=complex)
    Q = np.zeros((n1, 0), complex)
    rounds, spaces = [], [("base", Z)]
    for rnd in range(a.rounds + 1):
        rows, Csp = reduced_beyn(Z, f"round {rnd}")
        rounds.append(dict(dim=int(Z.shape[1]), rows=rows))
        if Csp.shape[1] == 0 or rnd == a.rounds:
            break
        Q = np.linalg.qr(np.hstack([Q, Csp]))[0]
        Z = sla.null_space(Q.conj().T)                                 # orthonormal complement (n1 - k columns)
        spaces.append((f"defl{rnd + 1}", Z))
    log(f"deflated {Q.shape[1]} spurious directions in {len(spaces) - 1} rounds")

    # ---- fine grid
    lo, hi, st = (float(x) for x in a.grid.split(":"))
    grid = np.round(np.arange(lo, hi + 0.5 * st, st), 6)
    exact = json.load(open(a.exact_cache)) if os.path.exists(a.exact_cache) else {}
    tags = [t for t, _ in spaces]
    out_rows = []
    for x in grid:
        E = Ec + x
        SL, SR = sint(E)
        B0 = mb.injection(SL)[0]
        b = np.zeros((dev.n, B0.shape[1]), complex); b[dev.bL] = B0
        GR = 1j * (SR - SR.conj().T)
        Tof = lambda X: float(np.real(np.einsum("ic,ij,jc->", X[dev.bR].conj(), GR, X[dev.bR])))
        A = Ac(E, SL, SR)
        Ub = np.vstack([Us[s_].conj().T @ b[dev.sb[s_]:dev.sb[s_ + 1]] for s_ in range(len(Us))])
        row = dict(dE=float(x))
        key = f"{name}:{a.center:g}:{x:.6f}"
        if key in exact:
            row["T_exact"] = exact[key]
        for t, Zt in spaces:
            y = Zt @ np.linalg.solve(Zt.conj().T @ A @ Zt, Zt.conj().T @ Ub)
            row[f"T_{t}"] = Tof(fine_vec_block(Us, y, off))
        out_rows.append(row)
        line = f"Ec{x:+.3f}" + (f" exact {row['T_exact']:.6f}" if "T_exact" in row else "")
        for t in tags:
            v = row[f"T_{t}"]
            line += f" | {t}: {v:.6f}" + (f" ({(v - row['T_exact']) / row['T_exact']:+.1e})" if "T_exact" in row else "")
        log(line)
    summ = {}
    if all("T_exact" in r for r in out_rows):
        for t in tags:
            e = np.array([abs(r[f"T_{t}"] - r["T_exact"]) / r["T_exact"] for r in out_rows])
            summ[t] = dict(median=float(np.median(e)), max=float(e.max()), at=float(grid[np.argmax(e)]))
            log(f"{t}: MS T error median {np.median(e):.1e} max {e.max():.1e} at Ec{grid[np.argmax(e)]:+.3f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(a), N1=N1, ndefl=int(Q.shape[1]), rounds=rounds, rows=out_rows, summary=summ),
              open(f"{a.out}.json", "w"), indent=1)
    np.save(f"{a.out}_Q.npy", Q)
    log(f"-> {a.out}.json")


if __name__ == "__main__":
    main()
