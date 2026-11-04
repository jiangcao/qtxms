"""Galerkin test of local bases (step 1b): does the mode-space solution reach what the basis span allows?

For each basis U (old tile U1 / U1 V, direct level 2, and the ideal local snapshot bases of span_bench: SVD of the exact
psi on each region at the training energies, k_R from the held-out tolerance) and each test energy:
    x_MS = U A_c^-1 U^H b,  b = A psi_exact
    span = |(I - U U^H) psi| / |psi|,  err = |psi - x_MS| / |psi|,  amp = err / span (Galerkin amplification),
    T error, sigma_min(A_c) (spurious coarse states at E), lambda_min(U^H S U).
Regions nest in the slabs, so U per slab = block-diagonal over the regions of that slab (orthonormal columns).

    python -m qtx_ms.galerkin_bench --inputs inputs/mos2_hfo2 --psi results_ms/span_bench/mos2_hfo2_psi
"""
import argparse, json, os, time
import numpy as np
import scipy.linalg as sla

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb
from qtx_ms.run_l2_clean import Coarse
from qtx_ms.span_bench import load_psi, regions, ideal_curve, kneed

log = mb.log


def dense_Ac(cs, E, SL, SR):
    blk, off, n = cs.blk, cs.off, len(cs.V)
    A = np.zeros((cs.N, cs.N), complex)
    for s in range(n):
        a = slice(off[s], off[s + 1])
        A[a, a] = E * blk["SD"][s] - blk["HD"][s]
        if s < n - 1:
            b = slice(off[s + 1], off[s + 2])
            A[a, b] = E * blk["SU"][s] - blk["HU"][s]
            A[b, a] = (E * blk["SU"][s] - blk["HU"][s]).conj().T
    A[off[0]:off[1], off[0]:off[1]] -= cs.PL.conj().T @ SL @ cs.PL
    A[off[-2]:off[-1], off[-2]:off[-1]] -= cs.PR.conj().T @ SR @ cs.PR
    return A


def ideal_basis(dev, rl, Ptr, Pte, groups, tol):
    """Per-slab U from the regions rl (label, device orbitals): top-k_R left singular vectors of the training psi."""
    sb = dev.sb
    nsl = len(dev.ns)
    cols = [[] for _ in range(nsl)]
    for lab, o in rl:
        s = int(np.searchsorted(sb, o[0], side="right") - 1)
        assert o[-1] < sb[s + 1]
        W, sv, _ = np.linalg.svd(Ptr[o], full_matrices=False)
        _, et, _ = ideal_curve(Ptr[o], Pte[o], groups)
        k = kneed(et, tol)
        k = len(sv) if k is None else max(k, 1)
        X = np.zeros((sb[s + 1] - sb[s], k), complex)
        X[o - sb[s]] = W[:, :k]
        cols[s].append(X)
    return [np.hstack(c) for c in cols]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--psi", required=True)
    p.add_argument("--tiles", default="results_ms/mos2_hfo2_nocore_tc6_sk600/tiles")
    p.add_argument("--test", default="0.01,0.05,0.07,0.15,0.25,0.31,0.39,0.45", help="test energies - Ec (held out)")
    p.add_argument("--bases", default="tile_t1e-5=results_ms/mos2_hfo2_vcycle/tile_t1e-5,"
                   "tile_t1e-6=results_ms/mos2_hfo2_vcycle/tile_t1e-6,"
                   "direct_n400=results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead",
                   help="name=prefix: <prefix>_U1/_V_slab<s>.npy (tile levels: name_U1, name_U1V) or <prefix>_slab<s>.npy")
    p.add_argument("--smin-max", type=int, default=6000, help="sigma_min(A_c) by dense SVD only up to this N")
    p.add_argument("--regions", default="slab,tile,x2,x2tile,x4")
    p.add_argument("--tols", default="1e-2,1e-3")
    p.add_argument("--train-step", type=float, default=0.02)
    p.add_argument("--out", help="default <psi>/galerkin_bench.json")
    p.add_argument("--save", help="folder: write the ideal bases as <folder>/<name>_slab<s>.npy (schwarz --modes)")
    p.add_argument("--save-only", action="store_true", help="stop after --save")
    a = p.parse_args()
    inp = read_inputs(a.inputs)
    name = os.path.basename(os.path.normpath(a.inputs))
    dev = mb.Device(inp, obc="folded")
    dev.sigma_cache = os.path.join("results_ms", "sigma", name)
    Ec = inp["meta"]["edges"]["Ec_lead"]
    P = load_psi(a.psi)
    istr = [abs(d / a.train_step - round(d / a.train_step)) < 1e-6 for d, _ in P]
    Ptr = np.hstack([ps for (d, ps), t in zip(P, istr) if t])
    te = [ps for (d, ps), t in zip(P, istr) if not t]
    Pte = np.hstack(te)
    gb = np.concatenate([[0], np.cumsum([x.shape[1] for x in te])])
    groups = list(zip(gb[:-1], gb[1:]))
    nsl = len(dev.ns)

    bases = {}
    for spec in a.bases.split(","):
        nm, pre = spec.split("=")
        if os.path.exists(f"{pre}_U1_slab0.npy"):
            U1 = [np.load(f"{pre}_U1_slab{s}.npy") for s in range(nsl)]
            V = [np.load(f"{pre}_V_slab{s}.npy") for s in range(nsl)]
            bases[nm + "_U1"] = U1
            bases[nm + "_U1V"] = [u @ v for u, v in zip(U1, V)]
        else:
            bases[nm] = [np.load(f"{pre}_slab{s}.npy") for s in range(nsl)]
    R = regions(dev, inp, a.tiles)
    for rn in a.regions.split(","):
        for tol in (float(t) for t in a.tols.split(",")):
            bases[f"ideal_{rn}_{tol:g}"] = ideal_basis(dev, R[rn], Ptr, Pte, groups, tol)
    del Ptr, Pte
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        for nm, V in bases.items():
            if nm.startswith("ideal_"):
                for s, v in enumerate(V):
                    np.save(os.path.join(a.save, f"{nm}_slab{s}.npy"), v)
        log(f"ideal bases -> {a.save}")
        if a.save_only:
            return
    coarse = {}
    for nm, V in bases.items():
        t0 = time.time()
        cs = Coarse(dev, V)
        Sc = cs.Sc.toarray()
        lmin = float(np.linalg.eigvalsh(0.5 * (Sc + Sc.conj().T))[0])
        coarse[nm] = cs
        log(f"basis {nm:22s} N {cs.N:5d} {[v.shape[1] for v in V]}, lambda_min(U^H S U) {lmin:.2e}; {time.time() - t0:.0f} s")
        coarse[nm].lmin = lmin

    psiE = {round(d, 4): ps for d, ps in P}
    res = {nm: dict(N=cs.N, lmin_S=cs.lmin, E={}) for nm, cs in coarse.items()}
    for dE in (float(x) for x in a.test.split(",")):
        t0 = time.time()
        E = Ec + dE
        psi = psiE[round(dE, 4)]
        SL, SR = dev.sigma(E + 1e-6j)
        A = dev.A(E, SL, SR)
        b = A @ psi
        nL = dev.inject(SL).shape[1]
        GR = 1j * (SR - SR.conj().T)
        T = lambda y: float(np.real(np.einsum("ic,ij,jc->", y[dev.bR, :nL].conj(), GR, y[dev.bR, :nL])))
        Tex = T(psi)
        npsi = np.linalg.norm(psi)
        line = []
        for nm, cs in coarse.items():
            x = cs.prolong(cs.Ac(E, SL, SR).solve(cs.restrict(b)))
            span = np.linalg.norm(psi - cs.prolong(cs.restrict(psi))) / npsi
            err = np.linalg.norm(psi - x) / npsi
            smin = float(sla.svdvals(dense_Ac(cs, E, SL, SR))[-1]) if cs.N <= a.smin_max else float("nan")
            r = dict(span=float(span), err=float(err), amp=float(err / span), dT=(T(x) - Tex) / Tex, smin=smin)
            res[nm]["E"][f"{dE:.3f}"] = r
            line.append(f"{nm} span {span:.1e} err {err:.1e} amp {err / span:.0f} dT {r['dT']:+.0e} smin {smin:.1e}")
        log(f"E = Ec {dE:+.3f} ({psi.shape[1]} columns, {time.time() - t0:.0f} s):\n    " + "\n    ".join(line))
    for nm, r in res.items():
        e = np.array([v["err"] for v in r["E"].values()])
        s = np.array([v["span"] for v in r["E"].values()])
        log(f"{nm:22s} N {r['N']:5d}  span med {np.median(s):.1e}  MS err med {np.median(e):.1e} max {e.max():.1e}  "
            f"amp med {np.median(e / s):.0f} max {(e / s).max():.0f}  lambda_min(S_c) {r['lmin_S']:.1e}")
    out = a.out or os.path.join(a.psi, "galerkin_bench.json")
    json.dump(res, open(out, "w"), indent=1)
    log(f"-> {out}")


if __name__ == "__main__":
    main()
