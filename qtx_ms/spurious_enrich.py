"""Residual enrichment of a mode-space basis from its spurious device resonances.

Per round, on every --source input set (flat band and/or gated device):
  find the resonances z of the open MS device near the --find energies (qtx_ms.ms_deflate.find_spurious), keep the
  spurious ones (fine residual > --thr eV), r = (z S - H - Sigma) U y with |U y|_S = 1;
  per slab s: e_s = r_s (corr raw) or A_ss(Re z)^-1 r_s (corr slab, block Jacobi; dense LU per slab and energy),
  blocks with |r_s| < --bfrac max_s |r_s| skipped; orthogonalize against V_s, SVD, append the directions with
  singular value > --rcut (at most --nadd per slab and round). U stays block-diagonal.
After every round: modes saved (<out>/r<i>_slab<s>.npy) and T_MS vs exact on every --eval input set.

    python -m qtx_ms.spurious_enrich --modes <prefix> --source inputs/mos2_hfo2 \\
        --eval inputs/mos2_hfo2:results_ms/exact/mos2_hfo2.json,inputs/mos2_hfo2_gate0.4:results_ms/exact/mos2_hfo2_gate0.4.json \\
        --Ec 0.50435 --rounds 3 --out results_ms/gate_off/enrich/<tag>
"""
import argparse, json, os, time
import numpy as np, scipy.linalg as sla

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb
from qtx_ms.run_l2_clean import Coarse, fine_residual, append
from qtx_ms.ms_deflate import find_spurious

log = mb.log


def device(path, sig):
    d = mb.Device(read_inputs(path), obc="folded")
    d.sigma_cache = sig
    return d


def spurious_residuals(dev, c, Ec, Es, st, k, thr, imax):
    """slab blocks of the fine residual of the spurious resonances: list of (E, z, R)"""
    rows, Y = find_spurious(dev, c, Ec, Es, st, k, thr, imax)
    out = []
    zs = [complex(q[1] + Ec, q[2]) for q in rows if q[3] > thr]
    Es_ = [q[0] + Ec for q in rows if q[3] > thr]
    for E, z, j in zip(Es_, zs, range(Y.shape[1])):
        SL, SR = dev.sigma(E + 1e-6j)
        X = c.prolong(Y[:, j:j + 1])
        out.append((E, z, fine_residual(dev, X, np.array([z]), SL, SR)[:, 0]))
    return out, len(rows)


def evaluate(dev, c, Ec, exact):
    ex = json.load(open(exact))["E"]
    res = {}
    for key in sorted(ex, key=float):
        if "T_exact" not in ex[key] or not 0 <= float(key) <= 0.5:
            continue
        E = Ec + float(key)
        SL, SR = dev.sigma(E + 1e-6j)
        B = dev.inject(SL)
        GR = 1j * (SR - SR.conj().T)
        b = np.zeros((dev.n, B.shape[1]), complex); b[dev.bL] = B
        x = c.prolong(c.Ac(E, SL, SR).solve(c.restrict(b)))
        T = float(np.real(np.einsum("ic,ij,jc->", x[dev.bR].conj(), GR, x[dev.bR])))
        res[key] = dict(T_ms=T, T_exact=ex[key]["T_exact"], err=T / ex[key]["T_exact"] - 1)
    e = np.abs([r["err"] for r in res.values()])
    return res, float(np.median(e)), float(e.max())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", required=True)
    p.add_argument("--source", required=True, help="input sets whose spurious resonances enrich the basis (comma list)")
    p.add_argument("--eval", required=True, help="inputs:exact_json pairs (comma list)")
    p.add_argument("--Ec", type=float, required=True)
    p.add_argument("--find", default="0:0.5:0.01")
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--thr", type=float, default=5.0)
    p.add_argument("--imax", type=float, default=0.05)
    p.add_argument("--corr", default="raw", choices=["raw", "slab"])
    p.add_argument("--bfrac", type=float, default=0.1)
    p.add_argument("--rcut", type=float, default=1e-2)
    p.add_argument("--nadd", type=int, default=40)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--sigma-cache", default="results_ms/sigma/mos2_hfo2")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    srcs = {s: device(s, a.sigma_cache) for s in a.source.split(",")}
    evs = []
    for q in a.eval.split(","):
        i, x = q.split(":")
        evs.append((i, srcs[i] if i in srcs else device(i, a.sigma_cache), x))
    d0 = next(iter(srcs.values()))
    V = [np.load(f"{a.modes}_slab{s}.npy").astype(complex) for s in range(len(d0.ns))]
    lo, hi, st = (float(x) for x in a.find.split(":"))
    Es = np.round(np.arange(lo, hi + st / 2, st), 6)
    hist = []
    for rnd in range(a.rounds + 1):
        rec = dict(round=rnd, N=[v.shape[1] for v in V], eval={})
        for i, dev, x in evs:
            res, med, mx = evaluate(dev, Coarse(dev, V), a.Ec, x)
            rec["eval"][i] = dict(median=med, max=mx, E=res)
            log(f"round {rnd} N {sum(rec['N'])} {rec['N']}: {os.path.basename(i)} T err median {med:.2e} max {mx:.2e}")
        if rnd == a.rounds:
            hist.append(rec); break
        t = time.time()
        cols = [[] for _ in V]
        nsp = 0
        for s_in, dev in srcs.items():
            c = Coarse(dev, V)
            R, nres = spurious_residuals(dev, c, a.Ec, Es, st, a.k, a.thr, a.imax)
            nsp += len(R)
            log(f"  {os.path.basename(s_in)}: {nres} resonances, {len(R)} spurious")
            lus = {}
            for E, z, r in R:
                nb = np.array([np.linalg.norm(r[dev.sb[s]:dev.sb[s + 1]]) for s in range(len(V))])
                for s in np.nonzero(nb > a.bfrac * nb.max())[0]:
                    rs = r[dev.sb[s]:dev.sb[s + 1]]
                    if a.corr == "slab":
                        if (E, s) not in lus:
                            sl = slice(dev.sb[s], dev.sb[s + 1])
                            Ass = (E * dev.S[sl, sl] - dev.H[sl, sl]).toarray()
                            lus = {kk: vv for kk, vv in lus.items() if kk[0] == E}
                            lus[(E, s)] = sla.lu_factor(Ass, check_finite=False)
                        rs = sla.lu_solve(lus[(E, s)], rs.astype(complex), check_finite=False)
                    cols[s].append(rs)
        rec.update(spurious=nsp, t_find=time.time() - t)
        if nsp == 0:
            log("  no spurious resonances left"); hist.append(rec); break
        add = []
        for s in range(len(V)):
            if not cols[s]:
                add.append(0); continue
            X = np.array(cols[s]).T
            Vn, n = append(V[s], X, a.rcut)
            n = min(n, a.nadd)
            V[s] = Vn[:, :V[s].shape[1] + n]
            add.append(n)
        rec["added"] = add
        log(f"  round {rnd + 1}: {nsp} spurious resonances, added {add} ({time.time() - t:.0f} s)")
        for s, v in enumerate(V):
            np.save(f"{a.out}/r{rnd + 1}_slab{s}.npy", v)
        hist.append(rec)
        json.dump(hist, open(f"{a.out}/history.json", "w"), indent=1)
    json.dump(hist, open(f"{a.out}/history.json", "w"), indent=1)
    log(f"done -> {a.out}/history.json")


if __name__ == "__main__":
    main()
