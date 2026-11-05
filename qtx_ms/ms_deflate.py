"""Remove spurious device states from a mode-space basis by deflating them from the coarse space.

1. Find: resonances z of the open MS device, A_c(E) y = 0 near every energy of --find (Coarse.ritz, k nearest),
   fine residual |(z S - H - Sigma) U y| with |U y|_S = 1. Physical resonances: residual 0.2-3 eV; spurious ones
   (no fine counterpart): ~10 eV. Keep y with residual > --thr and |Re z - E| < step/2 + 1 meV, |Im z| < --imax.
2. Deflate: Q = basis of {x : x^H S_c Y = 0} (S_c-orthogonal complement of the spurious vectors). Galerkin solve in
   U Q: x = U Q (Q^H A_c(E) Q)^-1 Q^H U^H b, one fixed Q for all energies (dense, N - n_sp).
   Deflation fails (gated device: T 100x too large; flat band err 1e-6 -> 1e-3): the spurious vectors overlap the
   physical space.
3. Absorb (--absorb eta,...): keep the space, add an absorbing term on the spurious subspace only,
   A_c + i eta W, W = S_c Y_o Y_o^H S_c (Y_o: S_c-orthonormal basis of span(Y)): the spurious resonances get width
   ~eta while the physical states orthogonal to them are untouched.
4. Scan: T_MS raw, deflated and absorbed on --dE vs the exact json.

    python -m qtx_ms.ms_deflate --inputs inputs/mos2_hfo2_gate0.4 --modes <prefix> --Ec 0.50435 \\
        --exact results_ms/exact/mos2_hfo2_gate0.4.json --out results_ms/gate_off/deflate/<tag>.json
"""
import argparse, json, os, time
import numpy as np, scipy.linalg as sla

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb
from qtx_ms.run_l2_clean import Coarse, fine_residual
from qtx_ms.device_spurious import dense_coarse
from qtx_ms.ms_scan import grid

log = mb.log


def find_spurious(dev, c, Ec, Es, step, k, thr, imax):
    rows, Y = [], []
    for dE in Es:
        E = Ec + dE
        SL, SR = dev.sigma(E + 1e-6j)
        z, Yz = c.ritz(E, SL, SR, k)
        X = c.prolong(Yz)
        nrm = np.sqrt(np.abs(np.real(np.einsum("ic,ic->c", X.conj(), dev.S @ X))))
        X /= nrm[None]
        r = np.linalg.norm(fine_residual(dev, X, z, SL, SR), axis=0)
        wb = c.block_norms(X) ** 2
        for j in range(len(z)):
            near = abs(z[j].real - E) < step / 2 + 1e-3 and abs(z[j].imag) < imax
            if near:
                rows.append([dE, z[j].real - Ec, z[j].imag, r[j], *wb[:, j]])
                if r[j] > thr:
                    Y.append(Yz[:, j] / nrm[j])
        nsp = sum(1 for q in rows if q[0] == dE and q[3] > thr)
        log(f"find Ec{dE:+.3f}: {sum(1 for q in rows if q[0] == dE)} resonances near, {nsp} spurious")
    return np.array(rows), (np.array(Y).T if Y else np.zeros((c.N, 0), complex))


def complement(Sc, Y, tol=1e-8):
    """columns Q (orthonormal) with Q^H Sc Y = 0; the span of Y is first reduced to its numerical rank."""
    if Y.shape[1] == 0:
        return np.eye(Sc.shape[0], dtype=complex), 0
    W = Sc @ Y
    u, s, _ = np.linalg.svd(W, full_matrices=True)
    r = int((s > tol * s[0]).sum())
    return u[:, r:], r


def sc_orth(Sc, Y, tol=1e-8):
    """S_c-orthonormal basis of span(Y)"""
    if Y.shape[1] == 0:
        return Y
    w, X = np.linalg.eigh(Y.conj().T @ Sc @ Y)
    k = w > tol * w[-1]
    return Y @ (X[:, k] / np.sqrt(w[k]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--modes", required=True)
    p.add_argument("--Ec", type=float, required=True)
    p.add_argument("--find", default="-0.1:0.55:0.01", help="lo:hi:step, energies - Ec where resonances are searched")
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--thr", type=float, default=5.0, help="fine residual [eV] above which a resonance is spurious")
    p.add_argument("--imax", type=float, default=0.05)
    p.add_argument("--dE", default="0:0.5:0.01")
    p.add_argument("--exact", required=True)
    p.add_argument("--sigma-cache", default="results_ms/sigma/mos2_hfo2")
    p.add_argument("--out", required=True)
    p.add_argument("--absorb", default="", help="eta values [eV] for the absorbing term on the spurious subspace")
    p.add_argument("--no-defl", action="store_true", help="skip the deflated solve")
    p.add_argument("--reuse", action="store_true", help="read the spurious vectors from <out>_res.npz of an earlier run")
    a = p.parse_args()
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc="folded")
    dev.sigma_cache = a.sigma_cache
    V = [np.load(f"{a.modes}_slab{s}.npy") for s in range(len(dev.ns))]
    c = Coarse(dev, V)
    Hc, Sc = dense_coarse(c)
    t = time.time()
    lo, hi, st = (float(x) for x in a.find.split(":"))
    Es = np.round(np.arange(lo, hi + st / 2, st), 6)
    fres = a.out.replace(".json", "_res.npz")
    if a.reuse and os.path.exists(fres):
        z = np.load(fres); rows, Y = z["rows"], z["Y"]
        log(f"spurious vectors from {fres}")
    else:
        rows, Y = find_spurious(dev, c, a.Ec, Es, st, a.k, a.thr, a.imax)
    Q, nsp = complement(Sc, Y)
    log(f"{a.modes}: N {c.N}, {len(rows)} resonances, {Y.shape[1]} spurious vectors, rank {nsp} -> N {Q.shape[1]}; "
        f"{time.time() - t:.0f} s")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(fres, rows=rows, Y=Y)
    Yo = sc_orth(Sc, Y)
    W = (Sc @ Yo) @ (Sc @ Yo).conj().T
    etas = [float(x) for x in a.absorb.split(",")] if a.absorb else []
    ex = json.load(open(a.exact))["E"]
    n0, n1 = c.off[1], c.off[-2]
    res = dict(modes=a.modes, inputs=a.inputs, N=c.N, N_defl=int(Q.shape[1]), thr=a.thr, Ec=a.Ec, E={})
    for dE in grid(a.dE, 0.05):
        key, E = f"{dE:.4f}", a.Ec + dE
        SL, SR = dev.sigma(E + 1e-6j)
        B = dev.inject(SL)
        GR = 1j * (SR - SR.conj().T)
        b = np.zeros((dev.n, B.shape[1]), complex); b[dev.bL] = B
        Tof = lambda x: float(np.real(np.einsum("ic,ij,jc->", x[dev.bR].conj(), GR, x[dev.bR])))
        Ac = E * Sc - Hc
        Ac[:n0, :n0] -= c.PL.conj().T @ SL @ c.PL
        Ac[n1:, n1:] -= c.PR.conj().T @ SR @ c.PR
        bc = c.restrict(b)
        x0 = c.prolong(np.linalg.solve(Ac, bc))
        r = dict(E=E, T_ms=Tof(x0))
        if not a.no_defl:
            r["T_defl"] = Tof(c.prolong(Q @ np.linalg.solve(Q.conj().T @ Ac @ Q, Q.conj().T @ bc)))
        for eta in etas:
            r[f"T_abs{eta:g}"] = Tof(c.prolong(np.linalg.solve(Ac + 1j * eta * W, bc)))
        tk = [q for q in r if q.startswith("T_") and q != "T_ms"]
        if "T_exact" in ex.get(key, {}):
            Te = ex[key]["T_exact"]; r["T_exact"] = Te
            r["err"] = r["T_ms"] / Te - 1
            for q in tk:
                r["err" + q[1:]] = r[q] / Te - 1
        log(f"E = Ec{dE:+.3f}: T_exact {r.get('T_exact', float('nan')):.4e}  err MS {r.get('err', float('nan')):+.2e}  "
            + "  ".join(f"{q[2:]} {r.get('err' + q[1:], r[q]):+.2e}" for q in tk))
        res["E"][key] = r
        json.dump(res, open(a.out, "w"), indent=1)
    log(f"done -> {a.out}")


if __name__ == "__main__":
    main()
