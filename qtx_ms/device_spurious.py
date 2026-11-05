"""Spurious device states of a mode-space basis (closed and open MS device) on a given input set.

Closed: generalized eigenpairs (H_c, S_c) of U = diag(V_s), fine residual |(H - lam S) U c| (c^H S_c c = 1), slab weights.
Open: resonances of A_c(E) near the given energies (Coarse.ritz) with their fine residual.

    python -m qtx_ms.device_spurious --inputs inputs/mos2_hfo2_gate0.4 --modes <prefix> --Ec 0.50435 --win -0.15,0.6 \\
        --open 0.06,0.17,0.36 --out results_ms/gate_off/spurious/<tag>.npz
"""
import argparse, os, time
import numpy as np, scipy.linalg as sla

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb
from qtx_ms.run_l2_clean import Coarse, fine_residual

log = mb.log


def dense_coarse(c):
    n, off = len(c.V), c.off
    Hc = np.zeros((c.N, c.N), complex); Sc = np.zeros_like(Hc)
    for M, kD, kU in ((Hc, "HD", "HU"), (Sc, "SD", "SU")):
        for s in range(n):
            M[off[s]:off[s + 1], off[s]:off[s + 1]] = c.blk[kD][s]
            if s < n - 1:
                M[off[s]:off[s + 1], off[s + 1]:off[s + 2]] = c.blk[kU][s]
                M[off[s + 1]:off[s + 2], off[s]:off[s + 1]] = c.blk[kU][s].conj().T
    return 0.5 * (Hc + Hc.conj().T), 0.5 * (Sc + Sc.conj().T)


def closed_residual(dev, c, lam, Y):
    X = c.prolong(Y)
    R = (dev.S @ X) * lam[None] - dev.H @ X
    return np.linalg.norm(R, axis=0), c.block_norms(X) ** 2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--modes", required=True)
    p.add_argument("--Ec", type=float, required=True)
    p.add_argument("--win", default="-0.15,0.6", help="closed eigenvalues reported in Ec + lo..hi")
    p.add_argument("--open", default="", help="energies - Ec for open resonances (Coarse.ritz)")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--sigma-cache", default="results_ms/sigma/mos2_hfo2")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc="folded")
    dev.sigma_cache = a.sigma_cache
    V = [np.load(f"{a.modes}_slab{s}.npy") for s in range(len(dev.ns))]
    c = Coarse(dev, V)
    t = time.time()
    Hc, Sc = dense_coarse(c)
    ws = np.linalg.eigvalsh(Sc)
    log(f"{a.modes}: N {c.N}, S_c eigenvalues {ws[0]:.3e} .. {ws[-1]:.3e}")
    lam, Y = sla.eigh(Hc, Sc)
    lo, hi = (float(x) for x in a.win.split(","))
    m = (lam > a.Ec + lo) & (lam < a.Ec + hi)
    lam, Y = lam[m], Y[:, m]
    res, w = closed_residual(dev, c, lam, Y)
    log(f"closed MS device: {m.sum()} eigenvalues in Ec{lo:+}..{hi:+}, {time.time() - t:.0f} s")
    for q in (0.1, 0.5, 1, 2, 5):
        log(f"   residual > {q} eV: {(res > q).sum()}")
    out = dict(lam=lam - a.Ec, res=res, w=w, N=c.N)
    for i in np.argsort(lam):
        log(f"   {lam[i] - a.Ec:+.4f}  res {res[i]:.2e}  slab weights " + " ".join(f"{x:.2f}" for x in w[:, i]))
    if a.open:
        E0 = [float(x) for x in a.open.split(",")]
        ro = []
        for dE in E0:
            E = a.Ec + dE
            SL, SR = dev.sigma(E + 1e-6j)
            z, Yz = c.ritz(E, SL, SR, a.k)
            X = c.prolong(Yz)
            X /= np.sqrt(np.real(np.einsum("ic,ic->c", X.conj(), dev.S @ X)))[None]
            r = np.linalg.norm(fine_residual(dev, X, z, SL, SR), axis=0)
            wb = c.block_norms(X) ** 2
            for j in np.argsort(abs(z - E)):
                log(f"   open near Ec{dE:+.3f}: z - Ec {z[j].real - a.Ec:+.5f} {z[j].imag:+.2e}i  res {r[j]:.2e}  slabs "
                    + " ".join(f"{x:.2f}" for x in wb[:, j]))
                ro.append([dE, z[j].real - a.Ec, z[j].imag, r[j], *wb[:, j]])
        out["open"] = np.array(ro)
    np.savez(a.out, **out)
    log(f"-> {a.out}")


if __name__ == "__main__":
    main()
