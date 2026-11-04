"""Local-solve diagnostics on slabs and slab interfaces, before any two-level iteration.

From the mode-space solution x_MS: r = b - A x_MS, true error e = psi - x_MS (exact psi of a psi file). A subdomain
set D with local solves gives the correction z = sum_j R_j^T B_j R_j r (restricted: each orbital from one subdomain);
reported per set: |e - z| / |e| (how much of the error one local step removes; < 1 is useful, > 1 amplifies),
|r - A z| / |r|, and the same restricted to the interface zones and to the slab interiors. Subdomain sets:
  slab        each device slab (block), no overlap
  iface<w>    windows of width w Å centred on the slab interfaces (orbitals elsewhere untouched)
  slab+iface  multiplicative: slab step, new residual, interface step
Local solves B_j:
  lu          exact local inverse of A_jj (E S - H, with Sigma on the lead faces)
  defl<d>     closed (H_jj, S_jj) eigenpairs, inverse on |E - lambda| > d only (states near E left out)
  shift<s>    exact local inverse of A_jj + i s S_jj (complex shift damping the local resonances)
  cap<s>/<d>  exact local inverse of A_jj + i s P S_jj P, P a quadratic ramp over the outer d Å of the window
              towards its cuts (absorbing layer on the artificial boundaries only; lead faces keep Sigma)

    python -m qtx_ms.smoother_diag --inputs inputs/mos2_hfo2 --psi results_ms/qtbm_cmp/mos2_hfo2_eta0/psi_+0.55434897.npz \\
        --sigma-cache results_ms/sigma_qtx_eta0/mos2_hfo2 --modes results_ms/mos2_hfo2_l2direct/modes/mu-0.15_0.6_n400_lead
"""
import argparse, json, os, time
import numpy as np
import scipy.linalg as sla

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb
from qtx_ms.plot_residual import ms_solution

log = mb.log


_EIG = {}


def local_op(A, S, H, idx, method, E, x=None, xmin=None, xmax=None):
    """Callable rj -> B_j rj for one subdomain (closed eigenpairs cached per subdomain)."""
    Ajj = A[idx][:, idx].toarray()
    if method == "lu":
        lu = sla.lu_factor(Ajj, check_finite=False)
        return lambda rj: sla.lu_solve(lu, rj, check_finite=False)
    if method.startswith("shift"):
        s = float(method[5:])
        lu = sla.lu_factor(Ajj + 1j * s * S[idx][:, idx].toarray(), check_finite=False)
        return lambda rj: sla.lu_solve(lu, rj, check_finite=False)
    if method.startswith("cap"):
        s, d = (float(v) for v in method[3:].split("/"))
        xi = x[idx]
        lo, hi = xi.min(), xi.max()
        ramp = np.zeros(len(idx))
        if lo > xmin + 1e-6:                                   # a cut on the left (not the lead face)
            ramp = np.maximum(ramp, np.clip((lo + d - xi) / d, 0, 1) ** 2)
        if hi < xmax - 1e-6:
            ramp = np.maximum(ramp, np.clip((xi - (hi - d)) / d, 0, 1) ** 2)
        P = np.sqrt(ramp)
        Sjj = S[idx][:, idx].toarray()
        lu = sla.lu_factor(Ajj + 1j * s * (P[:, None] * Sjj * P[None, :]), check_finite=False)
        return lambda rj: sla.lu_solve(lu, rj, check_finite=False)
    if method.startswith("defl"):
        d = float(method[4:])
        key = (int(idx[0]), len(idx))
        if key not in _EIG:
            _EIG[key] = sla.eigh(H[idx][:, idx].toarray(), S[idx][:, idx].toarray(), check_finite=False)
        lam, V = _EIG[key]
        f = np.where(np.abs(E - lam) > d, 1 / (E - lam + (np.abs(E - lam) <= d)), 0.0)
        return lambda rj: V @ (f[:, None] * (V.conj().T @ rj))
    raise ValueError(method)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--psi", required=True)
    p.add_argument("--sigma-cache", required=True)
    p.add_argument("--modes", required=True)
    p.add_argument("--sets", default="slab,iface6,iface10,slab+iface6")
    p.add_argument("--methods", default="lu,shift0.05,defl0.1,defl0.3")
    p.add_argument("--zone", type=float, default=3.0, help="half width [Å] of the interface zones in the report")
    p.add_argument("--out", help="json (default <psi>_smoother.json)")
    a = p.parse_args()
    inp = read_inputs(a.inputs)
    c = np.load(a.psi)
    dev, E, b, A, x0, cs, T = ms_solution(inp, c, a.modes, a.sigma_cache)
    A = A.tocsr()
    H, S = dev.H.tocsr(), dev.S.tocsr()
    psi = c["psi_m"]
    r, e = b - A @ x0, psi - x0
    ax = inp["meta"]["axis"]
    atom = np.repeat(np.arange(len(inp["norb"])), inp["norb"])
    x = inp["xyz"][atom[dev.dev], ax]
    sb = dev.sb
    slabs = [np.arange(sb[s], sb[s + 1]) for s in range(len(dev.ns))]
    xi = [0.5 * (x[slabs[s]].max() + x[slabs[s + 1]].min()) for s in range(len(slabs) - 1)]
    zone = np.zeros(dev.n, bool)
    for q in xi:
        zone |= np.abs(x - q) < a.zone
    ne = np.linalg.norm(e)
    log(f"E = Ec {E - inp['meta']['edges']['Ec_lead']:+.3f}: |e|/|psi| {ne / np.linalg.norm(psi):.2e}, "
        f"|r|/|b| {np.linalg.norm(r) / np.linalg.norm(b):.2e}; interfaces at x = {', '.join(f'{q:.1f}' for q in xi)} Å; "
        f"zones ±{a.zone} Å hold {zone.mean():.0%} of the orbitals, {np.linalg.norm(e[zone]) ** 2 / ne ** 2:.0%} of |e|², "
        f"{np.linalg.norm(r[zone]) ** 2 / np.linalg.norm(r) ** 2:.0%} of |r|²")

    def subdomains(name):
        if name == "slab":
            return [(i, np.ones(len(i), bool)) for i in slabs]
        w = float(name[5:])
        win = [np.where(np.abs(x - q) < w / 2)[0] for q in xi]
        return [(i, np.ones(len(i), bool)) for i in win]

    res = []
    for method in a.methods.split(","):
        for st in a.sets.split(","):
            t = time.time()
            z = np.zeros_like(r)
            rr = r.copy()
            for part in st.split("+"):
                dz = np.zeros_like(r)
                for idx, core in subdomains(part):
                    B = local_op(A, S, H, idx, method, E, x, x.min(), x.max())
                    dz[idx[core]] += B(rr[idx])[core]
                z = z + dz
                rr = r - A @ z
            d = e - z
            row = dict(method=method, set=st, err=float(np.linalg.norm(d) / ne),
                       err_zone=float(np.linalg.norm(d[zone]) / np.linalg.norm(e[zone])),
                       err_rest=float(np.linalg.norm(d[~zone]) / np.linalg.norm(e[~zone])),
                       res=float(np.linalg.norm(rr) / np.linalg.norm(r)),
                       T_err=float((T(x0 + z) - T(psi)) / T(psi)), time=time.time() - t)
            res.append(row)
            log(f"  {method:9s} {st:12s}: |e - z|/|e| {row['err']:.2e} (zones {row['err_zone']:.2e}, rest "
                f"{row['err_rest']:.2e}), |r - Az|/|r| {row['res']:.2e}, T {row['T_err']:+.1e}; {row['time']:.0f} s")
    out = a.out or os.path.splitext(a.psi)[0] + "_smoother.json"
    json.dump(dict(E=E, rows=res), open(out, "w"), indent=1)
    log(f"-> {out}")


if __name__ == "__main__":
    main()
