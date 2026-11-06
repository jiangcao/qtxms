"""Level-1 local basis of one block (tile or slab): density-matrix contour vs FEAST-style subspace filter.

All methods share the same CAP+OBC embedding (msbasis._tile_embedding with the lead Sigma on the face orbitals,
as run_ms --tile-obc), the same contour (upper arc mu1..mu2, Gauss-Legendre in theta) and one dense LU of
A(z) = z S - H + iW - Sigma(z) per contour point; they differ only in the right-hand sides solved with it:

  dm     local density matrix, identity columns on the block P (m solves): rho_PP = -Im int G_PP dz / pi, eigh,
         keep occupations > cut  (the production level-1 basis)
  rf     randomized range finder on rho_PP: Omega (m x r Gaussian) on P, Y = rho_PP Omega, one-pass Nystrom
         (msbasis.tile_modes sketch): rho_PP ~ F F^T, F = Y (Omega^T Y)^-1/2, SVD, weights s^2
  feast  FEAST filter: Y (n x M Gaussian on the whole embedding), Q = -Im int G(z) S Y dz / pi = rho S Y, rows P,
         S_PP-orthonormalized: S_PP = L L^T, L^T Q = U s W^T, V = L^-T U (V^T S_PP V = I), weights s^2 / M
  feastP the same with Y supported on P only (S Y spreads to the first neighbours of P)
  feastL right-hand sides L Y with S_e = L L^T instead of S Y: probes with covariance S, E[Q Q^T] = (rho S rho)_PP,
         = rho_PP for a projector (rho S rho = rho); S Y gives (rho S^2 rho)_PP

Probe counts are nested (the first M columns of one draw), so one contour pass gives every M.
Accuracy against dm (eigenpairs lam_i, x_i): missed occupation sum_i lam_i |(1 - Pi_V) x_i|^2 over lam_i > cut
(Pi_V = Euclidean projector on span V), the worst |(1 - Pi_V) x_i| for lam_i > cut, for V cut by the method's own
weights ("self") and for the top k_dm(cut) vectors ("count"). Flops: dense LU 8/3 n^3, solve 8 n^2 per column (real
flops, complex arithmetic). --modes-out writes slab mode files (this block's slab replaced by the top k_dm(cut) vectors
of the method, the others copied from --modes-ref) for ms_scan.

    python -m qtx_ms.feast_bench --inputs inputs/mos2_hfo2 --block 0 --M 400,800,1200,1600 --workers 4 --threads 6
"""
import argparse, json, os, time
import numpy as np
import scipy.linalg as sla

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb

log = mb.log


def contour_pass(e, obc, zs, wz, groups, workers, threads):
    """acc[g] = sum_k w_k G(z_k)[P, :] R_g (complex), timings per group; contour points over forked processes."""
    He, Se, W, pidx = e["He"], e["Se"], e["W"], e["pidx"]

    def run(ks):
        acc = {g: np.zeros((len(pidx), R.shape[1]), complex) for g, R in groups.items()}
        tm = {"lu": 0.0, **{g: 0.0 for g in groups}}
        for k in ks:
            z = zs[k]
            t = time.time()
            A = (z * Se - He + 1j * W + mb._sig_emb(obc, e, z)).toarray()
            lu = sla.lu_factor(A, check_finite=False, overwrite_a=True)
            tm["lu"] += time.time() - t
            for g, R in groups.items():
                t = time.time()
                acc[g] += sla.lu_solve(lu, R, check_finite=False)[pidx] * wz[k]
                tm[g] += time.time() - t
            del lu, A
        return acc, tm

    import multiprocessing as mp
    from threadpoolctl import threadpool_limits
    if workers <= 1:
        with threadpool_limits(threads):
            return run(range(len(zs)))
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    parts = [list(range(j, len(zs), workers)) for j in range(workers)]

    def child(j):
        try:
            with threadpool_limits(threads):
                q.put((j, *run(parts[j]), None))
        except Exception as ex:                                            # noqa: BLE001
            q.put((j, None, None, repr(ex)))
    ps = [ctx.Process(target=child, args=(j,)) for j in range(workers)]
    for p_ in ps:
        p_.start()
    acc, tm = None, None
    for _ in ps:
        j, a_, t_, err = q.get()
        if err:
            raise RuntimeError(f"worker {j}: {err}")
        if acc is None:
            acc, tm = a_, t_
        else:
            for g in acc:
                acc[g] += a_[g]
            for g in tm:
                tm[g] += t_[g]
    for p_ in ps:
        p_.join()
    return acc, tm


def capture(V, lam, X, cut):
    """missed occupation and worst missed fraction of the dm eigenvectors with lam > cut by span(V)."""
    sel = lam > cut
    if V.shape[1] == 0:
        return float(lam[sel].sum()), 1.0
    Qv, _ = np.linalg.qr(V)
    R = X[:, sel] - Qv @ (Qv.T @ X[:, sel])
    f = np.einsum("ij,ij->j", R, R)
    return float((lam[sel] * f).sum()), float(np.sqrt(f.max()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--tile-orbs", type=int, default=100000, help="partition (100000: one block per slab)")
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--mu", default="-0.15,0.6", help="window relative to Ec_lead [eV]")
    p.add_argument("--npts", type=int, default=24)
    p.add_argument("--d-emb", type=float, default=12.0)
    p.add_argument("--d-cap", type=float, default=3.0)
    p.add_argument("--eta-cap", type=float, default=0.3)
    p.add_argument("--M", default="400,800,1200,1600", help="probe counts (rf: r, feast: M)")
    p.add_argument("--methods", default="dm,rf,feast,feastP")
    p.add_argument("--cuts", default="1e-4,1e-6,1e-8")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--sigma-cache", help="default results_ms/sigma/<inputs name>")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--schur", action="store_true", help="also time the production dm path (msbasis.tile_rho_contour, "
                   "Schur G_PP, double) and compare its rho")
    p.add_argument("--out", default="results_ms/feast_bench")
    p.add_argument("--modes-ref", help="slab mode prefix whose other slabs are copied into --modes-out files")
    p.add_argument("--modes-out", default="", help="'method:M:cut,...' -> <out>/modes/<method>_M<M>_c<cut>_slab<s>.npy")
    a = p.parse_args()

    name = os.path.basename(os.path.normpath(a.inputs))
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc="folded")
    dev.sigma_cache = a.sigma_cache or os.path.join("results_ms", "sigma", name)
    Ec = inp["meta"]["edges"]["Ec_lead"]
    blocks, _, _ = mb.partition(dev, cores=False, tile_orbs=a.tile_orbs)
    b = blocks[a.block]
    geo = mb.Geometry(inp["xyz"], inp["meta"].get("periodic"))
    e = mb._tile_embedding(inp, geo, geo.tree(), b["atoms"], a.d_emb, a.d_cap, a.eta_cap, obc=dev)
    n, m = e["He"].shape[0], len(e["pidx"])
    pidx = e["pidx"]
    Ms = [int(x) for x in a.M.split(",")]
    Mmax = max(Ms)
    methods = a.methods.split(",")
    cuts = [float(x) for x in a.cuts.split(",")]
    mu1, mu2 = Ec + float(a.mu.split(",")[0]), Ec + float(a.mu.split(",")[1])
    c, rr = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    x, wq = np.polynomial.legendre.leggauss(a.npts)
    th, wq = 0.5 * np.pi * (x + 1), 0.5 * np.pi * wq
    zs = c + rr * np.exp(1j * th)
    wz = -1j * rr * np.exp(1j * th) * wq
    log(f"{name} block {a.block} (slab {b['slab']}): m {m} orbitals, embedding n {n} ({len(e['emb'])} atoms); "
        f"contour {a.npts} pts [{mu1:.3f}, {mu2:.3f}]; probes {Ms}; {a.workers} x {a.threads} threads")

    rng = np.random.default_rng(a.seed)
    groups = {}
    if "dm" in methods:
        R = np.zeros((n, m)); R[pidx, np.arange(m)] = 1.0; groups["dm"] = R
    if "rf" in methods:
        Om = rng.standard_normal((m, Mmax)); R = np.zeros((n, Mmax)); R[pidx] = Om; groups["rf"] = R
    if "feast" in methods:
        Y = rng.standard_normal((n, Mmax)); groups["feast"] = np.asarray(e["Se"] @ Y)
    if "feastL" in methods:                                               # Y ~ N(0, S_e): E[Q Q^T] = rho S rho ~ rho
        ds, Ws = np.linalg.eigh(e["Se"].toarray())                       # S_e is indefinite here (device S)
        log(f"S_e: lambda_min {ds[0]:.2e}, {int((ds <= 0).sum())} eigenvalues <= 0 clipped for the S_e^1/2 probes")
        Le = Ws * np.sqrt(np.clip(ds, 0, None))
        groups["feastL"] = Le @ rng.standard_normal((n, Mmax))
        del Le, Ws
    if "feastP" in methods:
        Y = np.zeros((n, Mmax)); Y[pidx] = rng.standard_normal((m, Mmax)); groups["feastP"] = np.asarray(e["Se"] @ Y)
    t0 = time.time()
    acc, tm = contour_pass(e, dev, zs, wz, groups, a.workers, a.threads)
    twall = time.time() - t0
    flop_lu, flop_col = 8 / 3 * n ** 3 * a.npts, 8 * n ** 2 * a.npts
    log(f"contour pass {twall:.0f} s wall; summed over workers: LU {tm['lu']:.0f} s ("
        + ", ".join(f"{g} {tm[g]:.0f} s for {R.shape[1]} cols" for g, R in groups.items()) + ")")

    if a.schur:
        t = time.time()
        acc_s, _ = mb.tile_rho_contour(inp, geo, geo.tree(), b["atoms"], (mu1, mu2, a.npts), a.d_emb, a.d_cap,
                                       a.eta_cap, dev, gpp="schur", env_prec="double", workers=a.workers,
                                       threads=a.threads)
        t_schur = time.time() - t
        d_schur = float(np.abs(acc_s - acc["dm"]).max() / np.abs(acc["dm"]).max())
        log(f"dm via Schur G_PP (production path): {t_schur:.0f} s wall, max |acc - acc_LU| / max {d_schur:.1e}")
    assert "dm" in acc, "the reference needs method dm"
    rho = -acc["dm"].imag / np.pi
    rho = 0.5 * (rho + rho.T)
    t = time.time()
    lam, X = np.linalg.eigh(rho)
    lam, X = lam[::-1], X[:, ::-1]
    t_dm_post = time.time() - t
    SPP = e["Se"][pidx][:, pidx].toarray()
    Lc = np.linalg.cholesky(SPP)
    res = dict(inputs=name, block=a.block, slab=int(b["slab"]), m=m, n=n, npts=a.npts, mu=[mu1, mu2], Ec=Ec,
               t_wall=twall, t_lu=tm["lu"], t_schur=t_schur if a.schur else None, t_solve={g: tm[g] for g in groups}, flops=dict(lu=flop_lu, per_col=flop_col),
               dm=dict(k={f"{c_:g}": int((lam > c_).sum()) for c_ in cuts}, trace=float(lam.sum()), t_post=t_dm_post,
                       cols=m, flops_solve=flop_col * m), methods={})
    log(f"dm: trace {lam.sum():.2f}, kept " + ", ".join(f"{c_:g}: {int((lam > c_).sum())}" for c_ in cuts)
        + f"; solves {flop_col * m:.2e} flops, LU {flop_lu:.2e}")
    bases = {("dm", m): (X, lam)}
    for g in [g for g in methods if g != "dm"]:
        for M in Ms:
            t = time.time()
            Yg = -acc[g][:, :M].imag / np.pi
            if g == "rf":
                Om = groups["rf"][pidx][:, :M]
                d, V = np.linalg.eigh(0.5 * (Om.T @ Yg + Yg.T @ Om))
                ok = d > 1e-12 * d.max()
                F = Yg @ (V[:, ok] / np.sqrt(d[ok]))
                U, sv, _ = np.linalg.svd(F, full_matrices=False)
                w = sv ** 2
            else:                                                          # S_PP = L L^T, SVD of L^T Q: no squared
                Bq = Lc.T @ Yg                                             # condition number (Gram eigh loses 1e-8)
                Ub, sv, _ = np.linalg.svd(Bq, full_matrices=False)
                ok = sv > 1e-15 * sv.max()
                U = sla.solve_triangular(Lc.T, Ub[:, ok], lower=False)     # U^T S_PP U = I
                w = sv[ok] ** 2 / M
            tp = time.time() - t
            bases[(g, M)] = (U, w)
            r = dict(M=M, t_post=tp, flops_solve=flop_col * M, cuts={})
            for c_ in cuts:
                kd = int((lam > c_).sum())
                ks = int((w > c_).sum())
                ms_, ws_ = capture(U[:, :ks], lam, X, c_)
                mc_, wc_ = capture(U[:, :min(kd, U.shape[1])], lam, X, c_)
                r["cuts"][f"{c_:g}"] = dict(k_dm=kd, k_self=ks, miss_self=ms_, worst_self=ws_, miss_count=mc_,
                                            worst_count=wc_, occ_dm=float(lam[lam > c_].sum()))
            res["methods"].setdefault(g, []).append(r)
            log(f"{g:6s} M {M:5d}: " + " | ".join(
                f"{k}: k {v['k_self']}/{v['k_dm']} miss {v['miss_self']:.1e} worst {v['worst_self']:.2f}, "
                f"top-k miss {v['miss_count']:.1e} worst {v['worst_count']:.2f}" for k, v in r["cuts"].items())
                + f"; post {tp:.1f} s")
    os.makedirs(a.out, exist_ok=True)
    fjs = os.path.join(a.out, f"{name}_b{a.block}.json")
    json.dump(res, open(fjs, "w"), indent=1)
    log(f"-> {fjs}")
    if a.modes_out:
        assert a.modes_ref, "--modes-out needs --modes-ref"
        os.makedirs(os.path.join(a.out, "modes"), exist_ok=True)
        for spec in a.modes_out.split(","):
            g, M, cut = spec.split(":")
            U, w = bases[(g, m if g == "dm" else int(M))]
            k = min(int((lam > float(cut)).sum()), U.shape[1])                # N matched to dm at this cut
            pre = os.path.join(a.out, "modes", f"{name}_b{a.block}_{g}_M{M}_c{cut}")
            for s in range(len(dev.ns)):
                if s == b["slab"]:
                    Z = np.zeros((dev.ns[s], k), U.dtype)
                    Z[b["orbs"]] = U[:, :k]
                else:
                    Z = np.load(f"{a.modes_ref}_slab{s}.npy")
                np.save(f"{pre}_slab{s}.npy", Z)
            log(f"modes {pre}: slab {b['slab']} {k} columns")


if __name__ == "__main__":
    main()
