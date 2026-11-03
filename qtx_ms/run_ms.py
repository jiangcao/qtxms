"""Two-level mode-space basis (qtmode method, no enrichment) as stand-alone mode space and as the coarse
level of the two-level solver, on a prepared input set (python -m mgms.prep.si_sheet ...).

    python -m mgms.run_ms --inputs inputs/si3x5_sp3s [--no-cores] [--dE 0.01,0.05,0.1,0.2,0.35]

Per energy E (relative to the lead conduction edge in meta.json):
  exact : direct sparse solve of A = E - H - Sigma_L - Sigma_R for the injection columns
  MS1   : Galerkin in U1 (level 1),   MS2 : Galerkin in U = U1 U2 (level 2)
  MG1 / MG2 : flexible GMRES with the balancing two-level preconditioner, coarse = U1 / U, GMRES smoother
T = sum over injection columns of psi_R^H Gamma_R psi_R. Bases and Sigma are cached in --out.
"""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp

from mgms.prep.inputs import read_inputs
from mgms import msbasis as mb
from mgms.twolevel import TwoLevel

log = mb.log
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--inputs", required=True)
p.add_argument("--sigma-cache", help="lead self-energy cache, shared by all runs on these inputs "
               "(default results_ms/sigma/<inputs name>)")
p.add_argument("--out", help="cache / results folder (default results_ms/<inputs name>[_nocore])")
p.add_argument("--no-cores", action="store_true", help="tiles only (ab initio mode)")
p.add_argument("--direct-l2", action="store_true",
               help="no level 1 (U1 = I): level 2 from the slab density matrix of the full orbital system")
p.add_argument("--l2-metric", choices=["euclid", "S"], default="euclid",
               help="level-2 eigenproblem: rho (euclid) or S^1/2 rho S^1/2 with the level-1 slab overlap (S)")
p.add_argument("--d-core", type=float, default=6.0)
p.add_argument("--tile-orbs", type=int, default=2500)
p.add_argument("--tile-dE", default="0.005,0.245,0.495,0.745,0.995", help="tile energies - Ec_lead [eV]")
p.add_argument("--tile-thr", type=float, default=1e-4)
p.add_argument("--core-win", default="-0.05,1.1", help="core window: Ec_bulk + lo, Ec_lead + hi [eV]")
p.add_argument("--core-thr", type=float, default=1e-2)
p.add_argument("--mu", default="0,0.395", help="level-2 window - Ec_lead [eV]")
p.add_argument("--npts", type=int, default=24, help="contour points")
p.add_argument("--l2-thr", type=float, default=1e-5)
p.add_argument("--dE", default="0.01,0.03,0.06,0.10,0.20,0.35", help="test energies - Ec_lead [eV]")
p.add_argument("--E", help="absolute test energies [eV] (overrides --dE)")
p.add_argument("--k", type=int, default=10, help="GMRES smoother steps")
p.add_argument("--tol", type=float, default=1e-8)
p.add_argument("--maxit", type=int, default=100, help="FGMRES iteration cap per channel")
p.add_argument("--mg-cols", type=int, default=0, help="MG on the n strongest channels only (0: all)")
p.add_argument("--inj-tol", type=float, default=1e-4, help="Gamma_L eigenvalue cut [eV] for injection")
a = p.parse_args()

name = os.path.basename(os.path.normpath(a.inputs)) + ("_l2direct" if a.direct_l2 else "_nocore" if a.no_cores else "")
out = a.out or os.path.join("results_ms", name)
os.makedirs(os.path.join(out, "tiles"), exist_ok=True)
t0 = time.time()
inp = read_inputs(a.inputs)
meta = inp["meta"]
Ec = meta["edges"]["Ec_lead"]
dev = mb.Device(inp)
dev.sigma_cache = a.sigma_cache or os.path.join("results_ms", "sigma", os.path.basename(os.path.normpath(a.inputs)))
log(f"{name}: device {dev.n} orbitals, slabs {dev.ns}, |bL| {len(dev.bL)}, |bR| {len(dev.bR)}, "
    f"lead faces {len(dev.lead.Lf)}/{len(dev.lead.Rf)}, interior {len(dev.lead.I)}")

# ------------------------------------------------------------------ level 1
t = time.time()
if a.direct_l2:                     # level 2 straight on the orbitals: U1 = I per slab
    Us = [sp.identity(n, dtype=complex, format="csr") for n in dev.ns]
else:
    blocks, core, dmin = mb.partition(dev, cores=not a.no_cores, d_core=a.d_core, tile_orbs=a.tile_orbs)
    cores = [b for b in blocks if b["kind"] == "core"]
    log(f"partition: {len(cores)} core blocks ({sum(len(b['orbs']) for b in cores)} orb), "
        f"{len(blocks) - len(cores)} tiles, orbitals per tile {min(len(b['orbs']) for b in blocks if b['kind'] == 'tile')}"
        f"..{max(len(b['orbs']) for b in blocks if b['kind'] == 'tile')}")
    bases = [None] * len(blocks)
    fcore = os.path.join(out, "cores.npz")
    if cores:
        if os.path.exists(fcore):
            z = np.load(fcore)
            for i, b in enumerate(blocks):
                if b["kind"] == "core":
                    bases[i] = z[f"b{i}"]
        else:
            t = time.time()
            bulk = mb.BulkFromDevice(inp, core, dmin)
            ks = np.linspace(0, 1, 401)[:, None] * np.array([2 * np.pi / bulk.a_lat, 0, 0])
            Ek = np.linalg.eigvalsh(bulk.hk(ks))
            Eg = np.linalg.eigvalsh(bulk.hk(np.zeros((1, 3))))[0]
            mid = 0.5 * meta["edges"]["Ec_bulk"]
            Ecb = Ek[Ek > mid].min()
            log(f"bulk H(k) from core atoms {bulk.atoms}: a = {bulk.a_lat:.4f} A, {bulk.ns} orb, {len(bulk.terms)} terms, "
                f"VBM {Eg[Eg < mid].max():.4f}, CBM (Delta) {Ecb:.4f} eV (meta Ec_bulk {meta['edges']['Ec_bulk']:.4f})")
            lo, hi = (float(x) for x in a.core_win.split(","))
            ewin = (meta["edges"]["Ec_bulk"] + lo, Ec + hi)
            cpos = inp["xyz"][np.where(core)[0]]
            K = bulk.kgrid(8 * bulk.a_lat, np.ptp(cpos[:, 1]), np.ptp(cpos[:, 2]))
            kst, cst = bulk.states(K, ewin)
            log(f"core window [{ewin[0]:.3f}, {ewin[1]:.3f}] eV: {len(K)} k points, {len(kst)} Bloch states")
            for i, b in enumerate(blocks):
                if b["kind"] == "core":
                    bases[i], _ = mb.core_basis(bulk, inp["xyz"][b["atoms"]], kst, cst, thr=a.core_thr)
            np.savez(fcore, **{f"b{i}": bases[i] for i, b in enumerate(blocks) if b["kind"] == "core"})
            log(f"core bases: {[bases[i].shape for i, b in enumerate(blocks) if b['kind'] == 'core']}, {time.time() - t:.0f} s")
    geo = mb.Geometry(inp["xyz"], meta.get("periodic"))
    tree = geo.tree()
    tE = [Ec + float(x) for x in a.tile_dE.split(",")]
    for i, b in enumerate(blocks):
        if b["kind"] != "tile":
            continue
        f = os.path.join(out, "tiles", f"tile_{i:03d}.npz")
        if not os.path.exists(f):
            tt = time.time()
            Q, s, nemb = mb.tile_modes(inp, geo, tree, b["atoms"], tE, tile_thr=a.tile_thr)
            np.savez(f, U=Q, s=s, atoms=b["atoms"])
            log(f"  tile {i} (slab {b['slab']}): {len(b['orbs'])} orb, embedding {nemb} atoms -> {Q.shape[1]} modes, "
                f"{time.time() - tt:.0f} s")
        z = np.load(f)
        assert np.array_equal(z["atoms"], b["atoms"]), f"{f}: built on another partition"
        bases[i] = z["U"]
    Us, own = mb.slab_U(dev, blocks, bases)
N1 = [u.shape[1] for u in Us]
blk1 = mb.slab_blocks(dev, Us)
PL = Us[0][dev.bL].toarray()
PR = Us[-1][dev.bR - dev.sb[-2]].toarray()
log(f"level 1: N1 = {sum(N1)} {N1} ({time.time() - t:.0f} s for tiles + assembly)")

# ------------------------------------------------------------------ level 2
mu = [Ec + float(x) for x in a.mu.split(",")]
frho = os.path.join(out, f"rho_{mu[0]:.4f}_{mu[1]:.4f}_{a.npts}.npz")
if os.path.exists(frho) and np.array_equal(np.load(frho)["N1"], N1):
    rho = [np.load(frho)[f"r{s}"] for s in range(len(Us))]
else:
    t = time.time()
    rho = mb.rho_contour(dev, blk1, PL, PR, mu[0], mu[1], a.npts)
    np.savez(frho, N1=N1, **{f"r{s}": r for s, r in enumerate(rho)})
    log(f"contour rho [{mu[0]:.4f}, {mu[1]:.4f}] eV, {a.npts} pts: {time.time() - t:.0f} s; "
        f"trace {np.round([np.trace(r).real for r in rho], 2)}")
V = mb.level2_basis(rho, a.l2_thr, S=blk1["SD"] if a.l2_metric == "S" else None)
N2 = [v.shape[1] for v in V]
blk2 = mb.project(blk1, V)
PL2, PR2 = PL @ V[0], PR @ V[-1]
U1 = sp.block_diag(Us, format="csr")
U2 = sp.block_diag([sp.csr_matrix(Us[s] @ V[s]) for s in range(len(Us))], format="csr")
log(f"level 2 (thr {a.l2_thr:g}, {a.l2_metric} metric): N2 = {sum(N2)} {N2}; setup {time.time() - t0:.0f} s")


# ------------------------------------------------------------------ transmission
def sigma(E):
    return dev.sigma(E + 1e-6j)


def T_of(psi, GR):
    pr = psi[dev.bR]
    return float(np.real(np.einsum("ic,ij,jc->", pr.conj(), GR, pr)))


fres = os.path.join(out, "results" + ("" if a.l2_metric == "euclid" else f"_{a.l2_metric}") +
                    ("" if a.l2_thr == 1e-5 else f"_thr{a.l2_thr:g}") +
                    ("" if a.mu == "0,0.395" else f"_mu{a.mu.replace(',', '_')}") + ".json")
res = json.load(open(fres)) if os.path.exists(fres) else {}
res.update(N1=N1, N2=N2, n=dev.n, slabs=dev.ns, cores=not a.no_cores)
res.setdefault("E", {})
Elist = [float(x) for x in a.E.split(",")] if a.E else [Ec + float(x) for x in a.dE.split(",")]
for E in Elist:
    dE = E - Ec
    t = time.time()
    SL, SR = sigma(E)
    ts = time.time() - t
    GR = 1j * (SR - SR.conj().T)
    B, lam, X = mb.injection(SL, a.inj_tol)
    nc = B.shape[1]
    r = dict(dE=dE, E=E, channels=nc, t_sigma=ts)
    if nc == 0:
        log(f"E - Ec = {dE:+.3f}: no propagating channel")
        res["E"][f"{dE:.4f}"] = r
        continue
    b = np.zeros((dev.n, nc), complex)
    b[dev.bL] = B
    A = dev.A(E, SL, SR)
    t = time.time()
    cs = mb.CSolver(A)
    psi = cs.solve(b)
    # all Gamma_L columns above 1e-7 lambda_max (qtmode reference), to check the injection cut
    kf = lam > 1e-7 * lam.max()
    bf = np.zeros((dev.n, kf.sum()), complex)
    bf[dev.bL] = X[:, kf] * np.sqrt(lam[kf])
    T_full = T_of(cs.solve(bf), GR)
    cs.free()
    r.update(T_exact=T_of(psi, GR), T_exact_allcols=T_full, cols_all=int(kf.sum()),
             res_exact=float(np.linalg.norm(A @ psi - b) / np.linalg.norm(b)), t_exact=time.time() - t)
    Te = r["T_exact"]
    rel = lambda x, ref=Te: abs(x - ref) / max(ref, 1e-12)
    log(f"E = {E:.4f} (Ec {dE:+.3f}): {nc} channels, T_exact {Te:.5f} (all {r['cols_all']} Gamma_L cols "
        f"{T_full:.5f}), residual {r['res_exact']:.1e}; sigma {ts:.0f} s, exact {r['t_exact']:.0f} s")
    sel = np.arange(nc) if a.mg_cols <= 0 else np.arange(nc)[::-1][:a.mg_cols]   # strongest channels last in B
    Te_sel = T_of(psi[:, sel], GR)
    r.update(mg_cols=len(sel), T_exact_mgcols=Te_sel)
    levels = (("1", U1, blk1, PL, PR), ("2", U2, blk2, PL2, PR2))[1 if a.direct_l2 else 0:]   # U1 = I: MS1 exact
    for tag, U, blk, P_L, P_R in levels:
        t = time.time()
        Ac = mb.BlockTridiag(blk, E, P_L.conj().T @ SL @ P_L, P_R.conj().T @ SR @ P_R)
        x_ms = U @ Ac.solve(U.conj().T @ b)
        r[f"T_MS{tag}"] = T_of(x_ms, GR)
        r[f"t_coarse{tag}"] = time.time() - t
        log(f"  MS{tag}: T {r[f'T_MS{tag}']:.5f}, rel err {rel(r[f'T_MS{tag}']):.1e}, {r[f't_coarse{tag}']:.0f} s")
        tl = TwoLevel(A, U, smoother="gmres", k=a.k, coarse=Ac)
        its, nfail, xs = [], 0, np.zeros((dev.n, len(sel)), complex)
        t = time.time()
        for i, c in enumerate(sel):
            x, hist, _ = tl.solve(b[:, c], tol=a.tol, maxit=a.maxit)
            xs[:, i] = x; its.append(len(hist) - 1); nfail += hist[-1] > a.tol
            if i < 3 or (i + 1) % 10 == 0:
                log(f"    MG{tag} channel {i + 1}/{len(sel)}: {len(hist) - 1} its, residual {hist[-1]:.1e}, "
                    f"{time.time() - t:.0f} s")
        r[f"T_MG{tag}"] = T_of(xs, GR)
        r[f"its_MG{tag}"] = float(np.mean(its))
        r[f"maxits_MG{tag}"] = int(max(its))
        r[f"fail_MG{tag}"] = int(nfail)
        r[f"mv_MG{tag}"] = tl.nmv / len(sel)
        r[f"t_MG{tag}"] = time.time() - t
        log(f"  MG{tag}: {r[f'its_MG{tag}']:.1f} its (max {max(its)}, {nfail} not converged), "
            f"{r[f'mv_MG{tag}']:.0f} matvecs per channel, T err {rel(r[f'T_MG{tag}'], Te_sel):.1e} "
            f"(on {len(sel)} channels), {r[f't_MG{tag}']:.0f} s")
    res["E"][f"{dE:.4f}"] = r
    json.dump(res, open(fres, "w"), indent=1)
log(f"done, {time.time() - t0:.0f} s -> {fres}")
