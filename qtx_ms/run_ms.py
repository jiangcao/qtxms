"""Two-level mode-space basis (qtmode method) as stand-alone mode space and as the coarse level of the
two-level solver, on a prepared input set (python -m qtx_ms.prep.si_sheet ...).

    python -m qtx_ms.run_ms --inputs inputs/si3x5_sp3s [--no-cores] [--dE 0.01,0.05,0.1,0.2,0.35]
                          [--enrich 3 --enrich-width 1]

--enrich n: up to n rounds of residual enrichment of the level-1 blocks before level 2 (msbasis.enrich_round:
window Ritz pairs of closed clusters, residuals on the cluster centre only; --enrich-mode slab: 2 width + 1
slabs around slab s, --enrich-width -1 = one global H1 eigenproblem as qtmode.enrich; --enrich-mode tile: a
tile and all tiles coupled to it).

Per energy E (relative to the lead conduction edge in meta.json):
  exact : direct sparse solve of A = E - H - Sigma_L - Sigma_R for the injection columns
  MS1   : Galerkin in U1 (level 1),   MS2 : Galerkin in U = U1 U2 (level 2)
  MG1 / MG2 : flexible GMRES with the balancing two-level preconditioner, coarse = U1 / U, GMRES smoother
T = sum over injection columns of psi_R^H Gamma_R psi_R. Bases and Sigma are cached in --out.
"""
import sys, argparse, json, os, time
import numpy as np
import scipy.sparse as sp

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb
from qtx_ms.twolevel import TwoLevel

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
p.add_argument("--tile-xsplit", type=int, default=1, help="cut every tile into this many pieces along x (equal orbitals)")
p.add_argument("--tile-cut", default="", help="tiles by planes instead of bisection: '<axis>:<c1>,<c2>@<slabs>' (msbasis.parse_cut), "
               "e.g. '2:8.5,15.5@1,2'; other slabs one tile")
p.add_argument("--tile-dE", default="0.005,0.245,0.495,0.745,0.995", help="tile energies - Ec_lead [eV]")
p.add_argument("--tile-thr", type=float, default=1e-4)
p.add_argument("--tile-eta", type=float, default=0.0, help="tile energies E + i eta [eV] (width for CAP-isolated states)")
p.add_argument("--tile-contour", type=int, default=0,
               help="> 0: tile basis from the tile density matrix of --tile-win on an arc with this many points "
                    "(instead of A(E) at --tile-dE)")
p.add_argument("--tile-win", default=None, help="tile contour window - Ec_lead [eV] (default: --mu)")
p.add_argument("--tile-rho-thr", type=float, default=1e-3,
               help="tile contour: keep occupations above this (applied when the cached tiles are loaded, so a "
                    "cache built with a lower cut, e.g. by --tile-dir, can be reused)")
p.add_argument("--tile-boundary", choices=["cap", "schur"], default="cap",
               help="tile embedding: CAP on its outer shell, or the Schur-complement self-energy of the closed "
                    "environment (needs --tile-contour or --tile-eta > 0)")
p.add_argument("--tile-dir", help="tile cache directory (default <out>/tiles)")
p.add_argument("--tile-obc", type=int, default=0, metavar="NPTS",
               help="> 0: tiles whose embedding reaches a lead face are built with the lead self-energy in the "
                    "embedding (tile_modes(obc=dev): lead-block atoms dropped, -Sigma_L/R on the face orbitals, CAP on "
                    "the other cuts), exact tile rho of --tile-win on an NPTS-point arc (NPTS = --npts reuses the "
                    "level-2 Sigma cache); the other tiles come from --tile-dir as usual (needs --tile-contour)")
p.add_argument("--tile-obc-dir", help="OBC tile cache directory (default <out>/tiles_obc)")
p.add_argument("--tile-obc-slabs", default="", help="with --tile-obc: only tiles of these slabs get the lead Sigma "
               "('ends' = first and last slab, or a comma list); default: every tile whose embedding reaches a lead face")
p.add_argument("--tile-worker", metavar="K,N",
               help="only build the missing tiles assigned to worker K of N (largest first), then exit")
p.add_argument("--tile-fast", choices=["", "full", "schur"], default="",
               help="contour tile rho via msbasis.tile_rho_contour: 'schur' = G_PP by the Schur complement of the "
                    "environment (only m columns solved), 'full' = whole-embedding LU; '' = the original serial path")
p.add_argument("--tile-env-prec", choices=["single", "double"], default="single", help="--tile-fast schur: environment LU precision")
p.add_argument("--tile-refine", type=int, default=1, help="--tile-fast schur, single: refinement steps of A_EE^-1 A_EP")
p.add_argument("--tile-emb-far", default="", help="--tile-fast: smaller embeddings off the axis, 'imz:d,...' "
               "(points with Im z >= imz use d [A]; e.g. 0.1:8,0.2:6)")
p.add_argument("--tile-pts-workers", type=int, default=1, help="--tile-fast: contour points over this many forked processes")
p.add_argument("--tile-pts-threads", type=int, default=0, help="--tile-fast: BLAS threads per process (0: unchanged)")
p.add_argument("--tile-sketch", type=int, default=0, help="tile contour: Nystrom sketch with this many probes (0: exact)")
p.add_argument("--tile-probes", choices=["gauss", "sign"], default="gauss", help="tile sketch probes")
p.add_argument("--core-win", default="-0.05,1.1", help="core window: Ec_bulk + lo, Ec_lead + hi [eV]")
p.add_argument("--core-thr", type=float, default=1e-2)
p.add_argument("--enrich", type=int, default=0, help="max rounds of level-1 residual enrichment")
p.add_argument("--enrich-mode", choices=["slab", "tile"], default="slab",
               help="clusters: slabs s +- width (slab) or a tile and all tiles coupled to it (tile)")
p.add_argument("--enrich-slabs", choices=["all", "interior"], default="all",
               help="enrich the blocks of every slab, or of the interior slabs only (end slabs: --lead-clean)")
p.add_argument("--enrich-width", type=int, default=1, help="slab mode: cluster half width in slabs (-1: whole device)")
p.add_argument("--enrich-tol", type=float, default=0.0, help="tile mode: ignore block couplings with max |H|+|S| <= tol")
p.add_argument("--enrich-win", default="-0.45,0.40", help="Ritz window - Ec_lead [eV]")
p.add_argument("--enrich-rthr", type=float, default=1.0, help="residual flagging a Ritz pair [eV]")
p.add_argument("--enrich-smin", type=float, default=0.1, help="SVD cut of the block residuals [eV]")
p.add_argument("--lead-clean", action="store_true",
               help="remove the spurious lead bands of the end-slab level-1 bases (lead_rms.RMS, as run_l2_clean)")
p.add_argument("--lead-clean-slabs", choices=["ends", "all"], default="ends",
               help="lead clean on the end slabs only, or on every slab (interior slabs: the level-1 basis of all its "
                    "tiles restricted to the lead-cell atoms matched in the slab, RMS vectors split back onto the tiles)")
p.add_argument("--lead-win", default="-0.6,0.45", help="lead spurious-state window - Ec_lead [eV]")
p.add_argument("--lead-tol", type=float, default=0.005, help="lead: spurious if > tol [eV] from every exact state")
p.add_argument("--lead-nk", type=int, default=25, help="k points on [0, pi] for the lead bands")
p.add_argument("--lead-kstep", type=int, default=2, help="RMS fit on every kstep-th k point")
p.add_argument("--lead-maxit", type=int, default=150)
p.add_argument("--map-tol", type=float, default=0.6, help="max displacement [A] of a lead-matched atom")
p.add_argument("--lead-clean-l2", action="store_true",
               help="after the level-2 cut: RMS on the end-slab level-2 bases, candidates = the rest of the level-1 "
                    "span (U1 (1 - V V^H)) on the lead cell, so U = U1 V stays nested")
p.add_argument("--lead-l2-drop", action="store_true", help="let level 2 drop the RMS vectors (default: kept)")
p.add_argument("--mu", default="0,0.395", help="level-2 window - Ec_lead [eV]")
p.add_argument("--npts", type=int, default=24, help="contour points")
p.add_argument("--l2-thr", type=float, default=1e-5)
p.add_argument("--dE", default="0.01,0.03,0.06,0.10,0.20,0.35", help="test energies - Ec_lead [eV]")
p.add_argument("--res-enrich", type=int, default=0,
               help="rounds of slab-basis enrichment from the residual of the original open system (after level 1)")
p.add_argument("--res-dE", default="0.0125:0.5125:0.025", help="training energies - Ec_lead: lo:hi:step or list")
p.add_argument("--res-corr", choices=["raw", "slab", "cluster", "exact"], default="cluster",
               help="residual -> correction: raw r, slab block Jacobi, overlapping slab clusters (s +- 1), exact A^-1 r")
p.add_argument("--res-rhs", choices=["inject", "random"], default="inject",
               help="residual right-hand sides: QTBM injection, or --res-probes Gaussian probes on all orbitals")
p.add_argument("--res-probes", type=int, default=20, help="random probes per energy")
p.add_argument("--res-contour", type=int, default=0,
               help="> 0: training points = this many points of the level-2 arc (--mu) instead of --res-dE")
p.add_argument("--res-integrate", action="store_true",
               help="with --res-contour and random probes: use the arc-integrated correction (rho - rho_MS) omega")
p.add_argument("--res-tol", type=float, default=1e-2, help="keep correction singular values above this (relative to |psi|)")
p.add_argument("--res-add", type=int, default=40, help="max vectors appended per slab and round")
p.add_argument("--Ec", type=float, help="reference edge for --dE and --exact-from (default Ec_lead; windows stay "
                                       "relative to Ec_lead), as in exact_T / run_l2_clean --Ec")
p.add_argument("--E", help="absolute test energies [eV] (overrides --dE)")
p.add_argument("--k", type=int, default=10, help="GMRES smoother steps")
p.add_argument("--tol", type=float, default=1e-8)
p.add_argument("--maxit", type=int, default=100, help="FGMRES iteration cap per channel")
p.add_argument("--mg-cols", type=int, default=0, help="MG on the n strongest channels only (0: all)")
p.add_argument("--mg", default="1,2", help="levels to run the two-level solver with ('' : none)")
p.add_argument("--save-basis", metavar="PREFIX", help="save level 1 (<PREFIX>_U1_slab<s>.npy, device orbitals x N1_s) "
               "and level 2 (<PREFIX>_V_slab<s>.npy, N1_s x N2_s, level-1 coordinates) for qtx_ms.schwarz --levels 3")
p.add_argument("--basis-only", action="store_true", help="stop after building (and saving) the basis")
p.add_argument("--exact-from", help="json with T_exact per dE (results_ms/exact/<name>.json): skip the exact "
               "solve at the energies it has (test energies Ec_lead + dE must match its Ec)")
p.add_argument("--inj-tol", type=float, default=1e-4, help="Gamma_L eigenvalue cut [eV] for injection")
p.add_argument("--obc", default="auto", choices=["auto", "qttools", "folded"], help="lead OBC on the real axis: qttools (QuaTrEx spectral OBC; auto falls back to the folded lead when it does not import) or mgms (folded lead); --exact-from values must come from the same OBC")
p.add_argument("--gpu", action="store_true", help="dense parts on the GPU (cupy): folded-lead Sigma, dense tile rho "
               "(contour points serial, --tile-pts-workers ignored), level-2 contour rho")
a = p.parse_args()
if a.gpu:
    from qtx_ms import gpu
    gpu.enable()
    print(f"backend: {gpu.describe()}", flush=True)

name = os.path.basename(os.path.normpath(a.inputs)) + ("_l2direct" if a.direct_l2 else "_nocore" if a.no_cores else "")
assert not (a.enrich and a.direct_l2), "--enrich needs a level-1 basis"
etag = "" if not a.enrich else (f"_enr{a.enrich}" +
                                (f"w{a.enrich_width}" if a.enrich_mode == "slab" else
                                 "tile" + ("" if a.enrich_tol == 0 else f"{a.enrich_tol:g}")) +
                                ("" if a.enrich_win == "-0.45,0.40" else f"_win{a.enrich_win.replace(',', '_')}") +
                                ("" if a.enrich_rthr == 1.0 else f"_r{a.enrich_rthr:g}") +
                                ("" if a.enrich_smin == 0.1 else f"_s{a.enrich_smin:g}") +
                                ("" if a.enrich_slabs == "all" else "_int"))
tile_win = a.tile_win or a.mu
ttag = (f"_tc{a.tile_contour}" + ("" if a.tile_win is None else f"_win{tile_win.replace(',', '_')}") +
        ("" if a.tile_rho_thr == 1e-3 else f"_t{a.tile_rho_thr:g}") +
        ("" if not a.tile_sketch else f"_sk{a.tile_sketch}") +
        ("" if a.tile_probes == "gauss" else f"_{a.tile_probes}")) if a.tile_contour else (
       ("" if a.tile_eta == 0 else f"_teta{a.tile_eta:g}") +
       ("" if a.tile_dE == "0.005,0.245,0.495,0.745,0.995" else f"_tdE{a.tile_dE.replace(',', '_')}"))
ttag += "" if a.tile_boundary == "cap" else "_schur"
ttag += "" if a.tile_xsplit == 1 else f"_tx{a.tile_xsplit}"
ttag += "" if not a.tile_cut else "_cut" + a.tile_cut.replace(":", "_").replace(",", "_").replace("@", "s")
assert not a.tile_obc or a.tile_contour, "--tile-obc: needs --tile-contour (and its --tile-rho-thr cut)"
ttag += "" if not a.tile_obc else f"_tobc{a.tile_obc}" + ("" if not a.tile_obc_slabs else f"s{a.tile_obc_slabs.replace(',', '_')}")
name += ttag
out = a.out or os.path.join("results_ms", name)
ltag = "" if not a.lead_clean else ("_lc" + ("d" if a.lead_l2_drop else "") + ("" if a.lead_win == "-0.6,0.45" else f"{a.lead_win.replace(',', '_')}") +
                                    ("" if a.lead_tol == 0.005 else f"_t{a.lead_tol:g}") +
                                    ("" if a.lead_clean_slabs == "ends" else "_all"))
tdir = a.tile_dir or os.path.join(out, "tiles")
os.makedirs(tdir, exist_ok=True)
os.makedirs(out, exist_ok=True)
t0 = time.time()
inp = read_inputs(a.inputs)
meta = inp["meta"]
Ec = meta["edges"]["Ec_lead"]
dev = mb.Device(inp, obc=a.obc)
log(f"lead OBC: {'qttools on the real axis' if dev.qtt is not None else 'folded lead'}")
dev.sigma_cache = a.sigma_cache or os.path.join("results_ms", "sigma", os.path.basename(os.path.normpath(a.inputs)))
log(f"{name}: device {dev.n} orbitals, slabs {dev.ns}, |bL| {len(dev.bL)}, |bR| {len(dev.bR)}, "
    f"lead faces {len(dev.lead.Lf)}/{len(dev.lead.Rf)}, interior {len(dev.lead.I)}")

# ------------------------------------------------------------------ level 1
t = time.time()
if a.direct_l2:                     # level 2 straight on the orbitals: U1 = I per slab
    Us = [sp.identity(n, dtype=complex, format="csr") for n in dev.ns]
else:
    blocks, core, dmin = mb.partition(dev, cores=not a.no_cores, d_core=a.d_core, tile_orbs=a.tile_orbs, xsplit=a.tile_xsplit,
                                     cut=mb.parse_cut(a.tile_cut))
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
    tw = [float(x) for x in tile_win.split(",")]
    odir = a.tile_obc_dir or os.path.join(out, "tiles_obc")
    obc_t = set()
    if a.tile_obc:                                           # tiles whose embedding reaches a lead-face atom
        os.makedirs(odir, exist_ok=True)
        aoff = np.concatenate([[0], np.cumsum(inp["norb"])])
        face_atoms = np.unique(np.searchsorted(aoff, dev.dev[np.concatenate([dev.bL, dev.bR])], side="right") - 1)
        for i, b in enumerate(blocks):
            if b["kind"] == "tile":
                near = tree.query_ball_point(geo.points(b["atoms"]), 12.0)
                if np.isin(np.unique(np.concatenate([np.asarray(q, int) for q in near])), face_atoms).any():
                    obc_t.add(i)
        if a.tile_obc_slabs:
            osl = ({0, len(dev.ns) - 1} if a.tile_obc_slabs == "ends" else
                   {int(x) % len(dev.ns) for x in a.tile_obc_slabs.split(",")})
            obc_t = {i for i in obc_t if blocks[i]["slab"] in osl}
        log(f"tiles with the lead Sigma in the embedding ({a.tile_obc}-point exact contour, {odir}): {sorted(obc_t)}")
    tfile = {i: os.path.join(odir, f"tile_obc_{i:03d}.npz") if i in obc_t else os.path.join(tdir, f"tile_{i:03d}.npz")
             for i, b in enumerate(blocks) if b["kind"] == "tile"}
    mine = set(tfile)
    if a.tile_worker:
        k, nw = (int(x) for x in a.tile_worker.split(","))
        order = sorted(tfile, key=lambda i: -len(blocks[i]["orbs"]) - 10 ** 6 * (i in obc_t))
        mine = {i for j, i in enumerate(order) if j % nw == k}
    fast = None if not a.tile_fast else dict(
        gpp=a.tile_fast, env_prec=a.tile_env_prec, refine=a.tile_refine, workers=a.tile_pts_workers,
        threads=a.tile_pts_threads or None,
        emb_far=[tuple(float(v) for v in x.split(":")) for x in a.tile_emb_far.split(",")] if a.tile_emb_far else None)
    for i, b in enumerate(blocks):
        if b["kind"] != "tile":
            continue
        f = tfile[i]
        if not os.path.exists(f) and i in mine:
            tt = time.time()
            if i in obc_t:
                Q, s, nemb = mb.tile_modes(inp, geo, tree, b["atoms"], None, contour=(Ec + tw[0], Ec + tw[1], a.tile_obc),
                                           rho_thr=min(a.tile_rho_thr, 1e-8), obc=dev, fast=fast)
            else:
                Q, s, nemb = mb.tile_modes(inp, geo, tree, b["atoms"], tE, tile_thr=a.tile_thr, eta=a.tile_eta,
                                           contour=(Ec + tw[0], Ec + tw[1], a.tile_contour) if a.tile_contour else None,
                                           rho_thr=min(a.tile_rho_thr, 1e-6), sketch=a.tile_sketch,
                                           probes=a.tile_probes, boundary=a.tile_boundary,
                                           fast=fast if a.tile_contour and a.tile_boundary == "cap" else None)
            np.savez(f, U=Q, s=s, atoms=b["atoms"])
            log(f"  {'OBC ' if i in obc_t else ''}tile {i} (slab {b['slab']}): {len(b['orbs'])} orb, embedding {nemb} "
                f"atoms -> {Q.shape[1]} modes, {time.time() - tt:.0f} s")
        if a.tile_worker:
            continue
        z = np.load(f)
        assert np.array_equal(z["atoms"], b["atoms"]), f"{f}: built on another partition"
        bases[i] = z["U"]
        if a.tile_contour:                                   # occupations descending, columns = the leading ones
            bases[i] = bases[i][:, :int(np.sum(z["s"][:bases[i].shape[1]] > a.tile_rho_thr))]
    if a.tile_worker:
        log(f"worker {a.tile_worker}: tiles done, {time.time() - t0:.0f} s")
        sys.exit(0)
    lwin = tuple(float(x) for x in a.enrich_win.split(","))
    for it in range(a.enrich):
        f = os.path.join(out, f"enrich{etag.replace(f'_enr{a.enrich}', '_')}_round{it}.npz")
        if os.path.exists(f):
            z = np.load(f)
            assert np.array_equal(z["N"], [q.shape[1] for q in bases]), f"{f}: built on another level-1 basis"
            added = {int(k[1:]): z[k] for k in z.files if k.startswith("b")}
            st = dict(added=int(sum(q.shape[1] for q in added.values())))
        else:
            added, st = mb.enrich_round(dev, blocks, bases, Ec, lwin, a.enrich_rthr, a.enrich_smin, a.enrich_width,
                                        a.enrich_mode, a.enrich_tol,
                                        slabs=None if a.enrich_slabs == "all" else range(1, len(dev.ns) - 1))
            np.savez(f, N=[q.shape[1] for q in bases], **{f"b{i}": q for i, q in added.items()})
            log(f"  enrichment round {it}: N1 {st['N1']}, {st['n_clusters']} clusters of {st['cluster_blocks']} blocks, "
                f"size mean {st['n_cluster_mean']:.0f} max {st['n_cluster_max']}, {st['n_window']} Ritz pairs "
                f"in window, {st['n_spurious']} flagged (|r| hist {st['res_hist']}), {st['t']:.0f} s")
        for i, q in added.items():
            bases[i] = np.hstack([bases[i], q])
        log(f"  enrichment round {it}: +{st['added']} columns in {len(added)} blocks")
        if not added:
            break
    if a.lead_clean:
        # end slabs: the level-1 basis on the lead cell, spurious reduced lead bands in the window removed by RMS
        from qtx_ms.lead_rms import LeadCell, RMS, slab_cell_map, to_cell, to_slab, band_stats
        cell = LeadCell(inp["lead"])
        ks = np.linspace(0, np.pi, a.lead_nk)
        Eex = cell.bands(ks, cache=os.path.join(out, f"bands_{a.lead_nk}.npz"))
        w1, w2 = (Ec + float(x) for x in a.lead_win.split(","))
        lead_added = {}                                     # slab -> {tile: number of RMS columns appended}
        clean_slabs = sorted({0, len(dev.ns) - 1}) if a.lead_clean_slabs == "ends" else list(range(len(dev.ns)))
        for s in clean_slabs:
            tl = [i for i, b in enumerate(blocks) if b["slab"] == s]
            X = np.zeros((dev.ns[s], sum(bases[i].shape[1] for i in tl)), np.result_type(*(bases[i] for i in tl)))
            c0 = 0
            for i in tl:
                X[blocks[i]["orbs"], c0:c0 + bases[i].shape[1]] = bases[i]
                c0 += bases[i].shape[1]
            cmap, mi = slab_cell_map(inp, dev, s, tol=a.map_tol, return_info=True)
            u, sv, _ = np.linalg.svd(to_cell(X, cmap), full_matrices=False)
            Phi = u[:, sv > 1e-6 * sv[0]]
            rms = RMS(cell, Phi, ks, (w1, w2))
            st0 = band_stats([rms.eig(q)[0] for q in ks], Eex, w1, w2, a.lead_tol)
            frms = os.path.join(out, f"rms_slab{s}{etag}{ltag}.npz")
            if os.path.exists(frms):
                z = np.load(frms)
                assert z["Xi"].shape == rms.Xi.shape, f"{frms}: built on another basis"
                rms.Cs = z["C"]
            else:
                tt = time.time()
                ki = np.arange(0, a.lead_nk, a.lead_kstep)
                rms.ks = ks[ki]
                hist = rms.run(Eex[ki], tol=a.lead_tol, maxit=a.lead_maxit)
                rms.ks = ks
                np.savez(frms, Xi=rms.Xi, C=rms.Cs, hist=json.dumps(hist))
                log(f"  slab {s} lead RMS: {len(hist)} steps, {time.time() - tt:.0f} s")
            st1 = band_stats([rms.eig(q)[0] for q in ks], Eex, w1, w2, a.lead_tol)
            lead_added[s] = {}
            if rms.Cs.shape[1]:
                Ys = to_slab(rms.added(), cmap, dev.ns[s])
                Ys = Ys / np.linalg.norm(Ys, axis=0)
                for i in tl:                                 # each tile spans the RMS vectors' part on its orbitals
                    Y = Ys[blocks[i]["orbs"]]
                    if Y.dtype != bases[i].dtype and not np.iscomplexobj(bases[i]):
                        bases[i] = bases[i].astype(Y.dtype)
                    for _ in range(2):
                        Y = Y - bases[i] @ (bases[i].conj().T @ Y)
                    uy, sy, _ = np.linalg.svd(Y, full_matrices=False)
                    nadd = int((sy > 1e-3).sum())
                    bases[i] = np.hstack([bases[i], uy[:, :nadd]])
                    lead_added[s][i] = nadd
            log(f"  slab {s} lead clean: basis on the cell {Phi.shape[1]} of {X.shape[1]} (singular values of the "
                f"restriction > 1e-1 / 1e-2 / 1e-3: {[int((sv > x).sum()) for x in (1e-1, 1e-2, 1e-3)]}; unmatched cell atoms "
                f"{mi['missing']}, slab atoms without a cell atom {mi['extra']}), spurious lead bands in "
                f"[{w1:.3f}, {w2:.3f}] {st0['spurious']} -> {st1['spurious']} (p95 {st1['p95_meV']:.3f} meV), "
                f"+{sum(lead_added[s].values())} columns {lead_added[s]}")
    Us, own = mb.slab_U(dev, blocks, bases)
rtag = "" if not a.res_enrich else f"_res{a.res_enrich}{a.res_corr}" + (
    "" if a.res_dE == "0.0125:0.5125:0.025" else f"_rdE{a.res_dE.replace(',', '_').replace(':', '_')}") + (
    "" if a.res_tol == 1e-2 else f"_rt{a.res_tol:g}") + ("" if a.res_add == 40 else f"_ra{a.res_add}") + (
    "" if a.res_rhs == "inject" else f"_rnd{a.res_probes}") + ("" if not a.res_contour else f"_rc{a.res_contour}") + (
    "_int" if a.res_integrate else "")
if a.res_enrich:
    if ":" in a.res_dE:
        lo_, hi_, st_ = (float(x) for x in a.res_dE.split(":"))
        rdE = np.arange(lo_, hi_ + 0.5 * st_, st_)
    else:
        rdE = np.array([float(x) for x in a.res_dE.split(",")])
    for it in range(a.res_enrich):
        f = os.path.join(out, f"resenr{etag}{ltag}{rtag.replace(f'_res{a.res_enrich}', '_res')}_round{it}.npz")
        if os.path.exists(f):
            z = np.load(f)
            assert np.array_equal(z["N"], [u.shape[1] for u in Us]), f"{f}: built on another basis"
            added = [z[f"a{s_}"] for s_ in range(len(Us))]
        else:
            if a.res_contour:                                # Gauss-Legendre points of the level-2 arc
                m1, m2 = (Ec + float(x) for x in a.mu.split(","))
                th_ = 0.5 * np.pi * (np.polynomial.legendre.leggauss(a.res_contour)[0] + 1)
                Etr = 0.5 * (m1 + m2) + 0.5 * (m2 - m1) * np.exp(1j * th_)
                wq_ = 0.5 * np.pi * np.polynomial.legendre.leggauss(a.res_contour)[1]
                dz_ = -1j * 0.5 * (m2 - m1) * np.exp(1j * th_) * wq_
            else:
                Etr = Ec + rdE + 1e-6j
            added, st = mb.residual_enrich(dev, Us, Etr, dev.sigma, a.res_corr, tol=a.res_tol, nadd=a.res_add,
                                           inj_tol=a.inj_tol, rhs=a.res_rhs, nprobe=a.res_probes,
                                           rng=np.random.default_rng(it),
                                           weights=dz_ if a.res_integrate else None)
            np.savez(f, N=[u.shape[1] for u in Us], st=json.dumps(st), **{f"a{s_}": q for s_, q in enumerate(added)})
            log(f"  residual round {it}: |e|/|psi| median {np.median(st['err']):.2e} max {np.max(st['err']):.2e} "
                f"over {len(st['E'])} energies; leading sv per slab {[round(v[0], 3) if v else 0 for v in st['sv']]}; "
                f"{st['t']:.0f} s")
        Us = [sp.csr_matrix(sp.hstack([u, sp.csr_matrix(q)])) for u, q in zip(Us, added)]
        log(f"  residual round {it}: +{[q.shape[1] for q in added]} -> N1 {[u.shape[1] for u in Us]}")
        if not any(q.shape[1] for q in added):
            break
etag += rtag
N1 = [u.shape[1] for u in Us]
blk1 = mb.slab_blocks(dev, Us)
PL = Us[0][dev.bL].toarray()
PR = Us[-1][dev.bR - dev.sb[-2]].toarray()
log(f"level 1: N1 = {sum(N1)} {N1} ({time.time() - t:.0f} s for tiles + assembly)")

# ------------------------------------------------------------------ level 2
mu = [Ec + float(x) for x in a.mu.split(",")]
frho = os.path.join(out, f"rho{etag}{ltag}_{mu[0]:.4f}_{mu[1]:.4f}_{a.npts}.npz")
if os.path.exists(frho) and np.array_equal(np.load(frho)["N1"], N1):
    rho = [np.load(frho)[f"r{s}"] for s in range(len(Us))]
else:
    t = time.time()
    rho = mb.rho_contour(dev, blk1, PL, PR, mu[0], mu[1], a.npts)
    np.savez(frho, N1=N1, **{f"r{s}": r for s, r in enumerate(rho)})
    log(f"contour rho [{mu[0]:.4f}, {mu[1]:.4f}] eV, {a.npts} pts: {time.time() - t:.0f} s; "
        f"trace {np.round([np.trace(r).real for r in rho], 2)}")
V = mb.level2_basis(rho, a.l2_thr, S=blk1["SD"] if a.l2_metric == "S" else None)
if a.lead_clean and not a.lead_l2_drop:
    # the RMS vectors have little occupation in the window, so the rho threshold drops them and the spurious
    # lead bands come back: keep them (the last columns of the end-slab level-1 basis) in level 2
    for s_, per_tile in lead_added.items():
        cols, c0 = [], 0                                     # slab_U column order: the slab's tiles in block order
        for i in [i for i, b in enumerate(blocks) if b["slab"] == s_]:
            c0 += bases[i].shape[1]
            cols += list(range(c0 - per_tile.get(i, 0), c0))
        if cols:
            E_ = np.zeros((V[s_].shape[0], len(cols)), V[s_].dtype)
            E_[cols, np.arange(len(cols))] = 1.0
            for _ in range(2):
                E_ = E_ - V[s_] @ (V[s_].conj().T @ E_)
            ue, se, _ = np.linalg.svd(E_, full_matrices=False)
            V[s_] = np.hstack([V[s_], ue[:, se > 1e-3]])
if a.lead_clean and a.lead_clean_l2:
    for s_ in sorted({0, len(Us) - 1}):
        U1d = Us[s_].toarray()
        cmap = slab_cell_map(inp, dev, s_, tol=a.map_tol)
        Qv, _ = np.linalg.qr(V[s_])
        u, sv, _ = np.linalg.svd(to_cell(U1d @ Qv, cmap), full_matrices=False)
        Phi = u[:, sv > 1e-6 * sv[0]]
        Kc = np.linalg.svd(Qv, full_matrices=True)[0][:, Qv.shape[1]:]          # level-1 complement of V
        M = to_cell(U1d @ Kc, cmap)
        for _ in range(2):
            M = M - Phi @ (Phi.conj().T @ M)
        Wm, sg, Zh = np.linalg.svd(M, full_matrices=False)
        keep = sg > 1e-3 * sg[0]
        rms = RMS(cell, Phi, ks, (w1, w2), Xi=Wm[:, keep])
        st0 = band_stats([rms.eig(q)[0] for q in ks], Eex, w1, w2, a.lead_tol)
        frms = os.path.join(out, f"rms_l2_slab{s_}{etag}{ltag}" + ("" if a.l2_thr == 1e-5 else f"_thr{a.l2_thr:g}") + ".npz")
        if os.path.exists(frms):
            z = np.load(frms)
            assert z["Xi"].shape == rms.Xi.shape, f"{frms}: built on another basis"
            rms.Cs = z["C"]
        else:
            tt = time.time()
            ki = np.arange(0, a.lead_nk, a.lead_kstep)
            rms.ks = ks[ki]
            hist = rms.run(Eex[ki], tol=a.lead_tol, maxit=a.lead_maxit)
            rms.ks = ks
            np.savez(frms, Xi=rms.Xi, C=rms.Cs, hist=json.dumps(hist))
            log(f"  slab {s_} level-2 lead RMS: {len(hist)} steps, {time.time() - tt:.0f} s")
        st1 = band_stats([rms.eig(q)[0] for q in ks], Eex, w1, w2, a.lead_tol)
        nadd = 0
        if rms.Cs.shape[1]:
            Y = Kc @ ((Zh.conj().T[:, keep] / sg[keep]) @ rms.Cs)    # level-1 coefficients of the RMS vectors
            for _ in range(2):
                Y = Y - Qv @ (Qv.conj().T @ Y)
            uy, sy, _ = np.linalg.svd(Y / np.linalg.norm(Y, axis=0), full_matrices=False)
            nadd = int((sy > 1e-3).sum())
            V[s_] = np.hstack([V[s_], uy[:, :nadd].astype(np.result_type(V[s_], uy))])
        log(f"  slab {s_} level-2 lead clean: N2 {Qv.shape[1]}, {int(keep.sum())} candidate directions from level 1, "
            f"spurious lead bands {st0['spurious']} -> {st1['spurious']} (p95 {st1['p95_meV']:.3f} meV), +{nadd} columns")
N2 = [v.shape[1] for v in V]
blk2 = mb.project(blk1, V)
PL2, PR2 = PL @ V[0], PR @ V[-1]
U1 = sp.block_diag(Us, format="csr")
U2 = sp.block_diag([sp.csr_matrix(Us[s] @ V[s]) for s in range(len(Us))], format="csr")
log(f"level 2 (thr {a.l2_thr:g}, {a.l2_metric} metric): N2 = {sum(N2)} {N2}; setup {time.time() - t0:.0f} s")
if a.save_basis:
    os.makedirs(os.path.dirname(a.save_basis) or ".", exist_ok=True)
    for s_ in range(len(Us)):
        np.save(f"{a.save_basis}_U1_slab{s_}.npy", Us[s_].toarray())
        np.save(f"{a.save_basis}_V_slab{s_}.npy", V[s_])
    with open(f"{a.save_basis}_partition.json", "w") as f:       # tile partition, for tile-aligned level-1 windows
        json.dump(dict(tile_orbs=a.tile_orbs, xsplit=a.tile_xsplit, cores=not a.no_cores, d_core=a.d_core, cut=a.tile_cut), f)
    log(f"basis -> {a.save_basis}_{{U1,V}}_slab<s>.npy, _partition.json")
if a.basis_only:
    sys.exit(0)
if a.lead_clean:
    for s_ in clean_slabs:                                  # do the cleaned lead bands survive level 2?
        cmap = slab_cell_map(inp, dev, s_, tol=a.map_tol)
        u, sv, _ = np.linalg.svd(to_cell(Us[s_] @ V[s_], cmap), full_matrices=False)
        r2 = RMS(cell, u[:, sv > 1e-6 * sv[0]], ks, (w1, w2))
        st = band_stats([r2.eig(q)[0] for q in ks], Eex, w1, w2, a.lead_tol)
        log(f"  slab {s_} level 2 on the lead cell: {st['spurious']} spurious lead bands (p95 {st['p95_meV']:.3f} meV)")


# ------------------------------------------------------------------ transmission
def sigma(E):
    return dev.sigma(E + 1e-6j)


def T_of(psi, GR):
    pr = psi[dev.bR]
    return float(np.real(np.einsum("ic,ij,jc->", pr.conj(), GR, pr)))


fres = os.path.join(out, "results" + etag + ltag + ("_l2c" if a.lead_clean and a.lead_clean_l2 else "") + ("" if a.l2_metric == "euclid" else f"_{a.l2_metric}") +
                    ("" if a.l2_thr == 1e-5 else f"_thr{a.l2_thr:g}") +
                    ("" if a.mu == "0,0.395" else f"_mu{a.mu.replace(',', '_')}") + ".json")
res = json.load(open(fres)) if os.path.exists(fres) else {}
res.update(N1=N1, N2=N2, n=dev.n, slabs=dev.ns, cores=not a.no_cores, enrich=a.enrich and dict(
    rounds=a.enrich, mode=a.enrich_mode, width=a.enrich_width, tol=a.enrich_tol, win=a.enrich_win, rthr=a.enrich_rthr, smin=a.enrich_smin))
ex = {}
EcT = Ec if a.Ec is None else a.Ec        # test energies E = EcT + dE
if a.exact_from:
    z = json.load(open(a.exact_from))
    if abs(z.get("Ec", EcT) - EcT) > 1e-9:
        raise SystemExit(f"{a.exact_from}: written with test edge {z['Ec']}, this run uses {EcT} (--Ec)")
    ex = z["E"]
res.setdefault("E", {})
Elist = [float(x) for x in a.E.split(",")] if a.E else [EcT + float(x) for x in a.dE.split(",")]
for E in Elist:
    dE = E - EcT
    t = time.time()
    SL, SR = sigma(E)
    ts = time.time() - t
    GR = 1j * (SR - SR.conj().T)
    B, (_, lam, X) = dev.inject(SL, a.inj_tol), mb.injection(SL, a.inj_tol)
    nc = B.shape[1]
    r = dict(dE=dE, E=E, channels=nc, t_sigma=ts)
    if nc == 0:
        log(f"E - Ec = {dE:+.3f}: no propagating channel")
        res["E"][f"{dE:.4f}"] = r
        continue
    b = np.zeros((dev.n, nc), complex)
    b[dev.bL] = B
    A = dev.A(E, SL, SR)
    sel = np.arange(nc) if a.mg_cols <= 0 else np.arange(nc)[::-1][:a.mg_cols]   # strongest channels last in B
    exk = ex.get(f"{dE:.4f}")
    if exk is not None and len(sel) == nc:
        assert exk["channels"] == nc, f"--exact-from: {exk['channels']} channels at {dE:.4f}, here {nc}"
        r.update(T_exact=exk["T_exact"], exact_from=a.exact_from)
        Te = Te_sel = r["T_exact"]
        log(f"E = {E:.4f} (Ec {dE:+.3f}): {nc} channels, T_exact {Te:.5f} (from {a.exact_from}); sigma {ts:.0f} s")
    else:
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
        log(f"E = {E:.4f} (Ec {dE:+.3f}): {nc} channels, T_exact {Te:.5f} (all {r['cols_all']} Gamma_L cols "
            f"{T_full:.5f}), residual {r['res_exact']:.1e}; sigma {ts:.0f} s, exact {r['t_exact']:.0f} s")
        Te_sel = T_of(psi[:, sel], GR)
    rel = lambda x, ref=Te: abs(x - ref) / max(ref, 1e-12)
    r.update(mg_cols=len(sel), T_exact_mgcols=Te_sel)
    levels = (("1", U1, blk1, PL, PR), ("2", U2, blk2, PL2, PR2))[1 if a.direct_l2 else 0:]   # U1 = I: MS1 exact
    for tag, U, blk, P_L, P_R in levels:
        t = time.time()
        Ac = mb.BlockTridiag(blk, E, P_L.conj().T @ SL @ P_L, P_R.conj().T @ SR @ P_R)
        x_ms = U @ Ac.solve(U.conj().T @ b)
        r[f"T_MS{tag}"] = T_of(x_ms, GR)
        r[f"t_coarse{tag}"] = time.time() - t
        log(f"  MS{tag}: T {r[f'T_MS{tag}']:.5f}, rel err {rel(r[f'T_MS{tag}']):.1e}, {r[f't_coarse{tag}']:.0f} s")
        if tag not in a.mg.split(","):
            continue
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
