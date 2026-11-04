"""Accelerated tile local rho (msbasis.tile_rho_contour: Schur G_PP, low-precision environment, smaller embeddings
away from the axis, contour points in parallel) vs a cached reference tile_obc file (full-embedding double LU).
python -m qtx_ms.tile_fast_bench --inputs inputs/mos2_hfo2 --tile-orbs 100000 --tile 0
    --ref results_ms/mos2_hfo2_slabl1_tobc24/tiles_obc/tile_obc_000.npz --variants schur_d,schur_s1,schur_s0,far1
Metrics: wall time, |rho - rho_ref|_2 (rho_ref from the cached U, occupations > 1e-8), max occupation change,
subspace distance |(1 - U_ref U_ref^T) U| at the cuts 1e-5 / 1e-6 / 1e-7, kept counts."""
import argparse, json, os, time
import numpy as np
from qtx_ms.prep.inputs import read_inputs
import qtx_ms.msbasis as mb

VARIANTS = {
    "full":     dict(gpp="full"),
    "schur_d":  dict(gpp="schur", env_prec="double"),
    "schur_s0": dict(gpp="schur", env_prec="single", refine=0),
    "schur_s1": dict(gpp="schur", env_prec="single", refine=1),
    "far1":     dict(gpp="schur", env_prec="single", refine=1, emb_far=[(0.1, 8.0), (0.2, 6.0)]),
    "far2":     dict(gpp="schur", env_prec="single", refine=1, emb_far=[(0.05, 8.0), (0.15, 6.0), (0.3, 4.0)]),
    "far3":     dict(gpp="schur", env_prec="single", refine=1, emb_far=[(0.03, 6.0)]),
}
p = argparse.ArgumentParser()
p.add_argument("--inputs", required=True)
p.add_argument("--tile-orbs", type=int, default=2500)
p.add_argument("--tile", type=int, default=0)
p.add_argument("--ref", required=True)
p.add_argument("--mu", default="-0.15,0.6")
p.add_argument("--npts", type=int, default=24)
p.add_argument("--variants", default="schur_d,schur_s1")
p.add_argument("--workers", type=int, default=2)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--out", default="results_ms/tile_fast")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
inp = read_inputs(a.inputs)
dev = mb.Device(inp, obc="folded")
dev.sigma_cache = os.path.join("results_ms", "sigma", os.path.basename(os.path.normpath(a.inputs)))
Ec = inp["meta"]["edges"]["Ec_lead"]
blocks, _, _ = mb.partition(dev, cores=False, tile_orbs=a.tile_orbs)
b = blocks[a.tile]
geo = mb.Geometry(inp["xyz"], inp["meta"].get("periodic"))
tree = geo.tree()
z = np.load(a.ref)
assert np.array_equal(z["atoms"], b["atoms"]), "reference built on another partition"
Ur, sr = z["U"], z["s"]
rho_ref = (Ur * sr[:Ur.shape[1]]) @ Ur.T
mu = [Ec + float(x) for x in a.mu.split(",")]
print(f"tile {a.tile} (slab {b['slab']}): {len(b['orbs'])} orb; reference {a.ref}: {Ur.shape[1]} cols", flush=True)
res = {}
for v in a.variants.split(","):
    opt = dict(VARIANTS[v], workers=a.workers, threads=a.threads)
    acc, info = mb.tile_rho_contour(inp, geo, tree, b["atoms"], (mu[0], mu[1], a.npts), obc=dev, **opt)
    rho = -acc.imag / np.pi
    rho = 0.5 * (rho + rho.T)
    lam, X = np.linalg.eigh(rho)
    lam, X = lam[::-1], X[:, ::-1]
    r = dict(t=info["t"], n_emb=info["n_emb"], drho=float(np.linalg.norm(rho - rho_ref, 2)),
             docc=float(np.abs(lam[:200] - sr[:200]).max()))
    for cut in (1e-5, 1e-6, 1e-7):
        k, kr = int((lam > cut).sum()), int((sr > cut).sum())
        U, Uref = X[:, :k], Ur[:, :kr]
        r[f"{cut:g}"] = dict(k=k, k_ref=kr, dist=float(np.linalg.norm(U - Uref @ (Uref.T @ U), 2)) if k else 0.0)
    res[v] = r
    print(v, json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(a.out, f"bench_tile{a.tile}_{a.tile_orbs}.json"), "w"), indent=1)
