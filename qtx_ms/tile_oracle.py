"""Oracle level 1: tile bases from the exact tile density matrix rho_tt = (rho_ss)_tt, the tile block of the exact
contour slab rho of the open device (the direct level 2 eigenpairs of run_l2_clean). This is the best any tile
embedding (CAP, Schur, DMFT / DMET-style self-consistent bath) can do with the same occupation cut, so its MS1 T
error separates "tile embedding is wrong" from "block-diagonal tile basis / occupation criterion is the limit".

    python -m qtx_ms.tile_oracle --inputs inputs/mos2_hfo2_trim_c3 \
        --eig results_ms/mos2_hfo2_trim_c3_l2direct/eig_0.3545_1.1045_24_top800.npz \
        --cap-tiles results_ms/mos2_hfo2_trim_c3_nocore_tc12/tiles \
        --exact-from results_ms/exact/mos2_hfo2_trim_c3.json --Ec 0.5043489720738419

Per occupation cut (absolute, as run_ms --tile-rho-thr) MS1 T vs exact for
  oracle : eigenvectors of the exact rho_tt with occupation > cut
  cap    : the cached CAP tiles (run_ms --tile-contour) with occupation > cut (cuts >= their storage cut)
  slab   : eigenvectors of the exact slab rho with occupation > cut (no tile cut; direct level 2 by threshold)
and, per tile, how much exact occupation the CAP basis misses: lambda_max(Q_perp rho_tt Q_perp).
"""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb

log = mb.log
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--inputs", required=True)
p.add_argument("--eig", required=True, help="slab rho eigenpairs l<s>, X<s> (run_l2_clean eig_*_top<K>.npz)")
p.add_argument("--cap-tiles", help="run_ms tile cache (tile_XXX.npz with U, s, atoms) to compare with")
p.add_argument("--cap-min", type=float, default=1e-6, help="occupation cut the CAP tile cache was stored with")
p.add_argument("--thr", default="1e-3,1e-4,1e-5,1e-6,1e-7,1e-8", help="absolute occupation cuts")
p.add_argument("--tile-orbs", type=int, default=2500)
p.add_argument("--dE", default="0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
p.add_argument("--Ec", type=float, help="reference edge for --dE and --exact-from (default Ec_lead)")
p.add_argument("--exact-from", required=True)
p.add_argument("--inj-tol", type=float, default=1e-4)
p.add_argument("--obc", default="folded", choices=["auto", "qttools", "folded"])
p.add_argument("--out", default="results_ms/tile_oracle.json")
a = p.parse_args()

t0 = time.time()
inp = read_inputs(a.inputs)
dev = mb.Device(inp, obc=a.obc)
dev.sigma_cache = os.path.join("results_ms", "sigma", os.path.basename(os.path.normpath(a.inputs)))
blocks, _, _ = mb.partition(dev, cores=False, tile_orbs=a.tile_orbs)
nsl = len(dev.ns)
log(f"device {dev.n} orbitals, slabs {dev.ns}; {len(blocks)} tiles, per slab "
    f"{[sum(b['slab'] == s for b in blocks) for s in range(nsl)]}")
z = np.load(a.eig)
eig = [(z[f"l{s}"], z[f"X{s}"]) for s in range(nsl)]
assert [X.shape[0] for _, X in eig] == dev.ns, f"{a.eig}: slab sizes {[X.shape[0] for _, X in eig]}"
log(f"slab rho eigenpairs from {a.eig}: top {eig[0][0].size}, smallest kept {[f'{l[-1]:.0e}' for l, _ in eig]}")

# exact tile rho and its eigenpairs (descending)
orc = []
for b in blocks:
    l, X = eig[b["slab"]]
    Xt = X[b["orbs"]]
    R = (Xt * l) @ Xt.conj().T
    lam, Q = np.linalg.eigh(0.5 * (R + R.conj().T))
    orc.append((lam[::-1], Q[:, ::-1], R))
cap = None
if a.cap_tiles:
    cap = []
    for i, b in enumerate(blocks):
        zt = np.load(os.path.join(a.cap_tiles, f"tile_{i:03d}.npz"))
        assert np.array_equal(zt["atoms"], b["atoms"]), f"{a.cap_tiles}: tile {i} built on another partition"
        cap.append((zt["s"][:zt["U"].shape[1]], zt["U"]))

thrs = [float(x) for x in a.thr.split(",")]
res = dict(args=vars(a), tiles=[dict(slab=int(b["slab"]), norb=len(b["orbs"])) for b in blocks], cut={})
for t in thrs:
    r = res["cut"][f"{t:g}"] = {}
    r["N_oracle"] = [int(np.sum(o[0] > t)) for o in orc]
    r["N_slab"] = [int(np.sum(l > t)) for l, _ in eig]
    if cap is not None and t >= a.cap_min:
        r["N_cap"] = [int(np.sum(s > t)) for s, _ in cap]
        miss = []
        for (s, U), (lam, Q, R) in zip(cap, orc):
            Uk = U[:, s > t]
            Pp = np.eye(R.shape[0]) - Uk @ Uk.conj().T
            miss.append(float(np.linalg.eigvalsh(Pp @ R @ Pp)[-1]))
        r["cap_missed_occ"] = miss
    log(f"cut {t:g}: N oracle {sum(r['N_oracle'])} {r['N_oracle']}, slab {sum(r['N_slab'])}"
        + ("" if "N_cap" not in r else f", cap {sum(r['N_cap'])}; exact occupation missed by cap per tile "
           f"{[f'{m:.0e}' for m in r['cap_missed_occ']]}"))


def bases_for(kind, t):
    if kind == "oracle":
        return mb.slab_U(dev, blocks, [Q[:, lam > t] for lam, Q, _ in orc])[0]
    if kind == "cap":
        return mb.slab_U(dev, blocks, [U[:, s > t] for s, U in cap])[0]
    return [sp.csr_matrix(X[:, l > t]) for l, X in eig]


systems = []
for t in thrs:
    for kind in ("oracle", "cap", "slab"):
        if kind == "cap" and (cap is None or t < a.cap_min):
            continue
        Us = bases_for(kind, t)
        blk = mb.slab_blocks(dev, Us)
        systems.append(dict(kind=kind, cut=t, N=int(sum(u.shape[1] for u in Us)), U=sp.block_diag(Us, format="csr"),
                            blk=blk, PL=Us[0][dev.bL].toarray(), PR=Us[-1][dev.bR - dev.sb[-2]].toarray()))
log(f"{len(systems)} level-1 systems assembled, {time.time() - t0:.0f} s")

Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
ex = json.load(open(a.exact_from))
if abs(ex.get("Ec", Ec) - Ec) > 1e-9:
    raise SystemExit(f"{a.exact_from}: written with test edge {ex['Ec']}, this run uses {Ec} (--Ec)")
res["E"] = {}
for dE in (float(x) for x in a.dE.split(",")):
    E = Ec + dE
    exk = ex["E"].get(f"{dE:.4f}")
    if exk is None:
        log(f"dE {dE:.4f}: not in {a.exact_from}, skipped")
        continue
    SL, SR = dev.sigma(E + 1e-6j)
    GR = 1j * (SR - SR.conj().T)
    B = dev.inject(SL, a.inj_tol)
    assert B.shape[1] == exk["channels"], f"{B.shape[1]} channels at {dE:.4f}, exact {exk['channels']}"
    b = np.zeros((dev.n, B.shape[1]), complex)
    b[dev.bL] = B
    Te = exk["T_exact"]
    rE = res["E"][f"{dE:.4f}"] = dict(T_exact=Te, channels=B.shape[1])
    for sy in systems:
        Ac = mb.BlockTridiag(sy["blk"], E, sy["PL"].conj().T @ SL @ sy["PL"], sy["PR"].conj().T @ SR @ sy["PR"])
        pr = (sy["U"] @ Ac.solve(sy["U"].conj().T @ b))[dev.bR]
        T = float(np.real(np.einsum("ic,ij,jc->", pr.conj(), GR, pr)))
        rE[f"{sy['kind']}_{sy['cut']:g}"] = T
    log(f"dE {dE:.3f}: T_exact {Te:.5f}; rel err " + ", ".join(
        f"{sy['kind']} {sy['cut']:g} (N {sy['N']}) {abs(rE[sy['kind'] + '_' + format(sy['cut'], 'g')] - Te) / Te:.1e}"
        for sy in systems))
    json.dump(res, open(a.out, "w"), indent=1)

log("summary: relative T error over the energies (median / max), N1")
for sy in systems:
    k = f"{sy['kind']}_{sy['cut']:g}"
    e = [abs(v[k] - v["T_exact"]) / v["T_exact"] for v in res["E"].values()]
    res["cut"][f"{sy['cut']:g}"][f"err_{sy['kind']}"] = dict(median=float(np.median(e)), max=float(np.max(e)), N=sy["N"])
    log(f"  {sy['kind']:6s} cut {sy['cut']:7.0e}  N1 {sy['N']:5d}  median {np.median(e):.1e}  max {np.max(e):.1e}")
json.dump(res, open(a.out, "w"), indent=1)
log(f"done, {time.time() - t0:.0f} s -> {a.out}")
