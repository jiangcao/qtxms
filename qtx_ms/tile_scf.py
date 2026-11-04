"""Self-consistent tile embedding (DMFT / DMET-style bath from the mode-space system).

    python -m qtx_ms.tile_scf --inputs inputs/mos2_hfo2_trim_c3 --cap-tiles results_ms/mos2_hfo2_trim_c3_nocore_tc12/tiles \
        --exact-from results_ms/exact/mos2_hfo2_trim_c3.json --Ec 0.5043489720738419 \
        [--oracle results_ms/mos2_hfo2_trim_c3_l2direct/eig_0.3545_1.1045_24_top800.npz]

Iteration 0 (level 1): tiles whose 12 A embedding reaches a lead face are rebuilt with the exact lead Sigma in the
embedding (tile_modes(obc=dev): lead-block atoms dropped, -Sigma on the embedded face orbitals, CAP on the other
cuts); the interior tiles keep their cached CAP bases. Occupations > --rho-thr.
Each iteration k, on the level-2 arc (--mu, --npts; the same points as rho_contour):
  coarse   : A1(z) = U1^H A(z) U1 (level 1, exact lead Sigma projected), RGF over the slabs: G1 blocks |s - s'| <= 2
  level 2  : rho2_ss = -(1/pi) Im int G1_ss dz  ->  V (level2_basis, --l2-thr)
  tiles    : fine tile t + environment in level 1 (all other tiles, leads exact):
             G_tt = (A_tt - Delta_t)^-1,  Delta_t = A_tE U_E G1^(-t)_EE U_E^H A_Et,
             G1^(-t) = G1_EE - G1_Et G1_tt^-1 G1_tE (level-1 system with tile t removed; exact Schur identity)
             rho_tt = -(1/pi) Im int G_tt dz  ->  new U1_t (occupations > --rho-thr)
  checks   : MS1 / MS2 T vs exact at --dE, subspace change per tile lambda_max((1 - P_old) rho_new (1 - P_old)),
             and with --oracle the exact occupation each tile basis misses, lambda_max(Q_perp rho_exact,tt Q_perp).
The fixed point is exact when span(U1_E) holds the environment response A_EE^-1 A_Et near tile t.
"""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb

log = mb.log
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--inputs", required=True)
p.add_argument("--cap-tiles", required=True, help="run_ms tile cache of the CAP tiles (interior tiles)")
p.add_argument("--out", help="output folder (default results_ms/<inputs name>_tilescf)")
p.add_argument("--tile-orbs", type=int, default=2500)
p.add_argument("--d-emb", type=float, default=12.0, help="tile embedding radius [A] (as the CAP tiles)")
p.add_argument("--rho-thr", type=float, default=1e-6, help="tile occupations kept (absolute)")
p.add_argument("--mu", default="-0.15,0.6", help="window - Ec_lead [eV] (level 2 and tiles)")
p.add_argument("--npts", type=int, default=24)
p.add_argument("--l2-thr", type=float, default=1e-5)
p.add_argument("--iters", type=int, default=6)
p.add_argument("--tol", type=float, default=0.1, help="stop when every tile's subspace change < tol * rho-thr")
p.add_argument("--dE", default="0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
p.add_argument("--Ec", type=float, help="reference edge for --dE and --exact-from (default Ec_lead)")
p.add_argument("--exact-from", required=True)
p.add_argument("--oracle", help="exact slab rho eigenpairs (run_l2_clean eig_*.npz) for the missed-occupation check")
p.add_argument("--inj-tol", type=float, default=1e-4)
p.add_argument("--obc", default="folded", choices=["auto", "qttools", "folded"])
a = p.parse_args()

t0 = time.time()
name = os.path.basename(os.path.normpath(a.inputs))
out = a.out or os.path.join("results_ms", f"{name}_tilescf")
os.makedirs(out, exist_ok=True)
inp = read_inputs(a.inputs)
dev = mb.Device(inp, obc=a.obc)
dev.sigma_cache = os.path.join("results_ms", "sigma", name)
nsl = len(dev.ns)
EcL = inp["meta"]["edges"]["Ec_lead"]
mu1, mu2 = (EcL + float(x) for x in a.mu.split(","))
blocks, _, _ = mb.partition(dev, cores=False, tile_orbs=a.tile_orbs)
nt = len(blocks)
tdev = [dev.sb[b["slab"]] + b["orbs"] for b in blocks]                 # device orbitals of each tile
geo = mb.Geometry(inp["xyz"], inp["meta"].get("periodic"))
tree = geo.tree()

# ------------------------------------------------------------------ iteration 0: OBC tiles + CAP tiles
off = np.concatenate([[0], np.cumsum(inp["norb"])])
face_atoms = np.unique(np.searchsorted(off, dev.dev[np.concatenate([dev.bL, dev.bR])], side="right") - 1)
bases, kind = [], []
for i, b in enumerate(blocks):
    near = tree.query_ball_point(geo.points(b["atoms"]), a.d_emb)
    emb = np.unique(np.concatenate([np.asarray(q, int) for q in near]))
    if np.isin(emb, face_atoms).any():
        f = os.path.join(out, f"tile_obc_{i:03d}.npz")
        if not os.path.exists(f):
            t = time.time()
            Q, occ, nemb = mb.tile_modes(inp, geo, tree, b["atoms"], None, d_emb=a.d_emb, contour=(mu1, mu2, a.npts),
                                         rho_thr=min(a.rho_thr, 1e-8), obc=dev)
            np.savez(f, U=Q, s=occ, atoms=b["atoms"])
            log(f"  OBC tile {i} (slab {b['slab']}): {len(b['orbs'])} orb, embedding {nemb} atoms -> {Q.shape[1]} "
                f"modes > {min(a.rho_thr, 1e-8):g}, {time.time() - t:.0f} s")
        kind.append("obc")
    else:
        f = os.path.join(a.cap_tiles, f"tile_{i:03d}.npz")
        kind.append("cap")
    z = np.load(f)
    assert np.array_equal(z["atoms"], b["atoms"]), f"{f}: built on another partition"
    bases.append(z["U"][:, :int(np.sum(z["s"][:z["U"].shape[1]] > a.rho_thr))])
log(f"{name}: {nt} tiles ({kind.count('obc')} with the lead Sigma: "
    f"{[i for i, k in enumerate(kind) if k == 'obc']}), cut {a.rho_thr:g}; N1 per tile {[q.shape[1] for q in bases]}")

orc = None
if a.oracle:
    z = np.load(a.oracle)
    orc = []
    for b in blocks:
        Xt = z[f"X{b['slab']}"][b["orbs"]]
        orc.append((Xt * z[f"l{b['slab']}"]) @ Xt.conj().T)


def missed(R, Q):
    """Largest occupation of R outside span(Q)."""
    Y = R - Q @ (Q.conj().T @ R)
    Y = Y - (Y @ Q) @ Q.conj().T
    return float(np.linalg.eigvalsh(0.5 * (Y + Y.conj().T))[-1])


# ------------------------------------------------------------------ test energies (Sigma, injection) once
EcT = EcL if a.Ec is None else a.Ec
ex = json.load(open(a.exact_from))
if abs(ex.get("Ec", EcT) - EcT) > 1e-9:
    raise SystemExit(f"{a.exact_from}: written with test edge {ex['Ec']}, this run uses {EcT} (--Ec)")
tests = []
for dE in (float(x) for x in a.dE.split(",")):
    exk = ex["E"].get(f"{dE:.4f}")
    if exk is None:
        continue
    SL, SR = dev.sigma(EcT + dE + 1e-6j)
    B = dev.inject(SL, a.inj_tol)
    assert B.shape[1] == exk["channels"], f"{B.shape[1]} channels at {dE:.4f}, exact {exk['channels']}"
    tests.append(dict(dE=dE, E=EcT + dE, SL=SL, SR=SR, B=B, GR=1j * (SR - SR.conj().T), Te=exk["T_exact"]))


def ms_T(U, blk, PL, PR):
    """MS T relative errors at the test energies (U: slab bases, block-diagonal)."""
    err = []
    for t in tests:
        b = np.zeros((dev.n, t["B"].shape[1]), complex)
        b[dev.bL] = t["B"]
        Ac = mb.BlockTridiag(blk, t["E"], PL.conj().T @ t["SL"] @ PL, PR.conj().T @ t["SR"] @ PR)
        pr = (U @ Ac.solve(U.conj().T @ b))[dev.bR]
        T = float(np.real(np.einsum("ic,ij,jc->", pr.conj(), t["GR"], pr)))
        err.append(abs(T - t["Te"]) / t["Te"])
    return np.array(err)


# ------------------------------------------------------------------ iterations
c, rr = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
xg, wg = np.polynomial.legendre.leggauss(a.npts)
th, wg = 0.5 * np.pi * (xg + 1), 0.5 * np.pi * wg
hist = []
for it in range(a.iters + 1):
    ti = time.time()
    Us, own = mb.slab_U(dev, blocks, bases)
    blk = mb.slab_blocks(dev, Us)
    U1 = sp.block_diag(Us, format="csr")
    PL, PR = Us[0][dev.bL].toarray(), Us[-1][dev.bR - dev.sb[-2]].toarray()
    N1 = [u.shape[1] for u in Us]
    cb = np.concatenate([[0], np.cumsum(N1)])
    tc = [np.where(own == i)[0] for i in range(nt)]                  # level-1 columns of each tile
    # neighbours of tile t: tiles in slabs s-1 .. s+1 other than t (A is block tridiagonal over the slabs);
    # G1 is needed on the window of slabs s-1 .. s+1 only (RGF blocks up to |s - s'| = 2)
    nb = [[j for j, bj in enumerate(blocks) if j != i and abs(bj["slab"] - b["slab"]) <= 1] for i, b in enumerate(blocks)]
    ni_ = [np.concatenate([tdev[j] for j in nb[i]]) for i in range(nt)]
    cn_ = [np.concatenate([tc[j] for j in nb[i]]) for i in range(nt)]
    win = [list(range(max(b["slab"] - 1, 0), min(b["slab"] + 1, nsl - 1) + 1)) for b in blocks]
    ci_ = [tc[i] - cb[win[i][0]] for i in range(nt)]                 # tile / neighbour columns inside the window
    cw_ = [cn_[i] - cb[win[i][0]] for i in range(nt)]
    UN_ = [U1[ni_[i]][:, cn_[i]].toarray() for i in range(nt)]
    acc2 = [np.zeros((n, n), complex) for n in N1]
    acct = [np.zeros((len(o), len(o)), complex) for o in tdev]
    for ip, (t_, w_) in enumerate(zip(th, wg)):
        z = c + rr * np.exp(1j * t_)
        dz = -1j * rr * np.exp(1j * t_) * w_
        SL, SR = dev.sigma(z)
        A = dev.A(z, SL, SR)
        G1 = mb.rgf_blocks(blk, z, PL.conj().T @ SL @ PL, PR.conj().T @ SR @ PR, width=2)
        for s in range(nsl):
            acc2[s] += G1[(s, s)] * dz
        Gw = {}
        for i in range(nt):
            w = tuple(win[i])
            if w not in Gw:
                Gw[w] = np.block([[G1[(p_, q_)] for q_ in w] for p_ in w])
            G, ci, cn = Gw[w], ci_[i], cw_[i]
            ni, UN = ni_[i], UN_[i]
            Gc = G[np.ix_(cn, cn)] - G[np.ix_(cn, ci)] @ np.linalg.solve(G[np.ix_(ci, ci)], G[np.ix_(ci, cn)])
            C = A[tdev[i]][:, ni] @ UN                                      # A_tE U_E
            Ct = UN.conj().T @ A[ni][:, tdev[i]]                            # U_E^H A_Et
            Att = A[tdev[i]][:, tdev[i]].toarray()
            acct[i] += np.linalg.inv(Att - C @ Gc @ Ct) * dz
        if ip % 6 == 5:
            log(f"  iteration {it}: contour point {ip + 1}/{a.npts}, {time.time() - ti:.0f} s")
    herm = lambda X: -((X - X.conj().T) / 2j) / np.pi
    rho2 = [herm(x) for x in acc2]
    V = mb.level2_basis(rho2, a.l2_thr)
    blk2 = mb.project(blk, V)
    U2 = sp.block_diag([sp.csr_matrix(Us[s] @ V[s]) for s in range(nsl)], format="csr")
    e1 = ms_T(U1, blk, PL, PR)
    e2 = ms_T(U2, blk2, PL @ V[0], PR @ V[-1])
    new, chg, occ = [], [], []
    for i in range(nt):
        R = herm(acct[i])
        lam, X = np.linalg.eigh(0.5 * (R + R.conj().T))
        lam, X = lam[::-1], X[:, ::-1]
        new.append(X[:, lam > a.rho_thr])
        chg.append(missed(R, bases[i]))
        occ.append(lam)
    rec = dict(it=it, N1=int(sum(N1)), N1_tiles=[q.shape[1] for q in bases], N2=int(sum(v.shape[1] for v in V)),
               err_MS1=e1.tolist(), err_MS2=e2.tolist(), change=chg, t=time.time() - ti)
    if orc is not None:
        rec["oracle_missed"] = [missed(R, q) for R, q in zip(orc, bases)]
    hist.append(rec)
    log(f"iteration {it}: N1 {rec['N1']} {rec['N1_tiles']}, N2 {rec['N2']}; T err MS1 median {np.median(e1):.1e} "
        f"max {e1.max():.1e}, MS2 median {np.median(e2):.1e} max {e2.max():.1e}; occupation outside the current tile "
        f"basis {[f'{x:.0e}' for x in chg]}" + ("" if orc is None else
        f"; exact occupation missed {[f'{x:.0e}' for x in rec['oracle_missed']]}") + f"; {rec['t']:.0f} s")
    log(f"  MS1 errors {[f'{x:.1e}' for x in e1]}")
    log(f"  MS2 errors {[f'{x:.1e}' for x in e2]}")
    np.savez(os.path.join(out, f"iter{it}_thr{a.rho_thr:g}.npz"), **{f"U{i}": q for i, q in enumerate(bases)},
             **{f"s{i}": o for i, o in enumerate(occ)})
    json.dump(dict(args=vars(a), kind=kind, hist=hist), open(os.path.join(out, f"hist_thr{a.rho_thr:g}.json"), "w"), indent=1)
    if it == a.iters or max(chg) < a.tol * a.rho_thr:
        break
    bases = new
log(f"done, {time.time() - t0:.0f} s -> {out}")
