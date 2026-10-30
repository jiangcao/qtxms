"""Coupled mode space on the rough Si nanosheet: conventional per-slab CMS vs the union-cross-section CMS
(mgms.cms_union), stand-alone mode-space error vs basis size M, optionally as the coarse level of the
two-level solver.

    python -m mgms.run_cms --inputs inputs/si3x5_sp3s [--M 16,32,64,128,256,512] [--mg-M 64,256]

Per energy: exact QTBM solve (sparse direct), and for every basis and M the Galerkin mode-space solution
psi_MS = U A_c^-1 U^T b, A_c = U^T (E - H - Sigma) U (block tridiagonal), T = sum psi_R^H Gamma_R psi_R.
Bases: conv (lowest M closed-slab eigenvectors above mid-gap of each slab), union (one reference slab with
the union channel, projected onto each slab), nominal (reference = flat lead block). Caches in --out.
"""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp

from mgms.prep.inputs import read_inputs
from mgms.prep import si_sheet, model
from mgms import msbasis as mb
from mgms import cms_union as cu
from mgms.twolevel import TwoLevel

log = mb.log
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--inputs", required=True)
p.add_argument("--out", help="cache / results folder (default results_cms/<inputs name>)")
p.add_argument("--M", default="16,32,64,128,256,512", help="modes per slab")
p.add_argument("--bases", default="conv,union,union-zero,nominal",
               help="also: union-bloch, nominal-bloch (union*/nominal*: missing channel Si damped exp(-kappa r), "
                    "union-zero: set to 0)")
p.add_argument("--E", default="1.21,1.24,1.27,1.30,1.35,1.44,1.53", help="absolute energies [eV]")
p.add_argument("--bloch-win", default="-0.05,0.6", help="Bloch-basis energy window - Ec_lead [eV]")
p.add_argument("--kappa", type=float, help="oxide decay constant [1/A] for missing channel Si (default: fit)")
p.add_argument("--nk", type=int, default=12, help="k points in [0, pi] for the Bloch bases")
p.add_argument("--tol", type=float, default=1e-3, help="SVD cut for the projected reference modes")
p.add_argument("--mg-M", default="", help="M values also run as coarse level of the two-level solver")
p.add_argument("--k", type=int, default=10, help="GMRES smoother steps")
p.add_argument("--mg-tol", type=float, default=1e-8)
p.add_argument("--sigma-cache", default=None, help="folder with sig_<E>.npz (default results_ms/<name>/sigma)")
a = p.parse_args()

name = os.path.basename(os.path.normpath(a.inputs))
out = a.out or os.path.join("results_cms", name)
os.makedirs(out, exist_ok=True)
Ms = [int(x) for x in a.M.split(",")]
Mmax = max(Ms)
bases = a.bases.split(",")
t0 = time.time()
inp = read_inputs(a.inputs)
meta = inp["meta"]
st, P = si_sheet.load_structure(a.inputs, meta)
Ec, Emid = meta["edges"]["Ec_lead"], 0.5 * meta["edges"]["Ec_bulk"]
dev = mb.Device(inp)
ns = len(dev.ns)
keys, _ = cu.atom_keys(st.pos, st.species, st.bonds, st.block)
slab_atoms = [np.where(st.block == b)[0] for b in dev.dev_blocks]
for s, at in enumerate(slab_atoms):
    assert np.array_equal(np.concatenate([np.arange(dev.off[i], dev.off[i + 1]) for i in at]), dev.slab_orbs[s])
nch = [int((st.species[at] == cu.SI_C).sum()) for at in slab_atoms]
log(f"{name}: {dev.n} orbitals, slabs {dev.ns}, channel Si per slab {nch}, Emid {Emid:.3f}, Ec_lead {Ec:.4f}")


def cached(f, build):
    f = os.path.join(out, f)
    if os.path.exists(f):
        return dict(np.load(f, allow_pickle=True))
    t = time.time()
    r = build()
    np.savez(f, **r)
    log(f"  built {os.path.basename(f)} ({time.time() - t:.0f} s)")
    return r


# ------------------------------------------------------------------ bases
Hs = [dev.H[dev.sb[s]:dev.sb[s + 1], dev.sb[s]:dev.sb[s + 1]] for s in range(ns)]
Hu = [dev.H[dev.sb[s]:dev.sb[s + 1], dev.sb[s + 1]:dev.sb[s + 2]] for s in range(ns - 1)]
modes = {}                                   # basis -> per slab dense mode matrix (Mmax columns)
info = {}
if "conv" in bases:
    Q = []
    for s in range(ns):
        z = cached(f"conv_slab{s}_M{Mmax}.npz", lambda: dict(zip(("w", "V"), cu.slab_modes(Hs[s], Mmax, Emid))))
        Q.append(z["V"])
        info.setdefault("conv_E", []).append([float(z["w"][0]), float(z["w"][-1])])
    modes["conv"] = Q
    log(f"conv: slab eigenvalues [lowest, M-th] {np.round(info['conv_E'], 3).tolist()}")
refs = {}
depth = None
for kind in ("union", "union-zero", "nominal", "union-bloch", "nominal-bloch"):
    if kind not in bases:
        continue
    rk = kind.split("-")[0]
    if rk not in refs:
        mask = cu.union_mask(st.pos, st.species, st.block, dev.dev_blocks) if rk == "union" else None
        refs[rk] = cu.reference_slab(meta, mask, P, model)
    ref = refs[rk]
    if kind.endswith("bloch"):
        lo, hi = (Ec + float(x) for x in a.bloch_win.split(","))

        def build():
            Q, sv, cnt = cu.bloch_modes(ref, lo, hi, a.nk)
            return dict(V=Q[:, :Mmax], w=sv[:Mmax], cnt=np.array(cnt))
        z = cached(f"{rk}-bloch_ref_M{Mmax}_nk{a.nk}_{a.bloch_win}.npz", build)
        log(f"{kind}: Bloch states per k {z['cnt'].tolist()} in [{lo:.3f}, {hi:.3f}] eV; singular values "
            f"(M-th / first) {z['w'][-1] / z['w'][0]:.1e}")
    else:
        z = cached(f"{rk}_ref_M{Mmax}.npz", lambda: dict(zip(("w", "V"), cu.slab_modes(ref["H"], Mmax, Emid))))
    if kind.endswith("zero"):
        kappa = None
    elif a.kappa is not None:
        kappa = a.kappa
    else:
        kappa, bins, prof = cu.fit_kappa(ref, z["V"][:, :min(64, Mmax)])
        log(f"{kind}: fitted kappa {kappa:.3f} 1/A (decay length {1 / kappa:.2f} A); mean log|psi|^2 per atom "
            f"vs depth into the oxide (1 A bins) {np.round([x if x is not None else np.nan for x in prof], 2).tolist()}")
    if depth is None:
        depth = cu.oxide_depth(st.pos, st.pos[st.species == cu.SI_C])
    maps, stats = [], []
    for s, at in enumerate(slab_atoms):
        om, fac, sts = cu.orbital_map(ref, [keys[i] for i in at], st.species[at], st.norb[at], depth[at], kappa)
        maps.append((om, fac)); stats.append(sts)
    modes[kind] = (z["V"], maps)
    info[kind] = dict(ref_orbitals=int(ref["H"].shape[0]), ref_channel_Si=ref["n_channel"], kappa=kappa,
                      E=[float(z["w"][0]), float(z["w"][-1])], map=stats)
    log(f"{kind}: reference slab {ref['H'].shape[0]} orbitals, {ref['n_channel']} channel Si, "
        f"[{z['w'][0]:.3f}, {z['w'][-1]:.3f}]; map per slab (matched / missing Si / other atoms): "
        f"{[(x['matched'], x['missing_si'], x['other']) for x in stats]}")


def basis(kind, M):
    if kind == "conv":
        return [q[:, :M] for q in modes["conv"]], None
    Q, maps = modes[kind]
    V, sv = [], []
    for s in range(ns):
        v, sg = cu.project_modes(Q[:, :M], maps[s][0], maps[s][1], a.tol)
        V.append(v); sv.append(sg)
    return V, sv


def coarse_blocks(V):
    HD = [V[s].T @ (Hs[s] @ V[s]) for s in range(ns)]
    blk = dict(HD=[0.5 * (h + h.T) for h in HD], SD=[np.eye(v.shape[1]) for v in V],
               HU=[V[s].T @ (Hu[s] @ V[s + 1]) for s in range(ns - 1)],
               SU=[np.zeros((V[s].shape[1], V[s + 1].shape[1])) for s in range(ns - 1)])
    return blk, V[0][dev.bL], V[-1][dev.bR - dev.sb[-2]]


combos = {}
for kind in bases:
    for M in Ms:
        t = time.time()
        V, sv = basis(kind, M)
        blk, PL, PR = coarse_blocks(V)
        N = [v.shape[1] for v in V]
        combos[(kind, M)] = dict(V=V, blk=blk, PL=PL, PR=PR, N=N)
        smin = "" if sv is None else f", min singular value {min(x.min() for x in sv):.1e}"
        log(f"  {kind} M={M}: N per slab {N}{smin} ({time.time() - t:.0f} s)")

# ------------------------------------------------------------------ transport
scache = a.sigma_cache or os.path.join("results_ms", name, "sigma")
os.makedirs(os.path.join(out, "sigma"), exist_ok=True)


def sigma(E):
    for d in (scache, os.path.join(out, "sigma")):
        f = os.path.join(d, f"sig_{E:.4f}.npz")
        if os.path.exists(f):
            z = np.load(f)
            return z["SL"], z["SR"]
    SL, SR = dev.sigma(E + 1e-6j)
    np.savez(os.path.join(out, "sigma", f"sig_{E:.4f}.npz"), SL=SL, SR=SR)
    return SL, SR


def T_of(pr, GR):
    return float(np.real(np.einsum("ic,ij,jc->", pr.conj(), GR, pr)))


mgM = [int(x) for x in a.mg_M.split(",") if x]
fres = os.path.join(out, "results.json")
res = dict(n=dev.n, slabs=dev.ns, channel_Si=nch, info=info, M=Ms, bases=bases, E={})
for E in [float(x) for x in a.E.split(",")]:
    t = time.time()
    SL, SR = sigma(E)
    GR = 1j * (SR - SR.conj().T)
    B, lam, X = mb.injection(SL)
    nc = B.shape[1]
    r = dict(dE=E - Ec, channels=nc)
    if nc == 0:
        res["E"][f"{E:.4f}"] = r
        continue
    b = np.zeros((dev.n, nc), complex)
    b[dev.bL] = B
    A = dev.A(E, SL, SR)
    cs = mb.CSolver(A)
    psi = cs.solve(b)
    cs.free()
    Te = T_of(psi[dev.bR], GR)
    r.update(T_exact=Te, t_exact=time.time() - t)
    line = []
    for (kind, M), c in combos.items():
        Ac = mb.BlockTridiag(c["blk"], E, c["PL"].conj().T @ SL @ c["PL"], c["PR"].conj().T @ SR @ c["PR"])
        rhs = np.zeros((sum(c["N"]), nc), complex)
        rhs[:c["N"][0]] = c["PL"].conj().T @ B
        x = Ac.solve(rhs)
        Tm = T_of(c["PR"] @ x[-c["N"][-1]:], GR)
        err = abs(Tm - Te) / Te
        r[f"{kind}_{M}"] = dict(T=Tm, err=err, N=int(sum(c["N"])))
        if M in mgM:
            U = sp.block_diag([sp.csr_matrix(v) for v in c["V"]], format="csr")
            tl = TwoLevel(A, U, smoother="gmres", k=a.k, coarse=Ac)
            its, xs = [], np.zeros_like(b)
            for col in range(nc):
                xx, hist, _ = tl.solve(b[:, col], tol=a.mg_tol)
                xs[:, col] = xx; its.append(len(hist) - 1)
            r[f"{kind}_{M}"].update(mg_its=float(np.mean(its)), mg_mv=tl.nmv / nc,
                                    mg_err=abs(T_of(xs[dev.bR], GR) - Te) / Te)
        line.append(f"{kind}{M}:{err:.1e}")
    log(f"E {E:.3f} (Ec{E - Ec:+.3f}) ch {nc:2d} T {Te:.4f} | " + " ".join(line) + f" ({time.time() - t:.0f} s)")
    res["E"][f"{E:.4f}"] = r
    json.dump(res, open(fres, "w"), indent=1)

# summary: max / mean relative T error over the energies with T > 0
log("\nbasis         M    N(total)  max err   mean err" + ("   MG its" if mgM else ""))
Es = [v for v in res["E"].values() if "T_exact" in v]
summ = {}
for (kind, M), c in combos.items():
    e = [v[f"{kind}_{M}"]["err"] for v in Es]
    its = [v[f"{kind}_{M}"].get("mg_its") for v in Es]
    summ[f"{kind}_{M}"] = dict(N=int(sum(c["N"])), max=float(np.max(e)), mean=float(np.mean(e)))
    mg = f"   {np.mean(its):.1f}" if its and its[0] is not None else ""
    log(f"{kind:13s} {M:4d} {sum(c['N']):8d}  {np.max(e):.2e}  {np.mean(e):.2e}{mg}")
res["summary"] = summ
json.dump(res, open(fres, "w"), indent=1)
log(f"done, {time.time() - t0:.0f} s -> {fres}")
