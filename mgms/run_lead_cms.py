"""Mode space from the lead: a CMS-type basis of the lead cell, cleaned of spurious bands, applied to every
device slab on the lead-like atoms only (coefficients on all other atoms, e.g. the oxide, are zero).

  1. lead basis Phi: k integral of the lead Bloch projectors with E_n(k) in [Ec + mu1, Ec + mu2], eigenvectors
     with occupation (S00 metric) above thr * max (lead_rms.lead_rho_basis)
  2. spurious bands of Phi^H H(k) Phi (vs the exact lead bands) in [Ec + win1, Ec + win2] removed by appending
     one optimized vector per iteration (lead_rms.RMS, qtmode / Huang et al. 2018)
  3. slab s: lead-cell atoms shifted to the slab and matched one-to-one to slab atoms of the same species
     (lead_rms.slab_cell_map); U_s = basis rows on the matched orbitals, zero elsewhere, re-orthonormalized
  4. per energy: T_MS from x = U A_c^-1 U^H b, A_c = U^H (E S - H - Sigma) U (block tridiagonal, exact Sigma
     projected), vs the exact T (sparse direct solve); both with Phi and with the cleaned basis

    python -m mgms.run_lead_cms --inputs inputs/mos2_hfo2 --dE 0.02,0.05,0.15,0.3
"""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp

from mgms.prep.inputs import read_inputs
from mgms import msbasis as mb
from mgms.lead_rms import LeadCell, RMS, lead_rho_basis, slab_cell_map, to_slab, band_stats, plot_bands

log = mb.log
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--inputs", required=True)
p.add_argument("--out", help="default results_ms/<inputs name>_leadcms")
p.add_argument("--mu", default="0,0.4", help="basis window - Ec_lead [eV]")
p.add_argument("--thr", type=float, default=1e-5, help="occupation cut (relative)")
p.add_argument("--nk-rho", type=int, default=32, help="k points for the lead density matrix")
p.add_argument("--win", default="-0.8,0.4", help="spurious-state window - Ec_lead [eV]")
p.add_argument("--nk", type=int, default=25, help="k points on [0, pi] for bands / evaluation")
p.add_argument("--kstep", type=int, default=2, help="every kstep-th k point is used in the optimization")
p.add_argument("--maxit", type=int, default=150)
p.add_argument("--tol", type=float, default=0.005, help="spurious: > tol [eV] from every exact state")
p.add_argument("--map-tol", type=float, default=0.6, help="max displacement [A] of a matched atom")
p.add_argument("--dE", default="0.02,0.05,0.15,0.3", help="test energies - Ec_lead [eV]")
p.add_argument("--no-exact", action="store_true")
a = p.parse_args()

name = os.path.basename(os.path.normpath(a.inputs))
out = a.out or os.path.join("results_ms", name + "_leadcms")
os.makedirs(out, exist_ok=True)
t0 = time.time()
inp = read_inputs(a.inputs)
Ec = inp["meta"]["edges"]["Ec_lead"]
dev = mb.Device(inp)
dev.sigma_cache = os.path.join("results_ms", "sigma", name)
cell = LeadCell(inp["lead"])
log(f"{name}: device {dev.n} orbitals, slabs {dev.ns}; lead cell {cell.n} orbitals, Ec {Ec:.4f} eV")

# ------------------------------------------------------------------ 1. lead basis
m1, m2 = (Ec + float(x) for x in a.mu.split(","))
fphi = os.path.join(out, f"phi_mu{a.mu.replace(',', '_')}_thr{a.thr:g}_nk{a.nk_rho}.npy")
if os.path.exists(fphi):
    Phi = np.load(fphi)
else:
    t = time.time()
    Phi = lead_rho_basis(cell, m1, m2, a.thr, nk=a.nk_rho)
    np.save(fphi, Phi)
    log(f"lead basis [{m1:.3f}, {m2:.3f}] eV, thr {a.thr:g}: {Phi.shape[1]} vectors, {time.time() - t:.0f} s")

# ------------------------------------------------------------------ 2. spurious bands
ks = np.linspace(0, np.pi, a.nk)
t = time.time()
Eex = cell.bands(ks, cache=os.path.join(out, f"bands_{a.nk}.npz"))
log(f"exact lead bands on {a.nk} k: {time.time() - t:.0f} s")
w1, w2 = (Ec + float(x) for x in a.win.split(","))
frms = os.path.join(out, os.path.basename(fphi).replace("phi_", "rms_").replace(".npy", f"_win{a.win.replace(',', '_')}.npz"))
rms = RMS(cell, Phi, ks, (w1, w2))
e0 = [rms.eig(q)[0] for q in ks]
st0 = dict(window=band_stats(e0, Eex, w1, w2, a.tol), near=band_stats(e0, Eex, Ec, m2, a.tol))
if os.path.exists(frms):
    z = np.load(frms)
    assert z["Xi"].shape == rms.Xi.shape
    rms.Cs = z["C"]
    hist = json.loads(str(z["hist"]))
else:
    ki = np.arange(0, a.nk, a.kstep)
    rms.ks = ks[ki]
    hist = rms.run(Eex[ki], tol=a.tol, maxit=a.maxit, grad_check=True)
    rms.ks = ks
    np.savez(frms, Xi=rms.Xi, C=rms.Cs, hist=json.dumps(hist))
e1 = [rms.eig(q)[0] for q in ks]
st1 = dict(window=band_stats(e1, Eex, w1, w2, a.tol), near=band_stats(e1, Eex, Ec, m2, a.tol))
log(f"lead bands, window [{w1:.3f}, {w2:.3f}]: N {Phi.shape[1]} -> {st0}\n"
    f"  + {rms.Cs.shape[1]} vectors (N {Phi.shape[1] + rms.Cs.shape[1]}) -> {st1}")
plot_bands(os.path.join(out, "lead_bands_rms.png"), ks, Eex, [(f"lead CMS, N {Phi.shape[1]}", e0),
           (f"+ {rms.Cs.shape[1]} vectors", e1)], {"Ec": Ec, "Ev": inp["meta"]["edges"]["Ev_lead"]},
           (inp["meta"]["edges"]["Ev_lead"] - 0.6, Ec + 1.0), f"{name} lead cell ({cell.n} orbitals)",
           [(w1, w2, "spurious window")], [Ec + float(x) for x in a.dE.split(",")], a.tol)
bases = dict(cms=Phi, cms_rms=np.linalg.qr(np.hstack([Phi, rms.added()]))[0])

# ------------------------------------------------------------------ 3. slab bases
maps = []
for s in range(len(dev.ns)):
    cmap, info = slab_cell_map(inp, dev, s, tol=a.map_tol, return_info=True)
    maps.append(cmap)
    log(f"slab {s}: {dev.ns[s]} orbitals, lead-cell atoms unmatched {info['missing']}, slab atoms without "
        f"basis {info['extra']}, max displacement {info['max_dist']:.3f} A")


def slab_bases(B):
    Us = []
    for s, cmap in enumerate(maps):
        X = to_slab(B, cmap, dev.ns[s])
        u, sv, _ = np.linalg.svd(X, full_matrices=False)          # orthonormal, drop columns lost with atoms
        Us.append(sp.csr_matrix(u[:, sv > 1e-8 * sv[0]]))
    return Us


sets = {}
for tag, B in bases.items():
    Us = slab_bases(B)
    blk = mb.slab_blocks(dev, Us)
    sets[tag] = dict(Us=Us, blk=blk, PL=Us[0][dev.bL].toarray(), PR=Us[-1][dev.bR - dev.sb[-2]].toarray(),
                     U=sp.block_diag(Us, format="csr"))
    log(f"{tag}: N per slab {[u.shape[1] for u in Us]}, total {sum(u.shape[1] for u in Us)}")

# ------------------------------------------------------------------ 4. transmission
fres = os.path.join(out, "results.json")
res = json.load(open(fres)) if os.path.exists(fres) else {}
res.update(N={k: int(v["U"].shape[1]) for k, v in sets.items()}, lead_bands=dict(cms=st0, cms_rms=st1),
           args=vars(a))
res.setdefault("E", {})
for dE in (float(x) for x in a.dE.split(",")):
    E = Ec + dE
    r = res["E"].get(f"{dE:.4f}", {})
    t = time.time()
    SL, SR = dev.sigma(E + 1e-6j)
    B = mb.injection(SL)[0]
    GR = 1j * (SR - SR.conj().T)
    b = np.zeros((dev.n, B.shape[1]), complex)
    b[dev.bL] = B
    Tof = lambda x: float(np.real(np.einsum("ic,ij,jc->", x[dev.bR].conj(), GR, x[dev.bR])))
    r.update(E=E, channels=int(B.shape[1]), t_sigma=time.time() - t)
    if not a.no_exact and "T_exact" not in r:
        t = time.time()
        cs = mb.CSolver(dev.A(E, SL, SR))
        r.update(T_exact=Tof(cs.solve(b)), res_exact=cs.res, t_exact=time.time() - t)
        cs.free()
    line = f"E = Ec {dE:+.3f}: {B.shape[1]} channels, T_exact {r.get('T_exact', float('nan')):.5f}"
    for tag, d in sets.items():
        Ac = mb.BlockTridiag(d["blk"], E, d["PL"].conj().T @ SL @ d["PL"], d["PR"].conj().T @ SR @ d["PR"])
        x = d["U"] @ Ac.solve(d["U"].conj().T @ b)
        r[f"T_{tag}"] = Tof(x)
        if "T_exact" in r:
            r[f"err_{tag}"] = (r[f"T_{tag}"] - r["T_exact"]) / r["T_exact"]
        line += f" | {tag}: T {r[f'T_{tag}']:.5f}" + (f" ({100 * r[f'err_{tag}']:+.2f}%)" if "T_exact" in r else "")
    log(line)
    res["E"][f"{dE:.4f}"] = r
    json.dump(res, open(fres, "w"), indent=1)
log(f"done, {time.time() - t0:.0f} s -> {fres}")
