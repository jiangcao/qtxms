"""Two-level (mode space as coarse grid) transport solve for an ab initio TB device.

Example:
  python -m mgms.run_tb --xyz device.xyz --H H.mtx --S S.mtx --norb "Si:10,H:1" --units eV \
                        --dE 0.02,0.06,0.10 --Eb 0.6 --basis ldm,slab
"""
import argparse, json, time
import numpy as np
import scipy.sparse as sp
from mgms.abtb import (read_xyz, parse_norb, read_sparse, TBDevice, UNITS, injection, rgf_transmission,
                       basis_slab_window, basis_ldm)
from mgms.twolevel import TwoLevel, build_A

p = argparse.ArgumentParser()
p.add_argument("--xyz", required=True)
p.add_argument("--H", required=True)
p.add_argument("--S", help="overlap; omit for orthogonal TB")
p.add_argument("--norb", required=True, help="int, 'Si:10,H:1', or file with one int per atom")
p.add_argument("--units", default="eV", choices=list(UNITS))
p.add_argument("--axis", default="x", choices=["x", "y", "z"])
p.add_argument("--slab-width", type=float, default=None, help="default: coupling range along transport")
p.add_argument("--Ec", type=float, default=None, help="conduction band edge (eV); default: from lead bands")
p.add_argument("--dE", default="0.02,0.06,0.10,0.14", help="energies above Ec (eV)")
p.add_argument("--Eb", type=float, default=0.6, help="basis window top above Ec (eV)")
p.add_argument("--basis", default="ldm,slab")
p.add_argument("--M", type=int, default=4, help="modes per slab for the 'slab' (CMS-like) basis")
p.add_argument("--w", type=int, default=2, help="LDM cluster half-width (slabs)")
p.add_argument("--smoother", default="gmres", choices=["gmres", "cheb"])
p.add_argument("--k", type=int, default=10, help="smoother steps")
p.add_argument("--tol", type=float, default=1e-8)
p.add_argument("--no-ref", action="store_true", help="skip RGF reference")
p.add_argument("--out", default="results_tb.json")
a = p.parse_args()

print("reading input")
species, xyz = read_xyz(a.xyz)
norb = parse_norb(a.norb, species)
n = int(norb.sum())
H = read_sparse(a.H, n) * UNITS[a.units]
S = read_sparse(a.S, n) if a.S else sp.identity(n, format="csr")
dev = TBDevice(H, S, species, xyz, norb, axis="xyz".index(a.axis), slab_width=a.slab_width)

# ---- lead band edges
ks, bands = dev.lead_bands()
if a.Ec is None:
    bmin, bmax = bands.min(0), bands.max(0)
    gaps = bmin[1:] - np.maximum.accumulate(bmax)[:-1]
    g = int(np.argmax(gaps))
    Ev, Ec = np.maximum.accumulate(bmax)[g], bmin[g + 1]
    print(f"lead: largest gap {gaps[g]:.3f} eV, Ev = {Ev:.4f}, Ec = {Ec:.4f} eV")
else:
    Ec = a.Ec
    Ev = bands[bands < Ec].max()
    print(f"lead: Ec = {Ec:.4f} (given), Ev = {Ev:.4f} eV")
mid = 0.5 * (Ev + Ec)
energies = Ec + np.array([float(v) for v in a.dE.split(",")])

# ---- coarse bases
bases = {"none": (sp.csr_matrix((n, 0)), 0.0)}
for name in a.basis.split(","):
    t0 = time.time()
    Us = basis_ldm(dev, mid, Ec + a.Eb, w=a.w) if name == "ldm" else basis_slab_window(dev, mid, a.M)
    U = sp.block_diag([sp.csr_matrix(u) if u.shape[1] else sp.csr_matrix((u.shape[0], 0)) for u in Us],
                      format="csr")
    bases[name] = (U, time.time() - t0)
    print(f"basis {name}: N = {U.shape[1]} ({U.shape[1] / dev.S_n:.1f}/slab), window [{mid:.3f}, {Ec + a.Eb:.3f}] eV, "
          f"build {bases[name][1]:.1f} s")

# smallest |eigenvalue| of -A on the complement ~ distance to the band edges outside the window (cheb only)
results = []
for E in energies:
    SigL, SigR = dev.sigma(E)
    B0 = injection(SigL)
    B = np.zeros((n, B0.shape[1]), complex); B[:dev.bounds[1]] = B0
    GamR = 1j * (SigR - SigR.conj().T)
    lo = dev.bounds[-2]
    Tpsi = lambda X: np.real(np.trace(X[lo:].conj().T @ GamR @ X[lo:]))
    Tref = np.nan
    if not a.no_ref:
        t0 = time.time(); Tref = rgf_transmission(dev, E, SigL, SigR); t_rgf = time.time() - t0
    A = build_A(dev, E, SigL, SigR)
    nk_cross = int(np.sum(np.diff(np.sign(bands - E), axis=0) != 0))
    if nk_cross != B0.shape[1]:
        print(f"  note: {B0.shape[1]} injected channels vs {nk_cross} band crossings in (0, pi)")
    row = dict(E=E, E_minus_Ec=E - Ec, channels=B0.shape[1], Tref=Tref, t_rgf=t_rgf if not a.no_ref else None)
    for name, (U, tb) in bases.items():
        beta = 1.05 * max(abs(E - dev.H.diagonal().min()), abs(dev.H.diagonal().max() - E)) * 3
        tl = TwoLevel(A, U, smoother=a.smoother, k=a.k, alpha=Ec + a.Eb - E, beta=beta)
        t0 = time.time()
        X, X0, its = [], [], []
        for c in range(B.shape[1]):
            x, h, x0 = tl.solve(B[:, c].copy(), tol=a.tol)
            X.append(x); X0.append(x0); its.append(len(h) - 1)
        ts = time.time() - t0
        Tms, Tmg = Tpsi(np.array(X0).T), Tpsi(np.array(X).T)
        r = dict(N=U.shape[1], T_ms=Tms, T=Tmg, its=its, matvec_per_rhs=tl.nmv / max(1, B.shape[1]),
                 t=ts, sigma_min_Ac=tl.sigma_min_coarse())
        row[name] = r
        err = lambda T: abs(T - Tref) / max(Tref, 1e-12)
        print(f"E-Ec={E - Ec:+.3f} ch={B0.shape[1]:2d} T_RGF={Tref:.5f} | {name:5s} N={U.shape[1]:5d} "
              f"MS err {err(Tms):.1e} -> {err(Tmg):.1e}  its {np.mean(its):5.1f}  "
              f"matvec/rhs {r['matvec_per_rhs']:5.0f}  {ts:.2f} s (RGF {row['t_rgf'] or 0:.2f} s)  "
              f"sigma_min(Ac) {r['sigma_min_Ac']:.1e}", flush=True)
    results.append(row)
json.dump(dict(n=n, slabs=dev.S_n, Ec=Ec, Ev=Ev, results=results), open(a.out, "w"), default=float, indent=1)
print("wrote", a.out)
