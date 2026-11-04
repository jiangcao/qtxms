import sys, time, json
import numpy as np
import scipy.sparse.linalg as spla
from qtx_ms.device import Device, rgf_column
from qtx_ms.em import basis_cms, basis_ldm, assemble_U, TwoLevel, transmission_from_psi

L = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
dev = Device(L=L)
H = dev.full_H()
print(f"grid: {dev.S} slabs x {dev.ns} = {dev.n} points, t = {dev.t:.3f} eV")
eps, _ = dev.lead_modes()
e1 = eps[0] - 2 * dev.t  # bottom of lowest lead subband
print("lead subband bottoms (meV above e1):", np.round((eps[:6] - 2 * dev.t - e1) * 1e3, 1))
energies = e1 + np.array([0.01, 0.04, 0.07, 0.10, 0.13, 0.16])
Emax = e1 + 0.30
beta = 1.05 * spla.eigsh(H, k=1, which="LA", return_eigenvectors=False)[0] - energies.min()
Hblocks = [dev.slab_H(i) for i in range(dev.S)]

# ---- reference
ref = {}
t0 = time.time()
for E in energies:
    G, T = rgf_column(dev, E, Hblocks)
    B, Sig = dev.gamma_inj(E)
    psi = np.vstack([g @ B[:dev.ns] for g in G])
    ref[E] = (T, transmission_from_psi(dev, psi, Sig), B.shape[1])
t_rgf = (time.time() - t0) / len(energies)
print(f"RGF: {t_rgf:.2f} s/energy;", [(round(v[0], 4), v[2]) for v in ref.values()])

# ---- bases
bases = {}
for M in (2, 4, 8, 16):
    t0 = time.time(); bases[f"CMS M={M}"] = (assemble_U(basis_cms(dev, M)), time.time() - t0)
import scipy.sparse as sp
bases["none"] = (sp.csr_matrix((dev.n, 0)), 0.0)
for Eb in (0.3, 0.6, 1.0):
    t0 = time.time(); bases[f"LDM Eb={Eb:g}"] = (assemble_U(basis_ldm(dev, e1 + Eb, thr=1e-3, Hfull=H)), time.time() - t0)
# smoother lower bound: where the complement spectrum is assumed to start
alpha_of = {k: (e1 + float(k.split("=")[1]) if k.startswith("LDM") else Emax) for k in bases}

out = []
for name, (U, tb) in bases.items():
    for deg in (8, 16):
        rows = []
        for E in energies:
            B, Sig = dev.gamma_inj(E)
            tl = TwoLevel(dev, H, U, E, Sig, cheb_deg=deg, Emax=alpha_of[name], beta=beta)
            if E == energies[0] and deg == 8:  # projector sanity check
                v = np.random.default_rng(0).standard_normal(dev.n) + 0j
                p1 = tl.Pc(tl.A @ v); p2 = tl.Pc(tl.A @ p1)
                idem = np.linalg.norm(p2 - p1) / np.linalg.norm(p1)
            t0 = time.time()
            X, X0, its = [], [], []
            for c in range(B.shape[1]):
                x, hist, x0 = tl.solve(B[:, c].copy())
                X.append(x); X0.append(x0); its.append(len(hist) - 1)
            ts = time.time() - t0
            Tms = transmission_from_psi(dev, np.array(X0).T, Sig)
            Tmg = transmission_from_psi(dev, np.array(X).T, Sig)
            Tr = ref[E][0]
            rows.append(dict(E=E - e1, Tref=Tr, err_ms=abs(Tms - Tr) / Tr, err_mg=abs(Tmg - Tr) / Tr,
                             its=its, nmv=tl.nmv, nrhs=B.shape[1], t=ts))
        r = dict(basis=name, N=U.shape[1], N_per_slab=U.shape[1] / dev.S, t_build=tb, deg=deg,
                 idem=idem, max_err_ms=max(x["err_ms"] for x in rows),
                 max_err_mg=max(x["err_mg"] for x in rows),
                 its_mean=np.mean([np.mean(x["its"]) for x in rows]),
                 its_max=max(max(x["its"]) for x in rows),
                 mv_per_rhs=np.mean([x["nmv"] / x["nrhs"] for x in rows]),
                 t_per_E=np.mean([x["t"] for x in rows]), rows=rows)
        out.append(r)
        print(f"{name:14s} N={r['N']:5d} ({r['N_per_slab']:.1f}/slab) deg={deg:2d} | "
              f"MS err {r['max_err_ms']:.2e} -> MG err {r['max_err_mg']:.2e} | "
              f"GMRES its mean {r['its_mean']:.1f} max {r['its_max']} | "
              f"matvec/rhs {r['mv_per_rhs']:.0f} | {r['t_per_E']:.2f} s/E | idem {idem:.1e}", flush=True)

json.dump(dict(n=dev.n, S=dev.S, ns=dev.ns, t_rgf=t_rgf, results=out), open(f"results_L{L:g}.json", "w"),
          default=float, indent=1)
