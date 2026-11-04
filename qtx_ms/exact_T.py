"""Exact QTBM transmission of a prepared input set (sparse direct solve, no mode space): checks a new input set.

    python -m qtx_ms.exact_T --inputs inputs/mos2_hfo2 --dE 0.05,0.15
"""
import argparse, json, os, time
import numpy as np
from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--inputs", required=True)
p.add_argument("--dE", default="0.05,0.15", help="energies - Ec_lead [eV]")
p.add_argument("--Ec", type=float, help="reference edge for --dE (default Ec_lead of the inputs; set it to compare "
               "input sets at the same absolute energies)")
p.add_argument("--json", help="results file, merged across runs (default results_ms/exact/<inputs name>.json); "
               "same layout as run_lead_cms results.json, so it can be passed to --exact-from")
p.add_argument("--sigma-cache", help="lead self-energy cache (default results_ms/sigma/<inputs name>; results_ms/sigma_qtx/<name> for QuaTrEx Sigma from qtx_ms.prep.quatrex_obc)")
p.add_argument("--save-psi", metavar="DIR", help="also save psi (device orbitals) per energy to DIR/psi_<E>.npz "
               "(E, psi_m, dev, obc), for qtx_ms.plot_psi --modes")
p.add_argument("--obc", default="auto", choices=["auto", "qttools", "folded"], help="lead OBC: qttools (QuaTrEx spectral OBC, real E; auto falls back to the folded-lead decimation when it is not available) or mgms")
a = p.parse_args()
name = os.path.basename(os.path.normpath(a.inputs))
scache = a.sigma_cache or os.path.join("results_ms", "sigma", name)
inp = read_inputs(a.inputs)
dev = mb.Device(inp, obc=a.obc)
dev.sigma_cache = scache
Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
tag = "" if a.sigma_cache is None else "_" + os.path.basename(os.path.dirname(os.path.normpath(scache)))
tag += "_qtt" if dev.uses_qttools(Ec + 1e-6j) else ""
fjs = a.json or os.path.join("results_ms", "exact", name + tag + ".json")   # other Sigma / OBC: own file
os.makedirs(os.path.dirname(fjs) or ".", exist_ok=True)
print(f"device {dev.n} orbitals, slabs {dev.ns}, |bL| {len(dev.bL)}, |bR| {len(dev.bR)}, "
      f"separate right lead {dev.leadR is not dev.lead}; OBC {'qttools' if tag.endswith('_qtt') else 'folded lead'} "
      f"-> {fjs}", flush=True)
for dE in (float(x) for x in a.dE.split(",")):
    E = Ec + dE
    t = time.time()
    SL, SR = dev.sigma(E + 1e-6j)
    ts = time.time() - t
    B = dev.inject(SL)
    nR = dev.inject(SR).shape[1]
    GR = 1j * (SR - SR.conj().T)
    b = np.zeros((dev.n, B.shape[1]), complex)
    b[dev.bL] = B
    cs = mb.CSolver(dev.A(E, SL, SR))
    psi = cs.solve(b)
    if a.save_psi:
        os.makedirs(a.save_psi, exist_ok=True)
        np.savez(os.path.join(a.save_psi, f"psi_{E:+.8f}.npz"), E=E, psi_m=psi, dev=dev.dev,
                 obc="qttools" if tag.endswith("_qtt") else "folded", eta=1e-6)
    T = float(np.real(np.einsum("ic,ij,jc->", psi[dev.bR].conj(), GR, psi[dev.bR])))
    print(f"E = Ec {dE:+.3f}: channels L {B.shape[1]} / R {nR}, T = {T:.5f}, residual {cs.res:.1e} "
          f"(GMRES its {cs.its}, perturbed pivots {cs.npert}); sigma {ts:.0f} s, total {time.time() - t:.0f} s",
          flush=True)
    res = json.load(open(fjs)) if os.path.exists(fjs) else {"E": {}}
    if res.get("Ec", Ec) != Ec:
        raise SystemExit(f"{fjs}: written with Ec {res['Ec']}, this run uses {Ec}")
    res["Ec"] = Ec
    res["E"][f"{dE:.4f}"] = dict(E=E, channels=int(B.shape[1]), T_exact=T, res_exact=cs.res)
    json.dump(res, open(fjs, "w"), indent=1)
    cs.free()
