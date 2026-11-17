"""Rebuild exact QTBM solutions psi (A psi = b, b = injected lead channels) into a psi directory and check them
against a recorded run: T_exact must match the reference (default tolerance 1e-8 relative) and the residual must be
small. File names as the readers expect them:
  release  psi_<E>_<eta>.npy        (qtx_modespace --psi-dir; qtx_ms.mlevel reads these too)
  qtx_ms   psi_<E>.npz, key psi_m   (qtx_ms.mlevel --psi-dir, run_exact_21600a.sh)

    python workflows/psi/rebuild_psi.py --stack release --inputs results_ms/gpu/wf_gh200/prep --gpu \
        --sigma-cache S/sigma_2808a --psi-dir S/psi_2808a --dE 0.05,0.15,0.2 \
        --ref results_ms/gpu/wf_gh200/A_slab_obc/transmission.json
    python workflows/psi/rebuild_psi.py --stack qtx_ms --inputs inputs/mos2_hfo2_21600a --gpu --obc folded \
        --sigma-cache /scratch/jiacao/mg_modespace/21600a/sigma --psi-dir /scratch/jiacao/mg_modespace/21600a/psi \
        --dE 0.15 --ref-T 0.937877 --ref-abs 5e-7

Existing files are reused (checked: residual) unless --force. Exit status 1 on a T mismatch.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import energies, load_device, reference, rel  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stack", choices=["release", "qtx_ms"], default="release")
    p.add_argument("--inputs", required=True)
    p.add_argument("--sigma-cache", required=True)
    p.add_argument("--psi-dir", required=True)
    p.add_argument("--dE", required=True)
    p.add_argument("--eta", type=float, default=None, help="Im z of Sigma [eV] (default: release 0, qtx_ms 1e-6)")
    p.add_argument("--obc", default="auto")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--ref", help="recorded transmission.json / exact json (T_exact per energy)")
    p.add_argument("--ref-T", help="recorded T_exact values, comma list in --dE order (instead of --ref)")
    p.add_argument("--ref-tol", type=float, default=1e-8, help="relative T tolerance")
    p.add_argument("--ref-abs", type=float, default=0.0, help="absolute T tolerance (references printed with few "
                   "digits, e.g. 5e-7 for 6 decimals)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", help="json with T, residual, reference per energy")
    a = p.parse_args()
    eta = a.eta if a.eta is not None else (0.0 if a.stack == "release" else 1e-6)
    ns, inp, dev, Ec = load_device(a.stack, a.inputs, a.obc, a.sigma_cache, a.gpu)
    ref = reference(a.ref)
    dEs = energies(a.dE)
    if a.ref_T:
        ref = {f"{d:.4f}": dict(T_exact=float(t)) for d, t in zip(dEs, a.ref_T.split(","))}
    os.makedirs(a.psi_dir, exist_ok=True)
    rows, bad = {}, 0
    for dE in dEs:
        E = Ec + dE
        t = time.time()
        SL, SR = dev.sigma(E + 1j * eta)
        B = dev.inject(SL)
        b = np.zeros((dev.n, B.shape[1]), complex)
        b[dev.bL] = B
        A = dev.A(E, SL, SR).tocsr()
        f = os.path.join(a.psi_dir, f"psi_{E:+.8f}_{eta:.2e}.npy" if a.stack == "release" else f"psi_{E:+.8f}.npz")
        X = None
        if os.path.exists(f) and not a.force:
            X = np.load(f) if f.endswith(".npy") else np.load(f)["psi_m"]
            res = float(np.linalg.norm(b - A @ X) / np.linalg.norm(b)) if X.shape == b.shape else np.inf
            if res > 1e-8:
                X = None
            src = "reused"
        if X is None:
            X, res = ns["exact_solve"](dev, A, b)
            tmp = f + ".tmp" + os.path.splitext(f)[1]
            np.save(tmp, X) if f.endswith(".npy") else np.savez(tmp, psi_m=X)
            os.replace(tmp, f)
            src = "computed"
        GR = 1j * (SR - SR.conj().T)
        T = float(np.real(np.einsum("ic,ij,jc->", X[dev.bR].conj(), GR, X[dev.bR])))
        r = ref.get(f"{dE:.4f}", {})
        Tr = r.get("T_exact")
        ok = Tr is None or abs(T - Tr) <= max(a.ref_tol * abs(Tr), a.ref_abs)
        bad += not ok
        rows[f"{dE:.4f}"] = dict(E=E, channels=int(B.shape[1]), T=T, residual=float(res), T_ref=Tr,
                                 rel_dT=None if Tr is None else rel(T, Tr), ok=ok, file=f, source=src)
        print(f"Ec {dE:+.4f}: {B.shape[1]} channels, T {T:.10f} (ref {Tr if Tr is not None else '-'}"
              + (f", rel {rel(T, Tr):.1e}" if Tr is not None else "") + f"), residual {res:.1e}, {src}, "
              f"{time.time() - t:.0f} s{'' if ok else '  MISMATCH'}", flush=True)
    if a.out:
        json.dump(dict(stack=a.stack, inputs=a.inputs, eta=eta, E=rows), open(a.out, "w"), indent=1)
    print(f"psi: {len(rows)} energies, {bad} T mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
