"""Rebuild a lead self-energy cache (Sigma_L, Sigma_R at z = Ec_lead + dE + i eta) and check it against a recorded run:
the number of injected channels must match, and the trace of Gamma_L is reported. The Sigma files themselves are a
cache (both stacks recompute a missing file automatically); this script only prefills a cache directory.

    python workflows/sigma/rebuild_sigma.py --stack release --inputs results_ms/gpu/wf_gh200/prep \
        --cache /scratch/jiacao/mg_modespace/verify/sigma_2808a --dE 0.05,0.15,0.2 --gpu \
        --ref results_ms/gpu/wf_gh200/A_slab_obc/transmission.json --out sigma_2808a.json
    python workflows/sigma/rebuild_sigma.py --stack qtx_ms --inputs inputs/mos2_hfo2 --obc qttools --eta 0 \
        --cache results_ms/sigma_qtx/mos2_hfo2 --dE 0.05,0.15 --ref results_ms/exact/mos2_hfo2.json

GPU hosts have no qttools: --obc auto falls back to the folded lead at E + 1e-6 i there (release) / use --eta 1e-6
(qtx_ms). Exit status 1 if a channel count differs from the reference.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import energies, load_device, reference  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stack", choices=["release", "qtx_ms"], default="release")
    p.add_argument("--inputs", required=True)
    p.add_argument("--cache", required=True, help="Sigma cache directory to fill")
    p.add_argument("--dE", required=True, help="energies - Ec_lead [eV]: list or lo:hi:step")
    p.add_argument("--eta", type=float, default=None, help="Im z [eV] (default: release 0, qtx_ms 1e-6)")
    p.add_argument("--obc", default="auto")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--ref", help="recorded transmission.json / exact json with 'channels' per energy")
    p.add_argument("--out", help="json with channels and tr Gamma_L per energy")
    a = p.parse_args()
    eta = a.eta if a.eta is not None else (0.0 if a.stack == "release" else 1e-6)
    _, inp, dev, Ec = load_device(a.stack, a.inputs, a.obc, a.cache, a.gpu)
    ref = reference(a.ref)
    rows, bad = {}, 0
    for dE in energies(a.dE):
        t = time.time()
        SL, SR = dev.sigma(Ec + dE + 1j * eta)
        nch = int(dev.inject(SL).shape[1])
        trG = float(np.real(np.trace(1j * (SL - SL.conj().T))))
        r = ref.get(f"{dE:.4f}", {})
        ok = "channels" not in r or r["channels"] == nch
        bad += not ok
        rows[f"{dE:.4f}"] = dict(E=Ec + dE, channels=nch, trGammaL=trG, ref_channels=r.get("channels"), ok=ok)
        print(f"Ec {dE:+.4f}: {nch} channels (ref {r.get('channels', '-')}), tr Gamma_L {trG:.6e}, "
              f"{time.time() - t:.1f} s{'' if ok else '  MISMATCH'}", flush=True)
    if a.out:
        json.dump(dict(stack=a.stack, inputs=a.inputs, cache=a.cache, eta=eta, E=rows), open(a.out, "w"), indent=1)
    print(f"sigma: {len(rows)} energies, {bad} channel mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
