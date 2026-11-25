"""Exact driver (user, 2026-11-25): the fine QTBM system A psi = b solved directly (msbasis.exact_solve: sparse direct on the
CPU, block-tridiagonal LU over the slabs on one GPU, factors streamed from the host when they do not fit), no basis. One process.
Output as qtx_ms.exact_T (key T_exact, usable with --exact-from), default results_ms/exact/<name>[...].json, refused for
another Ec; --save-psi DIR: psi_<E>.npz, readable by qtx_ms.mlevel --psi-dir. With --dev-store the sliced H, S are read
whole. Energies already in the output are skipped (--redo: solved again).

    python -m qtx_ms.mlexact --inputs DIR --dE 0.15,0.50 --gpu --sigma-cache DIR [--dev-store DIR] [--save-psi DIR]
"""
import gc, json, os, time
import numpy as np

from qtx_ms import gpu, mldrive as md, msbasis as mb
from qtx_ms.mlevel import guarded, start
from qtx_ms.dist import Comm

log = md.log


def parse(argv=None):
    p = md.new_parser(__doc__)
    md.device_args(p)
    p.add_argument("--save-psi", metavar="DIR", help="psi of the device orbitals per energy")
    p.add_argument("--redo", action="store_true", help="energies already in --out are solved again")
    return p.parse_args(argv)


def solve(a):
    t00 = time.time()
    log(f"backend: {gpu.describe()}")
    gpu.track_peak()
    inp, dev, name, Ec = md.read_device(a)
    if not a.out:
        tag = "" if a.sigma_cache is None else "_" + os.path.basename(os.path.dirname(os.path.normpath(a.sigma_cache)))
        tag += "_qtt" if not a.dev_store and dev.uses_qttools(Ec + 1j * a.eta) else ""
        a.out = os.path.join("results_ms", "exact", name + tag + ".json")
    md.check_out(a, Ec)
    log(f"  -> {a.out}")
    old = json.load(open(a.out))["E"] if os.path.exists(a.out) else {}
    if a.dev_store:
        dev.hold(None)
    res = dict(Ec=Ec, E={})
    for dE in md.energies(a):
        E, t = Ec + dE, time.time()
        k = f"{dE:.4f}"
        if k in old and not a.redo and "T_exact" in old[k]:
            log(f"Ec {dE:+.3f}: T {old[k]['T_exact']:.10f} (stored)")
            res["E"][k] = old[k]
            continue
        SL, SR, b, nch = md.lead(dev, E, a.eta, Comm())
        ts = time.time() - t
        psi, r = mb.exact_solve(dev, dev.A(E, SL, SR), b)
        GR = 1j * (SR - SR.conj().T)
        T = float(np.real(np.einsum("ic,ij,jc->", psi[dev.bR].conj(), GR, psi[dev.bR])))
        if a.save_psi:
            os.makedirs(a.save_psi, exist_ok=True)
            np.savez(os.path.join(a.save_psi, f"psi_{E:+.8f}.npz"), E=E, psi_m=psi, dev=dev.dev, obc="folded"
                     if a.obc == "folded" or a.dev_store else a.obc, eta=a.eta)
        log(f"Ec {dE:+.3f}: {nch} columns, T {T:.10f}, residual {r:.1e}; sigma {ts:.0f} s, total {time.time() - t:.0f} s")
        rec = dict(E=E, channels=int(nch), T_exact=T, res_exact=r, t=time.time() - t)
        res["E"][k] = rec
        md.write(a, Comm(), Ec, dE, rec)
        del psi, b
        gc.collect()
        gpu.free()
    log(f"done, {time.time() - t00:.0f} s")
    return res


def main(argv=None):
    a = parse(argv)
    start("", a.gpu)
    solve(a)


if __name__ == "__main__":
    guarded(main)
