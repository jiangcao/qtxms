"""Preprocess a QuaTrEx device folder once: read it (quatrex.read_quatrex: Gamma-point sum of the R blocks, Hermitian
completion, blocks, lead cell, lead band edges) and store the result in a prepared folder. Every later step takes the
prepared folder in place of the device folder and loads it in seconds instead of repeating this.

    python -m qtx_modespace.prep DEVICE_FOLDER --out PREP [--cells m] [--trim] [--potential]
    python -m qtx_modespace PREP --out ms_out --basis l1 ...          # as with DEVICE_FOLDER
    python -m qtx_modespace.obc_eig PREP ...

--cells, --trim and --potential are fixed at this step; later steps need not repeat them (and stop if they are given
with other values). qtx_modespace.to_quatrex still needs the QuaTrEx folder itself (it rewrites its files).
"""
import argparse, sys, time

from . import gpu
from .quatrex import read_quatrex, save_inputs, is_prepared


def main(argv=None):
    p = argparse.ArgumentParser(prog="qtx-modespace-prep", description=__doc__.split("\n\n")[0])
    p.add_argument("folder", help="QuaTrEx device folder (quatrex_config.toml, inputs/)")
    p.add_argument("--out", required=True, help="prepared folder to write")
    p.add_argument("--cells", type=int, help="contact cells per block (default: smallest covering the coupling range)")
    p.add_argument("--trim", action="store_true", help="drop H, S entries coupling non-neighbouring blocks")
    p.add_argument("--potential", action="store_true", help="add inputs/potential.npy as (V S + S V) / 2")
    p.add_argument("--nk", type=int, default=41, help="k points on [0, pi] for the lead band edges")
    p.add_argument("--gpu", action="store_true", help="lead band edges on the GPU (cupy)")
    p.add_argument("--sigma-cache", help="directory with / for lead_edges.json (the --sigma-cache of the other steps): "
                   "reuses lead band edges computed there")
    a = p.parse_args(argv)
    log = lambda *s: print(*s, flush=True)
    if is_prepared(a.folder):
        p.error(f"{a.folder} is already a prepared folder")
    if a.gpu:
        gpu.enable()
    t0 = time.time()
    inp = read_quatrex(a.folder, cells=a.cells, trim=a.trim, potential=a.potential, nk=a.nk,
                       cache_dir=a.sigma_cache, log=log)
    save_inputs(inp, a.out, options=dict(cells=a.cells, trim=a.trim, potential=a.potential, nk=a.nk),
                source=a.folder, log=log)
    log(f"done, {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
