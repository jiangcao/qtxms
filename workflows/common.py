"""Shared helpers of workflows/sigma, workflows/psi, workflows/bases: load a device from either code stack and read the
recorded reference values (T_exact, channels, ...) of an earlier run.

Stacks
  release  qtx_modespace (release/qtx_modespace/src): --inputs = QuaTrEx device folder or prepared folder
           (python -m qtx_modespace.prep); Sigma cache files sig_<Re z>_<Im z>.npz; psi files psi_<E>_<eta>.npy
  qtx_ms   research stack: --inputs = qtx_ms prepared inputs (inputs/mos2_hfo2*); Sigma cache sig_/qsig_<...>.npz;
           psi files psi_<E>.npz (key psi_m)
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "release", "qtx_modespace", "src"))
sys.path.insert(1, REPO)


def energies(spec):
    if ":" in spec:
        lo, hi, st = (float(x) for x in spec.split(":"))
        return list(np.round(np.arange(lo, hi + 0.5 * st, st), 6))
    return [float(x) for x in spec.split(",")]


def load_device(stack, inputs, obc="auto", sigma_cache=None, gpu=False):
    """(stack module namespace, inp, dev, Ec_lead). sigma_cache: directory (created on the first write)."""
    if stack == "release":
        from qtx_modespace import gpu as g
        from qtx_modespace.core import Device, exact_solve
        from qtx_modespace.quatrex import read_quatrex
        if gpu:
            g.enable()
        inp = read_quatrex(inputs, cache_dir=sigma_cache, log=lambda *a, **k: None)
        dev = Device(inp, sigma_cache=sigma_cache, obc=obc)
    elif stack == "qtx_ms":
        from qtx_ms import gpu as g, msbasis as mb
        from qtx_ms.prep.inputs import read_inputs
        exact_solve = mb.exact_solve
        if gpu:
            g.enable()
        inp = read_inputs(inputs)
        dev = mb.Device(inp, obc=obc)
        dev.sigma_cache = sigma_cache
    else:
        raise ValueError(stack)
    return dict(exact_solve=exact_solve, gpu=g), inp, dev, float(inp["meta"]["edges"]["Ec_lead"])


def reference(path):
    """{dE key '%.4f': record} from a release transmission.json or a qtx_ms results_ms/exact/<name>.json."""
    if not path:
        return {}
    d = json.load(open(path))
    return {f"{float(k):.4f}": v for k, v in d.get("E", {}).items()}


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)
