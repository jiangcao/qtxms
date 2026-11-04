"""Prepare transport inputs for the rough Si nanosheet with oxide shell (sp3s* or sp3d5s* TB).

Builds the open device  [L0 L1 L2 | D1 .. Dn | R0 R1 R2]  (blocks of 2 Si cells, 10.86 A along x),
its TB Hamiltonian and the periodic lead cell, and writes the generic input set of qtx_ms.prep.inputs
plus structure.npz (species codes, bonds, union sites, roughness profiles, TB parameters in meta.json)
for the Si-specific level-1 core basis of qtmode.cms.

    python -m qtx_ms.prep.si_sheet --model sp3d5s --xsec 3x5 --out inputs/si3x5_sp3d5s

Transport convention (as in the qtmode pipeline): device = blocks 2 .. nb-nl (L2 + D + R0), coupled to
lead cells L1 (left) and R1 (right); lead_H00 / lead_H01 are the R1 cell and its coupling to R2.
Energies in eV with bulk Si VBM = 0, lengths in A.
"""
import argparse
import dataclasses
import time
import numpy as np
from types import SimpleNamespace

from . import model, structure
from .structure import build_open_device, NAMES, L_BLOCK
from .inputs import write_inputs

LABELS = ["Si", "Siox", "O", "Au"]             # xyz labels of SI_C, SI_OX, OX, AU
ORBITALS = {"sp3s": {"Si": ["s", "px", "py", "pz", "s*"]},
            "sp3d5s": {"Si": ["s", "px", "py", "pz", "dxy", "dyz", "dzx", "dx2y2", "dz2", "s*"]}}
for _m in ORBITALS.values():
    _m.update(Siox=["s", "px", "py", "pz"], O=["s", "px", "py", "pz"], Au=["s"])


def params_dict(P):
    return dataclasses.asdict(P)


def params_from_dict(name, d):
    """Inverse of params_dict for the model `name` (sp3s -> TBParams, sp3d5s -> TB5Params)."""
    from .hamiltonian import TBParams
    if name == "sp3s":
        return TBParams(**d)
    from .hamiltonian5 import TB5Params
    return TB5Params(**dict(d, ox=TBParams(**d["ox"])))


def prepare(name, cfg, out, lead_edge=True, log=print):
    model.select(name)
    t = time.time()
    P = model.make_params()
    st = build_open_device(**cfg)
    H = model.build_H(st, P).tocsr()
    log(f"{name}: {len(st.pos)} atoms, {H.shape[0]} orbitals, nnz(H) = {H.nnz}, {st.nblocks} blocks, "
        f"{time.time() - t:.0f} s")
    H00, H01 = model.lead_cell(st, H, P)
    nl, nb = len(st.meta["lead_L"]), st.nblocks
    edges = dict(Ev_bulk=0.0, Ec_bulk=float(model.bulk_cbm(P)))
    if lead_edge:
        t = time.time()
        edges["Ec_lead"] = float(model.lead_edge(SimpleNamespace(H00=H00, H01=H01)))
        log(f"lead conduction edge {edges['Ec_lead']:.4f} eV ({time.time() - t:.0f} s)")
    m = st.meta
    meta = dict(
        name=f"Si nanosheet {name}", model=name, units=dict(energy="eV", length="Angstrom"),
        orthogonal=True, axis=0, block_length=L_BLOCK, nblocks=nb,
        species=LABELS, orbitals=ORBITALS[name],
        transport=dict(device_blocks=list(range(2, nb - nl + 1)), lead_left_block=1,
                       lead_right_block=nb - nl + 1, lead_cell="lead_right_block, coupling to the next block (+x)"),
        blocks=dict(lead_L=m["lead_L"], dev=m["dev"], lead_R=m["lead_R"], x0=list(map(float, m["x0"]))),
        edges=edges, structure=dict(cfg, nt=list(cfg["nt"]), half_width=list(cfg["half_width"]),
                                    center=list(map(float, m["center"]))),
        params=params_dict(P))
    write_inputs(out, np.array(LABELS)[st.species], st.pos, st.norb, st.block, H,
                 lead=dict(H00=H00, H01=H01), meta=meta)
    np.savez_compressed(f"{out}/structure.npz", pos=st.pos, species=st.species, bonds=st.bonds,
                        block=st.block, site=st.site, **{f"profile{i}": p for i, p in enumerate(m["profiles"])})
    log(f"wrote {out}: " + ", ".join(f"{NAMES[s]} {np.sum(st.species == s)}" for s in range(len(NAMES))))
    return st, H, P


def load_structure(d, meta):
    """Structure + TB parameters of a prepared Si sheet (selects its model, so st.norb is right)."""
    model.select(meta["model"])
    z = np.load(f"{d}/structure.npz")
    s, b = meta["structure"], meta["blocks"]
    md = dict(nt=tuple(s["nt"]), half_width=tuple(s["half_width"]), rough_rms=s["rough_rms"],
              rough_corr=s["rough_corr"], seed=s["seed"], lead_seed=s["lead_seed"], Lx=None,
              center=tuple(s["center"]), x0=np.array(b["x0"]), lead_L=b["lead_L"], dev=b["dev"],
              lead_R=b["lead_R"], profiles=[z[f"profile{i}"] for i in range(4)])
    st = structure.Structure(z["pos"], z["species"], z["bonds"], z["block"], z["site"], meta["nblocks"], md)
    return st, params_from_dict(meta["model"], meta["params"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["sp3s", "sp3d5s"], default="sp3s")
    p.add_argument("--xsec", choices=sorted(model._CFG), default="3x5", help="cross-section preset")
    p.add_argument("--nt", help="override: cells across y,z (e.g. 12,8)")
    p.add_argument("--half-width", help="override: channel half widths y,z [A] (e.g. 25,15)")
    p.add_argument("--n-dev", type=int, help="rough device blocks (default 4)")
    p.add_argument("--rms", type=float, help="roughness rms [A] (default 1.5)")
    p.add_argument("--corr", type=float, help="roughness correlation length [A] (default 8)")
    p.add_argument("--seed", type=int, help="device seed (default 7)")
    p.add_argument("--no-lead-edge", action="store_true", help="skip the lead conduction-edge search")
    p.add_argument("--out", help="output folder (default inputs/si<xsec>_<model>)")
    a = p.parse_args()
    cfg = model.device_cfg(a.xsec)
    for k, v in dict(n_dev_blocks=a.n_dev, rough_rms=a.rms, rough_corr=a.corr, seed=a.seed).items():
        if v is not None:
            cfg[k] = v
    if a.nt:
        cfg["nt"] = tuple(int(x) for x in a.nt.split(","))
    if a.half_width:
        cfg["half_width"] = tuple(float(x) for x in a.half_width.split(","))
    prepare(a.model, cfg, a.out or f"inputs/si{a.xsec}_{a.model}", lead_edge=not a.no_lead_edge)


if __name__ == "__main__":
    main()
