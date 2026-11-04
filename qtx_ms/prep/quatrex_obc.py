"""Lead self-energies of a QuaTrEx input set from QuaTrEx's own OBC solver, written into an mgms Sigma cache.

The drivers then read them instead of folding + Sancho-Rubio (exact_T / run_l2_clean / ms_scan / run_ms
--sigma-cache results_ms/sigma_qtx/<name>). Runs in the QuaTrEx environment with the repo on the path:

    PYTHONPATH=. ~/.conda/envs/quatrex-dev/bin/python -m qtx_ms.prep.quatrex_obc --inputs inputs/mos2_hfo2 --dE 0.05,0.15

QuaTrEx's contacts are the device's own end cells (contact.compute_boundary, spectral OBC, as in QuaTrEx's QTBM):
Sigma_q sits on the contact orbitals and stands for the leads beyond them. The mgms Sigma lives on the device
orbitals next to the lead block (Device.bL / bR), so the lead block l (which must contain the contact) is decimated:
    Sigma = A_dl (A_ll - Sigma_q)^-1 A_ld,   A = z S - H of the input set (its potential, trim, cut), z = E + i eta.
Sigma_q is computed at real E (QuaTrEx A(E) = E S - H - (V S + S V) / 2, potential.npy as loaded by QuaTrEx,
zeroed when the input set was prepared without --potential). Files are named by z like Device.sigma_cache.
Checks: QuaTrEx H, S vs the input set (orbital map perm.npy), channel counts (QuaTrEx injection vs Gamma
eigenvalues > 1e-4 eV of the decimated Sigma), Sigma vs the folded-lead cache results_ms/sigma/<name> where present.
Summary per energy in <out>/qtx_obc.json. Based on extern/generate_inputs.py, without its one-cell contact limit.
"""
import argparse, copy, json, os, time, tomllib
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from qtx_ms.prep.inputs import read_inputs
from qtx_ms import msbasis as mb


def load_config(f, nevp_location="numpy", overrides=()):
    """QuatrexConfig from a (deviceBuilder) toml; keys this QuaTrEx version rejects (e.g. device.broadening) are
    dropped and reported, the NEVP runs on nevp_location (deviceBuilder writes cupy); overrides: "a.b.c=value"
    with a TOML value, e.g. electron.obc.min_decay=1e-3."""
    from pydantic import ValidationError
    from quatrex.core.config import QuatrexConfig, _parse_config
    f = Path(f).resolve()
    d = tomllib.load(open(f, "rb"))
    d.setdefault("compute", {}).setdefault("nevp", {})["eig_compute_location"] = nevp_location
    for o in overrides:
        key, val = o.split("=", 1)
        *path, last = key.strip().split(".")
        x = d
        for k in path:
            x = x.setdefault(k, {})
        x[last] = tomllib.loads(f"v = {val.strip()}")["v"]
    dropped = []
    for _ in range(50):
        try:
            return QuatrexConfig(**_parse_config(f, copy.deepcopy(d))), dropped
        except ValidationError as e:
            extra = [er["loc"] for er in e.errors() if er["type"] == "extra_forbidden"]
            if not extra:
                raise
            for loc in extra:
                x = d
                for k in loc[:-1]:
                    x = x[k]
                x.pop(loc[-1])
                dropped.append(".".join(map(str, loc)))
    raise RuntimeError(f"{f}: config still invalid after dropping {dropped}")


def full_matrix(mats):
    """Gamma-point matrix from QuaTrEx's H(R) / S(R) dict: load_matrices keeps the upper triangle of every R and
    drops R with a transport component, so sum and complete the lower triangle."""
    M = None
    for m in mats.values():
        M = m.tocsr() if M is None else M + m.tocsr()
    return (M + M.conj().T - sp.diags(M.diagonal())).tocsr()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True, help="input set written by qtx_ms.prep.quatrex (needs perm.npy)")
    p.add_argument("--config", help="QuaTrEx config (default <meta source folder>/quatrex_config.toml)")
    p.add_argument("--dE", default="0.05,0.15", help="energies - Ec [eV]")
    p.add_argument("--Ec", type=float, help="reference edge for --dE (default Ec_lead of the inputs)")
    p.add_argument("--eta", type=float, default=1e-6, help="Im z of the decimation and the cache file name "
                   "(the drivers ask for Sigma(E + 1e-6 i))")
    p.add_argument("--out", help="Sigma cache directory (default results_ms/sigma_qtx/<inputs name>)")
    p.add_argument("--folded", help="folded-lead cache to compare with (default results_ms/sigma/<inputs name>)")
    p.add_argument("--nevp-location", default="numpy", help="QuaTrEx eig_compute_location (numpy | cupy)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config entry (TOML value), e.g. --set electron.obc.min_decay=1e-3; repeatable")
    p.add_argument("--redo", action="store_true", help="recompute energies already in --out")
    a = p.parse_args()

    from qttools import xp
    from quatrex.core.config import setup_context
    from quatrex.device import create_device
    assert xp is np, "CPU (numpy) backend expected"

    t0 = time.time()
    name = os.path.basename(os.path.normpath(a.inputs))
    out = a.out or os.path.join("results_ms", "sigma_qtx", name)
    folded = a.folded or os.path.join("results_ms", "sigma", name)
    os.makedirs(out, exist_ok=True)
    inp = read_inputs(a.inputs)
    meta = inp["meta"]
    perm = inp["perm"]
    assert perm is not None, f"{a.inputs}: no perm.npy (not written by qtx_ms.prep.quatrex?)"
    Ec = meta["edges"]["Ec_lead"] if a.Ec is None else a.Ec
    dev = mb.Device(inp)
    n = len(perm)

    cfg_file = a.config or os.path.join(meta["source"]["folder"], "quatrex_config.toml")
    cfg, dropped = load_config(cfg_file, a.nevp_location, a.set)
    if dropped:
        print(f"  {cfg_file}: dropped keys unknown to this QuaTrEx: {dropped}")
    if cfg.qtbm.low_rank_obc:
        raise SystemExit("low_rank_obc: QuaTrEx returns no Sigma, set it to false")
    setup_context(cfg)
    qd = create_device(cfg)
    assert qd.hamiltonians[(0, 0, 0)].shape == (n, n), f"QuaTrEx {qd.hamiltonians[(0, 0, 0)].shape}, inputs {n}"
    print(f"OBC: {cfg.electron.obc}")
    print(f"QuaTrEx device: H keys {list(qd.hamiltonians)}, {len(qd.contacts)} contacts; {time.time() - t0:.0f} s")

    # QuaTrEx H (+ potential term), S in QuaTrEx order; compared with the input set in its order
    Hq, Sq = full_matrix(qd.hamiltonians), full_matrix(qd.overlap_matrices)
    V = np.asarray(qd.potential) if qd.potential is not None else np.zeros(n)
    if meta["source"].get("potential") is None and np.abs(V).max() > 0:
        print(f"  inputs without potential: QuaTrEx potential [{V.min():.3f}, {V.max():.3f}] eV set to 0")
        V = np.zeros(n)
    if np.abs(V).max() > 0:
        D = sp.diags(V)
        Hq = (Hq + 0.5 * (D @ Sq + Sq @ D)).tocsr()
    H, S = inp["H"].tocsr(), inp["S"].tocsr()
    check = dict(max_dH=float(abs(Hq[perm][:, perm] - H).max()), max_dS=float(abs(Sq[perm][:, perm] - S).max()))
    print(f"  QuaTrEx vs input set: max |dH| {check['max_dH']:.2e}, max |dS| {check['max_dS']:.2e} eV "
          f"(trim / cut of the input set show up here)")

    # contacts -> lead blocks of the input set
    q2o = np.empty(n, int)
    q2o[perm] = np.arange(n)
    ob = np.repeat(inp["blocks"], inp["norb"])
    tr = meta["transport"]
    sides = {}
    for c in qd.contacts:
        ny, nz = c.transverse_repetition_grid
        if (ny, nz) != (1, 1):
            raise SystemExit(f"contact {c.name}: transverse repetition grid ({ny}, {nz}) not supported")
        oc = q2o[c.orbital_indices]
        for side, blk, b in (("L", tr["lead_left_block"], dev.bL), ("R", tr["lead_right_block"], dev.bR)):
            lb = np.where(ob == blk)[0]
            if np.isin(oc, lb).all():
                pos = np.searchsorted(lb, oc)
                sides[side] = dict(contact=c, lb=lb, pos=pos, d=dev.dev[b])
                print(f"  contact {c.name}: {len(oc)} orbitals ({c.transport_repetitions} cells) in lead block "
                      f"{blk} ({len(lb)} orbitals) -> Sigma_{side} on {len(b)} device orbitals")
                break
        else:
            raise SystemExit(f"contact {c.name}: orbitals not inside a lead block of the input set "
                             f"(contact longer than the block? prepare the inputs with --cells >= "
                             f"{c.transport_repetitions})")
    assert set(sides) == {"L", "R"}, f"contacts found for {sorted(sides)}"

    fjs = os.path.join(out, "qtx_obc.json")
    summary = json.load(open(fjs)) if os.path.exists(fjs) else {}
    summary.update(inputs=os.path.abspath(a.inputs), config=os.path.abspath(cfg_file), overrides=a.set, Ec=Ec, eta=a.eta,
                   dropped_config_keys=dropped, inputs_check=check, potential=bool(np.abs(V).max() > 0))
    summary.setdefault("E", {})
    for dE in (float(x) for x in a.dE.split(",")):
        E = Ec + dE
        z = complex(E, a.eta)
        f = os.path.join(out, f"sig_{z.real:+.8f}_{z.imag:.8e}.npz")
        if os.path.exists(f) and not a.redo:
            print(f"E = Ec {dE:+.3f}: {f} exists")
            continue
        t = time.time()
        M = (E * Sq - Hq).astype(complex).tocsr()
        Az = (z * S - H).tocsr()
        res, sig = {}, {}
        for side, s in sides.items():
            r = s["contact"].compute_boundary(M, False, [0.0, 0.0, 0.0], return_modes_only=False)
            Sq_c = sum(v[0] for v in r.sigma_obc_k.values())
            lb, pos, d = s["lb"], s["pos"], s["d"]
            All = Az[lb][:, lb].toarray()
            All[np.ix_(pos, pos)] -= Sq_c
            sig[side] = Az[d][:, lb].toarray() @ np.linalg.solve(All, Az[lb][:, d].toarray())
            res[side] = dict(channels_qtx=int(r.injection[0].shape[1]),
                             channels=int(mb.injection(sig[side])[0].shape[1]))
        np.savez(f, SL=sig["L"], SR=sig["R"])
        ff = os.path.join(folded, os.path.basename(f))
        if os.path.exists(ff):
            c = np.load(ff)
            for side in "LR":
                ref = c["S" + side]
                res[side]["rel_diff_folded"] = float(np.linalg.norm(sig[side] - ref) / np.linalg.norm(ref))
        print(f"E = Ec {dE:+.3f}: " + ", ".join(
            f"{side}: channels QuaTrEx {r['channels_qtx']} / Gamma {r['channels']}"
            + (f", |dSigma|/|Sigma| vs folded {r['rel_diff_folded']:.1e}" if "rel_diff_folded" in r else "")
            for side, r in res.items()) + f"; {time.time() - t:.0f} s", flush=True)
        summary["E"][f"{dE:.4f}"] = dict(E=E, time=time.time() - t, **{side: r for side, r in res.items()})
        json.dump(summary, open(fjs, "w"), indent=1)
    print(f"-> {out}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
