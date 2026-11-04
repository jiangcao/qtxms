"""QuaTrEx QTBM (extern/generate_inputs.py style) vs mgms QTBM on one input set: injection and psi.

Two stages, since QuaTrEx and pypardiso live in different environments:

    PYTHONPATH=. ~/.conda/envs/quatrex-dev/bin/python -m qtx_ms.prep.compare_qtbm obc --inputs inputs/mos2_hfo2 --dE 0.05,0.15
    .venv/bin/python -m qtx_ms.prep.compare_qtbm compare --inputs inputs/mos2_hfo2 --dE 0.05,0.15

obc: QuaTrEx contact.compute_boundary at real E (A = E S - H - (V S + S V) / 2) for both contacts; saves the injection
  columns B_q (= -A_ji b_inj, flux-normalized modes), Sigma_q and the contact orbitals (input order) to
  <out>/obc_<E>.npz.
compare: per energy
  - contact level: Gamma_q = i (Sigma_q - Sigma_q^H) on the contact; channels (eigenvalues > tol), |B_q B_q^H - Gamma_q|,
    B_q = X U fit on the eigenchannels X sqrt(lam) (span residual, U^H U - I).
  - QuaTrEx system: full input-order A - Sigma_q on the contact orbitals, psi_q = A^-1 B_q (left), T_q on the right contact.
  - mgms system: device orbitals, Sigma(E + i eta) = decimated Sigma_q (results_ms/sigma_qtx/<name>, from
    qtx_ms.prep.quatrex_obc --eta; --eta 0 = real E as QuaTrEx),
    B_m = Gamma_L eigenchannels, psi_m, T_m.
  - psi_q restricted to the device vs psi_m: psi_m U = psi_q fit (span residual, U^H U - I), T_q vs T_m.
psi_q (all orbitals, input order), psi_m (device orbitals dev) in <out>/psi_<E>.npz; summary in <out>/compare.json.
"""
import argparse, json, os, time
import numpy as np
import scipy.sparse as sp

from qtx_ms.prep.inputs import read_inputs


def fit(X, Y):
    """Y ~ X U: relative residual, |U^H U - I| (U unitary on Y's columns means the same flux normalization),
    singular values of U."""
    U = np.linalg.lstsq(X, Y, rcond=None)[0]
    res = np.linalg.norm(X @ U - Y) / np.linalg.norm(Y)
    dev = np.linalg.norm(U.conj().T @ U - np.eye(U.shape[1]))
    return dict(span_res=float(res), unitary_dev=float(dev), sv_U=np.linalg.svd(U, compute_uv=False).tolist())


def stage_obc(a, inp, out, Ec):
    from qtx_ms.prep.quatrex_obc import load_config, full_matrix
    from qttools import xp
    from quatrex.core.config import setup_context
    from quatrex.device import create_device
    assert xp is np, "CPU (numpy) backend expected"
    meta, perm = inp["meta"], inp["perm"]
    n = len(perm)
    cfg_file = a.config or os.path.join(meta["source"]["folder"], "quatrex_config.toml")
    cfg, dropped = load_config(cfg_file, "numpy", a.set)
    setup_context(cfg)
    qd = create_device(cfg)
    Hq, Sq = full_matrix(qd.hamiltonians), full_matrix(qd.overlap_matrices)
    V = np.asarray(qd.potential) if qd.potential is not None else np.zeros(n)
    if meta["source"].get("potential") is None:
        V = np.zeros(n)                     # same rule as quatrex_obc: inputs without potential
    if np.abs(V).max() > 0:
        D = sp.diags(V)
        Hq = (Hq + 0.5 * (D @ Sq + Sq @ D)).tocsr()
    q2o = np.empty(n, int)
    q2o[perm] = np.arange(n)
    ob = np.repeat(inp["blocks"], inp["norb"])
    tr = meta["transport"]
    sides = {}
    for c in qd.contacts:
        oc = q2o[c.orbital_indices]
        for side, blk in (("L", tr["lead_left_block"]), ("R", tr["lead_right_block"])):
            if np.isin(oc, np.where(ob == blk)[0]).all():
                sides[side] = (c, oc)
    assert set(sides) == {"L", "R"}, f"contacts found for {sorted(sides)}"
    for dE in (float(x) for x in a.dE.split(",")):
        E = Ec + dE
        f = os.path.join(out, f"obc_{E:+.8f}.npz")
        if os.path.exists(f) and not a.redo:
            print(f"E = Ec {dE:+.3f}: {f} exists", flush=True)
            continue
        t = time.time()
        M = (E * Sq - Hq).astype(complex).tocsr()
        d = {}
        for side, (c, oc) in sides.items():
            r = c.compute_boundary(M, False, [0.0, 0.0, 0.0], return_modes_only=False)
            d["o" + side] = oc
            d["B" + side] = np.asarray(r.injection[0])
            d["S" + side] = sum(np.asarray(v[0]) for v in r.sigma_obc_k.values())
        np.savez(f, E=E, **d)
        print(f"E = Ec {dE:+.3f}: injected L {d['BL'].shape[1]} / R {d['BR'].shape[1]}; {time.time() - t:.0f} s",
              flush=True)


def stage_compare(a, inp, out, Ec):
    from qtx_ms import msbasis as mb
    name = os.path.basename(os.path.normpath(a.inputs))
    dev = mb.Device(inp)
    dev.sigma_cache = a.sigma_cache or os.path.join("results_ms", "sigma_qtx", name)
    H, S = inp["H"].tocsr(), inp["S"].tocsr()
    N = H.shape[0]
    fjs = os.path.join(out, "compare.json")
    summary = json.load(open(fjs)) if os.path.exists(fjs) else {}
    summary.update(inputs=os.path.abspath(a.inputs), Ec=Ec, sigma_cache=dev.sigma_cache, tol=a.tol, eta=a.eta)
    summary.setdefault("E", {})
    for dE in (float(x) for x in a.dE.split(",")):
        E = Ec + dE
        f = os.path.join(a.obc or out, f"obc_{E:+.8f}.npz")
        if not os.path.exists(f):
            print(f"E = Ec {dE:+.3f}: {f} missing (run the obc stage)", flush=True)
            continue
        t = time.time()
        c = np.load(f)
        r = dict(E=E)

        # contact level: QuaTrEx injection vs Gamma_q eigenchannels
        G = {}
        for side in "LR":
            Sg, B = c["S" + side], c["B" + side]
            Gm = 1j * (Sg - Sg.conj().T)
            G[side] = 0.5 * (Gm + Gm.conj().T)
            lam, X = np.linalg.eigh(G[side])
            k = lam > a.tol
            Xc = X[:, k] * np.sqrt(lam[k])
            r[side] = dict(inj_qtx=int(B.shape[1]), channels_gamma=int(k.sum()),
                           gamma_top=np.sort(lam)[::-1][:30].tolist(),
                           BBh_vs_gamma=float(np.linalg.norm(B @ B.conj().T - G[side]) / np.linalg.norm(G[side])),
                           contact_fit=fit(Xc, B) if B.shape[1] else None)

        # QuaTrEx system: all orbitals, Sigma_q on the contacts, left injection
        rows, cols, vals = [], [], []
        for side in "LR":
            o = c["o" + side]
            ii, jj = np.meshgrid(o, o, indexing="ij")
            rows.append(ii.ravel()); cols.append(jj.ravel()); vals.append(c["S" + side].ravel())
        Sig = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
        oL, oR = c["oL"], c["oR"]
        bq = np.zeros((N, c["BL"].shape[1]), complex)
        bq[oL] = c["BL"]
        cs = mb.CSolver((E * S - H - Sig).tocsr())
        psi_q = cs.solve(bq)
        res_q = cs.res
        cs.free()
        Tq_col = np.real(np.einsum("ic,ij,jc->c", psi_q[oR].conj(), G["R"], psi_q[oR]))

        # mgms system: device orbitals, decimated QuaTrEx Sigma, Gamma_L eigenchannels
        SL, SR = dev.sigma(E + 1j * a.eta)
        Bm = mb.injection(SL, a.tol)[0]
        bm = np.zeros((dev.n, Bm.shape[1]), complex)
        bm[dev.bL] = Bm
        cs = mb.CSolver(dev.A(E, SL, SR))
        psi_m = cs.solve(bm)
        res_m = cs.res
        cs.free()
        GR = 1j * (SR - SR.conj().T)
        Tm_col = np.real(np.einsum("ic,ij,jc->c", psi_m[dev.bR].conj(), GR, psi_m[dev.bR]))
        # QuaTrEx psi on the device orbitals, T with the mgms Gamma_R (same physics if Sigma's agree)
        pq = psi_q[dev.dev]
        np.savez(os.path.join(out, f"psi_{E:+.8f}.npz"), E=E, psi_q=psi_q, psi_m=psi_m, dev=dev.dev)
        Tq_dev = np.real(np.einsum("ic,ij,jc->c", pq[dev.bR].conj(), GR, pq[dev.bR]))

        r.update(T_qtx=float(Tq_col.sum()), T_qtx_cols=Tq_col.tolist(), T_qtx_devGamma=float(Tq_dev.sum()),
                 T_mgms=float(Tm_col.sum()), T_mgms_cols=Tm_col.tolist(), res_qtx=float(res_q), res_mgms=float(res_m),
                 psi_fit=fit(psi_m, pq) if pq.shape[1] else None,
                 norm2_qtx=float(np.linalg.norm(pq) ** 2), norm2_mgms=float(np.linalg.norm(psi_m) ** 2),
                 time=time.time() - t)
        print(f"E = Ec {dE:+.3f}: L inj QuaTrEx {r['L']['inj_qtx']} / Gamma {r['L']['channels_gamma']}, "
              f"|BB^H - Gamma|/|Gamma| {r['L']['BBh_vs_gamma']:.2e}; contact fit {r['L']['contact_fit']}\n"
              f"   T QuaTrEx {r['T_qtx']:.5f} (dev Gamma_R {r['T_qtx_devGamma']:.5f}), T mgms {r['T_mgms']:.5f}; "
              f"psi fit {r['psi_fit']}; |psi|^2 dev QuaTrEx {r['norm2_qtx']:.4e} mgms {r['norm2_mgms']:.4e}; "
              f"{r['time']:.0f} s", flush=True)
        summary["E"][f"{dE:.4f}"] = r
        json.dump(summary, open(fjs, "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["obc", "compare"])
    p.add_argument("--inputs", required=True, help="input set written by qtx_ms.prep.quatrex (needs perm.npy)")
    p.add_argument("--config", help="QuaTrEx config (default <meta source folder>/quatrex_config.toml)")
    p.add_argument("--dE", default="0.05,0.15", help="energies - Ec [eV]")
    p.add_argument("--Ec", type=float, help="reference edge for --dE (default Ec_lead of the inputs)")
    p.add_argument("--tol", type=float, default=1e-4, help="Gamma eigenvalue threshold for channels [eV]")
    p.add_argument("--sigma-cache", help="decimated QuaTrEx Sigma for mgms (default results_ms/sigma_qtx/<name>)")
    p.add_argument("--eta", type=float, default=1e-6, help="mgms Sigma at E + i eta (cache file name; 0 = real E as "
                   "QuaTrEx, needs qtx_ms.prep.quatrex_obc --eta 0)")
    p.add_argument("--obc", help="directory of the obc stage files (default --out)")
    p.add_argument("--out", help="default results_ms/qtbm_cmp/<inputs name>")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="QuaTrEx config override")
    p.add_argument("--redo", action="store_true")
    a = p.parse_args()
    inp = read_inputs(a.inputs)
    assert inp["perm"] is not None, f"{a.inputs}: no perm.npy"
    name = os.path.basename(os.path.normpath(a.inputs))
    out = a.out or os.path.join("results_ms", "qtbm_cmp", name)
    os.makedirs(out, exist_ok=True)
    Ec = inp["meta"]["edges"]["Ec_lead"] if a.Ec is None else a.Ec
    (stage_obc if a.stage == "obc" else stage_compare)(a, inp, out, Ec)


if __name__ == "__main__":
    main()
