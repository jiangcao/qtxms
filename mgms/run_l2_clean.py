"""Direct level-2 mode space (slab density matrix of the full orbital device, U1 = I) with spurious-state
cleanup, for devices a lead-derived (CMS) basis cannot represent (e.g. MoS2 on HfO2: oxide atoms, displaced
channel atoms).

  1. V_s: eigenvectors of the contour slab density matrix rho_ss of the open device (window
     [Ec + mu1, Ec + mu2]) with lambda > thr * lambda_max (cache shared with run_ms --direct-l2)
  2. lead: V_s of the slabs next to the leads put on the lead cell (lead_rms.to_cell, rows of atoms the lead
     cell does not have, e.g. oxide, dropped; orthonormalized); spurious reduced lead bands in
     [Ec + win1, Ec + win2] removed by RMS (lead_rms.RMS), added vectors mapped back to the slab
  3. block residual: Ritz pairs of the open coarse system at probe energies E_j,
        (H_c + Sigma_c(E_j)) y = e S_c y,   |Re e - E_j| < dE_probe / 2   (shift-invert Arnoldi on A_c(E_j)),
     fine residual r = (e S - H - Sigma(E_j)) U y. Galerkin gives U^H r = 0, so r is entirely outside the
     basis; split into slab blocks r_s. With |U y| = 1, a pair with max_s |r_s| > res_tol [eV] is spurious
     (the fine operator does not support it) and r_s / |r_s| of each block above res_tol is appended to U_s
     (a Davidson step: the Ritz value of a spurious state leaves, a poorly resolved true state converges).
     Repeated until no pair is flagged.
  2a. --lead-cms: end slabs get the cleaned lead-CMS basis (run_lead_cms) on their lead-matched atoms and the
     level-2 modes restricted to the other atoms (e.g. oxide); interior slabs keep the level-2 modes
  4. per test energy: T of x = U A_c^-1 U^H b against the exact T, and the block residual |(b - A x)_s| / |b|,
     for every basis stage; with --mg also the two-level solve (twolevel.TwoLevel, coarse = the stage basis,
     GMRES(k) smoother, flexible GMRES from x0 = the mode-space solution): iterations, matvecs, T error
  The modes of every stage are saved as <out>/modes/mu<mu>_<stage>_slab<s>.npy (slab-local orbitals x modes).
  --l2-thr / --nmodes take comma lists: the stages 1-2 run for each basis size (rho eigendecomposed once).

    python -m mgms.run_l2_clean --inputs inputs/mos2_hfo2 --exact-from results_ms/mos2_hfo2_leadcms/results.json
    python -m mgms.run_l2_clean --inputs inputs/mos2_hfo2 --nmodes 100,200,400 --no-res --mg \
        --exact-from results_ms/mos2_hfo2_leadcms/results.json
"""
import argparse, json, os, time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mgms.prep.inputs import read_inputs
from mgms import msbasis as mb
from mgms.twolevel import TwoLevel
from mgms.lead_rms import LeadCell, RMS, slab_cell_map, to_cell, to_slab, band_stats, plot_bands

log = mb.log


def coarse_blocks(dev, V):
    """Blocks of U^H H U, U^H S U for U = diag(V_s) (dense V_s on slab-local orbitals)."""
    b, n = dev.sb, len(V)
    out = dict(HD=[], SD=[], HU=[], SU=[])
    for M, kD, kU in ((dev.H, "HD", "HU"), (dev.S, "SD", "SU")):
        for s in range(n):
            X = V[s].conj().T @ (M[b[s]:b[s + 1], b[s]:b[s + 1]] @ V[s])
            out[kD].append(0.5 * (X + X.conj().T))
            if s < n - 1:
                out[kU].append(V[s].conj().T @ (M[b[s]:b[s + 1], b[s + 1]:b[s + 2]] @ V[s + 1]))
    return out


def append(V, X, rcut=1e-3):
    """[V, X] with X orthogonalized against V (twice) and cut to the directions with s > rcut (columns of X
    normalized first)."""
    X = X / np.linalg.norm(X, axis=0)
    for _ in range(2):
        X = X - V @ (V.conj().T @ X)
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    return np.hstack([V, u[:, s > rcut]]), int((s > rcut).sum())


class Coarse:
    """Coarse system of the basis V = [V_s]: blocks, lead projections, S_c, and fine-level residuals."""

    def __init__(self, dev, V):
        self.dev, self.V = dev, V
        self.blk = coarse_blocks(dev, V)
        self.PL, self.PR = V[0][dev.bL], V[-1][dev.bR - dev.sb[-2]]
        n = len(V)
        self.off = np.concatenate([[0], np.cumsum([v.shape[1] for v in V])])
        self.N = int(self.off[-1])
        B = [[None] * n for _ in range(n)]
        for s in range(n):
            B[s][s] = sp.csr_matrix(self.blk["SD"][s])
            if s < n - 1:
                B[s][s + 1] = sp.csr_matrix(self.blk["SU"][s])
                B[s + 1][s] = sp.csr_matrix(self.blk["SU"][s].conj().T)
        self.Sc = sp.bmat(B, format="csr")

    def Ac(self, E, SL, SR):
        return mb.BlockTridiag(self.blk, E, self.PL.conj().T @ SL @ self.PL, self.PR.conj().T @ SR @ self.PR)

    def prolong(self, Y):
        return np.vstack([self.V[s] @ Y[self.off[s]:self.off[s + 1]] for s in range(len(self.V))])

    def restrict(self, X):
        b = self.dev.sb
        return np.vstack([self.V[s].conj().T @ X[b[s]:b[s + 1]] for s in range(len(self.V))])

    def ritz(self, E, SL, SR, k):
        """k eigenpairs of (H_c + Sigma_c(E), S_c) nearest to E: A_c(E)^-1 S_c y = y / (E - e)."""
        Ac = self.Ac(E, SL, SR)
        op = spla.LinearOperator((self.N, self.N), matvec=lambda y: Ac.solve((self.Sc @ y)[:, None])[:, 0],
                                 dtype=complex)
        th, Y = spla.eigs(op, k=min(k, self.N - 2), which="LM", tol=1e-8)
        return E - 1 / th, Y / np.linalg.norm(Y, axis=0)

    def block_norms(self, R):
        b = self.dev.sb
        return np.array([np.linalg.norm(R[b[s]:b[s + 1]], axis=0) for s in range(len(self.V))])


def fine_residual(dev, X, e, SL, SR):
    """(e S - H - Sigma) X, one e per column."""
    R = (dev.S @ X) * e[None] - dev.H @ X
    R[dev.bL] -= SL @ X[dev.bL]
    R[dev.bR] -= SR @ X[dev.bR]
    return R


def cell_bands(cell, P, ks):
    """bands of the lead cell in the basis P (orthonormal columns on the cell orbitals), one list entry per k"""
    out = []
    for q in ks:
        H, S = cell.hk(q)
        h, s = P.conj().T @ H @ P, P.conj().T @ S @ P
        out.append(sla.eigh(0.5 * (h + h.conj().T), 0.5 * (s + s.conj().T), eigvals_only=True))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", help="default results_ms/<inputs name>_l2direct (rho cache of run_ms --direct-l2)")
    p.add_argument("--mu", default="-0.15,0.6", help="level-2 window - Ec_lead [eV]")
    p.add_argument("--npts", type=int, default=24, help="contour points")
    p.add_argument("--l2-thr", default="1e-7", help="occupation threshold(s) lambda / lambda_max, comma list = sweep")
    p.add_argument("--nmodes", help="modes per slab instead of --l2-thr (largest occupations), comma list = sweep")
    p.add_argument("--no-lead", action="store_true", help="skip the lead-band cleanup")
    p.add_argument("--lead-slabs", default="0,-1", help="slabs cleaned against the lead bands")
    p.add_argument("--win", default="-0.6,0.45", help="lead spurious-state window - Ec_lead [eV]")
    p.add_argument("--nk", type=int, default=25, help="k points on [0, pi] for the lead bands")
    p.add_argument("--kstep", type=int, default=2)
    p.add_argument("--lead-maxit", type=int, default=150)
    p.add_argument("--lead-tol", type=float, default=0.005, help="lead: spurious if > tol [eV] from every exact state")
    p.add_argument("--map-tol", type=float, default=0.6, help="max displacement [A] of a matched atom")
    p.add_argument("--bands", help="exact lead bands cache (npz with k, E)")
    p.add_argument("--lead-cms", help="PHI.npy[,RMS.npz] from run_lead_cms: the end slabs (--lead-slabs) get this "
                   "cleaned lead basis on their lead-matched atoms and the level-2 modes restricted to the other "
                   "atoms (stage <spec>_leadcms)")
    p.add_argument("--extra-rcut", type=float, default=1e-2,
                   help="leadcms: keep level-2 directions on the non-lead atoms with singular value > rcut")
    p.add_argument("--no-res", action="store_true", help="skip the block-residual cleanup")
    p.add_argument("--probe", default="-0.1:0.55:0.05", help="probe energies - Ec_lead: lo:hi:step or list")
    p.add_argument("--nritz", type=int, default=60, help="Ritz pairs computed per probe energy")
    p.add_argument("--res-tol", type=float, default=0.05, help="block residual [eV] flagging a spurious pair")
    p.add_argument("--res-im", type=float, default=0.05, help="only pairs with |Im e| < res_im [eV]")
    p.add_argument("--res-add", type=int, default=30, help="max vectors added per slab and iteration")
    p.add_argument("--res-maxit", type=int, default=10)
    p.add_argument("--dE", default="0.02,0.05,0.15,0.3", help="test energies - Ec_lead [eV]")
    p.add_argument("--exact-from", help="results json with T_exact per dE (exact_T --json / run_lead_cms / run_ms)")
    p.add_argument("--Ec", type=float, help="reference edge for --dE (default Ec_lead of the inputs, as in exact_T --Ec)")
    p.add_argument("--mg", action="store_true", help="also solve each stage with twolevel.TwoLevel (GMRES smoother)")
    p.add_argument("--k", type=int, default=10, help="GMRES smoother steps")
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--maxit", type=int, default=100, help="FGMRES iteration cap per channel")
    p.add_argument("--mg-cols", type=int, default=0,
                   help="MG on the n strongest channels only, checked against their exact solutions (0: all)")
    p.add_argument("--mg-stages", help="comma list of stages solved with MG (default all)")
    a = p.parse_args()

    name = os.path.basename(os.path.normpath(a.inputs))
    out = a.out or os.path.join("results_ms", name + "_l2direct")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    inp = read_inputs(a.inputs)
    ed = inp["meta"]["edges"]
    Ec, Ev = ed["Ec_lead"], ed.get("Ev_lead", ed.get("Ev_bulk", ed["Ec_lead"] - 1.0))
    dev = mb.Device(inp)
    dev.sigma_cache = os.path.join("results_ms", "sigma", name)
    nsl = len(dev.ns)
    sigma = lambda E: dev.sigma(E + 1e-6j)
    log(f"{name}: device {dev.n} orbitals, slabs {dev.ns}; Ec {Ec:.4f}, Ev {Ev:.4f} eV")
    stages = {}

    # ------------------------------------------------------------------ 1. direct level 2
    mu = [Ec + float(x) for x in a.mu.split(",")]
    frho = os.path.join(out, f"rho_{mu[0]:.4f}_{mu[1]:.4f}_{a.npts}.npz")
    # leading eigenpairs of rho_ss, largest occupation first; with --nmodes only the top max(nmodes) are
    # computed and cached (eig_<rho>_top<K>.npz)
    K = max(int(n) for n in a.nmodes.split(",")) if a.nmodes else None
    feig = frho.replace("rho_", "eig_").replace(".npz", f"_top{K}.npz") if K else None
    if feig and os.path.exists(feig):
        z = np.load(feig)
        eig = [(z[f"l{s}"], z[f"X{s}"]) for s in range(nsl)]
        log(f"slab rho eigenpairs (top {K}) from {feig}")
    else:
        if os.path.exists(frho):
            z = np.load(frho)
            assert list(z["N1"]) == dev.ns, f"{frho}: not a direct-l2 rho of these slabs"
            rho = [z[f"r{s}"] for s in range(nsl)]
        else:
            t = time.time()
            Us = [sp.identity(n, dtype=complex, format="csr") for n in dev.ns]
            rho = mb.rho_contour(dev, mb.slab_blocks(dev, Us), Us[0][dev.bL].toarray(),
                                 Us[-1][dev.bR - dev.sb[-2]].toarray(), mu[0], mu[1], a.npts)
            np.savez(frho, N1=dev.ns, **{f"r{s}": r for s, r in enumerate(rho)})
            log(f"contour rho [{mu[0]:.4f}, {mu[1]:.4f}] eV, {a.npts} pts: {time.time() - t:.0f} s")
        t = time.time()
        eig = []
        for s in range(nsl):
            R, n = 0.5 * (rho[s] + rho[s].conj().T), dev.ns[s]
            sub = dict(subset_by_index=[n - min(K, n), n - 1], driver="evr") if K else {}
            lam, X = sla.eigh(R, **sub)
            eig.append((lam[::-1], X[:, ::-1]))
            rho[s] = None
        del rho
        if feig:
            np.savez(feig, **{f"l{s}": e[0] for s, e in enumerate(eig)}, **{f"X{s}": e[1] for s, e in enumerate(eig)})
        log(f"slab rho eigendecompositions{f' (top {K})' if K else ''}: {time.time() - t:.0f} s; occupations "
            f"lambda_1 {[f'{e[0][0]:.3g}' for e in eig]}")
    if a.nmodes:
        specs = [(f"n{n}", lambda e, n=int(n): min(n, len(e))) for n in a.nmodes.split(",")]
    else:
        specs = [(f"thr{float(x):g}", lambda e, x=float(x): int(np.sum(e > x * e[0]))) for x in a.l2_thr.split(",")]
    assert a.no_res or len(specs) == 1, "the block-residual cleanup runs on a single basis (one --l2-thr)"
    info = dict(args=vars(a), N={})
    lead_slabs = sorted({int(x) % nsl for x in a.lead_slabs.split(",")})
    if a.lead_cms:
        fphi, *frms_ = a.lead_cms.split(",")
        Bl = np.load(fphi)
        if frms_:
            z = np.load(frms_[0])
            Bl = np.hstack([Bl, z["Xi"] @ z["C"]])
        Bl = np.linalg.qr(Bl)[0]
        lmaps = {s: slab_cell_map(inp, dev, s, tol=a.map_tol, return_info=True) for s in lead_slabs}
        log(f"lead CMS basis {a.lead_cms}: {Bl.shape[1]} vectors; end slabs {lead_slabs}: non-lead atoms "
            f"{ {s: m[1]['extra'] for s, m in lmaps.items()} }")
    if not a.no_lead or a.lead_cms:
        cell = LeadCell(inp["lead"])
        ks = np.linspace(0, np.pi, a.nk)
        Eex = cell.bands(ks, cache=a.bands or os.path.join(out, f"bands_{a.nk}.npz"))
        w1, w2 = (Ec + float(x) for x in a.win.split(","))
        info["lead"] = {}

    for spec, count in specs:
        tag = f"mu{a.mu.replace(',', '_')}_{spec}"
        V = [X[:, :count(lam)] for lam, X in eig]
        stages[spec] = V
        info["N"][spec] = [v.shape[1] for v in V]
        log(f"level 2 [{mu[0]:.3f}, {mu[1]:.3f}] eV, {spec}: N {info['N'][spec]}, total {sum(info['N'][spec])}; "
            f"smallest kept occupation / lambda_1 {[f'{lam[v.shape[1] - 1] / lam[0]:.1e}' for (lam, _), v in zip(eig, V)]}")

        # -------------------------------------------------------------- 2a. lead CMS on the end slabs
        if a.lead_cms:
            Vc = list(V)
            red = []
            for s in lead_slabs:
                cmap = lmaps[s][0]
                Xl = to_slab(Bl, cmap, dev.ns[s])
                u, sv, _ = np.linalg.svd(Xl, full_matrices=False)
                Xl = u[:, sv > 1e-8 * sv[0]]
                other = np.setdiff1d(np.arange(dev.ns[s]), cmap[cmap >= 0])
                Y = np.zeros_like(V[s])
                Y[other] = V[s][other]                      # level-2 modes on the non-lead atoms only
                ne = 0
                if len(other):
                    u, sv, _ = np.linalg.svd(Y, full_matrices=False)
                    ne = int((sv > a.extra_rcut).sum())
                    Xl = np.hstack([Xl, u[:, :ne]])         # disjoint rows: orthogonal to the lead part
                Vc[s] = Xl
                e0 = cell_bands(cell, np.linalg.qr(to_cell(V[s], cmap))[0], ks)
                e1 = cell_bands(cell, np.linalg.qr(to_cell(Xl[:, :Xl.shape[1] - ne], cmap))[0], ks)   # lead part only
                st0, st1 = band_stats(e0, Eex, w1, w2, a.lead_tol), band_stats(e1, Eex, w1, w2, a.lead_tol)
                log(f"slab {s} lead CMS: {Xl.shape[1] - ne} lead + {ne} non-lead-atom vectors (was {V[s].shape[1]}); "
                    f"bands on the lead cell: {st0['spurious']} spurious (p95 {st0['p95_meV']:.3f} meV) -> "
                    f"{st1['spurious']} (p95 {st1['p95_meV']:.3f} meV)")
                red += [(f"slab {s}, level 2 N {V[s].shape[1]}", e0), (f"slab {s}, lead CMS N {Xl.shape[1] - ne}", e1)]
            plot_bands(os.path.join(out, f"lead_bands_{tag}_leadcms.png"), ks, Eex, red, {"Ec": Ec, "Ev": Ev},
                       (Ev - 0.4, Ec + 0.9), f"{name}: end-slab bases on the lead cell, level 2 ({spec}) vs lead CMS",
                       [(w1, w2, "spurious window")], [Ec + float(x) for x in a.dE.split(",")], a.lead_tol)
            stages[spec + "_leadcms"] = Vc
            info["N"][spec + "_leadcms"] = [v.shape[1] for v in Vc]

        # -------------------------------------------------------------- 2b. lead bands (RMS on the level-2 modes)
        if a.no_lead:
            continue
        V = list(V)
        info["lead"][spec] = {}
        red = []
        for s in lead_slabs:
            cmap, mi = slab_cell_map(inp, dev, s, tol=a.map_tol, return_info=True)
            X = to_cell(V[s], cmap)
            u, sv, _ = np.linalg.svd(X, full_matrices=False)
            Phi = u[:, sv > 1e-6 * sv[0]]
            rms = RMS(cell, Phi, ks, (w1, w2))
            e0 = [rms.eig(q)[0] for q in ks]
            st0 = band_stats(e0, Eex, w1, w2, a.lead_tol)
            frms = os.path.join(out, f"rms_slab{s}_{tag}_win{a.win.replace(',', '_')}_tol{a.lead_tol:g}.npz")
            if os.path.exists(frms):
                z = np.load(frms)
                assert z["Xi"].shape == rms.Xi.shape
                rms.Cs = z["C"]
            else:
                log(f"slab {s}: lead-cell atoms unmatched {mi['missing']}, slab atoms without lead image "
                    f"{mi['extra']}; basis on the cell {Phi.shape[1]} of {V[s].shape[1]}; window "
                    f"[{w1:.3f}, {w2:.3f}]: {st0}")
                ki = np.arange(0, a.nk, a.kstep)
                rms.ks = ks[ki]
                hist = rms.run(Eex[ki], tol=a.lead_tol, maxit=a.lead_maxit)
                rms.ks = ks
                np.savez(frms, Xi=rms.Xi, C=rms.Cs, hist=json.dumps(hist))
            e1 = [rms.eig(q)[0] for q in ks]
            st1 = band_stats(e1, Eex, w1, w2, a.lead_tol)
            nadd = 0
            if rms.Cs.shape[1]:
                V[s], nadd = append(V[s], to_slab(rms.added(), cmap, dev.ns[s]))
            info["lead"][spec][s] = dict(N_cell=Phi.shape[1], added=nadd, before=st0, after=st1)
            log(f"slab {s} lead bands: {st0['spurious']} spurious -> {st1['spurious']} with {rms.Cs.shape[1]} "
                f"vectors (p95 {st1['p95_meV']:.3f} meV); N {V[s].shape[1]}")
            red += [(f"slab {s}, N {Phi.shape[1]}", e0), (f"slab {s} + {rms.Cs.shape[1]}", e1)]
        plot_bands(os.path.join(out, f"lead_bands_{tag}.png"), ks, Eex, red, {"Ec": Ec, "Ev": Ev},
                   (Ev - 0.4, Ec + 0.9), f"{name}: direct level-2 slab bases ({spec}) on the lead cell "
                   f"({cell.n} orbitals)", [(w1, w2, "spurious window")], [Ec + float(x) for x in a.dE.split(",")],
                   a.lead_tol)
        stages[spec + "_lead"] = V
        info["N"][spec + "_lead"] = [v.shape[1] for v in V]
    del eig

    # ------------------------------------------------------------------ 3. block residual
    if not a.no_res:
        if ":" in a.probe:
            lo, hi, st = (float(x) for x in a.probe.split(":"))
            probe = np.round(np.arange(lo, hi + 0.5 * st, st), 4)
        else:
            probe = np.array([float(x) for x in a.probe.split(",")])
        half = 0.5 * np.min(np.diff(np.sort(probe))) if len(probe) > 1 else 0.05
        V = list(V)
        fres = os.path.join(out, f"res_{'nolead_' if a.no_lead else ''}{tag}_tol{a.res_tol:g}.npz")
        prev = np.load(fres) if os.path.exists(fres) else None
        if prev is not None and all(np.allclose(prev[f"V{s}"][:, :V[s].shape[1]], V[s]) for s in range(nsl)):
            V = [prev[f"V{s}"] for s in range(nsl)]
            hist = json.loads(str(prev["hist"]))
            log(f"block residual: cached, N {[v.shape[1] for v in V]}")
        else:
            hist = []
            for it in range(1, a.res_maxit + 1):
                t = time.time()
                cs = Coarse(dev, V)
                cand = [[] for _ in range(nsl)]
                nflag, npair, rmax_all = 0, 0, []
                for dE in probe:
                    E = Ec + float(dE)
                    SL, SR = sigma(E)
                    e, Y = cs.ritz(E, SL, SR, a.nritz)
                    keep = (np.abs(e.real - E) < half) & (np.abs(e.imag) < a.res_im)
                    if not keep.any():
                        continue
                    e, Y = e[keep], Y[:, keep]
                    X = cs.prolong(Y)
                    R = fine_residual(dev, X, e, SL, SR)
                    rn = cs.block_norms(R)                                  # (slab, pair)
                    rmax = rn.max(0)
                    rmax_all += list(rmax)
                    npair += len(e)
                    fl = rmax > a.res_tol
                    nflag += int(fl.sum())
                    b = dev.sb
                    for i in np.where(fl)[0]:
                        for s in np.where(rn[:, i] > a.res_tol)[0]:
                            cand[s].append((rn[s, i], R[b[s]:b[s + 1], i]))
                    if it == 1 or fl.any():
                        log(f"    probe Ec {dE:+.3f}: {len(e)} pairs, {int(fl.sum())} flagged; max block residual "
                            f"[eV] {np.array2string(np.sort(rmax)[::-1][:6], precision=3)}; flagged slabs "
                            f"{np.bincount(rn[:, fl].argmax(0), minlength=nsl).tolist() if fl.any() else '-'}")
                rq = np.percentile(rmax_all, [50, 90, 100]) if rmax_all else [np.nan] * 3
                added = []
                for s in range(nsl):
                    if not cand[s]:
                        added.append(0)
                        continue
                    cand[s].sort(key=lambda c: -c[0])
                    X = np.stack([c[1] for c in cand[s][:a.res_add]], axis=1)
                    V[s], k = append(V[s], X)
                    added.append(k)
                hist.append(dict(it=it, pairs=npair, flagged=nflag, added=added, r50=rq[0], r90=rq[1],
                                 rmax=rq[2], t=time.time() - t))
                log(f"  residual iteration {it}: {npair} Ritz pairs in the probe windows, {nflag} flagged "
                    f"(block residual median {rq[0]:.3f}, p90 {rq[1]:.3f}, max {rq[2]:.3f} eV); added {added} -> "
                    f"N {sum(v.shape[1] for v in V)}, {time.time() - t:.0f} s")
                if nflag == 0:
                    break
            np.savez(fres, hist=json.dumps(hist), **{f"V{s}": V[s] for s in range(nsl)})
        stages[list(stages)[-1] + "_res"] = V
        info["res"] = hist
        info["N"][list(stages)[-1]] = [v.shape[1] for v in V]

    # ------------------------------------------------------------------ 4. transmission
    # mode-space modes: modes/<stage>_slab<s>.npy, slab-local orbitals x modes (orthonormal columns)
    dmodes = os.path.join(out, "modes")
    os.makedirs(dmodes, exist_ok=True)
    for k, v in stages.items():
        for s in range(nsl):
            np.save(os.path.join(dmodes, f"mu{a.mu.replace(',', '_')}_{k}_slab{s}.npy"), v[s])
    log(f"modes of stages {list(stages)} -> {dmodes}")
    EcT = Ec if a.Ec is None else a.Ec        # test energies E = EcT + dE; windows stay relative to Ec_lead
    ex = {}
    if a.exact_from:
        z = json.load(open(a.exact_from))
        if z.get("Ec", EcT) != EcT:
            raise SystemExit(f"{a.exact_from}: written with test edge {z['Ec']}, this run uses {EcT}")
        ex = z["E"]
    fout = os.path.join(out, f"results_clean_mu{a.mu.replace(',', '_')}_{'_'.join(sp_ for sp_, _ in specs)}"
                        f"{'_mg' if a.mg else ''}.json")
    res = json.load(open(fout)) if os.path.exists(fout) else {}
    res.update(info)
    res.setdefault("E", {})
    if res.get("Ec", EcT) != EcT:
        raise SystemExit(f"{fout}: written with test edge {res['Ec']}, this run uses {EcT}")
    res["Ec"] = EcT
    cs = {k: Coarse(dev, v) for k, v in stages.items()}
    for dE in (float(x) for x in a.dE.split(",")):
        E = EcT + dE
        key = f"{dE:.4f}"
        r = dict(E=E)
        SL, SR = sigma(E)
        B = mb.injection(SL)[0]
        GR = 1j * (SR - SR.conj().T)
        b = np.zeros((dev.n, B.shape[1]), complex)
        b[dev.bL] = B
        Tof = lambda x: float(np.real(np.einsum("ic,ij,jc->", x[dev.bR].conj(), GR, x[dev.bR])))
        r["channels"] = int(B.shape[1])
        if key in ex and "T_exact" in ex[key]:
            r["T_exact"] = ex[key]["T_exact"]
        else:
            t = time.time()
            so = mb.CSolver(dev.A(E, SL, SR))
            r.update(T_exact=Tof(so.solve(b)), t_exact=time.time() - t)
            so.free()
        log(f"E = Ec {dE:+.3f}: {B.shape[1]} channels, T_exact {r['T_exact']:.5f}")
        nb = np.linalg.norm(b)
        A = dev.A(E, SL, SR) if a.mg else None
        if a.mg:      # MG columns: the n strongest channels (last in B), exact solutions for them as reference
            sel = np.arange(b.shape[1]) if a.mg_cols <= 0 else np.arange(b.shape[1])[::-1][:a.mg_cols]
            if len(sel) < b.shape[1]:
                t = time.time()
                so = mb.CSolver(A)
                r["T_exact_mgcols"] = Tof(so.solve(b[:, sel]))
                so.free()
                log(f"    exact T on the {len(sel)} MG channels: {r['T_exact_mgcols']:.5f}, {time.time() - t:.0f} s")
            else:
                r["T_exact_mgcols"] = r["T_exact"]
            mg_stages = set(a.mg_stages.split(",")) if a.mg_stages else set(cs)
        for k, c in cs.items():
            t = time.time()
            Ac = c.Ac(E, SL, SR)
            x = c.prolong(Ac.solve(c.restrict(b)))
            T = Tof(x)
            Rb = b - fine_residual(dev, x, np.full(x.shape[1], E), SL, SR)
            rs = np.linalg.norm(c.block_norms(Rb), axis=1) / nb
            r[k] = dict(T=T, err=(T - r["T_exact"]) / r["T_exact"], res_blocks=rs.tolist(), t_ms=time.time() - t)
            log(f"    {k:14s} N {c.N:5d}: MS T {T:.5f} ({100 * r[k]['err']:+.3f}%), block residual "
                f"{np.array2string(rs, precision=2)}")
            if not a.mg or k not in mg_stages:
                continue
            # two-level solve: coarse = this basis (Galerkin, x0 = MS solution), GMRES(k) smoother on A
            t = time.time()
            tl = TwoLevel(A, sp.block_diag(c.V, format="csr"), smoother="gmres", k=a.k, coarse=Ac)
            xs, its, nfail = np.zeros((dev.n, len(sel)), complex), [], 0
            for i, c in enumerate(sel):
                xs[:, i], hist, _ = tl.solve(b[:, c], tol=a.tol, maxit=a.maxit)
                its.append(len(hist) - 1); nfail += hist[-1] > a.tol
            T, Tx = Tof(xs), r["T_exact_mgcols"]
            r[k].update(T_mg=T, err_mg=(T - Tx) / Tx, mg_cols=len(sel), its_mg=float(np.mean(its)),
                        maxits_mg=int(max(its)), fail_mg=int(nfail), mv_mg=tl.nmv / len(sel),
                        t_mg=time.time() - t)
            log(f"    {'':14s} MG ({len(sel)} channels): T {T:.5f} ({r[k]['err_mg']:+.1e}), {r[k]['its_mg']:.1f} its (max {max(its)}, "
                f"{nfail} not converged), {r[k]['mv_mg']:.0f} matvecs per channel, {r[k]['t_mg']:.0f} s")
        res["E"][key] = r
        json.dump(res, open(fout, "w"), indent=1)
    log(f"done, {time.time() - t0:.0f} s -> {fout}")


if __name__ == "__main__":
    main()
