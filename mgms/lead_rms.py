"""Lead-cell bands in a slab basis and removal of spurious bands by basis optimization (the qtmode recipe of
qtmode_cms/studies/09_cross_sections/lead_rms.py: Huang, Ilatikhameneh, Povolotskyi, Klimeck, JAP 123, 044303
(2018)), generalized to a non-orthogonal S and complex bases.

Lead cell: H(k) = H00 + H01 e^{ik} + H01^H e^{-ik}, S(k) likewise (k in units of 1 / cell length).
A slab basis V_s (rows = slab orbitals) is put on the lead cell by matching the slab atoms to the cell atoms
(shift by the block index along the transport axis, minimum image across the periodic axis); orbitals of
missing atoms (vacancies) get zero rows. Reduced bands: eig(Phi^H H(k) Phi, Phi^H S(k) Phi).
A reduced state more than tol from every exact state at the same k is spurious.

Optimization: one vector xi = Xi C per iteration (C real), Xi = orth[(1 - Phi Phi^H) H(q) Phi, q = 0, pi],
C minimizing over the nq k with most spurious states
    F(C) = Re sum_q (1 / 2nz) sum_k (z_k - e_c) sigma'(z_k) / sigma(z_k) + (C^T C - 1)^2,
sigma(z) = xi^H (zS - H) xi - xi^H (zS - H) Phi [Phi^H (zS - H) Phi]^-1 Phi^H (zS - H) xi (Schur complement of xi),
z_k on the circle through the window. (1 / 2 pi i) oint sigma'/sigma dz is the change in the number of reduced
eigenvalues inside the circle when xi is added, so minimizing it pushes spurious eigenvalues out of the window.
Later C are kept orthogonal to earlier ones.

    python -m mgms.lead_rms --inputs inputs/mos2_20nm --rho results_ms/mos2_20nm_l2direct/rho_....npz --slab 0
"""
import argparse, json, os, time
import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize


def log(*a):
    print(*a, flush=True)


class LeadCell:
    def __init__(self, lead):
        self.H00, self.H01 = lead["H00"].toarray(), lead["H01"].toarray()
        self.S00, self.S01 = lead["S00"].toarray(), lead["S01"].toarray()
        self.n = len(self.H00)

    def hk(self, k):
        ph = np.exp(1j * k)
        H = self.H00 + self.H01 * ph + self.H01.conj().T / ph
        S = self.S00 + self.S01 * ph + self.S01.conj().T / ph
        return 0.5 * (H + H.conj().T), 0.5 * (S + S.conj().T)

    def bands(self, ks, cache=None):
        """All exact eigenvalues at ks (cached in an npz keyed by the k list)."""
        if cache and os.path.exists(cache):
            z = np.load(cache)
            if np.allclose(z["k"], ks):
                return z["E"]
        E = np.array([sla.eigh(*self.hk(k), eigvals_only=True) for k in ks])
        if cache:
            np.savez(cache, k=ks, E=E)
        return E


def slab_cell_map(inp, dev, s, tol=0.6):
    """Slab-local orbital index for each lead-cell orbital (-1: atom missing in slab s)."""
    tr, meta = inp["meta"]["transport"], inp["meta"]
    xyz, blocks, norb = inp["xyz"], inp["blocks"], inp["norb"]
    off = np.concatenate([[0], np.cumsum(norb)])
    ax, L = meta["axis"], meta["block_length"]
    per = meta.get("periodic")
    cb, sb = tr["lead_left_block"], tr["device_blocks"][s]
    ca, sa = np.where(blocks == cb)[0], np.where(blocks == sb)[0]
    d = xyz[sa][None] - xyz[ca][:, None]
    d[..., ax] -= (sb - cb) * L
    if per:
        d[..., per["axis"]] -= per["length"] * np.round(d[..., per["axis"]] / per["length"])
    dist = np.linalg.norm(d, axis=2)
    j = dist.argmin(1)
    ok = dist[np.arange(len(ca)), j] < tol
    loc = dev.loc
    m = []
    for i, a in enumerate(ca):
        oc = np.arange(off[a], off[a + 1])
        if ok[i]:
            b = sa[j[i]]
            assert norb[b] == norb[a]
            m.append(loc[np.arange(off[b], off[b + 1])])
        else:
            m.append(-np.ones(len(oc), int))
    return np.concatenate(m)


def to_cell(V, cmap):
    P = np.zeros((len(cmap), V.shape[1]), complex)
    P[cmap >= 0] = V[cmap[cmap >= 0]]
    return P


def to_slab(X, cmap, n_slab):
    P = np.zeros((n_slab, X.shape[1]), complex)
    P[cmap[cmap >= 0]] = X[cmap >= 0]
    return P


class RMS:
    def __init__(self, cell, Phi, ks, window, nz=3):
        self.cell, self.Phi, self.ks = cell, Phi, np.asarray(ks)
        self.E1, self.E2 = window
        self.ec, self.rho = 0.5 * (self.E1 + self.E2), 0.5 * (self.E2 - self.E1)
        self.nz = nz
        self.zk = self.ec + self.rho * np.exp(1j * np.pi * (np.arange(1, 2 * nz + 1) - 0.5) / nz)
        R = np.hstack([cell.hk(q)[0] @ Phi for q in (0.0, np.pi)])
        for _ in range(2):
            R -= Phi @ (Phi.conj().T @ R)
        Ux, sx, _ = np.linalg.svd(R, full_matrices=False)
        self.Xi = Ux[:, sx > 1e-8 * sx[0]]
        self.c = {}
        for q in self.ks:
            H, S = cell.hk(q)
            HX, SX = H @ self.Xi, S @ self.Xi
            self.c[q] = dict(h0=Phi.conj().T @ H @ Phi, s0=Phi.conj().T @ S @ Phi, PH=Phi.conj().T @ HX,
                             PS=Phi.conj().T @ SX, KH=self.Xi.conj().T @ HX, KS=self.Xi.conj().T @ SX)
        self.Cs = np.zeros((self.Xi.shape[1], 0))

    def reduced(self, q, Cs=None):
        """h, s of [Phi, Xi Cs] and Phi_ext^H H Xi, Phi_ext^H S Xi at k = q."""
        Cs = self.Cs if Cs is None else Cs
        c = self.c[q]
        if Cs.shape[1] == 0:
            return c["h0"], c["s0"], c["PH"], c["PS"]
        hb, sb = c["PH"] @ Cs, c["PS"] @ Cs
        h = np.block([[c["h0"], hb], [hb.conj().T, Cs.T @ c["KH"] @ Cs]])
        s = np.block([[c["s0"], sb], [sb.conj().T, Cs.T @ c["KS"] @ Cs]])
        return h, s, np.vstack([c["PH"], Cs.T @ c["KH"]]), np.vstack([c["PS"], Cs.T @ c["KS"]])

    def eig(self, q):
        h, s = self.reduced(q)[:2]
        return sla.eigh(0.5 * (h + h.conj().T), 0.5 * (s + s.conj().T))

    def setup(self, q):
        h, s, PH, PS = self.reduced(q)
        e, W = sla.eigh(0.5 * (h + h.conj().T), 0.5 * (s + s.conj().T))
        c = self.c[q]
        return dict(e=e, A=W.conj().T @ PS, B=W.conj().T @ PH, KSr=c["KS"].real, KHr=c["KH"].real)

    def cost(self, C, S):
        Cs = self.Cs
        Ce = C - Cs @ (Cs.T @ C) if Cs.shape[1] else C
        nn = Ce @ Ce
        f, g = (nn - 1) ** 2, 4 * (nn - 1) * Ce
        w = 1 / (2 * self.nz)
        z = self.zk[None, :]
        for d in S:
            e = d["e"][:, None]
            al, be = d["A"] @ Ce, d["B"] @ Ce
            p, qq, r = np.abs(al) ** 2, 2 * np.real(al.conj() * be), np.abs(be) ** 2
            kS, kH = d["KSr"] @ Ce, d["KHr"] @ Ce
            cS, cH = Ce @ kS, Ce @ kH
            u1, u2 = 1 / (z - e), 1 / (z - e) ** 2
            sig = self.zk * cS - cH - ((z ** 2 * p[:, None] - z * qq[:, None] + r[:, None]) * u1).sum(0)
            dsig = cS - ((z ** 2 * p[:, None] - 2 * z * e * p[:, None] + qq[:, None] * e - r[:, None]) * u2).sum(0)
            wk = w * (self.zk - self.ec)
            f += float(np.real(np.sum(wk * dsig / sig)))
            gk, hk = wk / sig, -wk * dsig / sig ** 2
            cp = (gk * (-(z ** 2 - 2 * z * e) * u2) + hk * (-z ** 2 * u1)).sum(1).real
            cq = (gk * (-e * u2) + hk * (z * u1)).sum(1).real
            cr = (gk * u2 + hk * (-u1)).sum(1).real
            cSc, cHc = np.real(np.sum(gk + hk * self.zk)), np.real(np.sum(-hk))
            g += 2 * cSc * kS + 2 * cHc * kH
            g += 2 * np.real(d["A"].T @ (cp * al.conj()))
            g += 2 * np.real(d["B"].T @ (cq * al.conj()) + d["A"].T @ (cq * be.conj()))
            g += 2 * np.real(d["B"].T @ (cr * be.conj()))
        if Cs.shape[1]:
            g = g - Cs @ (Cs.T @ g)
        return f, g

    def flag(self, e, Eex, tol):
        """Reduced eigenvalues in the window and the spurious mask."""
        ee = Eex[(Eex > self.E1 - 0.01) & (Eex < self.E2 + 0.01)]
        wi = e[(e > self.E1) & (e < self.E2)]
        if not len(wi):
            return wi, np.zeros(0, bool)
        return wi, (np.abs(wi[:, None] - ee[None]).min(1) > tol) if len(ee) else np.ones(len(wi), bool)

    def initial(self, q, lam):
        h, s, PH, PS = self.reduced(q)
        e, W = sla.eigh(0.5 * (h + h.conj().T), 0.5 * (s + s.conj().T))
        j = int(np.argmin(np.abs(e - lam)))
        C0 = np.real((PH - e[j] * PS).conj().T @ W[:, j])
        if self.Cs.shape[1]:
            C0 -= self.Cs @ (self.Cs.T @ C0)
        return C0 / np.linalg.norm(C0)

    def run(self, Eex, tol=0.005, nq=3, maxit=100, grad_check=False, log=log):
        """Eex: exact eigenvalues at self.ks. Returns the history; added vectors are Xi @ self.Cs."""
        hist = []
        for it in range(1, maxit + 1):
            t = time.time()
            fl = [self.flag(self.eig(q)[0], Eex[i], tol) for i, q in enumerate(self.ks)]
            nflag = np.array([m.sum() for _, m in fl])
            if nflag.sum() == 0:
                log(f"  iteration {it}: no spurious state in [{self.E1:.3f}, {self.E2:.3f}] on {len(self.ks)} k")
                break
            pick = np.argsort(-nflag)[:nq]
            pick = pick[nflag[pick] > 0]
            S = [self.setup(self.ks[j]) for j in pick]
            wi, m = fl[pick[0]]
            lam = wi[m][np.argmin(np.abs(wi[m] - self.ec))]
            C0 = self.initial(self.ks[pick[0]], lam)
            f0 = self.cost(C0, S)[0]
            if grad_check and it == 1:
                dC = np.random.default_rng(1).normal(size=len(C0)); hh = 1e-6
                fd = (self.cost(C0 + hh * dC, S)[0] - self.cost(C0 - hh * dC, S)[0]) / (2 * hh)
                log(f"  gradient check: analytic {self.cost(C0, S)[1] @ dC:.6e}, finite difference {fd:.6e}")
            r = minimize(self.cost, C0, args=(S,), jac=True, method="L-BFGS-B", options=dict(maxiter=300))
            Ce = r.x - self.Cs @ (self.Cs.T @ r.x) if self.Cs.shape[1] else r.x
            self.Cs = np.hstack([self.Cs, (Ce / np.linalg.norm(Ce))[:, None]])
            hist.append(dict(it=it, flagged=int(nflag.sum()), q=[float(self.ks[j] / np.pi) for j in pick],
                             f0=float(f0), f=float(r.fun), t=time.time() - t))
            log(f"  iteration {it}: {nflag.sum()} spurious on {len(self.ks)} k (targets k/pi "
                f"{np.round(hist[-1]['q'], 3).tolist()}), dF {f0:+.2f} -> {r.fun:+.2f}, {hist[-1]['t']:.1f} s")
        return hist

    def added(self):
        return self.Xi @ self.Cs


def band_stats(e_red, Eex, lo, hi, tol=0.005):
    """Spurious count and p95 / max error of matched exact states in [lo, hi] (meV)."""
    nsp, acc = 0, []
    for w, ee in zip(e_red, Eex):
        w = w[(w >= lo) & (w <= hi)]; ee = ee[(ee >= lo) & (ee <= hi)]
        if len(w) and len(ee):
            nsp += int((np.abs(w[:, None] - ee[None]).min(1) > tol).sum())
            acc += list(np.abs(ee[:, None] - w[None]).min(1))
        elif len(w):
            nsp += len(w)
    acc = np.array(acc) if acc else np.array([np.nan])
    return dict(spurious=nsp, p95_meV=float(1e3 * np.nanpercentile(acc, 95)), max_meV=float(1e3 * np.nanmax(acc)))


def plot_bands(fname, ks, Eex, red, edges, ylim, title, windows=(), energies=(), tol=0.005):
    """Exact lead bands (ink lines) and reduced bands per panel (dots; spurious ones in orange)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, blue, orange, grid = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#e4e3df"
    npan = 1 + len(red)
    fig, axs = plt.subplots(1, npan, figsize=(4.4 * npan + 1.2, 5.6), sharey=True, squeeze=False)
    x = ks / np.pi
    for ip, ax in enumerate(axs[0]):
        ax.set_facecolor("#fcfcfb")
        sel = (Eex > ylim[0] - 0.2) & (Eex < ylim[1] + 0.2)
        for b in range(Eex.shape[1]):
            if sel[:, b].any():
                ax.plot(x, Eex[:, b], color=ink if ip == 0 else "#a3a29b", lw=1.0 if ip == 0 else 0.9, zorder=2)
        for iw, (lo, hi, lab) in enumerate(windows):
            xw = 1.03 + 0.05 * iw                                    # window brackets right of the axis
            ax.plot([xw, xw], [lo, hi], color=blue, lw=2.0, solid_capstyle="butt", clip_on=False, transform=
                    ax.get_yaxis_transform())
            if ip == 0:
                ax.text(xw + 0.01, hi, lab, color=muted, fontsize=7, rotation=90, va="top", ha="left",
                        transform=ax.get_yaxis_transform())
        for k_, v in edges.items():
            ax.axhline(v, color=muted, lw=0.8, ls=(0, (4, 3)), zorder=1)
        if ip == 0:
            for k_, v in edges.items():
                below = v >= np.mean(list(edges.values()))                # label on the gap side
                ax.text(0.5, v - 0.03 if below else v + 0.03, k_, color=muted, fontsize=8,
                        va="top" if below else "bottom", ha="center",
                        bbox=dict(fc="#fcfcfb", ec="none", pad=1))
            for E in energies:
                ax.plot([0.0, 0.03], [E, E], color="#eb6834", lw=1.4, zorder=5)
            if energies:
                ax.text(0.04, min(energies) - 0.06, "test energies", color="#eb6834", fontsize=7, va="top",
                        bbox=dict(fc="#fcfcfb", ec="none", pad=1))
            ax.set_title("exact lead bands", fontsize=10, color=ink, loc="left")
        else:
            lab, er = red[ip - 1]
            xs, ys, sp = [], [], []
            for i, w in enumerate(er):
                w = w[(w > ylim[0]) & (w < ylim[1])]
                d = np.abs(w[:, None] - Eex[i][None]).min(1) if len(w) else np.zeros(0)
                xs += [x[i]] * len(w); ys += list(w); sp += list(d > tol)
            xs, ys, sp = map(np.array, (xs, ys, sp))
            ax.scatter(xs[~sp], ys[~sp], s=9, color=blue, zorder=3, lw=0, label="reduced, matches exact")
            ax.scatter(xs[sp], ys[sp], s=14, color=orange, zorder=4, lw=0, label=f"spurious (> {1e3 * tol:.0f} meV off)")
            if windows:
                lo_, hi_ = windows[0][0], windows[0][1]
                nin = int((sp & (ys > lo_) & (ys < hi_)).sum())
                ax.set_title(f"{lab}: {nin} spurious in window", fontsize=10, color=ink, loc="left")
            else:
                ax.set_title(f"{lab}: {int(sp.sum())} spurious", fontsize=10, color=ink, loc="left")
            if ip == npan - 1:
                ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.set_xlim(0, 1); ax.set_ylim(*ylim)
        ax.set_xlabel("k (π / cell length)", fontsize=9, color=muted)
        ax.grid(color=grid, lw=0.6); ax.set_axisbelow(True)
        for sp_ in ax.spines.values():
            sp_.set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    axs[0][0].set_ylabel("E (eV)", fontsize=9, color=muted)
    fig.suptitle(title, fontsize=11, color=ink, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(fname, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    from mgms.prep.inputs import read_inputs
    from mgms import msbasis as mb
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--rho", help="level-2 density matrix (run_ms cache, direct-l2) for the slab basis")
    p.add_argument("--l2-thr", type=float, default=1e-5)
    p.add_argument("--lead-rho", action="store_true",
                   help="basis from the pristine lead instead: k integral of the window Bloch projectors (--mu), S00 metric")
    p.add_argument("--mu", default="0,0.395", help="level-2 window - Ec_lead [eV] (--lead-rho)")
    p.add_argument("--slab", type=int, default=0)
    p.add_argument("--win", default="-0.8,0.4", help="spurious-state window - Ec_lead [eV]")
    p.add_argument("--nk", type=int, default=49, help="k points on [0, pi] (plot / evaluation)")
    p.add_argument("--kstep", type=int, default=4, help="every kstep-th k point is used for the optimization")
    p.add_argument("--maxit", type=int, default=100)
    p.add_argument("--tol", type=float, default=0.005)
    p.add_argument("--out", default="results_ms/lead_rms")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(os.path.normpath(a.inputs))
    inp = read_inputs(a.inputs)
    ed = inp["meta"]["edges"]
    Ec, Ev = ed["Ec_lead"], ed["Ev_lead"]
    cell = LeadCell(inp["lead"])
    ks = np.linspace(0, np.pi, a.nk)
    t = time.time()
    Eex = cell.bands(ks, cache=os.path.join(a.out, f"{name}_bands_{a.nk}.npz"))
    log(f"{name}: lead cell {cell.n} orbitals, exact bands on {a.nk} k: {time.time() - t:.0f} s; Ev {Ev:.3f}, Ec {Ec:.3f}")
    ylim = (Ev - 0.8, Ec + 1.0)
    tag = f"{name}_bands"
    red, info = [], {}
    if a.rho or a.lead_rho:
        if a.lead_rho:
            m1, m2 = (Ec + float(x) for x in a.mu.split(","))
            Rl = np.zeros((cell.n, cell.n), complex)
            kk = np.linspace(-np.pi, np.pi, 64, endpoint=False)
            for k in kk:
                e, Z = sla.eigh(*cell.hk(k))
                sel = (e > m1) & (e < m2)
                Rl += Z[:, sel] @ Z[:, sel].conj().T / len(kk)
            Phi = mb.level2_basis([Rl], a.l2_thr, S=[cell.S00])[0]
            cmap = np.arange(cell.n)
            a.slab = "lead"
        else:
            dev = mb.Device(inp)
            z = np.load(a.rho)
            V = mb.level2_basis([z[f"r{a.slab}"]], a.l2_thr)[0]
            cmap = slab_cell_map(inp, dev, a.slab)
            Phi = to_cell(V, cmap)
        lo, hi = (Ec + float(x) for x in a.win.split(","))
        ki = np.arange(0, a.nk, a.kstep)
        rms = RMS(cell, Phi, ks, (lo, hi))
        e0 = [rms.eig(q)[0] for q in ks]
        info["start"] = dict(N=Phi.shape[1], window=band_stats(e0, Eex, lo, hi, a.tol),
                             near=band_stats(e0, Eex, Ec - 0.1, Ec + 0.4, a.tol))
        log(f"slab {a.slab}: basis {Phi.shape[1]} ({(cmap < 0).sum()} cell orbitals missing), trial space "
            f"{rms.Xi.shape[1]}; window [{lo:.3f}, {hi:.3f}]: {info['start']}")
        red.append((f"{'lead rho' if a.lead_rho else 'slab ' + str(a.slab)} basis, N {Phi.shape[1]}", e0))
        rms.ks = ks[ki]
        info["hist"] = rms.run(Eex[ki], tol=a.tol, maxit=a.maxit, grad_check=True)
        rms.ks = ks
        e1 = [rms.eig(q)[0] for q in ks]
        info["end"] = dict(N=Phi.shape[1] + rms.Cs.shape[1], window=band_stats(e1, Eex, lo, hi, a.tol),
                           near=band_stats(e1, Eex, Ec - 0.1, Ec + 0.4, a.tol))
        log(f"after {rms.Cs.shape[1]} vectors: {info['end']}")
        red.append((f"+ {rms.Cs.shape[1]} vectors", e1))
        tag = f"{name}_{'lead' if a.lead_rho else 'slab' + str(a.slab)}_thr{a.l2_thr:g}_rms"
        np.savez(os.path.join(a.out, tag + ".npz"), X=rms.added(), cmap=cmap)
        json.dump(info, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
        win = [(lo, hi, "spurious window")]
    else:
        win = [(Ec, Ec + 0.395, "level-2 window"), (Ec - 0.15, Ec + 0.6, "wide window")]
    Etest = [Ec + float(x) for x in "0.01,0.03,0.06,0.10,0.20,0.35".split(",")]
    plot_bands(os.path.join(a.out, tag + ".png"), ks, Eex, red, {"Ec": Ec, "Ev": Ev}, ylim,
               f"{name} lead cell ({cell.n} orbitals, k_y = 0)", win, Etest, a.tol)
    log(f"-> {os.path.join(a.out, tag + '.png')}")
