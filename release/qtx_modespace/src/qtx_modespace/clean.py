"""Spurious lead bands of an end-slab basis and their removal (RMS: Huang, Ilatikhameneh, Povolotskyi, Klimeck,
JAP 123, 044303 (2018), generalized to a non-orthogonal S and complex bases).

Lead cell: H(k) = H00 + H01 e^{ik} + H01^H e^{-ik}, S(k) likewise. An end-slab basis V_s is put on the lead cell by
matching slab atoms to lead-cell atoms (shift by the block distance along the transport axis, same species, closest
pairs first); rows of slab atoms without a lead image (e.g. oxide) are dropped. Reduced bands:
eig(Phi^H H(k) Phi, Phi^H S(k) Phi); a reduced state more than tol from every exact state at that k is spurious.

Optimization: one vector xi = Xi C per iteration (C real), Xi = orth[(1 - Phi Phi^H) H(q) Phi, q = 0, pi], C minimizing
over the k points with most spurious states
    F(C) = Re sum_q (1 / 2nz) sum_k (z_k - e_c) sigma'(z_k) / sigma(z_k) + (C^T C - 1)^2,
sigma(z) = Schur complement of xi in [Phi, xi]^H (z S - H) [Phi, xi], z_k on the circle through the window:
(1 / 2 pi i) oint sigma'/sigma dz counts the reduced eigenvalues entering the circle when xi is added, so minimizing it
pushes spurious eigenvalues out of the window. Later C are kept orthogonal to earlier ones.
"""
import os, time
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


def slab_cell_map(inp, dev, s, tol=0.6, return_info=False):
    """Slab-local orbital index for each lead-cell orbital (-1: no atom of that species within tol in slab s).
    Lead-cell atoms are shifted by the block distance and matched one-to-one to slab atoms of the same label
    (closest pairs first); slab atoms left unmatched (e.g. oxide) get no basis coefficient.
    return_info: also dict(missing = unmatched cell atoms, extra = unmatched slab atoms per label, max_dist)."""
    tr, meta = inp["meta"]["transport"], inp["meta"]
    xyz, blocks, norb, lab = inp["xyz"], inp["blocks"], inp["norb"], inp["labels"]
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
    dist[lab[ca][:, None] != lab[sa][None]] = np.inf
    pi, pj = np.unravel_index(np.argsort(dist, axis=None), dist.shape)
    keep = dist[pi, pj] < tol
    pi, pj = pi[keep], pj[keep]
    match, used, dmax = -np.ones(len(ca), int), np.zeros(len(sa), bool), 0.0
    for i, j in zip(pi, pj):                                 # greedy one-to-one, closest first
        if match[i] < 0 and not used[j]:
            match[i], used[j] = j, True
            dmax = max(dmax, dist[i, j])
    m = []
    for i, a in enumerate(ca):
        if match[i] >= 0:
            b = sa[match[i]]
            assert norb[b] == norb[a]
            m.append(dev.loc[np.arange(off[b], off[b + 1])])
        else:
            m.append(-np.ones(norb[a], int))
    cmap = np.concatenate(m)
    if not return_info:
        return cmap
    ex = lab[sa][~used]
    info = dict(missing=int((match < 0).sum()), extra={str(k): int(v) for k, v in zip(*np.unique(ex, return_counts=True))},
                max_dist=float(dmax))
    return cmap, info


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


def append(V, X, rcut=1e-3):
    """[V, X] with X orthogonalized against V (twice) and cut to the directions with s > rcut (columns of X
    normalized first)."""
    X = X / np.linalg.norm(X, axis=0)
    for _ in range(2):
        X = X - V @ (V.conj().T @ X)
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    return np.hstack([V, u[:, s > rcut]]), int((s > rcut).sum())


def clean_slab(inp, dev, cell, V, s, ks, Eex, window, tol=0.005, kstep=2, maxit=150, map_tol=0.6, log=log):
    """Remove the spurious lead bands of slab s's basis V (slab-local orbitals x modes) in window = (E1, E2).
    Returns the extended basis and a summary dict."""
    cmap, mi = slab_cell_map(inp, dev, s, tol=map_tol, return_info=True)
    u, sv, _ = np.linalg.svd(to_cell(V, cmap), full_matrices=False)
    Phi = u[:, sv > 1e-6 * sv[0]]
    rms = RMS(cell, Phi, ks, window)
    st0 = band_stats([rms.eig(q)[0] for q in ks], Eex, *window, tol)
    log(f"  slab {s}: lead-cell atoms unmatched {mi['missing']}, slab atoms without lead image {mi['extra']}; "
        f"basis on the cell {Phi.shape[1]} of {V.shape[1]}; window [{window[0]:.3f}, {window[1]:.3f}]: {st0}")
    ki = np.arange(0, len(ks), kstep)
    rms.ks = ks[ki]
    hist = rms.run(Eex[ki], tol=tol, maxit=maxit, log=log)
    rms.ks = ks
    st1 = band_stats([rms.eig(q)[0] for q in ks], Eex, *window, tol)
    nadd = 0
    if rms.Cs.shape[1]:
        V, nadd = append(V, to_slab(rms.added(), cmap, V.shape[0]))
    log(f"  slab {s} lead bands: {st0['spurious']} spurious -> {st1['spurious']} with {rms.Cs.shape[1]} vectors "
        f"(p95 {st1['p95_meV']:.3f} meV); N {V.shape[1]}")
    return V, dict(N_cell=Phi.shape[1], vectors=int(rms.Cs.shape[1]), added=nadd, before=st0, after=st1,
                   map=mi, iterations=len(hist))
