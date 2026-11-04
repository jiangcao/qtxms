"""Nonlinear OBC eigenproblem T(z) v = 0, T(z) = z S - H - Sigma_L(z) - Sigma_R(z), by Beyn's contour integral method,
and the span of its eigenvectors (resonant / bound states) against a mode-space basis U1.

Second sheet.  The resonances of the open device lie below the real axis on the analytic continuation of the
retarded Sigma^R(E + i0) (the physical-sheet Sigma(z) of the decimation is analytic off the real axis and has no
device eigenvalues there).  The continuation is built from the lead Bloch modes of the folded chain
(FoldedLead: A00 = A_eff(z), A01, A10 on the faces F):
    F_R = -g_R A10   (psi_{n+1} = F_R psi_n, right-extending chain, eigenvalues lambda, |lambda| < 1)
    F_L = -g_L A01   (psi_{n-1} = F_L psi_n, left-extending chain, eigenvalues mu = 1/lambda)
from one Sancho decimation at z.  Below the axis (Im z < 0) the physical right set contains the propagating
modes with v < 0 (|lambda| = exp(-gamma/|v|) < 1) where the retarded sheet needs the v > 0 ones (|lambda| > 1,
found in the left set), and vice versa for the left lead.  Near-propagating modes (|ln|lambda|| < kappa) are
classified by their flux J = 2 Im(lambda phi^H A01(Re z) phi) / phi^H phi, and swapped:
    F_2 = F + (Phi' L' - F Phi') (W_K^H Phi')^-1 W_K^H
(W_K = left eigenvectors of the removed modes K, Phi', L' = the modes put in), g_R2 = (A00 + A01 F_R2)^-1,
g_L2 = (A00 + A10 F_L2)^-1.  Above the axis the physical sheet is the retarded one (no swap; checked: the rule
then finds nothing to swap).  The continuation is analytic between lead band edges only (branch points on the
real axis), so a Beyn contour should sit inside one interval between consecutive lead subband edges.

Beyn (block Hankel, kmom moments):  A_p = (1/2 pi i) oint z^p T(z)^-1 V dz,  B0 = [A_{i+j}], B1 = [A_{i+j+1}],
B0 = W0 s V0^H (rank k), eig of W0^H B1 V0 s^-1 -> lambda, eigenvectors = W0[:n] y.

CLI (per-point solves are cached, so a run can be resumed / extended with more probes or points):
  python -m qtx_ms.obc_eig --inputs inputs/mos2_hfo2 --center 0.53 --a 0.03 --b 0.02 --npts 24 --probes 300 \
      --U1 results_ms/mos2_hfo2_vcycle/tobc_t1e-6_lc --out results_ms/obc_eig/mos2_hfo2_c0.53
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.linalg as sla

from . import msbasis as mb

log = mb.log


# ------------------------------------------------------------------ continued (retarded-sheet) lead Sigma
def folded_blocks(lead, z):
    """A00 = A_eff(z) (interior folded onto the faces F), A01, A10 of the folded chain (FoldedLead general path)."""
    assert not lead.fast, "continued sheet needs the general FoldedLead path (fast=False)"
    A00 = z * lead.SFF - lead.HFF
    if len(lead.I):
        cs = mb.CSolver(z * lead.SII - lead.HII)
        X = cs.solve(z * lead.SIF - lead.HIF)
        cs.free()
        A00 = A00 - (z * lead.SFI - lead.HFI) @ X
    A01 = z * lead.S01 - lead.H01
    A10 = z * lead.S01.conj().T - lead.H01.conj().T
    return A00, A01, A10


def _flux(lam, Phi, A01E):
    """J = 2 Im(lambda phi^H A01(E) phi) / phi^H phi per column (positive = towards +x, see check_flux_sign)."""
    num = np.einsum("ij,ij->j", Phi.conj(), A01E @ Phi)
    return 2.0 * np.imag(lam * num) / np.einsum("ij,ij->j", Phi.conj(), Phi).real


def _modes(F, zero=1e-10):
    w, vl, vr = sla.eig(F, left=True, right=True, check_finite=False)
    nz = np.abs(w) > zero
    return w[nz], vl[:, nz], vr[:, nz]


def _swap(F, Wk, Pin, Lin):
    """F with the invariant subspace of the left eigenvectors Wk replaced by the modes (Pin, Lin)."""
    if Pin.shape[1] == 0:
        return F
    C = Wk.conj().T @ Pin
    return F + (Pin * Lin - F @ Pin) @ np.linalg.solve(C, Wk.conj().T)


def continued_gfs(lead, z, kappa=0.3, jrel=1e-6, nprop=None, info=None):
    """Surface Green's functions g_L, g_R (on F) on the sheet continued from the retarded Sigma^R(E + i0).
    nprop = (n_plus, n_minus): number of right- / left-moving propagating channels in the band interval of the
    contour (channel_counts); then exactly that many modes are swapped (the ones with the wrong flux sign and
    smallest |ln|lambda||), which keeps Sigma_2 analytic over the whole contour. Without nprop: every mode with
    |ln|lambda|| < kappa and the wrong flux sign. info (dict) receives the mode counts of the swap."""
    z = complex(z)
    A00, A01, A10 = folded_blocks(lead, z)
    gL, gR = mb.sancho_both(A00, A01, A10)
    if z.imag >= 0:
        if info is not None:
            info.update(swapped=0)
        return gL, gR
    FR, FL = -gR @ A10, -gL @ A01
    lR, WR, PR = _modes(FR)                  # lambda (right-decaying set)
    mL, WL, PL = _modes(FL)                  # mu = 1/lambda (left-decaying set)
    A01E = z.real * lead.S01 - lead.H01
    JR, JL = _flux(lR, PR, A01E), _flux(1.0 / mL, PL, A01E)
    jmax = max(np.abs(JR).max(initial=0), np.abs(JL).max(initial=0))
    lnR, lnL = np.abs(np.log(np.abs(lR))), np.abs(np.log(np.abs(mL)))
    cR = np.where(JR < -jrel * jmax)[0]                      # v < 0 in the right set: candidates to remove
    cL = np.where(JL > jrel * jmax)[0]                       # v > 0 in the left set
    cR, cL = cR[np.argsort(lnR[cR])], cL[np.argsort(lnL[cL])]
    if nprop is None:
        kR, kL = cR[lnR[cR] < kappa], cL[lnL[cL] < kappa]
    else:
        kR, kL = cR[:nprop[1]], cL[:nprop[0]]
    FR2 = _swap(FR, WR[:, kR], PL[:, kL], 1.0 / mL[kL])
    FL2 = _swap(FL, WL[:, kL], PR[:, kR], 1.0 / lR[kR])
    if info is not None:
        nxt = lambda c, ln, k: float(ln[c[len(k)]]) if len(c) > len(k) else float("inf")
        info.update(swapped=len(kR), swapped_L=len(kL),
                    ln_max_swapped=float(max(lnR[kR].max(initial=0), lnL[kL].max(initial=0))),
                    ln_next=float(min(nxt(cR, lnR, kR), nxt(cL, lnL, kL))))
    if len(kR) != len(kL) and nprop is None:
        log(f"  continued_gfs z={z:.5f}: {len(kR)} modes out of the right set, {len(kL)} out of the left "
            "(branch point nearby?)")
    gR2 = np.linalg.inv(A00 + A01 @ FR2)
    gL2 = np.linalg.inv(A00 + A10 @ FL2)
    return gL2, gR2


def channel_counts(lead, E, eta=1e-4, kappa=0.05):
    """(n_plus, n_minus): propagating channels moving towards +x / -x at E (lead modes at E + i eta)."""
    JR, JL = check_flux_sign(lead, E, eta, kappa)
    assert (JR > 0).all() and (JL < 0).all(), "flux sign check failed"
    return len(JR), len(JL)


def check_flux_sign(lead, E, eta=1e-4, kappa=0.3):
    """At z = E + i eta the right set holds the outgoing (v > 0) propagating modes: their J must be > 0."""
    z = E + 1j * eta
    A00, A01, A10 = folded_blocks(lead, z)
    gL, gR = mb.sancho_both(A00, A01, A10)
    lR, _, PR = _modes(-gR @ A10)
    mL, _, PL = _modes(-gL @ A01)
    A01E = E * lead.S01 - lead.H01
    JR, JL = _flux(lR, PR, A01E), _flux(1.0 / mL, PL, A01E)
    nR, nL = np.abs(np.log(np.abs(lR))) < kappa, np.abs(np.log(np.abs(mL))) < kappa
    return JR[nR], JL[nL]


class ContinuedSigma:
    """Sigma_L, Sigma_R of a msbasis.Device on the continued retarded sheet; cached in cache/sig2_<z>.npz."""

    def __init__(self, dev, kappa=0.3, cache=None, nprop=None, nprop_R=None):
        self.dev, self.kappa, self.cache = dev, kappa, cache
        self.nprop, self.nprop_R = nprop, (nprop if nprop_R is None else nprop_R)
        self.info = {}

    def key(self):
        return f"k{self.kappa:g}" if self.nprop is None else f"n{self.nprop[0]}-{self.nprop[1]}"

    def __call__(self, z):
        z = complex(z)
        f = self.cache and os.path.join(self.cache, f"sig2_{self.key()}_{z.real:+.8f}_{z.imag:+.8e}.npz")
        if f and os.path.exists(f):
            c = np.load(f)
            self.info = json.loads(str(c["info"]))
            return c["SL"], c["SR"]
        d = self.dev
        info = {}
        gL, gR = continued_gfs(d.lead, z, self.kappa, nprop=self.nprop, info=info)
        if d.leadR is not d.lead:
            iR = {}
            gR = continued_gfs(d.leadR, z, self.kappa, nprop=self.nprop_R, info=iR)[1]
            info["right_lead"] = iR
        out = []
        for g, (Hld, Sld, Hdl, Sdl) in ((gL, d.cL), (gR, d.cR)):
            out.append((z * Sdl - Hdl) @ g @ (z * Sld - Hld))
        self.info = info
        if f:
            os.makedirs(self.cache, exist_ok=True)
            np.savez(f, SL=out[0], SR=out[1], info=json.dumps(info))
        return tuple(out)


# ------------------------------------------------------------------ Beyn
def ellipse(center, a, b, npts):
    th = 2 * np.pi * (np.arange(npts) + 0.5) / npts
    z = center + a * np.cos(th) + 1j * b * np.sin(th)
    dz = (-a * np.sin(th) + 1j * b * np.cos(th)) * (2 * np.pi / npts)
    return z, dz / (2j * np.pi)                          # nodes, weights of (1/2 pi i) oint f dz


def dense_blocks(dev):
    """Dense slab blocks HD, SD (s, s), HU, SU (s, s+1) of the fine device H, S (for msbasis.BlockTridiag)."""
    b, n = dev.sb, len(dev.ns)
    out = dict(HD=[], SD=[], HU=[], SU=[])
    for M, kD, kU in ((dev.H, "HD", "HU"), (dev.S, "SD", "SU")):
        for s_ in range(n):
            out[kD].append(M[b[s_]:b[s_ + 1], b[s_]:b[s_ + 1]].toarray())
            if s_ < n - 1:
                out[kU].append(M[b[s_]:b[s_ + 1], b[s_ + 1]:b[s_ + 2]].toarray())
    return out


def block_solver(dev, blk, z, SL, SR):
    """Dense block-tridiagonal LU of T(z) = z S - H - Sigma_L - Sigma_R (Sigma on the end slabs)."""
    SL1 = np.zeros((dev.ns[0], dev.ns[0]), complex)
    SL1[np.ix_(dev.bL, dev.bL)] = SL
    SR1 = np.zeros((dev.ns[-1], dev.ns[-1]), complex)
    r = dev.bR - dev.sb[-2]
    SR1[np.ix_(r, r)] = SR
    return mb.BlockTridiag(blk, z, SL1, SR1)


def solve_point(dev, sig, z, V, cache=None, tag="", blk=None, ncheck=4):
    """X = T(z)^-1 V (cached in cache/X_<tag>_<z>.npy). blk (dense_blocks): dense block-tridiagonal LU, else PARDISO
    (CSolver). The residual is checked on ncheck columns with the sparse T(z)."""
    f = cache and os.path.join(cache, f"X{tag}_{z.real:+.8f}_{z.imag:+.8e}.npy")
    if f and os.path.exists(f):
        X = np.load(f)
        if X.shape == V.shape:
            return X
    t0 = time.time()
    SL, SR = sig(z)
    t1 = time.time()
    A = dev.A(z, SL, SR)
    if blk is not None:
        X = block_solver(dev, blk, z, SL, SR).solve(V)
        c = np.arange(min(ncheck, V.shape[1]))
        res = float((np.linalg.norm(A @ X[:, c] - V[:, c], axis=0) / np.linalg.norm(V[:, c], axis=0)).max())
    else:
        cs = mb.CSolver(A)
        X = cs.solve(V)
        res = cs.res
        cs.free()
    log(f"  z = {z.real:.5f}{z.imag:+.5f}i: Sigma {t1 - t0:.0f} s, solve {time.time() - t1:.0f} s, res {res:.1e}, "
        f"sheet {json.dumps(sig.info)}")
    if f:
        os.makedirs(cache, exist_ok=True)
        np.save(f, X)
    return X


def beyn(Xs, zs, ws, kmom=1, rank_tol=1e-8, center=0.0, scale=1.0):
    """Eigenpairs from the solves Xs[j] = T(z_j)^-1 V. Moments in the shifted, scaled variable (z - c)/scale."""
    n, m = Xs[0].shape
    A = [sum(w * ((z - center) / scale) ** p * X for z, w, X in zip(zs, ws, Xs)) for p in range(2 * kmom)]
    B0 = np.block([[A[i + j] for j in range(kmom)] for i in range(kmom)])
    B1 = np.block([[A[i + j + 1] for j in range(kmom)] for i in range(kmom)])
    W, s, Vh = np.linalg.svd(B0, full_matrices=False)
    k = int((s > rank_tol * s[0]).sum())
    W, sk, Vk = W[:, :k], s[:k], Vh[:k].conj().T
    M = (W.conj().T @ B1 @ Vk) / sk
    lam, Y = np.linalg.eig(M)
    vec = W[:n] @ Y
    vec /= np.linalg.norm(vec, axis=0)
    return center + scale * lam, vec, s, k


def residuals(dev, sig, lam, vec):
    """|T(lambda) v| / (|T|_F-ish |v|) at each eigenvalue (one Sigma per eigenvalue: expensive)."""
    out = []
    for l, v in zip(lam, vec.T):
        SL, SR = sig(l)
        A = dev.A(l, SL, SR)
        out.append(np.linalg.norm(A @ v) / (abs(A).max() * np.linalg.norm(v)))
    return np.array(out)


# ------------------------------------------------------------------ span against U1
def load_U1(prefix, ns):
    return [np.load(f"{prefix}_U1_slab{s}.npy") for s in range(ns)]


def span_missing(dev, Us, vec, S_metric=False):
    """Per eigenvector: |(I - P) v| / |v| with P the slab-block projector on span(U1) (Euclidean, U1 orthonormal per
    slab), and the per-slab share of the missing part."""
    sb = dev.sb
    miss_s = np.zeros((len(Us), vec.shape[1]))
    for s, U in enumerate(Us):
        v = vec[sb[s]:sb[s + 1]]
        r = v - U @ (U.conj().T @ v)
        miss_s[s] = np.linalg.norm(r, axis=0) ** 2
    nv = np.linalg.norm(vec, axis=0) ** 2
    return np.sqrt(miss_s.sum(0) / nv), np.sqrt(miss_s / nv)


# ------------------------------------------------------------------ driver
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--center", type=float, required=True, help="contour centre (eV, absolute)")
    p.add_argument("--a", type=float, required=True, help="real semi-axis (eV)")
    p.add_argument("--b", type=float, required=True, help="imaginary semi-axis (eV)")
    p.add_argument("--npts", type=int, default=24)
    p.add_argument("--probes", type=int, default=200)
    p.add_argument("--kmom", type=int, default=1)
    p.add_argument("--rank-tol", type=float, default=1e-8)
    p.add_argument("--kappa", type=float, default=0.3, help="|ln|lambda|| below which lead modes count as propagating")
    p.add_argument("--kappa-rule", action="store_true", help="swap by |ln|lambda|| < kappa instead of the channel count "
                   "at the contour centre (default: count, analytic over a contour inside one band interval)")
    p.add_argument("--solver", choices=["block", "pardiso"], default="block",
                   help="T(z)^-1 V by dense block-tridiagonal LU over the slabs (default) or PARDISO")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--points", help="only solve these contour point indices (i,j,... or i:j), for parallel runs")
    p.add_argument("--U1", help="prefix of saved level-1 bases (<prefix>_U1_slab<s>.npy)")
    p.add_argument("--resid", type=int, default=0, help="check |T(lambda) v| for the n eigenvalues closest to the centre")
    p.add_argument("--sigma-cache", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from .prep.inputs import read_inputs
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc="folded")
    os.makedirs(a.out, exist_ok=True)
    cache = a.sigma_cache or os.path.join("results_ms", "sigma", os.path.basename(os.path.normpath(a.inputs)))
    nprop = None
    fch = os.path.join(a.out, f"channels_{a.center:+.8f}.json")
    if not a.kappa_rule and os.path.exists(fch):
        nprop, nR = (tuple(x) for x in json.load(open(fch)))
    elif not a.kappa_rule:
        nprop = channel_counts(dev.lead, a.center)
        nR = nprop if dev.leadR is dev.lead else channel_counts(dev.leadR, a.center)
        json.dump([nprop, nR], open(fch, "w"))
    if nprop is not None:
        log(f"propagating channels at the contour centre (n+, n-): left lead {nprop}, right lead {nR}")
    sig = ContinuedSigma(dev, a.kappa, cache, nprop=nprop, nprop_R=None if nprop is None else nR)
    zs, ws = ellipse(a.center, a.a, a.b, a.npts)
    V = np.random.default_rng(a.seed).normal(size=(dev.n, a.probes)) + 0j
    tag = f"_p{a.probes}_s{a.seed}"
    xdir = os.path.join(a.out, "X")
    idx = range(a.npts)
    if a.points:
        idx = (range(*map(int, a.points.split(":"))) if ":" in a.points else [int(x) for x in a.points.split(",")])
    log(f"{a.inputs}: n = {dev.n}, slabs {dev.ns}, contour c {a.center} a {a.a} b {a.b}, {a.npts} pts, "
        f"{a.probes} probes, kappa {a.kappa}")
    blk = dense_blocks(dev) if a.solver == "block" else None
    Xs = [solve_point(dev, sig, zs[j], V, xdir, tag, blk) for j in idx]
    if a.points:
        log("points done")
        return
    lam, vec, s, k = beyn(Xs, zs, ws, a.kmom, a.rank_tol, a.center, max(a.a, a.b))
    inside = ((lam.real - a.center) / a.a) ** 2 + (lam.imag / a.b) ** 2 < 1
    log(f"Beyn: rank {k} of {len(s)} (s[k-1]/s0 {s[k - 1] / s[0]:.1e}, s[k]/s0 "
        f"{(s[k] / s[0]) if k < len(s) else 0:.1e}), {inside.sum()} eigenvalues inside")
    if k >= 0.8 * min(a.probes * a.kmom, len(s)):
        log("  WARNING: rank close to the number of probes x moments; increase --probes / --kmom")
    np.savez(os.path.join(a.out, "eigpairs_all.npz"), lam=lam, vec=vec, inside=inside, sv=s, rank=k)
    lam, vec = lam[inside], vec[:, inside]
    o = np.argsort(lam.real)
    lam, vec = lam[o], vec[:, o]
    out = dict(lam_re=lam.real.tolist(), lam_im=lam.imag.tolist(), sv=s.tolist(), rank=k,
               contour=dict(center=a.center, a=a.a, b=a.b, npts=a.npts, probes=a.probes, kmom=a.kmom, kappa=a.kappa))
    if a.resid:
        j = np.argsort(np.abs(lam - a.center))[:a.resid]
        r = residuals(dev, sig, lam[j], vec[:, j])
        out["resid"] = {int(i): float(x) for i, x in zip(j, r)}
        log("residuals |T v|/(max|T| |v|): " + " ".join(f"{x:.1e}" for x in r))
    if a.U1:
        Us = load_U1(a.U1, len(dev.ns))
        m, ms = span_missing(dev, Us, vec)
        out["U1"], out["missing"], out["missing_slab"] = a.U1, m.tolist(), ms.tolist()
        log(f"span(U1 = {a.U1}, N1 {[u.shape[1] for u in Us]}): missing |(I-P)v|/|v| per eigenvector")
        for l, x, xs in zip(lam, m, ms.T):
            log(f"  {l.real:.5f} {l.imag:+.2e}i  {x:.2e}   slabs " + " ".join(f"{y:.1e}" for y in xs))
    np.savez(os.path.join(a.out, "eigpairs.npz"), lam=lam, vec=vec, dev_orbitals=dev.dev, slab_bounds=dev.sb,
             **({"missing": out["missing"], "missing_slab": out["missing_slab"]} if a.U1 else {}))
    with open(os.path.join(a.out, "eig.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    log(f"-> {a.out}")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ cross-check: qttools spectral OBC at complex z
def qtt_sigma(dev, z, min_decay=1e-3, leads=None, **opts):
    """Sigma_L, Sigma_R (on bL, bR) from qttools' spectral OBC at complex z. qttools keeps a mode as propagating when
    |Im k| < min_decay (per primitive cell) and picks its direction by Re dE/dk, so a min_decay above the gamma/v of
    the propagating modes gives the continued retarded sheet below the axis; the default 1e-3 the decaying one.
    dev needs the qttools path (Device(obc="auto"|"qttools"), QTTOOLS_SRC). leads: cache dict {min_decay: QttLead}.
    Returns (SL, SR), warnings."""
    import warnings
    z = complex(z)
    if leads is not None and min_decay in leads:
        lead = leads[min_decay]
    else:
        m, src = dev.meta, dev.inp["lead"]
        sec = int(m.get("source", {}).get("cells_per_block", 1))
        ob = np.repeat(dev.inp["blocks"], dev.inp["norb"])
        O0 = np.where(ob == m["transport"]["lead_left_block"])[0]
        perm = mb.lead_cell_perm(dev.inp, O0, sec)
        lead = mb.QttLead(src["H00"], src["H01"], src["S00"], src["S01"], sec, perm, min_decay=min_decay, **opts)
        if leads is not None:
            leads[min_decay] = lead
    assert dev.leadR is dev.lead, "qtt_sigma: identical left / right leads only"
    out, warn = [], []
    for side, (Hld, Sld, Hdl, Sdl) in (("L", dev.qL), ("R", dev.qR)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g = _qtt_g(lead, z, side)
        warn += [f"{side}: {x.message}" for x in w]
        out.append((z * Sdl - Hdl) @ g @ (z * Sld - Hld))
    return tuple(out), warn


def _qtt_g(lead, z, side):
    """QttLead.modes without the real-E assumption and without raising on recursion warnings; returns g only."""
    a00 = z * lead.S00 - lead.H00
    a01 = z * lead.S01 - lead.H01
    a10 = z * lead.S01.conj().T - lead.H01.conj().T
    a_ji, a_ij = (a10, a01) if side == "L" else (a01, a10)
    r = np.argsort(lead.p["L"])[lead.p[side]]
    a_xx = tuple(M[np.ix_(r, r)][None] for M in (a_ji, a00, a_ij))
    if lead.sections > 1:
        from qttools.toeplitz.toeplitz import periodize_layer
        a_xx = tuple(np.ascontiguousarray(x) for x in periodize_layer(a_xx, lead.sections))
    g = lead.solver(a_xx, "")[0]
    ip = np.argsort(lead.p[side])
    return g[np.ix_(ip, ip)]
