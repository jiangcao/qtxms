"""Push the spurious level-1 OBC eigenstates out of the energy window by basis optimization (the lead-band RMS method,
Huang et al. JAP 123, 044303 (2018), carried over from the lead cell to the open device), with reduced Beyn as detector.

Level-1 open system T1(z) = U1^H T(z) U1, T(z) = z S - H - Sigma(z) (continued retarded sheet at the nodes z_k of a Beyn
contour inside one lead band interval; Sigma' at the nodes from the barycentric interpolant). One vector per step:
  1. reduced Beyn on T1 -> eigenpairs inside the contour; fine residual |T(lambda) U1 c| flags the spurious ones
     (|r| > --rthr; physical resonances 0.1-0.4 eV, spurious ~7 eV on mos2_hfo2)
  2. candidates per slab s: Xi_s = orth[(1 - U1_s U1_s^H) r_i|_s] from the residuals r_i of the spurious states
     (the analogue of (1 - Phi Phi^H) H Phi of the lead RMS), xi = Xi_s C
  3. Schur complement of xi:  sigma(z) = xi^H T xi - xi^H T U1 T1^-1 U1^H T xi = C^H M(z) C  (m x m matrices M, M' per
     node), and  N(C) = Re sum_k w_k sigma'(z_k) / sigma(z_k)  = change in the number of eigenvalues inside the contour
     when xi is appended (argument principle). Minimize N over C (complex, normalized; several starts), per slab
  4. append the best xi (most negative N) to its slab; repeat until no spurious eigenvalue is left inside
Physical resonances have small residuals, so xi^H T v is small for them and they barely move. Nothing is removed from
the basis: the spurious combination stays in the span, only its eigenvalue leaves the window.
At the end: MS T on a fine grid inside the contour vs exact (cache of ms_absorb), for the bases after each step.

  python -m qtx_ms.ms_push --inputs inputs/mos2_hfo2 --center 0.52135 --a 0.015 --b 0.01 --npts 24 \
      --U1 results_ms/mos2_hfo2_vcycle/tobc_t1e-6_lc --fine results_ms/obc_eig/c1_Ec+0.017 --out results_ms/obc_eig/push/c1_base
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize

from . import msbasis as mb
from . import obc_eig as oe
from .obc_eig_l1 import SigmaInterp, apply_T, apply_dT

log = mb.log


def node_derivs(zs, ws, Ss):
    """S'(z_j) of the barycentric (Cauchy) interpolant at its own nodes: -(1/w_j) sum_{k!=j} w_k (S_j - S_k)/(z_j - z_k)."""
    out = []
    for j in range(len(zs)):
        acc = 0
        for k in range(len(zs)):
            if k != j:
                acc = acc + ws[k] * (Ss[j] - Ss[k]) / (zs[j] - zs[k])
        out.append(-acc / ws[j])
    return out


class Level1:
    """Level-1 open system in a dense global basis Uf (device orbitals x N1, columns = the slab bases followed by the
    appended vectors in order, so every earlier basis is a leading block). H1, S1 kept and bordered on append."""

    def __init__(self, dev, Us, zs, SLs, SRs, dSLs, dSRs):
        self.dev, self.zs = dev, zs
        self.SLs, self.SRs, self.dSLs, self.dSRs = SLs, SRs, dSLs, dSRs
        self.N0 = [u.shape[1] for u in Us]
        self.Uf = np.zeros((dev.n, sum(self.N0)), complex)
        c = 0
        for s, u in enumerate(Us):
            self.Uf[dev.sb[s]:dev.sb[s + 1], c:c + u.shape[1]] = u
            c += u.shape[1]
        self.slab = list(np.repeat(np.arange(len(Us)), self.N0))
        t = time.time()
        HU, SU = dev.H @ self.Uf, dev.S @ self.Uf
        self.H1 = self.Uf.conj().T @ HU
        self.S1 = self.Uf.conj().T @ SU
        self.H1, self.S1 = 0.5 * (self.H1 + self.H1.conj().T), 0.5 * (self.S1 + self.S1.conj().T)
        log(f"  level-1 H1, S1 (N1 {self.Uf.shape[1]}): {time.time() - t:.0f} s")
        self.factor()

    def proj(self, SL, SR, U=None):
        """U^H Sigma U (dense N x N) for Sigma on the lead faces."""
        U = self.Uf if U is None else U
        UL, UR = U[self.dev.bL], U[self.dev.bR]
        return UL.conj().T @ SL @ UL + UR.conj().T @ SR @ UR

    def Ac(self, z, SL, SR, n=None):
        n = self.Uf.shape[1] if n is None else n
        return z * self.S1[:n, :n] - self.H1[:n, :n] - self.proj(SL, SR, self.Uf[:, :n])

    def factor(self):
        self.A, self.dA, self.lu = [], [], []
        for z, SL, SR, dSL, dSR in zip(self.zs, self.SLs, self.SRs, self.dSLs, self.dSRs):
            A = z * self.S1 - self.H1 - self.proj(SL, SR)
            self.A.append(A)
            self.dA.append(self.S1 - self.proj(dSL, dSR))
            self.lu.append(sla.lu_factor(A, check_finite=False))

    def append(self, x, s):
        d = self.dev
        x = x - self.Uf @ (self.Uf.conj().T @ x)
        x = x - self.Uf @ (self.Uf.conj().T @ x)
        x /= np.linalg.norm(x)
        Hx, Sx = d.H @ x, d.S @ x
        h, sv = self.Uf.conj().T @ Hx, self.Uf.conj().T @ Sx
        n = self.Uf.shape[1]
        H1 = np.zeros((n + 1, n + 1), complex); S1 = np.zeros((n + 1, n + 1), complex)
        H1[:n, :n], S1[:n, :n] = self.H1, self.S1
        H1[:n, n], S1[:n, n] = h, sv
        H1[n, :n], S1[n, :n] = h.conj(), sv.conj()
        H1[n, n], S1[n, n] = np.real(np.vdot(x, Hx)), np.real(np.vdot(x, Sx))
        self.H1, self.S1 = H1, S1
        self.Uf = np.hstack([self.Uf, x[:, None]])
        self.slab.append(s)
        self.factor()

    def schur(self, Xi):
        """Per node M, M' (m x m) with sigma(z) = C^H M C, sigma'(z) = C^H M' C for xi = Xi C (fine vectors)."""
        d = self.dev
        Ms, dMs = [], []
        UH = self.Uf.conj().T
        for k, z in enumerate(self.zs):
            SL, SR, dSL, dSR = self.SLs[k], self.SRs[k], self.dSLs[k], self.dSRs[k]
            TX = apply_T(d, z, SL, SR, Xi)
            dTX = apply_dT(d, dSL, dSR, Xi)
            THX = apply_T(d, np.conj(z), SL.conj().T, SR.conj().T, Xi)
            dTHX = apply_dT(d, dSL.conj().T, dSR.conj().T, Xi)
            p, dp = UH @ TX, UH @ dTX
            q, dq = (UH @ THX).conj().T, (UH @ dTHX).conj().T
            X = sla.lu_solve(self.lu[k], p, check_finite=False)
            Y = sla.lu_solve(self.lu[k], dp, check_finite=False)
            W = sla.lu_solve(self.lu[k], self.dA[k] @ X, check_finite=False)
            Ms.append(Xi.conj().T @ TX - q @ X)
            dMs.append(Xi.conj().T @ dTX - dq @ X + q @ W - q @ Y)
        return np.array(Ms), np.array(dMs)

    def slab_bases(self):
        """Per slab: its columns of Uf restricted to the slab rows (for saving in the <prefix>_U1_slab<s> format)."""
        sl = np.array(self.slab)
        return [self.Uf[self.dev.sb[s]:self.dev.sb[s + 1], sl == s] for s in range(len(self.dev.ns))]


def count_change(C, Ms, dMs, ws):
    num = np.einsum("i,kij,j->k", C.conj(), dMs, C)
    den = np.einsum("i,kij,j->k", C.conj(), Ms, C)
    return float(np.real(np.sum(ws * num / den)))


def optimize(Ms, dMs, ws, starts):
    m = Ms.shape[1]
    f = lambda x: count_change((x[:m] + 1j * x[m:]) / np.linalg.norm(x), Ms, dMs, ws)
    best = None
    for x0 in starts:
        r = minimize(f, x0, method="BFGS", options=dict(maxiter=200, gtol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    C = (x[:m] + 1j * x[m:]) / np.linalg.norm(x)
    return C, best.fun


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--center", type=float, required=True)
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--b", type=float, required=True)
    p.add_argument("--npts", type=int, default=24)
    p.add_argument("--probes", type=int, default=300)
    p.add_argument("--rank-tol", type=float, default=1e-8)
    p.add_argument("--U1", required=True)
    p.add_argument("--fine", help="obc_eig dir of the same contour (channel counts, full-size eigenpairs)")
    p.add_argument("--rthr", type=float, default=2.0, help="spurious: |T v| / |v| above this [eV]")
    p.add_argument("--maxit", type=int, default=30, help="max vectors added")
    p.add_argument("--accept", type=float, default=-0.5, help="append xi only if its count change is below this")
    p.add_argument("--grid", default="0.003:0.031:0.001", help="lo:hi:step, E - Ec [eV]")
    p.add_argument("--exact-cache", default="results_ms/obc_eig/absorb/exact_grid.json")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from .prep.inputs import read_inputs
    inp = read_inputs(a.inputs)
    dev = mb.Device(inp, obc="folded")
    Ec = inp["meta"]["edges"]["Ec_lead"]
    name = os.path.basename(os.path.normpath(a.inputs))
    cache = os.path.join("results_ms", "sigma", name)
    fch = a.fine and os.path.join(a.fine, f"channels_{a.center:+.8f}.json")
    if fch and os.path.exists(fch):
        nprop, nR = (tuple(x) for x in json.load(open(fch)))
    else:
        nprop = oe.channel_counts(dev.lead, a.center); nR = nprop
    sig = oe.ContinuedSigma(dev, cache=cache, nprop=nprop, nprop_R=nR)
    zs, ws = oe.ellipse(a.center, a.a, a.b, a.npts)
    SLs, SRs = zip(*[sig(z) for z in zs])
    dSLs, dSRs = node_derivs(zs, ws, SLs), node_derivs(zs, ws, SRs)
    sint = SigmaInterp(zs, ws, SLs, SRs)
    fine = None
    if a.fine and os.path.exists(os.path.join(a.fine, "eigpairs.npz")):
        f = np.load(os.path.join(a.fine, "eigpairs.npz"))
        fine = dict(lam=f["lam"], vec=f["vec"] / np.linalg.norm(f["vec"], axis=0))
    Us = [np.asarray(u, complex) for u in oe.load_U1(a.U1, len(dev.ns))]
    log(f"{a.inputs}: level 1 {a.U1} N1 {[u.shape[1] for u in Us]}; contour c {a.center} a {a.a} b {a.b}, "
        f"{a.npts} pts, channels {nprop}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    def detect(L, tag):
        t0 = time.time()
        n1 = L.Uf.shape[1]
        V = np.random.default_rng(1).normal(size=(n1, a.probes)) + 0j
        Xs = [sla.lu_solve(lu, V, check_finite=False) for lu in L.lu]
        lam, C, s, k = oe.beyn(Xs, zs, ws, 1, a.rank_tol, a.center, max(a.a, a.b))
        ins = ((lam.real - a.center) / a.a) ** 2 + (lam.imag / a.b) ** 2 < 1
        lam, C = lam[ins], C[:, ins]
        o = np.argsort(lam.real)
        lam, C = lam[o], C[:, o]
        rows, R = [], []
        for l, c in zip(lam, C.T):
            v = L.Uf @ c
            v /= np.linalg.norm(v)
            r = apply_T(dev, l, *sint(l), v)
            row = dict(lam=[float(l.real), float(l.imag)], dE=float(l.real - Ec), res=float(np.linalg.norm(r)))
            if fine is not None and len(fine["lam"]):
                j = int(np.argmin(np.abs(fine["lam"] - l)))
                row.update(dlam=float(abs(fine["lam"][j] - l)),
                           sin=float(np.sqrt(max(0.0, 1 - abs(np.vdot(fine["vec"][:, j], v)) ** 2))))
            rows.append(row); R.append(r)
        spur = [i for i, r in enumerate(rows) if r["res"] > a.rthr]
        log(f"[{tag}] reduced Beyn N1 {n1}: rank {k}, {len(lam)} inside, {len(spur)} spurious ({time.time() - t0:.0f} s): " +
            ", ".join(f"Ec{r['dE']:+.5f}{r['lam'][1]:+.1e}i |r| {r['res']:.2f}" +
                      (f" (phys: dlam {r['dlam']:.1e} sin {r['sin']:.1e})" if r['res'] <= a.rthr and 'dlam' in r else "")
                      for r in rows))
        return rows, [R[i] for i in spur]

    hist = []
    L = Level1(dev, Us, zs, SLs, SRs, dSLs, dSRs)
    n_base = L.Uf.shape[1]
    for it in range(a.maxit + 1):
        rows, Rsp = detect(L, f"step {it}")
        hist.append(dict(N1=int(L.Uf.shape[1]), rows=rows))
        if not Rsp or it == a.maxit:
            break
        t0 = time.time()
        best = None
        for s in range(len(dev.ns)):
            Ufs = L.Uf[dev.sb[s]:dev.sb[s + 1]]
            Rs = np.array([r[dev.sb[s]:dev.sb[s + 1]] for r in Rsp]).T
            for _ in range(2):
                Rs = Rs - Ufs @ (Ufs.conj().T @ Rs)
            Q, sv, _ = np.linalg.svd(Rs, full_matrices=False)
            Q = Q[:, sv > 1e-8 * sv[0]] if sv[0] > 0 else Q[:, :0]
            if Q.shape[1] == 0:
                continue
            Xi = np.zeros((dev.n, Q.shape[1]), complex)
            Xi[dev.sb[s]:dev.sb[s + 1]] = Q
            t1 = time.time()
            Ms, dMs = L.schur(Xi)
            t2 = time.time()
            m = Q.shape[1]
            starts = [np.r_[np.eye(m)[i], np.zeros(m)] for i in range(m)]
            C, fval = optimize(Ms, dMs, ws, starts)
            log(f"  slab {s}: {m} candidates, best count change {fval:+.3f} (Schur {t2 - t1:.0f} s, "
                f"optimize {time.time() - t2:.0f} s)")
            if best is None or fval < best[0]:
                best = (fval, s, Xi @ C)
        if best is None or best[0] > a.accept:
            log(f"step {it}: no candidate reduces the count (best {None if best is None else round(best[0], 3)}); stop")
            break
        fval, s, x = best
        L.append(x, s)
        log(f"step {it}: + vector in slab {s} (count change {fval:+.3f}), {time.time() - t0:.0f} s")
        hist[-1].update(added_slab=int(s), count_change=fval)
    pre = f"{a.out}_final"
    for s_, u in enumerate(L.slab_bases()):
        np.save(f"{pre}_U1_slab{s_}.npy", u)

    # ---- fine grid: base, final, and a few intermediate bases
    lo, hi, st = (float(x) for x in a.grid.split(":"))
    grid = np.round(np.arange(lo, hi + 0.5 * st, st), 6)
    exact = json.load(open(a.exact_cache)) if os.path.exists(a.exact_cache) else {}
    nk = L.Uf.shape[1] - n_base
    sizes = sorted({n_base, n_base + nk} | {n_base + k for k in range(0, nk + 1, max(1, nk // 4))})
    tags = ["base" if n == n_base else f"push{n - n_base}" for n in sizes]
    out_rows = []
    for x in grid:
        t = time.time()
        E = Ec + x
        SL, SR = sint(E)
        B0 = mb.injection(SL)[0]
        b = np.zeros((dev.n, B0.shape[1]), complex); b[dev.bL] = B0
        GR = 1j * (SR - SR.conj().T)
        Tof = lambda X: float(np.real(np.einsum("ic,ij,jc->", X[dev.bR].conj(), GR, X[dev.bR])))
        row = dict(dE=float(x))
        key = f"{name}:{a.center:g}:{x:.6f}"
        if key in exact:
            row["T_exact"] = exact[key]
        A = L.Ac(E, SL, SR)
        Ub = L.Uf.conj().T @ b
        for tg, n in zip(tags, sizes):
            row[f"T_{tg}"] = Tof(L.Uf[:, :n] @ np.linalg.solve(A[:n, :n], Ub[:n]))
        out_rows.append(row)
        line = f"Ec{x:+.3f}" + (f" exact {row['T_exact']:.6f}" if "T_exact" in row else "")
        for tg in tags:
            v = row[f"T_{tg}"]
            line += f" | {tg}: {v:.6f}" + (f" ({(v - row['T_exact']) / row['T_exact']:+.1e})" if "T_exact" in row else "")
        log(line + f"  [{time.time() - t:.0f} s]")
    summ = {}
    if all("T_exact" in r for r in out_rows):
        for t in tags:
            e = np.array([abs(r[f"T_{t}"] - r["T_exact"]) / r["T_exact"] for r in out_rows])
            summ[t] = dict(median=float(np.median(e)), max=float(e.max()), at=float(grid[np.argmax(e)]))
            log(f"{t}: MS T error median {np.median(e):.1e} max {e.max():.1e} at Ec{grid[np.argmax(e)]:+.3f}")
    json.dump(dict(args=vars(a), hist=hist, rows=out_rows, summary=summ), open(f"{a.out}.json", "w"), indent=1)
    log(f"-> {a.out}.json, {pre}_U1_slab<s>.npy")


if __name__ == "__main__":
    main()
