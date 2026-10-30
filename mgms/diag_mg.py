"""Diagnostics for the two-level solver (MG1: coarse = level-1 basis U1) at given energies, strongest
injection channel. Uses the bases cached by run_ms (same --inputs / --no-cores / basis options).

    python -m mgms.diag_mg --inputs inputs/si3x5_sp3s --E 1.21 [--neig 30] [-- <extra run_ms options>]

Per energy:
  [exact]    PARDISO: perturbed pivots, residual vs refinement steps, CSolver with GMRES fallback
  [coarse]   consistency of the block-tridiagonal A_c with U^H A U, U orthonormality
  [galerkin] / [mr]  coarse correction Pc = U A_c^-1 U^H  /  U (W^H W)^-1 W^H, W = A U:
             projector pi^2 = pi; MS solution x0 = Pc b (T of the channel, residual, error);
             GMRES(k) smoother alone on the MS residual, then + coarse correction;
             FGMRES with the two-level preconditioner (k = 10, 30); deflated GMRES (--its)
  [eig]      eigenvalues of A nearest 0 (shift-invert) and how much of each eigenvector U1 misses
(The version that produced the first logs, with plain refinement only and the Galerkin coarse level, is
commit f10d477.)
"""
import argparse, sys, time
import numpy as np
import scipy.sparse.linalg as spla

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--inputs", required=True)
p.add_argument("--E", required=True, help="absolute energies [eV], comma separated")
p.add_argument("--neig", type=int, default=30, help="eigenvalues of A nearest 0 (0: skip)")
p.add_argument("--its", type=int, default=0, help="deflated GMRES iterations (0: skip)")
p.add_argument("--mg-its", type=int, default=60, help="FGMRES iterations")
d, rest = p.parse_known_args()

# basis setup and caches: the part of run_ms before the transmission loop
sys.argv = ["run_ms", "--inputs", d.inputs, "--E", d.E] + [x for x in rest if x != "--"]
src = open(__file__.replace("diag_mg.py", "run_ms.py")).read()
exec(compile(src.split("# " + "-" * 66 + " transmission")[0], "run_ms_setup", "exec"))
from mgms.twolevel import TwoLevel, gmres_fixed


def sigma(E):
    f = os.path.join(out, "sigma", f"sig_{E:.4f}.npz")
    if os.path.exists(f):
        z = np.load(f)
        return z["SL"], z["SR"]
    SL, SR = dev.sigma(E + 1e-6j)
    np.savez(f, SL=SL, SR=SR)
    return SL, SR


def gmres_hist(op, r0, nb, it):
    """Residual history (relative to nb) of full GMRES on op from residual r0."""
    V = np.zeros((len(r0), it + 1), complex); H = np.zeros((it + 1, it), complex)
    be = np.linalg.norm(r0); V[:, 0] = r0 / be; h = [be / nb]
    for j in range(it):
        w = op(V[:, j])
        for _ in range(2):
            c = V[:, :j + 1].conj().T @ w; H[:j + 1, j] += c; w -= V[:, :j + 1] @ c
        H[j + 1, j] = np.linalg.norm(w); V[:, j + 1] = w / H[j + 1, j]
        e1 = np.zeros(j + 2, complex); e1[0] = be
        y = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)[0]
        h.append(np.linalg.norm(H[:j + 2, :j + 1] @ y - e1) / nb)
    return h


G = (U1.conj().T @ U1).toarray()
Gi = np.linalg.inv(G)
for E in [float(x) for x in d.E.split(",")]:
    T0 = time.time()
    SL, SR = sigma(E)
    B, lam, X = mb.injection(SL, a.inj_tol)
    b = np.zeros(dev.n, complex); b[dev.bL] = B[:, -1]
    A = dev.A(E, SL, SR)
    nb = np.linalg.norm(b)
    rel = lambda x: np.linalg.norm(A @ x - b) / nb
    log(f"\n=== E = {E:.4f} (Ec {E - Ec:+.3f}), {B.shape[1]} channels, strongest channel")

    t = time.time()
    cs = mb.CSolver(A, n_ref=0, maxit=0)
    x = cs.factor_solve(b[:, None])[:, 0]
    log(f"[exact] factorization {time.time() - t:.0f} s, {cs.npert} perturbed pivots; residual, refinement 0: {rel(x):.1e}")
    for i in range(1, 6):
        x = x + cs.factor_solve((b - A @ x)[:, None])[:, 0]
        if i in (1, 2, 5):
            log(f"[exact]   refinement {i}: {rel(x):.1e}")
    cs.n_ref, cs.maxit = 2, 60
    xex = cs.solve(b)
    log(f"[exact] CSolver (2 refinements + GMRES): residual {rel(xex):.1e}, {cs.its} GMRES its")
    GR = 1j * (SR - SR.conj().T)
    Tch = lambda x: float(np.real(x[dev.bR].conj() @ GR @ x[dev.bR]))
    log(f"[exact] T of the strongest channel {Tch(xex):.6f}")

    Ac = mb.BlockTridiag(blk1, E, PL.conj().T @ SL @ PL, PR.conj().T @ SR @ PR)
    v = np.random.default_rng(1).standard_normal(U1.shape[1]) + 0j
    w = Ac.solve(U1.conj().T @ (A @ (U1 @ v)))
    log(f"[coarse] |Ac^-1 U^H A U v - v| / |v| = {np.linalg.norm(w - v) / np.linalg.norm(v):.1e}; "
        f"|U^H U - I|_max = {np.abs(G - np.eye(len(G))).max():.1e}")
    t = time.time()
    mr = mb.MRCoarse(dev, A, Us)
    log(f"[coarse] minimal-residual W^H W: Cholesky {time.time() - t:.0f} s")
    y = np.random.default_rng(2).standard_normal(dev.n) + 0j
    for kind, kw in (("galerkin", dict(coarse=Ac)), ("mr", dict(coarse=mr, restrict="AU"))):
        tl = TwoLevel(A, U1, smoother="gmres", k=10, **kw)
        p1 = tl.Pc(A @ y); p2 = tl.Pc(A @ p1)
        x0 = tl.Pc(b)
        e0 = xex - x0
        r1 = b - A @ x0
        log(f"[{kind}] projector |pi^2 y - pi y| / |pi y| = {np.linalg.norm(p2 - p1) / np.linalg.norm(p1):.1e}")
        log(f"[{kind}] MS: T {Tch(x0):.6f} (rel err {abs(Tch(x0) - Tch(xex)) / Tch(xex):.1e}); residual "
            f"{rel(x0):.2e}; error |x - x_MS| / |x| = {np.linalg.norm(e0) / np.linalg.norm(xex):.2e}")
        for k in (10, 30, 100):
            z = gmres_fixed(lambda v: A @ v, r1, k)
            r2 = r1 - A @ z
            r3 = r2 - A @ tl.Pc(r2)
            log(f"[{kind}] smoother GMRES({k}) on MS residual: {np.linalg.norm(r1) / nb:.2e} -> "
                f"{np.linalg.norm(r2) / nb:.2e} -> + coarse {np.linalg.norm(r3) / nb:.2e}; error |e - z| / |e| = "
                f"{np.linalg.norm(e0 - z) / np.linalg.norm(e0):.3f}")
        for k in (10, 30):
            t = time.time()
            s = TwoLevel(A, U1, smoother="gmres", k=k, **kw)
            xs, h, _ = s.solve(b, tol=1e-10, maxit=d.mg_its)
            log(f"[{kind}] FGMRES two-level k={k:<3d}: " + " ".join(f"{h[i]:.1e}" for i in range(0, len(h), 10)) +
                f" (every 10 its) | final {h[-1]:.1e} at {len(h) - 1} its, {s.nmv} mv, T err "
                f"{abs(Tch(xs) - Tch(xex)) / Tch(xex):.1e}, {time.time() - t:.0f} s")
        if d.its:
            t = time.time()
            h = gmres_hist(lambda v: (lambda u: u - A @ tl.Pc(u))(A @ v), r1, nb, d.its)
            marks = [i for i in (0, 25, 50, 100, 200, 300, 400, 600, 800) if i <= d.its]
            log(f"[{kind}] deflated GMRES (I - A Pc) A: " + " ".join(f"{h[i]:.1e}" for i in marks) +
                f" (its {marks}), {time.time() - t:.0f} s")
    tl = TwoLevel(A, U1, smoother="gmres", k=10, coarse=Ac)
    e0 = xex - tl.Pc(b)

    if d.neig:
        t = time.time()
        Ainv = spla.LinearOperator(A.shape, matvec=lambda v: cs.solve(v), dtype=complex)
        mu, W = spla.eigs(Ainv, k=d.neig, which="LM", ncv=3 * d.neig, tol=1e-6)
        lamA = 1 / mu
        o = np.argsort(abs(lamA)); lamA, W = lamA[o], W[:, o]
        log(f"[eig] {d.neig} eigenvalues of A nearest 0 ({time.time() - t:.0f} s); miss = |(I - Q_U1) w|, "
            f"end = weight in the end slabs L/R, err = |<w, e_MS>| / |e_MS|")
        for i in range(len(lamA)):
            wv = W[:, i] / np.linalg.norm(W[:, i])
            miss = np.linalg.norm(wv - U1 @ (Gi @ (U1.conj().T @ wv)))
            left, right = np.linalg.norm(wv[:dev.sb[1]]) ** 2, np.linalg.norm(wv[dev.sb[-2]:]) ** 2
            log(f"[eig] {lamA[i].real:+.3e} {lamA[i].imag:+.2e}i  miss {miss:.2e}  end {left:.2f}/{right:.2f}  "
                f"err {abs(np.vdot(wv, e0)) / np.linalg.norm(e0):.2e}")
    cs.free()
    log(f"E = {E:.4f} done, {time.time() - T0:.0f} s")
