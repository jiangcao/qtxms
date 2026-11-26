"""Profile of the smoother setup of qtx_ms.mlsolve (user, 2026-11-25: ways to reduce the smoother time), by patching from outside.
    python prof_smoother.py bench              LU of a level window vs block elimination (one GPU)
    python prof_smoother.py <mlsolve options>  per-phase times (GPU synchronized, mean / max over the ranks), distinct windows per level
FREEZE=1: the smoothers (S0, A_k) of the first energy are kept for all others (A, Galerkin operators, LU of the bottom: every energy)."""
import os, sys, time
from collections import defaultdict
sys.path.insert(0, os.getcwd())


def bench():
    import cupy as cp
    import cupyx.scipy.linalg as cla
    sy = cp.cuda.get_current_stream().synchronize
    m, nb = 3014, 5                                              # A1 window: 5 x positions, coupling |dx| <= 2
    rng = cp.random.default_rng(0)
    rnd = lambda: (rng.standard_normal((m, m), cp.float32) + 1j * rng.standard_normal((m, m), cp.float32)).astype(cp.complex64)
    B = {(i, j): rnd() / 60 + (cp.eye(m, dtype=cp.complex64) * 4 if i == j else 0) for i in range(nb) for j in range(nb) if abs(i - j) <= 2}
    n = m * nb
    A = cp.zeros((n, n), cp.complex64)
    for (i, j), b in B.items():
        A[i * m:(i + 1) * m, j * m:(j + 1) * m] = b

    def rows(nr, rep=2):
        for q in range(rep):
            D = A.copy(order="F"); sy(); t = time.time()
            lu = cla.lu_factor(D, overwrite_a=True, check_finite=False); sy(); t1 = time.time()
            E = cp.zeros((n, nr), cp.complex64, order="F"); E[2 * m + cp.arange(nr), cp.arange(nr)] = 1
            X = cla.lu_solve(lu, E, trans=2, check_finite=False); sy(); t2 = time.time()
            R = cp.ascontiguousarray(X.conj().T); sy(); t3 = time.time()
        print(f"full LU n {n}: factor {t1 - t:.2f} s, solve {nr} rows {t2 - t1:.2f} s, conj().T {t3 - t2:.2f} s", flush=True)
        return R
    rows(m // 2)
    R = rows(m)
    for q in range(2):
        W = {k: v.copy(order="F") for k, v in B.items()}; sy(); t = time.time()
        LU, done = {}, []
        for e, nbr in ((0, (1, 2)), (4, (3, 2)), (1, (2, 3)), (3, (2,))):      # fold the ends into position 2
            LU[e] = cla.lu_factor(W[e, e], check_finite=False)
            X = cla.lu_solve(LU[e], cp.hstack([W[e, j] for j in nbr]), check_finite=False)
            U = cp.vstack([W[i, e] for i in nbr]) @ X
            for a, i in enumerate(nbr):
                for b, j in enumerate(nbr):
                    W[i, j] = W[i, j] - U[a * m:(a + 1) * m, b * m:(b + 1) * m]
        S = cla.lu_factor(W[2, 2], check_finite=False)
        r = {2: cla.lu_solve(S, cp.eye(m, dtype=cp.complex64), check_finite=False)}
        for e, nbr in ((3, (2,)), (1, (2, 3)), (4, (3, 2)), (0, (1, 2))):
            Y = sum(r[j] @ W[j, e] for j in nbr)
            r[e] = -cla.lu_solve(LU[e], Y.conj().T, trans=2, check_finite=False).conj().T
        Rb = cp.hstack([r[i] for i in range(nb)]); sy(); t1 = time.time()
    print(f"block elimination, {m} rows: {t1 - t:.2f} s, difference to the full LU {float(cp.abs(Rb - R).max() / cp.abs(R).max()):.1e}", flush=True)
    import cupyx
    for q in range(2):
        t = time.time(); h = cupyx.empty_pinned(R.shape, R.dtype); t1 = time.time(); R.get(out=h); t2 = time.time()
        print(f"pinned {R.nbytes / 2**30:.2f} GB: allocate {t1 - t:.2f} s, copy {t2 - t1:.2f} s" + (" (pool)" if q else ""), flush=True)
        del h


if sys.argv[1] == "bench":
    bench(); sys.exit()

from qtx_ms import dist, gpu, mldrive as md, mlevel, mlsolve, msbasis, schwarz   # noqa: E402
T, N, FREEZE, SM = defaultdict(float), defaultdict(int), os.environ.get("FREEZE") == "1", {}
depth = [0]


def patch(obj, name, label):
    f = getattr(obj, name)

    def g(*a, **k):
        gpu.sync(); t, d0 = time.time(), dict(T)
        depth[0] += 1
        out = f(*a, **k)
        depth[0] -= 1
        gpu.sync()
        T[label] += time.time() - t; N[label] += 1
        return out
    setattr(obj, name, g)


for o, nm, lb in ((mlevel.DistOp, "sub", "assemble A_w"), (mlevel.Hierarchy, "cap_of", "CAP"), (gpu, "lu_factor", "LU"), (gpu, "lu_solve", "rows solve"),
                  (gpu, "pinned", "pinned alloc + copy"), (gpu, "free", "gpu.free"), (dist.FineA, "window", "fine window (CPU sparse)"),
                  (gpu, "csr_dense", "csr to dense"), (dist.FineOp, "__init__", "fine operator"), (msbasis.Device, "sigma", "dev.sigma"),
                  (msbasis.Device, "inject", "dev.inject"), (mlevel.Hierarchy, "ghost_rows", "ghost_rows"), (mlevel.Hierarchy, "basis_get", "basis_get"),
                  (mlevel.Hierarchy, "couplings", "coupling norms")):
    if hasattr(o, nm):
        patch(o, nm, lb)
    else:
        print("no", o, nm)


def report(comm, what, tot, before):
    d = {k: T[k] - before.get(k, 0.0) for k in T if T[k] - before.get(k, 0.0) > 0}
    al = comm.allgather((tot, d))
    keys = sorted({k for _, x in al for k in x}, key=lambda k: -max(x.get(k, 0) for _, x in al))
    md.log(f"PROF {what}: {sum(a for a, _ in al) / len(al):.1f} / {max(a for a, _ in al):.1f} s (mean / max rank); " + ", ".join(
        f"{k} {sum(x.get(k, 0) for _, x in al) / len(al):.1f} / {max(x.get(k, 0) for _, x in al):.1f}" for k in keys))


ras0, merge0, set0, free0, lead0 = (mlevel.Hierarchy.ras, mlevel.Hierarchy.merge, schwarz.CapSchwarz.set_energy, schwarz.CapSchwarz.free, md.lead)


def ras(self, L, wins, writes):
    if FREEZE and L.k in SM:
        md.log(f"PROF smoother of A{L.k}: kept from the first energy")
        return SM[L.k]
    mine = [tuple(int(b) for b in w) for w in wins if w is not None]
    al = self.comm.allgather(mine)
    md.log(f"PROF windows of A{L.k}: {sum(map(len, al))}, distinct {len(set(sum(al, [])))}; per rank {[len(a) for a in al]}, distinct per rank {[len(set(a)) for a in al]}")
    t, b = time.time(), dict(T)
    out = ras0(self, L, wins, writes)
    gpu.sync()
    report(self.comm, f"smoother of A{L.k}", time.time() - t, b)
    if FREEZE:
        SM[L.k] = out
    return out


def merge(self, *a, **k):
    self.comm.barrier(); t, b = time.time(), dict(T)
    out = merge0(self, *a, **k)
    gpu.sync()
    report(self.comm, f"whole merge to level {out.k}", time.time() - t, b)
    return out


def set_energy(self, E, A, **k):
    if FREEZE and "S0" in SM:
        return
    t, b = time.time(), dict(T)
    set0(self, E, A, **k)
    gpu.sync()
    report(self.comm or dist.Comm(), "S0 set_energy", time.time() - t, b)
    if FREEZE:
        SM["S0"] = True


def lead(dev, E, eta, comm, ncol=0):
    t, b = time.time(), dict(T)
    out = lead0(dev, E, eta, comm, ncol)
    report(comm, "lead", time.time() - t, b)
    return out


mlevel.Hierarchy.ras, mlevel.Hierarchy.merge, schwarz.CapSchwarz.set_energy, md.lead = ras, merge, set_energy, lead
if FREEZE:
    schwarz.CapSchwarz.free = lambda self: None
mlevel.guarded(lambda: mlsolve.main(sys.argv[1:]))
