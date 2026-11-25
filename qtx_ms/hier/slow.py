"""Coarse level from the slow error modes of the fine smoother S_0 (distributed form of schwarz.SlowCoarse), built at one energy.

Round 1: block subspace iteration (q steps from m random vectors) on the error operator of two smoothing steps, (I - S_0 A)^2: the m
error components the smoother damps worst. Round r > 1: the same on the two-level cycle with the space of the rounds before,
(I - S_0 A)(I - P_c A)(I - S_0 A). The new vectors are orthogonalized against Q and appended. rounds = 1: slow modes of the fine
smoother alone. Coarse level: A_c = Q^H A Q (global vectors), dense LU in double precision on every rank, P_c = Q A_c^-1 Q^H.
A Q is kept: the residual after the coarse correction needs no product with A, and a column costs two vectors of memory.
Vectors are the own rows of every rank (Fine.part)."""
import time
import numpy as np

from .. import gpu
from .fine import _xp


def tsqr(Y, comm):
    """Orthonormal basis (own rows) of the columns of a distributed Y: QR of the own rows, then of the stacked R factors."""
    xp = _xp()
    if comm.size == 1:
        return xp.linalg.qr(Y)[0]
    m = Y.shape[1]
    Q1, R = xp.linalg.qr(Y)
    rows = [0] + list(np.cumsum(comm.allgather(R.shape[0])))
    W = xp.zeros((rows[-1], m), Y.dtype)
    a, b = int(rows[comm.rank]), int(rows[comm.rank + 1])
    W[a:b] = R
    return Q1 @ xp.linalg.qr(comm.sum(W))[0][a:b]


class Slow:
    def __init__(self, A, S0, part, comm, m, q, rounds, seed=0, log=None):
        """A: fine operator (A @ own rows); S0(r): smoother."""
        xp, t0 = _xp(), time.time()
        assert m * rounds < part.n and q >= 1
        self.A, self.S0, self.comm = A, S0, comm
        n = part.nloc
        self.Q = xp.zeros((n, 0), complex)
        self.AQ, self.lu = self.Q, None
        self.info = dict(m=m, q=q, rounds=[])
        for r in range(rounds):
            t = time.time()
            rng = xp.random.default_rng(seed + 1000 * r + comm.rank)
            Y = tsqr(rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m)), comm)
            for k in range(q):
                Z = Y - self.cycle(A @ Y) if self.lu is not None else self._smooth2(Y)
                if k == q - 1:                                         # Ritz values of the error operator on span(Y)
                    th = np.abs(np.linalg.eigvals(gpu.asnumpy(self.dot(Y, Z))))
                Y = tsqr(Z, comm)
            for _ in range(2):                                         # against the rounds before, twice
                if self.Q.shape[1]:
                    Y = Y - self.Q @ self.dot(self.Q, Y)
                Y = tsqr(Y, comm)
            AY = A @ Y
            self.Q, self.AQ = xp.hstack([self.Q, Y]), xp.hstack([self.AQ, AY])
            del Y, Z, AY
            Ac = self.dot(self.Q, self.AQ)
            self.lu = gpu.lu_factor(xp.asfortranarray(Ac))
            d = dict(N=int(self.Q.shape[1]), theta_max=float(th.max()), n_gt1=int((th > 1).sum()), n_gt05=int((th > 0.5).sum()),
                     theta_min=float(th.min()), t=time.time() - t)
            self.info["rounds"].append(d)
            if log:
                log(f"    slow modes round {r + 1}: |theta| of the " + (f"cycle with N {d['N'] - m}" if r else "smoother (2 steps)")
                    + f": max {d['theta_max']:.2f}, min {d['theta_min']:.2f}, {d['n_gt1']} > 1, {d['n_gt05']} > 0.5 of {m} -> N {d['N']}; {d['t']:.1f} s")
            gpu.free()
        self.N = int(self.Q.shape[1])
        self.info.update(N=self.N, t_build=time.time() - t0)

    def dot(self, X, Y):
        """X^H Y over all ranks; by columns of X, without a conjugate copy of the whole X."""
        xp = _xp()
        out = [X[:, a:a + 128].conj().T @ Y for a in range(0, X.shape[1], 128)]
        return self.comm.sum(xp.ascontiguousarray(xp.concatenate(out)))

    def _smooth2(self, Y):
        """(I - S_0 A)^2 Y."""
        Y = Y - self.S0(self.A @ Y)
        return Y - self.S0(self.A @ Y)

    def coef(self, r):
        return gpu.lu_solve(self.lu, self.dot(self.Q, r))

    def __call__(self, r):
        """P_c r."""
        return self.Q @ self.coef(r)

    def cycle(self, r):
        """One two-level cycle from zero: S_0, coarse correction, S_0."""
        z = self.S0(r)
        d = r - self.A @ z
        c = self.coef(d)
        z = z + self.Q @ c
        return z + self.S0(d - self.AQ @ c)
