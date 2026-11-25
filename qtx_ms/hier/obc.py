"""Open boundary: Sigma_L(z), Sigma_R(z) and the injection (eigenpairs of Gamma_L) at the real energies and the contour points.
The points are dealt out to the ranks; decimation, products and eigh run on the GPU of the rank. Results are files in `cache`
(the format of the other drivers) or, without a cache, shared in memory."""
import os, time
import numpy as np

from .. import dist, gpu


def contour(mu1, mu2, npts):
    """Upper half circle over [mu1, mu2]: points z, weights dz (Gauss-Legendre in the angle)."""
    c, r = 0.5 * (mu1 + mu2), 0.5 * (mu2 - mu1)
    x, w = np.polynomial.legendre.leggauss(int(npts))
    th, w = 0.5 * np.pi * (x + 1), 0.5 * np.pi * w
    return c + r * np.exp(1j * th), -1j * r * np.exp(1j * th) * w


def slices(text, npts, Ec):
    """Contours "M1,M2[,NPTS[,WEIGHT]]" joined by ":", energies relative to Ec -> [(mu1, mu2, npts, weight)], absolute."""
    out = []
    for c in text.split(":"):
        v = c.split(",")
        assert 2 <= len(v) <= 4 and float(v[0]) < float(v[1]), f"contour {c!r}: M1,M2[,NPTS[,WEIGHT]]"
        out.append((Ec + float(v[0]), Ec + float(v[1]), int(v[2]) if len(v) > 2 else npts, float(v[3]) if len(v) > 3 else 1.0))
    return out


def contours(sl):
    """Points and steps of rho = sum_s w_s rho_s: the integral is linear, so a weight only scales the steps of its contour
    (the cut of the basis acts like cut / w_s on a state of the slice s alone)."""
    q = [contour(a, b, n) for a, b, n, _ in sl]
    return np.concatenate([z for z, _ in q]), np.concatenate([w * dz for (_, dz), (_, _, _, w) in zip(q, sl)])


class OBC:
    def __init__(self, dev, comm, cache=None, tol=1e-4, log=print):
        self.dev, self.comm, self.cache, self.tol, self.log = dev, comm, cache, tol, log
        self.mem = {}                                                  # without a cache: {z: (SL, SR)}, {("inj", z): B}

    def _f(self, z, pre):
        return os.path.join(self.cache, f"{pre}_{z.real:+.8f}_{z.imag:.8e}.npz")

    def _put(self, z, pre, **arrs):
        f = self._f(z, pre)
        tmp = dist.tmpname(f)
        np.savez(tmp, **arrs)
        os.replace(tmp, f)

    def prepare(self, zs, reals):
        """Sigma at the contour points zs and the energies reals (z = E + i eta) and the injection at reals. Collective."""
        t, me, size = time.time(), self.comm.rank, self.comm.size
        pts = [(complex(z), False) for z in zs] + [(complex(z), True) for z in reals]
        if self.cache:
            os.makedirs(self.cache, exist_ok=True)
            todo = [(z, r) for z, r in pts if not os.path.exists(self._f(z, "sig")) or (r and not os.path.exists(self._f(z, "inj")))]
            todo = self.comm.bcast(todo)
        else:
            todo = [(z, r) for z, r in pts if z not in self.mem]
        todo.sort(key=lambda p: p[0].imag)                             # the slow points (close to the axis) on different ranks
        mine = {}
        for z, real in todo[me::size]:
            if self.cache and os.path.exists(self._f(z, "sig")):
                c = np.load(self._f(z, "sig"))
                S = c["SL"], None
            else:
                S = self.dev._sigma(z)[0]
                if self.cache:
                    self._put(z, "sig", SL=S[0], SR=S[1])
            if not self.cache:
                mine[z] = S
            if real:
                lam, B = self.eig(S[0])
                if self.cache:
                    self._put(z, "inj", lam=lam, B=B)
                else:
                    mine[("inj", z)] = B
            gpu.free()
        if self.cache:
            self.comm.barrier()
            for z, real in todo:
                dist.wait_file(self._f(z, "sig"))
                if real:
                    dist.wait_file(self._f(z, "inj"))
        else:
            for d in self.comm.allgather(mine):
                self.mem.update(d)
        self.log(f"obc: {len(todo)} of {len(pts)} points computed ({len(zs)} contour, {len(reals)} real) on {size} ranks, {time.time() - t:.1f} s")

    def eig(self, SL):
        """Eigenpairs of Gamma_L = i (Sigma_L - Sigma_L^H) above tol: (values, columns B with Gamma_L ~ B B^H), ascending."""
        xp = gpu.cp if gpu.on else np
        S = xp.asarray(SL)
        G = 1j * (S - S.conj().T)
        lam, X = xp.linalg.eigh(0.5 * (G + G.conj().T))
        k = lam > self.tol
        return gpu.asnumpy(lam[k]), gpu.asnumpy(X[:, k] * xp.sqrt(lam[k]))

    def sigma(self, z, side):
        """Sigma of side 0 (left) / 1 (right): numpy."""
        z = complex(z)
        return np.load(self._f(z, "sig"))[("SL", "SR")[side]] if self.cache else self.mem[z][side]

    def inject(self, z):
        z = complex(z)
        return np.load(self._f(z, "inj"))["B"] if self.cache else self.mem[("inj", z)]
