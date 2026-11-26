"""Open boundary: Sigma_L(z), Sigma_R(z) and the injection at the real energies (folded: eigenpairs of Gamma_L; Beyn: the incoming
lead modes, no eigenproblem of Gamma) and the contour points.
The points are dealt out to the ranks; decimation, products and eigh run on the GPU of the rank. Results are files in `cache`
(the format of the other drivers) or, without a cache, shared in memory."""
import os, time
import numpy as np
import scipy.sparse as sp

from .. import dist, gpu
from .. import msbasis as mb


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


def smallest_cell(inp, orbs):
    """(sections, orbital order): the smallest cell whose repetition along the transport axis gives the lead cell of the orbitals orbs: the
    largest number of sections with which the atoms of all cells match (mb.lead_cell_perm, positions to 1e-2 and the same labels)."""
    m, norb = inp["meta"], inp["norb"]
    off = np.concatenate([[0], np.cumsum(norb)])
    atoms = np.unique(np.repeat(np.arange(len(norb)), norb)[orbs])
    L = m.get("block_length") or m.get("lead_cell_length")
    given = int(m.get("cells_per_block") or m.get("source", {}).get("cells_per_block", 1))
    lab = np.asarray(inp["labels"])
    for s in range(len(atoms), 1, -1):
        if len(atoms) % s or len(orbs) % s or s % given:
            continue
        try:
            p = mb.lead_cell_perm(dict(inp, meta=dict(m, lead_cell_length=L) if L else m), orbs, s)
        except (AssertionError, ValueError, IndexError):
            continue
        a = np.searchsorted(off, np.asarray(orbs)[p], side="right") - 1        # atom of every orbital, in the new order
        if (lab[a].reshape(s, -1) == lab[a].reshape(s, -1)[0]).all():
            return s, p
    if given > 1:
        raise ValueError(f"the lead cell is not {given} repeated cells")
    return 1, None


class OBC:
    """model: "folded" (lead cell as given, decimation), "beyn" (periodized lead of qtx_ms.obc_beyn at every z) or "beyn-real" (Beyn at
    the real energies only, the contour keeps the folded lead: the same bases as "folded"), or "closed" (no lead at any z: the Sigma are None, A = E S - H, also on
    the contour; the right-hand sides are unit vectors of the left contact face, run.py). Files of Beyn points: bsig, bmode (B of the
    incoming modes, B B^H = Gamma_L)."""
    real_eta = 1e-5                                                    # |Im z| of a real-axis point

    def __init__(self, dev, comm, cache=None, tol=1e-4, log=print, model="folded", basis_dir=None, shift_R=0.0):
        """basis_dir: folder of the coupling bases of the Beyn leads (they do not depend on z; None: not stored).
        shift_R: the right lead and the device next to it shifted rigidly by -shift_R S (another drain bias): z S - H' = (z + shift_R) S - H,
        so Sigma_R'(z) = Sigma_R(z + shift_R) of the given lead (Beyn only; the left lead and the injection are unchanged)."""
        assert model in ("folded", "beyn", "beyn-real", "closed"), model
        assert not shift_R or model == "beyn", "shift_R: --obc beyn"
        self.dev, self.comm, self.cache, self.tol, self.log, self.model, self._leads = dev, comm, cache, tol, log, model, {}
        self.basis_dir, self.shift = basis_dir, (0.0, float(shift_R))
        self.mem = {}                                                  # without a cache: {(z, side): Sigma}, {("inj", z): B}

    def is_beyn(self, z):
        return self.model == "beyn" or (self.model == "beyn-real" and abs(complex(z).imag) <= self.real_eta)

    def _f(self, z, pre, side=""):
        """File of a point: folded "sig" (SL, SR), "inj"; Beyn per side "bsigL" / "bsigR" (S), "bmode" (B); "bsig" (SL, SR) of older runs."""
        if self.is_beyn(z):
            pre = {"sig": "bsig", "inj": "bmode"}[pre]
        return os.path.join(self.cache, f"{pre}{side}_{z.real:+.8f}_{z.imag:.8e}.npz")

    def _put(self, f, **arrs):
        tmp = dist.tmpname(f)
        np.savez(tmp, **arrs)
        os.replace(tmp, f)

    def _sigfile(self, z, side):
        """(file, key) of Sigma of one side (0 / 1) at z, None if not there."""
        if self.is_beyn(z):
            f = self._f(z, "sig", "LR"[side])
            if os.path.exists(f):
                return f, "S"
        f = self._f(z, "sig")
        return (f, ("SL", "SR")[side]) if os.path.exists(f) else None

    def prepare(self, zs, reals):
        """Sigma at the contour points zs and the energies reals (z = E + i eta) and the injection at reals. Collective.
        Beyn: one task per point and side; the left lead on the even ranks, the right lead on the odd ranks (one lead and one coupling
        basis per rank). Folded: one task per point (both sides), on all ranks."""
        if self.model == "closed":
            self.log("obc: closed, no lead self-energy")
            return
        t, me, size = time.time(), self.comm.rank, self.comm.size
        pts = [(complex(z), False) for z in zs] + [(complex(z), True) for z in reals]
        tasks = []                                                     # (z, real, side): side 0 / 1 (Beyn) or None (folded, both)
        for z, real in pts:
            for side in ((0, 1) if self.is_beyn(z) else (None,)):
                tasks.append((z + self.shift[side or 0], real, side))
        if self.shift[1]:
            self.log(f"obc: right lead shifted by {-self.shift[1]:+.4f} S: Sigma_R(z) of the lead at z {self.shift[1]:+.4f}")
        if self.cache:
            os.makedirs(self.cache, exist_ok=True)

            def missing(z, real, side):
                if side is None:
                    return not os.path.exists(self._f(z, "sig")) or (real and not os.path.exists(self._f(z, "inj")))
                return self._sigfile(z, side) is None or (real and side == 0 and not os.path.exists(self._f(z, "inj")))
            todo = self.comm.bcast([q for q in tasks if missing(*q)])
        else:
            todo = [q for q in tasks if (q[0], q[2] or 0) not in self.mem]
        todo.sort(key=lambda q: q[0].imag)                             # the slow points (close to the axis) on different ranks
        group = {0: list(range(0, size, 2)), 1: list(range(1, size, 2)) or [0], None: list(range(size))}
        seen = {g: 0 for g in group}
        mine, files = {}, []
        for z, real, side in todo:
            ranks = group[side]
            owner = ranks[seen[side] % len(ranks)]
            seen[side] += 1
            if self.cache and side is None:
                files += [self._f(z, "sig")] + ([self._f(z, "inj")] if real else [])
            elif self.cache:
                files += [self._f(z, "sig", "LR"[side])] + ([self._f(z, "inj")] if real and side == 0 else [])
            if owner != me:
                continue
            B = None
            if side is None:
                S = self.dev._sigma(z)[0]
                if self.cache:
                    self._put(self._f(z, "sig"), SL=S[0], SR=S[1])
                mine.update({(z, 0): S[0], (z, 1): S[1]})
                if real:
                    t1 = time.time()
                    lam, B = self.eig(S[0])
                    self.log(f"obc: Gamma_L eigenproblem of {len(S[0])}: {time.time() - t1:.1f} s")
                    if self.cache:
                        self._put(self._f(z, "inj"), lam=lam, B=B)
            else:
                S, B = self._beyn_side(z, side)
                if self.cache:
                    self._put(self._f(z, "sig", "LR"[side]), S=S)
                mine[z, side] = S
                if real and side == 0 and self.cache:
                    self._put(self._f(z, "inj"), B=B)
            if real and B is not None and (side is None or side == 0):
                mine["inj", z] = B
            gpu.free()
        if self.cache:
            self.comm.barrier()
            for f in files:
                dist.wait_file(f)
        else:
            for d in self.comm.allgather(mine):
                self.mem.update(d)
        nb = sum(1 for q in todo if q[2] is not None)
        self.log(f"obc: {len(todo)} of {len(tasks)} tasks computed ({len(zs)} contour, {len(reals)} real points; {nb} Beyn tasks of one side) "
                 f"on {size} ranks, {time.time() - t:.1f} s")

    def _lead(self, side):
        """(BeynLead, couplings) of one lead side ("L" / "R"), made on first use."""
        from ..obc_beyn import BeynLead
        if side in self._leads:
            return self._leads[side]
        dev, inp, m = self.dev, self.dev.inp, self.dev.meta
        t0 = time.time()
        tr, ob = m["transport"], np.repeat(inp["blocks"], inp["norb"])
        blk, mat, cpl = {"L": ("lead_left_block", inp["lead"], dev.cL), "R": ("lead_right_block", inp.get("lead_right") or inp["lead"], dev.cR)}[side]
        P01 = abs(sp.csr_matrix(mat["H01"])) + abs(sp.csr_matrix(mat["S01"]))     # faces as mb.FoldedLead: F, or Rf (left) / Lf (right)
        Rf, Lf = (np.unique(q) for q in P01.nonzero())
        face = np.union1d(Rf, Lf)
        if len(face) != cpl[0].shape[0]:
            face = Rf if side == "L" else Lf
        assert len(face) == cpl[0].shape[0]
        sec, pm = smallest_cell(inp, np.where(ob == tr[blk])[0])
        n, full = mat["H00"].shape[0], []
        bl = BeynLead(mat["H00"], mat["H01"], mat["S00"], mat["S01"], sec, pm, sides=side, layers_on="host", fp_store="host",
                      m0=min(max(64, n // sec // 2), n // sec))    # the default of BeynLead, at most the primitive cell
        bl.basis_dir = self.basis_dir
        for q, c in enumerate(cpl):                            # lead face -> whole lead cell
            c = sp.csr_matrix(c)
            E = sp.csr_matrix((np.ones(len(face)), (np.asarray(face), np.arange(len(face)))), shape=(n, len(face)))
            full.append((E @ c if q < 2 else c @ E.T).tocsr())
        self._leads[side] = bl, full
        self.log(f"obc: Beyn, lead {side}: {'periodized, %d primitive cells of %d orbitals' % (sec, bl.m) if sec > 1 else 'cell as given'}; "
                 f"nonzero blocks H {sorted(bl.layer[side, 'H'])}, S {sorted(bl.layer[side, 'S'])}, model {self.model}; {time.time() - t0:.1f} s")
        return self._leads[side]

    def _beyn_side(self, z, side):
        """Sigma of one side (0 / 1) and, for the left side, B (incoming modes; real axis) from the periodized lead (obc_beyn); real axis at E = Re z."""
        sd = "LR"[side]
        bl, (Hld, Sld, Hdl, Sdl) = self._lead(sd)
        z = complex(z.real, 0.0) if abs(z.imag) <= self.real_eta else complex(z)
        t = time.time()
        nb = len(bl._bases)
        S, b = bl.sigma(z, sd, z * Sdl - Hdl, z * Sld - Hld)
        bs = bl._bases.get(sd, {})
        how = ("loaded" if bs.get("loaded") else "built") if len(bl._bases) > nb else "kept"
        self.log(f"obc: Beyn {sd} at {z.real:+.6f}{z.imag:+.1e}i: {time.time() - t:.1f} s (coupling basis {how}, {bs.get('t', 0):.1f} s; "
                 f"fixed point {bl.info.get('fp_it', 0)} its)" + (f", {b.shape[1]} incoming modes" if side == 0 else ""))
        gpu.free()
        return S, (b if side == 0 else None)

    def _beyn(self, z):
        """(Sigma_L, Sigma_R), B of both sides at z (one rank)."""
        (SL, B), (SR, _) = self._beyn_side(z, 0), self._beyn_side(z, 1)
        return (SL, SR), B

    def eig(self, SL):
        """Eigenpairs of Gamma_L = i (Sigma_L - Sigma_L^H) above tol: (values, columns B with Gamma_L ~ B B^H), ascending."""
        xp = gpu.cp if gpu.on else np
        S = xp.asarray(SL)
        G = 1j * (S - S.conj().T)
        lam, X = xp.linalg.eigh(0.5 * (G + G.conj().T))
        k = lam > self.tol
        return gpu.asnumpy(lam[k]), gpu.asnumpy(X[:, k] * xp.sqrt(lam[k]))

    def sigma(self, z, side):
        """Sigma of side 0 (left) / 1 (right): numpy (closed: None)."""
        if self.model == "closed":
            return None
        z = complex(z) + self.shift[side]
        if not self.cache:
            return self.mem[z, side]
        f, key = self._sigfile(z, side)
        return np.load(f)[key]

    def inject(self, z):
        z = complex(z)
        return np.load(self._f(z, "inj"))["B"] if self.cache else self.mem[("inj", z)]
