"""Tile-built level hierarchy for the multilevel solver of qtx_ms.schwarz (--basis, --levels 3).

Level 1: per-slab tile basis U1_s (run_ms --save-basis: <prefix>_U1_slab<s>.npy, orthonormal columns), A1 = U1^H A U1
from cached U1^H H U1, U1^H S U1 (msbasis.slab_blocks) + U1^H Sigma U1 on the lead faces per energy (dense, N1 x N1).
Level 2: V_s in level-1 coordinates (<prefix>_V_slab<s>.npy: contour slab rho of the open level-1 system),
A2 = V^H A1 V = U^H A U with U = U1 V; exact dense LU. No direct (fine-slab RGF) basis anywhere.
Level-1 smoother: restricted additive Schwarz on windows of level-1 unknowns, selected by the centroid along x of each
level-1 vector; absorbing layer projected from the fine level, A1_ww + i s U_w^H (P S P) U_w, P the quadratic ramp
over the outer d Å of the window's orbital extent towards each artificial cut (lead faces keep Sigma).

V-cycle (preconditioner of flexible GMRES on a fine residual r):
    z = S0 r;  r1 = U1^H (r - A z)
        y = S1 r1;  y += V A2^-1 V^H (r1 - A1 y);  y += S1 (r1 - A1 y)
    z += U1 y;  z += S0 (r - A z)
l1 = "exact": y = A1^-1 r1 (two-level with the tile basis as coarse space); "coarse": no level-1 smoother.
"""
import time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from qtx_ms import msbasis as mb

log = mb.log


class Levels:
    def __init__(self, dev, inp, prefix):
        nsl = len(dev.ns)
        self.dev = dev
        self.Us = [np.load(f"{prefix}_U1_slab{s}.npy") for s in range(nsl)]
        self.Vs = [np.load(f"{prefix}_V_slab{s}.npy") for s in range(nsl)]
        self.N1 = [u.shape[1] for u in self.Us]
        self.N2 = [v.shape[1] for v in self.Vs]
        self.off1 = np.concatenate([[0], np.cumsum(self.N1)])
        self.U1 = sp.block_diag([sp.csr_matrix(u) for u in self.Us], format="csr")
        self.U1H = self.U1.conj().T.tocsr()
        self.V = sp.block_diag([sp.csr_matrix(v) for v in self.Vs], format="csr").toarray()
        t = time.time()
        blk = mb.slab_blocks(dev, [sp.csr_matrix(u) for u in self.Us])
        n1 = self.off1[-1]
        self.H1, self.S1 = np.zeros((n1, n1), complex), np.zeros((n1, n1), complex)
        for s in range(nsl):
            a, b = slice(self.off1[s], self.off1[s + 1]), slice(self.off1[s + 1], self.off1[s + 2]) if s < nsl - 1 else None
            self.H1[a, a], self.S1[a, a] = blk["HD"][s], blk["SD"][s]
            if b is not None:
                self.H1[a, b], self.S1[a, b] = blk["HU"][s], blk["SU"][s]
                self.H1[b, a], self.S1[b, a] = blk["HU"][s].conj().T, blk["SU"][s].conj().T
        self.PL = self.U1[dev.bL].toarray()                   # lead-face rows of U1 (all level-1 columns)
        self.PR = self.U1[dev.bR].toarray()
        ax = inp["meta"]["axis"]
        atom = np.repeat(np.arange(len(inp["norb"])), inp["norb"])
        self.x = inp["xyz"][atom[dev.dev], ax]
        w = np.abs(self.U1.multiply(self.U1.conj())).tocsc()   # |U1|^2: centroid and orbital support per column
        self.xc = (w.T @ self.x) / np.asarray(w.sum(0)).ravel()
        self.W = w                                             # |U1|^2, device orbitals x level-1 columns
        self.inp, self.ax, self.prefix = inp, ax, prefix
        self.atom_d = atom[dev.dev]                            # atom of every device orbital
        self.Sf = dev.S.tocsr()
        log(f"Levels: N1 {n1} {self.N1}, N2 {sum(self.N2)} {self.N2}; level-1 blocks {time.time() - t:.0f} s")

    def set_energy(self, E, SL, SR, A=None):
        self.A1 = E * self.S1 - self.H1 - self.PL.conj().T @ SL @ self.PL - self.PR.conj().T @ SR @ self.PR
        if A is not None:                                      # consistency with the fine operator
            y = np.random.default_rng(0).standard_normal((self.A1.shape[0], 2)) + 0j
            d = np.linalg.norm(self.A1 @ y - self.U1H @ (A @ (self.U1 @ y))) / np.linalg.norm(self.A1 @ y)
            assert d < 1e-8, f"A1 != U1^H A U1 ({d:.1e})"
        self.A2 = self.V.conj().T @ self.A1 @ self.V
        self.lu2 = sla.lu_factor(self.A2, check_finite=False)

    def coarse2(self, r1):
        """Level-2 correction in level-1 coordinates: V A2^-1 V^H r1."""
        return self.V @ sla.lu_solve(self.lu2, self.V.conj().T @ r1, check_finite=False)

    def ms(self, b):
        """Tile mode-space solution U1 V A2^-1 V^H U1^H b (iteration 0)."""
        return self.U1 @ self.coarse2(self.U1H @ b)

    def exact1(self):
        lu = sla.lu_factor(self.A1, check_finite=False)
        return lambda r1: sla.lu_solve(lu, r1, check_finite=False)

    def regions(self):
        """Transverse tile regions per slab: for slab s a list of atom-id arrays, one per transverse piece (tpar) of the
        tile partition the basis was built on (<prefix>_partition.json; default tiles of ~2500 orbitals, no x split)."""
        import json, os
        f = f"{self.prefix}_partition.json"
        cfg = json.load(open(f)) if os.path.exists(f) else dict(tile_orbs=2500, xsplit=1, cores=False, d_core=6.0)
        blocks, _, _ = mb.partition(self.dev, cores=cfg["cores"], d_core=cfg["d_core"], tile_orbs=cfg["tile_orbs"],
                                    xsplit=cfg["xsplit"], axis=self.ax, cut=mb.parse_cut(cfg.get("cut", "")))
        reg = [dict() for _ in self.dev.ns]
        for b in blocks:
            if b["kind"] == "tile":
                reg[b["slab"]].setdefault(b["tpar"], []).append(b["atoms"])
        return [[np.concatenate(v) for _, v in sorted(r.items())] for r in reg], cfg

    def sigma_t(self, ref):
        """Signed transverse coordinate of every device orbital relative to each region of `ref` (list of atom arrays):
        (d_r - d_other) / 2 with d the minimum-image distance in the transverse plane (all x); < 0 inside region r."""
        from scipy.spatial import cKDTree
        tr = [i for i in range(3) if i != self.ax]
        per = dict(mb.periodic_axes(self.inp["meta"].get("periodic")))
        xyz = self.inp["xyz"]
        lo = xyz[:, tr].min(0) - 1.0
        box = np.array([per.get(t, 2 * np.ptp(xyz[:, t]) + 1e3) for t in tr])
        q = lambda a: (xyz[a][:, tr] - lo) % box
        pts = q(np.arange(len(xyz)))[self.atom_d]
        if len(ref) == 1:
            return np.full((1, len(pts)), -np.inf)
        dist = np.array([cKDTree(q(r), boxsize=box).query(pts)[0] for r in ref])
        out = np.empty_like(dist)
        for k in range(len(ref)):
            out[k] = 0.5 * (dist[k] - np.delete(dist, k, 0).min(0))
        return out

    def schwarz1(self, centres, width, s=2.0, d=2.0, split=False, tov=2.5, wmin=0.1):
        """Level-1 additive Schwarz (callable r1 -> y): windows = level-1 columns with centroid within width/2 of a
        centre (cores: nearest centre), absorbing term projected from the fine orbitals of the window.
        split: every x-window is also cut transversely along the tile cuts (regions = transverse tile pieces of the slab
        with the most pieces among the window's slabs); a column belongs to region r if its weight on the orbitals within
        tov Å of r (signed transverse coordinate < tov) is >= wmin, its core is the region holding most of its weight;
        absorbing ramp over the outer d Å of r + tov as well."""
        xc, x = self.xc, self.x
        near = np.argmin(np.abs(xc[:, None] - centres[None, :]), axis=1)
        idx = [np.where((np.abs(xc - c) < width / 2) | (near == k))[0] for k, c in enumerate(centres)]
        core = [near[i] == k for k, i in enumerate(idx)]
        tcut = [None] * len(idx)                               # per window: signed transverse coordinate per orbital
        if split:
            reg, cfg = self.regions()
            slab_of = np.repeat(np.arange(len(self.N1)), self.N1)
            cache = {}
            idx2, core2, tcut2 = [], [], []
            for i, c in zip(idx, core):
                sls = np.unique(slab_of[i])
                sref = int(sls[np.argmax([len(reg[q]) for q in sls])])
                if sref not in cache:
                    sg = self.sigma_t(reg[sref])
                    WT = self.W.T.tocsr()
                    cache[sref] = (sg, np.array([WT @ (g < tov).astype(float) for g in sg]),
                                   np.array([WT @ (g < 0).astype(float) for g in sg]))
                sg, win, own = cache[sref]
                win, own = win[:, i], own[:, i]
                home = np.argmax(own, axis=0)
                for r in range(len(sg)):
                    m = (win[r] >= wmin) | (home == r)
                    idx2.append(i[m])
                    core2.append(c[m] & (home[m] == r))
                    tcut2.append(sg[r])
            idx, core, tcut = idx2, core2, tcut2
        xmin, xmax = x.min(), x.max()
        ops = []
        for i, sg in zip(idx, tcut):
            Uw = self.U1[:, i].tocsc()
            rows = np.unique(Uw.nonzero()[0])
            xi = x[rows]
            lo, hi = xi.min(), xi.max()
            ramp = np.zeros(len(rows))
            if lo > xmin + 1e-6:
                ramp = np.maximum(ramp, np.clip((lo + d - xi) / d, 0, 1) ** 2)
            if hi < xmax - 1e-6:
                ramp = np.maximum(ramp, np.clip((xi - (hi - d)) / d, 0, 1) ** 2)
            if sg is not None and np.isfinite(sg).all():       # transverse cut: ramp over the outer d of r + tov
                ramp = np.maximum(ramp, np.clip((sg[rows] - (tov - d)) / d, 0, 1) ** 2)
            P = sp.diags(np.sqrt(ramp))
            Ur = Uw[rows].toarray()
            C = Ur.conj().T @ (P @ self.Sf[rows][:, rows] @ P @ Ur)
            ops.append(sla.lu_factor(self.A1[np.ix_(i, i)] + 1j * s * C, check_finite=False))
        ncore = np.zeros(self.A1.shape[0], int)
        for i, c in zip(idx, core):
            ncore[i[c]] += 1
        assert (ncore == 1).all(), f"level-1 Schwarz: {np.sum(ncore != 1)} columns without exactly one core window"
        log(f"  level-1 Schwarz: {len(idx)} windows of {width} Å at x = {', '.join(f'{c:.1f}' for c in centres)}"
            + (f", transverse tile cuts (+{tov} Å, column weight >= {wmin})" if split else "") +
            f", {min(map(len, idx))}..{max(map(len, idx))} unknowns (sum {sum(map(len, idx))}), absorbing {s} eV over {d} Å")

        def apply(r1):
            y = np.zeros_like(r1)
            for i, c, lu in zip(idx, core, ops):
                y[i[c]] = sla.lu_solve(lu, r1[i], check_finite=False)[c]
            return y
        return apply

    def cycle1(self, S1):
        """Level-1 correction for a fine residual: r -> U1 y, y from the level-1 two-grid cycle (S1 None: level 2 only;
        S1 'exact': A1^-1)."""
        def corr(r):
            r1 = self.U1H @ r
            if S1 is None:
                y = self.coarse2(r1)
            elif S1 == "exact":
                y = self.A1inv(r1)
            else:
                y = S1(r1)
                y = y + self.coarse2(r1 - self.A1 @ y)
                y = y + S1(r1 - self.A1 @ y)
            return self.U1 @ y
        if S1 == "exact":
            self.A1inv = self.exact1()
        return corr
