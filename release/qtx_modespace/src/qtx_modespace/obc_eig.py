"""Open-boundary eigenmodes of a QuaTrEx device: T(z) v = 0, T(z) = z S - H - Sigma_L(z) - Sigma_R(z), by Beyn's
contour-integral method, with the span of the eigenvectors against a saved mode-space basis and a plot of the modes.

The eigenvalues are the resonances (Im < 0) and bound states of the open device. They lie on the analytic continuation
of the retarded Sigma^R(E + i0) below the real axis; the decimation Sigma(z) is analytic off the axis on the physical
sheet and has no device eigenvalues there. The continuation is built from the lead Bloch modes of the folded chain
(core.FoldedLead: A00 = interior folded onto the faces F, A01, A10):
    F_R = -g_R A10  (right-extending chain, eigenvalues lambda, |lambda| < 1),  F_L = -g_L A01  (mu = 1 / lambda)
from one decimation at z. Below the axis the physical right set holds the propagating modes with v < 0 where the
retarded sheet needs the v > 0 ones (found in the left set), and vice versa. The modes are classified by their flux
J = 2 Im(lambda phi^H A01(Re z) phi) / phi^H phi and exactly n swapped (n = number of propagating channels at the
contour centre, counted on the real axis), F_2 = F + (Phi' L' - F Phi') (W_K^H Phi')^-1 W_K^H, g_R2 = (A00 + A01 F_R2)^-1,
g_L2 = (A00 + A10 F_L2)^-1. With a fixed count the continued Sigma is analytic between consecutive lead subband edges
(branch points on the real axis), so a contour must lie inside one such interval (a few meV away from the edges).

Beyn (block Hankel, kmom moments, r random probes V): A_p = (1/2 pi i) oint ((z - c)/scale)^p T(z)^-1 V dz,
B0 = [A_{i+j}] = W s V^H (rank k), eig of W^H B1 V s^-1 -> eigenvalues, eigenvectors W[:n] y. The rank must stay
below r * kmom (warned otherwise). T(z)^-1 V by a dense block-tridiagonal LU over the device slabs.

    python -m qtx_modespace.obc_eig DEVICE --out eig_c1 --center 0.017 --a 0.015 --b 0.01 --npts 24 --probes 200 \\
        --modes data/mos2_hfo2_n400/modes --plot
    # in parallel: the same command with --points 0,3,6,... per worker (solves cached in <out>/X), then once without

Output (<out>): eigpairs.npz (lam, vec: device orbitals x modes in core.Device order, orbitals: QuaTrEx index of each
row, missing / missing_slab with --modes), eig.json (eigenvalues, residuals, span, contour), eigmodes.png (--plot:
Mulliken weight per atom in two projections and the profile along transport per species), X/ (per-point solves),
channels_<centre>.json, and the continued Sigma in the --sigma-cache (sig2_*.npz).
"""
import argparse, json, os, time
import numpy as np
import scipy.linalg as sla

from .core import BlockTridiag, CSolver, Device, sancho_both, slab_blocks, log
from .quatrex import read_quatrex


# ------------------------------------------------------------------ continued (retarded-sheet) lead Sigma
def folded_blocks(lead, z):
    """A00 (interior folded onto the faces), A01, A10 of the folded lead chain at z."""
    A00 = z * lead.SFF - lead.HFF
    if len(lead.I):
        cs = CSolver(z * lead.SII - lead.HII)
        X = cs.solve(z * lead.SIF - lead.HIF)
        cs.free()
        A00 = A00 - (z * lead.SFI - lead.HFI) @ X
    return A00, z * lead.S01 - lead.H01, z * lead.S01.conj().T - lead.H01.conj().T


def _flux(lam, Phi, A01E):
    num = np.einsum("ij,ij->j", Phi.conj(), A01E @ Phi)
    return 2.0 * np.imag(lam * num) / np.einsum("ij,ij->j", Phi.conj(), Phi).real


def _modes(F, zero=1e-10):
    w, vl, vr = sla.eig(F, left=True, right=True, check_finite=False)
    nz = np.abs(w) > zero
    return w[nz], vl[:, nz], vr[:, nz]


def _swap(F, Wk, Pin, Lin):
    if Pin.shape[1] == 0:
        return F
    return F + (Pin * Lin - F @ Pin) @ np.linalg.solve(Wk.conj().T @ Pin, Wk.conj().T)


def continued_gfs(lead, z, nprop, jrel=1e-6, info=None):
    """Surface Green's functions g_L, g_R on the sheet continued from Sigma^R(E + i0); nprop = (n+, n-) channels."""
    z = complex(z)
    A00, A01, A10 = folded_blocks(lead, z)
    gL, gR = sancho_both(A00, A01, A10)
    if z.imag >= 0:
        if info is not None:
            info.update(swapped=0)
        return gL, gR
    FR, FL = -gR @ A10, -gL @ A01
    lR, WR, PR = _modes(FR)
    mL, WL, PL = _modes(FL)
    A01E = z.real * lead.S01 - lead.H01
    JR, JL = _flux(lR, PR, A01E), _flux(1.0 / mL, PL, A01E)
    jmax = max(np.abs(JR).max(initial=0), np.abs(JL).max(initial=0))
    lnR, lnL = np.abs(np.log(np.abs(lR))), np.abs(np.log(np.abs(mL)))
    cR = np.where(JR < -jrel * jmax)[0]
    cL = np.where(JL > jrel * jmax)[0]
    kR = cR[np.argsort(lnR[cR])][:nprop[1]]
    kL = cL[np.argsort(lnL[cL])][:nprop[0]]
    FR2 = _swap(FR, WR[:, kR], PL[:, kL], 1.0 / mL[kL])
    FL2 = _swap(FL, WL[:, kL], PR[:, kR], 1.0 / lR[kR])
    if info is not None:
        info.update(swapped=len(kR), swapped_L=len(kL))
    return np.linalg.inv(A00 + A10 @ FL2), np.linalg.inv(A00 + A01 @ FR2)


def check_flux_sign(lead, E, eta=1e-4, kappa=0.05):
    z = E + 1j * eta
    A00, A01, A10 = folded_blocks(lead, z)
    gL, gR = sancho_both(A00, A01, A10)
    lR, _, PR = _modes(-gR @ A10)
    mL, _, PL = _modes(-gL @ A01)
    A01E = E * lead.S01 - lead.H01
    JR, JL = _flux(lR, PR, A01E), _flux(1.0 / mL, PL, A01E)
    nR, nL = np.abs(np.log(np.abs(lR))) < kappa, np.abs(np.log(np.abs(mL))) < kappa
    return JR[nR], JL[nL]


def channel_counts(lead, E, eta=1e-4, kappa=0.05):
    """(n+, n-): propagating channels towards +x / -x at real E (lead modes at E + i eta)."""
    JR, JL = check_flux_sign(lead, E, eta, kappa)
    assert (JR > 0).all() and (JL < 0).all(), "flux sign check failed"
    return len(JR), len(JL)


class ContinuedSigma:
    """Sigma_L, Sigma_R of a core.Device on the continued retarded sheet (cache/sig2_n<n+>-<n->_<z>.npz)."""

    def __init__(self, dev, nprop, nprop_R=None, cache=None):
        self.dev, self.cache = dev, cache
        self.nprop, self.nprop_R = tuple(nprop), tuple(nprop if nprop_R is None else nprop_R)
        self.info = {}

    def __call__(self, z):
        z = complex(z)
        f = self.cache and os.path.join(self.cache, f"sig2_n{self.nprop[0]}-{self.nprop[1]}_{z.real:+.8f}_{z.imag:+.8e}.npz")
        if f and os.path.exists(f):
            c = np.load(f)
            self.info = json.loads(str(c["info"]))
            return c["SL"], c["SR"]
        d, info = self.dev, {}
        gL, gR = continued_gfs(d.lead, z, self.nprop, info=info)
        if d.leadR is not d.lead:
            iR = {}
            gR = continued_gfs(d.leadR, z, self.nprop_R, info=iR)[1]
            info["right_lead"] = iR
        out = tuple((z * Sdl - Hdl) @ g @ (z * Sld - Hld) for g, (Hld, Sld, Hdl, Sdl) in ((gL, d.cL), (gR, d.cR)))
        self.info = info
        if f:
            os.makedirs(self.cache, exist_ok=True)
            np.savez(f, SL=out[0], SR=out[1], info=json.dumps(info))
        return out


# ------------------------------------------------------------------ Beyn
def ellipse(center, a, b, npts):
    """Nodes and weights of (1/2 pi i) oint f dz on the ellipse center + a cos t + i b sin t (trapezoidal)."""
    th = 2 * np.pi * (np.arange(npts) + 0.5) / npts
    z = center + a * np.cos(th) + 1j * b * np.sin(th)
    dz = (-a * np.sin(th) + 1j * b * np.cos(th)) * (2 * np.pi / npts)
    return z, dz / (2j * np.pi)


def block_solver(dev, blk, z, SL, SR):
    """Dense block-tridiagonal LU of T(z) over the device slabs (Sigma on the end slabs)."""
    SL1 = np.zeros((dev.ns[0], dev.ns[0]), complex)
    SL1[np.ix_(dev.bL, dev.bL)] = SL
    SR1 = np.zeros((dev.ns[-1], dev.ns[-1]), complex)
    r = dev.bR - dev.sb[-2]
    SR1[np.ix_(r, r)] = SR
    return BlockTridiag(blk, z, SL1, SR1)


def solve_point(dev, sig, z, V, blk, cache=None, ncheck=4):
    """X = T(z)^-1 V, cached in cache/X_<z>.npy; residual checked on ncheck columns with the sparse T(z)."""
    f = cache and os.path.join(cache, f"X_p{V.shape[1]}_{z.real:+.8f}_{z.imag:+.8e}.npy")
    if f and os.path.exists(f):
        return np.load(f)
    t0 = time.time()
    SL, SR = sig(z)
    t1 = time.time()
    X = block_solver(dev, blk, z, SL, SR).solve(V)
    c = np.arange(min(ncheck, V.shape[1]))
    res = float((np.linalg.norm(dev.A(z, SL, SR) @ X[:, c] - V[:, c], axis=0) / np.linalg.norm(V[:, c], axis=0)).max())
    log(f"  z = {z.real:.5f}{z.imag:+.5f}i: Sigma {t1 - t0:.0f} s, solve {time.time() - t1:.0f} s, res {res:.1e}, "
        f"sheet {json.dumps(sig.info)}")
    if f:
        os.makedirs(cache, exist_ok=True)
        np.save(f, X)
    return X


def beyn(Xs, zs, ws, kmom=1, rank_tol=1e-8, center=0.0, scale=1.0):
    """Eigenpairs from Xs[j] = T(z_j)^-1 V: (lambda, unit eigenvectors, singular values of B0, rank)."""
    n = Xs[0].shape[0]
    A = [sum(w * ((z - center) / scale) ** p * X for z, w, X in zip(zs, ws, Xs)) for p in range(2 * kmom)]
    B0 = np.block([[A[i + j] for j in range(kmom)] for i in range(kmom)])
    B1 = np.block([[A[i + j + 1] for j in range(kmom)] for i in range(kmom)])
    W, s, Vh = np.linalg.svd(B0, full_matrices=False)
    k = int((s > rank_tol * s[0]).sum())
    W, sk, Vk = W[:, :k], s[:k], Vh[:k].conj().T
    lam, Y = np.linalg.eig((W.conj().T @ B1 @ Vk) / sk)
    vec = W[:n] @ Y
    return center + scale * lam, vec / np.linalg.norm(vec, axis=0), s, k


def residuals(dev, sig, lam, vec):
    """|T(lambda) v| / (max|T| |v|) (one continued Sigma per eigenvalue)."""
    out = []
    for l, v in zip(lam, vec.T):
        A = dev.A(l, *sig(l))
        out.append(float(np.linalg.norm(A @ v) / (abs(A).max() * np.linalg.norm(v))))
    return out


def span_missing(dev, V, vec):
    """|(I - P) v| / |v| per eigenvector for the block-diagonal basis U = diag(V_s) (orthonormal columns), and the
    per-slab share."""
    ms = np.zeros((len(V), vec.shape[1]))
    for s, U in enumerate(V):
        v = vec[dev.sb[s]:dev.sb[s + 1]]
        ms[s] = np.linalg.norm(v - U @ (U.conj().T @ v), axis=0) ** 2
    nv = np.linalg.norm(vec, axis=0) ** 2
    return np.sqrt(ms.sum(0) / nv), np.sqrt(ms / nv)


# ------------------------------------------------------------------ plot
def atom_weights(inp, dev, v):
    """Mulliken weight per atom, q_a = Re sum_{i in a} v_i^* (S v)_i, normalized to 1 (all atoms; 0 off the device)."""
    dorb = np.concatenate(dev.slab_orbs)
    atom = np.repeat(np.arange(len(inp["norb"])), inp["norb"])[dorb]
    q = np.real(v.conj() * (dev.S @ v))
    w = np.bincount(atom, q, minlength=len(inp["norb"]))
    return w / w.sum()


def plot_modes(fname, inp, dev, modes, axis=0):
    """One row per mode: weight per atom (log colour, marker size) in the (axis, t2) and (axis, t1) projections, and
    the weight per 1 A along the transport axis for each species; dashed lines = slab boundaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz, lab = inp["xyz"], np.asarray(inp["labels"])
    a2o = np.repeat(np.arange(len(inp["norb"])), inp["norb"])
    atoms = np.unique(a2o[np.concatenate(dev.slab_orbs)])
    t1, t2 = [i for i in range(3) if i != axis]
    x0 = xyz[atoms, axis].min()
    sb_x = [xyz[np.unique(a2o[o]), axis].min() - x0 for o in dev.slab_orbs[1:]]
    names = "xyz"
    n = len(modes)
    fig, ax = plt.subplots(n, 3, figsize=(16, 3.1 * n), squeeze=False, gridspec_kw=dict(width_ratios=[2.2, 2.2, 1.6]))
    sp_ = sorted(set(lab[atoms]))
    r = xyz[atoms]
    for i, (title, w) in enumerate(modes):
        wa = np.clip(w[atoms], 0, None)
        o = np.argsort(wa)
        for j, c2 in enumerate((t2, t1)):
            a = ax[i, j]
            a.scatter(r[:, axis] - x0, r[:, c2], s=0.3, c="0.85", lw=0)
            sc = a.scatter(r[o, axis] - x0, r[o, c2], s=2 + 120 * wa[o] / wa.max(), c=np.log10(wa[o] + 1e-12),
                           cmap="viridis", vmin=np.log10(wa.max()) - 3, vmax=np.log10(wa.max()), lw=0)
            for xb in sb_x:
                a.axvline(xb, color="r", lw=0.6, ls="--")
            a.set_xlabel(f"{names[axis]} (A)"); a.set_ylabel(f"{names[c2]} (A)")
            a.set_aspect("equal")
        plt.colorbar(sc, ax=ax[i, 1], label="log10 weight / atom", fraction=0.03)
        bins = np.arange(0, r[:, axis].max() - x0 + 1, 1.0)
        a = ax[i, 2]
        for s_ in sp_:
            m = lab[atoms] == s_
            h, _ = np.histogram(r[m, axis] - x0, bins, weights=wa[m])
            a.semilogy(0.5 * (bins[1:] + bins[:-1]), h + 1e-9, label=f"{s_} {100 * wa[m].sum():.0f}%")
        for xb in sb_x:
            a.axvline(xb, color="r", lw=0.6, ls="--")
        a.set_ylim(1e-6, 1); a.set_xlabel(f"{names[axis]} (A)"); a.set_ylabel("weight per 1 A")
        a.legend(fontsize=7, ncol=2)
        ax[i, 0].set_title(title, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ driver
def main(argv=None):
    p = argparse.ArgumentParser(prog="qtx_modespace.obc_eig", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="QuaTrEx device folder")
    p.add_argument("--out", required=True)
    p.add_argument("--cells", type=int)
    p.add_argument("--trim", action="store_true")
    p.add_argument("--potential", action="store_true")
    p.add_argument("--center", type=float, required=True, help="contour centre - Ec_lead (eV)")
    p.add_argument("--a", type=float, required=True, help="real semi-axis (eV)")
    p.add_argument("--b", type=float, required=True, help="imaginary semi-axis (eV)")
    p.add_argument("--npts", type=int, default=24)
    p.add_argument("--probes", type=int, default=200)
    p.add_argument("--kmom", type=int, default=1)
    p.add_argument("--rank-tol", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--points", help="only solve these contour points (i,j,... or i:j), for parallel runs")
    p.add_argument("--resid", type=int, default=-1, help="residual check of the n eigenvalues closest to the centre "
                   "(one continued Sigma each; default all)")
    p.add_argument("--modes", help="saved basis (slab<s>.npy, slab<s>_orbitals.npy): span of the eigenvectors")
    p.add_argument("--plot", action="store_true", help="eigmodes.png (needs matplotlib)")
    p.add_argument("--sigma-cache", help="lead self-energy cache (default <out>/sigma; 'none' = off)")
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    cache = None if a.sigma_cache == "none" else (a.sigma_cache or os.path.join(a.out, "sigma"))
    inp = read_quatrex(a.folder, cells=a.cells, trim=a.trim, potential=a.potential, cache_dir=cache, log=log)
    Ec = inp["meta"]["edges"]["Ec_lead"]
    dev = Device(inp, sigma_cache=cache)
    c = Ec + a.center
    fch = os.path.join(a.out, f"channels_{c:+.8f}.json")
    if os.path.exists(fch):
        nL, nR = (tuple(x) for x in json.load(open(fch)))
    else:
        nL = channel_counts(dev.lead, c)
        nR = nL if dev.leadR is dev.lead else channel_counts(dev.leadR, c)
        json.dump([nL, nR], open(fch, "w"))
    log(f"device {dev.n} orbitals, slabs {dev.ns}; contour Ec{a.center:+.4f} (a {a.a}, b {a.b}, {a.npts} points, "
        f"{a.probes} probes); propagating channels (n+, n-) left {nL}, right {nR}")
    sig = ContinuedSigma(dev, nL, nR, cache=cache)
    zs, ws = ellipse(c, a.a, a.b, a.npts)
    V = np.random.default_rng(a.seed).standard_normal((dev.n, a.probes)) + 0j
    blk = slab_blocks(dev)
    idx = range(a.npts)
    if a.points:
        idx = range(*map(int, a.points.split(":"))) if ":" in a.points else [int(x) for x in a.points.split(",")]
    xdir = os.path.join(a.out, "X")
    Xs = [solve_point(dev, sig, zs[j], V, blk, xdir) for j in idx]
    if a.points:
        log("points done")
        return
    lam, vec, s, k = beyn(Xs, zs, ws, a.kmom, a.rank_tol, c, max(a.a, a.b))
    inside = ((lam.real - c) / a.a) ** 2 + (lam.imag / a.b) ** 2 < 1
    log(f"Beyn: rank {k} of {len(s)}, {inside.sum()} eigenvalues inside")
    if k >= 0.8 * min(a.probes * a.kmom, len(s)):
        log("  WARNING: rank close to the number of probes x moments; increase --probes / --kmom")
    lam, vec = lam[inside], vec[:, inside]
    o = np.argsort(lam.real)
    lam, vec = lam[o], vec[:, o]
    out = dict(Ec=Ec, lam_re=lam.real.tolist(), lam_im=lam.imag.tolist(), dE=(lam.real - Ec).tolist(), sv=s.tolist(),
               rank=k, contour=dict(center=a.center, a=a.a, b=a.b, npts=a.npts, probes=a.probes, kmom=a.kmom),
               channels=dict(left=nL, right=nR))
    j = np.argsort(np.abs(lam - c))[:None if a.resid < 0 else a.resid]
    r = residuals(dev, sig, lam[j], vec[:, j])
    out["resid"] = {int(i): x for i, x in zip(j, r)}
    extra = {}
    if a.modes:
        Vb = []
        for s_ in range(len(dev.ns)):
            orb = np.load(os.path.join(a.modes, f"slab{s_}_orbitals.npy"))
            assert np.array_equal(orb, inp["perm"][dev.slab_orbs[s_]]), f"slab {s_} of {a.modes} does not match"
            Vb.append(np.load(os.path.join(a.modes, f"slab{s_}.npy")))
        m, ms = span_missing(dev, Vb, vec)
        out.update(modes=a.modes, missing=m.tolist(), missing_slab=ms.tolist())
        extra = dict(missing=m, missing_slab=ms)
    for i, l in enumerate(lam):
        log(f"  E = Ec{l.real - Ec:+.5f} {l.imag:+.2e}i eV" +
            (f", residual {out['resid'][i]:.1e}" if i in out["resid"] else "") +
            (f", missing in span(modes) {out['missing'][i]:.1e}" if a.modes else ""))
    np.savez(os.path.join(a.out, "eigpairs.npz"), lam=lam, vec=vec,
             orbitals=inp["perm"][np.concatenate(dev.slab_orbs)], slab_bounds=dev.sb, **extra)
    with open(os.path.join(a.out, "eig.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    if a.plot and len(lam):
        modes = [(f"OBC eigenmode {i}: E = Ec{l.real - Ec:+.4f} {l.imag:+.1e}i eV" +
                  (f", missing in span(modes) {out['missing'][i]:.1e}" if a.modes else ""), atom_weights(inp, dev, v))
                 for i, (l, v) in enumerate(zip(lam, vec.T))]
        plot_modes(os.path.join(a.out, "eigmodes.png"), inp, dev, modes, inp["meta"].get("axis", 0))
        log(f"-> {os.path.join(a.out, 'eigmodes.png')}")
    log(f"-> {a.out}")


if __name__ == "__main__":
    main()
