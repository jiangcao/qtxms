"""Prepare transport inputs for monolayer 2H-MoS2 from GPAW LCAO (non-orthogonal ab initio TB).

Stage 1, `dft` (GPAW environment, envs/gpaw/pixi.toml): SCF of the 6-atom orthorhombic cell, rotated so that
the armchair direction (a sqrt 3) is the transport axis x and the zigzag direction (a) is y. With symmetry
off, H(k), S(k) on the full odd k-mesh give the exact real-space blocks
    H(R) = (1/N) sum_k exp(2 pi i k.R) H(k)       (R in cell units; the mesh must exceed the coupling range)
(GPAW convention H(k) = sum_R exp(-2 pi i k.R) H(R)), saved with the mesh eigenvalues for the checks.

Stage 2, `device` (numpy / scipy only): slab = m transport cells x Ny transverse cells, folded at k_y = 0
(transverse period Ny a in meta["periodic"]); Ny must be a multiple of 3 for the K valley to fold to k_y = 0.
Structure [lead | device slabs | lead]. Three kinds of disorder in the device slabs only (leads pristine):
bond-length disorder (random displacements, hoppings scaled by (d0/d)^eta), S vacancies (orbitals removed,
unrelaxed, not self-consistent), and a background potential from oxide-interface roughness (Gaussian field)
and random interface dipoles. Lead band edges from the generalized eigenproblem of the lead cell.
Writes the generic input set (qtx_ms.prep.inputs) plus disorder.npz (V per atom, displacements, vacancy
positions, dipoles) and lead_bands.npz.

    pixi run --manifest-path envs/gpaw/pixi.toml python -m qtx_ms.prep.mos2 dft --out inputs/mos2_hr.npz
    uv run python -m qtx_ms.prep.mos2 device --hr inputs/mos2_hr.npz --length 200 --ny 9 --out inputs/mos2
"""
import argparse
import json
import time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla


# ------------------------------------------------------------------ stage 1: GPAW
def rect_cell(a=3.18, thickness=3.19, vacuum=10.0):
    """6-atom orthorhombic 2H-MoS2 cell: x = armchair (a sqrt 3), y = zigzag (a); atoms sorted S, Mo."""
    from ase.build import mx2, make_supercell
    hexa = mx2("MoS2", kind="2H", a=a, thickness=thickness, vacuum=vacuum)
    rect = make_supercell(hexa, [[1, 0, 0], [-1, 2, 0], [0, 0, 1]])        # same lattice as A2 = (0, a sqrt 3)
    c = rect.cell.lengths()[2]
    pos = rect.get_positions()
    rect.set_cell([[a * np.sqrt(3), 0, 0], [0, a, 0], [0, 0, c]], scale_atoms=False)
    rect.set_positions(np.c_[pos[:, 1], -pos[:, 0], pos[:, 2]])             # rotate by -90 deg about z
    rect.wrap()
    order = np.lexsort((rect.positions[:, 2], rect.positions[:, 1], rect.positions[:, 0], rect.numbers))
    return rect[order]


def dft(out, a=3.18, thickness=3.19, vacuum=10.0, kx=9, ky=15, basis="dzp", xc="PBE", txt="-"):
    from gpaw import GPAW, FermiDirac
    from gpaw.lcao.tools import get_lcao_hamiltonian
    rect = rect_cell(a, thickness, vacuum)
    assert kx % 2 and ky % 2, "odd k-mesh (symmetric R range)"
    calc = GPAW(mode="lcao", basis=basis, xc=xc, kpts=(kx, ky, 1), symmetry="off",
                occupations=FermiDirac(0.01), txt=txt)
    rect.calc = calc
    t = time.time()
    rect.get_potential_energy()
    Ef = calc.get_fermi_level()
    H_skMM, S_kMM = get_lcao_hamiltonian(calc)
    kpts = calc.get_ibz_k_points()                                          # = full mesh (symmetry off)
    assert len(kpts) == kx * ky
    eps = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(kpts))])
    Hk, Sk = H_skMM[0], S_kMM
    R = np.array([(i, j) for i in range(-(kx // 2), kx // 2 + 1) for j in range(-(ky // 2), ky // 2 + 1)])
    ph = np.exp(2j * np.pi * (kpts[:, :2] @ R.T))                            # (nk, nR)
    HR = np.einsum("kr,kmn->rmn", ph, Hk) / len(kpts)
    SR = np.einsum("kr,kmn->rmn", ph, Sk) / len(kpts)
    assert np.abs(HR.imag).max() < 1e-6 and np.abs(SR.imag).max() < 1e-8, "H(R) not real"
    norb = np.array([calc.wfs.setups[i].nao for i in range(len(rect))])
    np.savez(out, R=R, H=HR.real, S=SR.real, Ef=Ef, kpts=kpts, eps=eps, Hk=Hk, Sk=Sk,
             pos=gpaw_positions(rect.get_positions(), np.array(rect.cell)), numbers=rect.numbers, symbols=np.array(rect.get_chemical_symbols()),
             cell=np.array(rect.cell), norb=norb, info=json.dumps(dict(a=a, thickness=thickness, vacuum=vacuum,
                                                                        kpts=[kx, ky], basis=basis, xc=xc)))
    print(f"GPAW LCAO {basis} {xc}: {len(rect)} atoms, nao {Hk.shape[1]} {norb.tolist()}, E_F {Ef:.4f} eV, "
          f"{time.time() - t:.0f} s -> {out}")
    shell = {}
    for r, h, s in zip(R, HR.real, SR.real):
        shell[tuple(r)] = (np.abs(h).max(), np.abs(s).max())
    edge = [v for (i, j), v in shell.items() if abs(i) == kx // 2 or abs(j) == ky // 2]
    print(f"max |H|, |S| on the outer R shell (aliasing check): {max(e[0] for e in edge):.1e} eV, {max(e[1] for e in edge):.1e}")


def height_field(xy, L, rms, corr, rng):
    """Periodic random height h(x, y) on the torus L = (Lx, Ly): Fourier modes q = 2 pi (i/Lx, j/Ly) with
    amplitudes ~ exp(-q^2 corr^2 / 8) (Gaussian autocorrelation exp(-r^2 / corr^2)), scaled to the given rms.
    Returns h and grad h at the points xy."""
    nq = [int(np.ceil(4 * Li / (np.pi * corr))) + 1 for Li in L]
    q = np.array([(2 * np.pi * i / L[0], 2 * np.pi * j / L[1]) for i in range(-nq[0], nq[0] + 1)
                  for j in range(0, nq[1] + 1) if (j > 0 or i > 0)])
    amp = np.exp(-(q ** 2).sum(1) * corr ** 2 / 8) * rng.standard_normal(len(q))
    ph = rng.uniform(0, 2 * np.pi, len(q))
    arg = xy @ q.T + ph
    h = np.cos(arg) @ amp
    g = -(np.sin(arg) * amp) @ q
    f = rms / np.sqrt(0.5 * (amp ** 2).sum())                              # rms over the torus
    return f * h, f * g


def dft_super(out, nsc=(9, 9), rms=1.0, corr=12.0, seed=0, a=3.18, thickness=3.19, vacuum=10.0, basis="dzp",
              xc="PBE", h=0.2, txt="-"):
    """Corrugated monolayer: nsc[0] x nsc[1] orthorhombic cells, Mo mid-plane displaced by a periodic random
    height h(x, y) (rms, correlation length corr [A]); every atom keeps its distance to the mid-plane along the
    local normal n = (-h_x, -h_y, 1) / |.|, so the in-plane bonds stretch with the local slope (unrelaxed).
    GPAW LCAO at Gamma only: the supercell is longer than twice the coupling range (checked in stage 2), so
    H(Gamma) splits uniquely into the couplings to the neighbouring supercells along x, and is already folded
    at k_y = 0 along y. Run with MPI: mpiexec -n N python -m qtx_ms.prep.mos2 dft-super."""
    from gpaw import GPAW, FermiDirac
    from gpaw.lcao.tools import get_lcao_hamiltonian
    from gpaw.mpi import world
    rect = rect_cell(a, thickness, vacuum)
    at = rect.repeat((nsc[0], nsc[1], 1))
    order = np.lexsort((at.positions[:, 2], at.positions[:, 1], at.positions[:, 0], at.numbers))
    at = at[order]
    L = (at.cell[0, 0], at.cell[1, 1])
    flat = at.get_positions()
    zmid = flat[at.numbers == 42, 2].mean()
    hz, g = height_field(flat[:, :2], L, rms, corr, np.random.default_rng(seed))
    n = np.c_[-g, np.ones(len(at))]
    n /= np.linalg.norm(n, axis=1)[:, None]
    pos = np.c_[flat[:, :2], np.full(len(at), zmid) + hz] + (flat[:, 2] - zmid)[:, None] * n
    c = thickness + 2 * vacuum + 2 * np.abs(hz).max()
    at.set_cell([L[0], L[1], c], scale_atoms=False)
    pos[:, 2] += 0.5 * c - zmid
    flat[:, 2] += 0.5 * c - zmid
    at.set_positions(pos)
    calc = GPAW(mode="lcao", basis=basis, xc=xc, kpts=(1, 1, 1), symmetry="off", h=h,
                occupations=FermiDirac(0.01), parallel=dict(sl_auto=True), txt=txt)
    at.calc = calc
    t = time.time()
    at.get_potential_energy()
    Ef = calc.get_fermi_level()
    eps = calc.get_eigenvalues(kpt=0)
    H_skMM, S_kMM = get_lcao_hamiltonian(calc)
    if world.rank != 0:
        return
    Hg, Sg = H_skMM[0, 0], S_kMM[0]
    assert np.isrealobj(Hg) or np.abs(Hg.imag).max() < 1e-10
    Hg, Sg = Hg.real, Sg.real
    assert np.abs(Hg - Hg.T).max() < 1e-8 and np.abs(Sg - Sg.T).max() < 1e-10
    r, cc = np.nonzero((np.abs(Hg) > 1e-12) | (np.abs(Sg) > 1e-12))
    norb = np.array([calc.wfs.setups[i].nao for i in range(len(at))])
    cell = np.array(at.cell)
    np.savez(out, Hg_rows=r, Hg_cols=cc, Hg=Hg[r, cc], Sg=Sg[r, cc], nao=Hg.shape[0], Ef=Ef, eps=eps[None],
             pos=gpaw_positions(at.get_positions(), cell), pos_flat=flat, height=hz, numbers=at.numbers,
             symbols=np.array(at.get_chemical_symbols()), cell=cell, norb=norb, nsc=np.array(nsc),
             info=json.dumps(dict(a=a, thickness=thickness, vacuum=vacuum, kpts=[1, 1], basis=basis, xc=xc, h=h,
                                  nsc=list(nsc), corrugation=dict(rms=rms, corr=corr, seed=seed))))
    slope = np.linalg.norm(g, axis=1)
    print(f"GPAW LCAO {basis} {xc} corrugated {nsc[0]}x{nsc[1]}: {len(at)} atoms, nao {Hg.shape[0]}, height rms "
          f"{hz.std():.2f} A (max {np.abs(hz).max():.2f}), slope rms {np.sqrt((slope ** 2).mean()):.3f} max {slope.max():.3f}, "
          f"E_F {Ef:.4f} eV, {time.time() - t:.0f} s -> {out}")


# ------------------------------------------------------------------ stage 2: device
KE2 = 14.399645                                  # e^2 / (4 pi eps0) [eV A]


def gpaw_positions(pos, cell):
    """Atom positions as GPAW places the basis functions: scaled positions wrapped with % 1 (an atom at
    y = -1e-16 sits at y = a inside GPAW, which shifts its Bloch phases by one cell)."""
    return (np.linalg.solve(cell.T, pos.T).T % 1.0) @ cell


def load_hr(path, cut=1e-6):
    """Real-space blocks hr[R] = (H, S) in the physical convention: orbital mu of cell 0 couples to orbital
    nu of cell +R (verified by `check_convention`). GPAW's Bloch sum H(k) = sum_R exp(-2 pi i k.R) H_g(R)
    gives the transpose, hr[R] = H_g(-R) = H_g(R)^T. Elements below `cut` (|H| in eV and |S|) are dropped
    with a mask symmetric under R -> -R, transpose, and the pair is symmetrized, so supercells stay exactly
    Hermitian. z["pos"] is replaced by the GPAW-wrapped positions."""
    z = dict(np.load(path))
    if "Hg" in z:
        return z, split_gamma(z, cut)
    z["pos"] = gpaw_positions(z["pos"], z["cell"])
    raw = {tuple(int(v) for v in r): (h.T, s.T) for r, h, s in zip(z["R"], z["H"], z["S"])}
    hr = {}
    for (rx, ry), (h, s) in raw.items():
        hm, sm = raw[(-rx, -ry)]
        h, s = 0.5 * (h + hm.T), 0.5 * (s + sm.T)
        keep = (np.abs(h) > cut) | (np.abs(s) > cut)
        if keep.any():
            hr[(rx, ry)] = (np.where(keep, h, 0.0), np.where(keep, s, 0.0))
    return z, hr


def split_gamma(z, cut=1e-6, margin=1.0):
    """hr[(R, 0)], R = -1, 0, 1, from the Gamma-point H, S of a supercell (dft_super): each atom pair is
    assigned to the x image with the shortest distance (minimum image along y = the k_y = 0 fold).
    Unique when every nonzero pair is shorter than the second-nearest image by `margin` [A] (asserted).
    Also sets z["cell_range"]: coupling range in primitive cells along x (from the flat positions)."""
    nao = int(z["nao"])
    pos, cell, norb = z["pos"], z["cell"], z["norb"]
    Lx, Ly = cell[0, 0], cell[1, 1]
    at = np.repeat(np.arange(len(norb)), norb)
    H = sp.csr_matrix((z["Hg"], (z["Hg_rows"], z["Hg_cols"])), shape=(nao, nao)).toarray()
    S = sp.csr_matrix((z["Sg"], (z["Hg_rows"], z["Hg_cols"])), shape=(nao, nao)).toarray()
    d = pos[None, :, :] - pos[:, None, :]                                   # (a, b): b - a
    d[..., 1] -= Ly * np.round(d[..., 1] / Ly)
    dist = np.stack([np.linalg.norm(d + np.array([R * Lx, 0, 0]), axis=-1) for R in (-1, 0, 1)])
    Rbest = np.argmin(dist, 0) - 1
    ds = np.sort(dist, 0)
    # second-nearest y image, for the k_y = 0 fold
    dy2 = np.abs(np.abs(d[..., 1]) - Ly)
    nzat = np.zeros((len(norb), len(norb)), bool)
    keep = (np.abs(H) > cut) | (np.abs(S) > cut)
    ii, jj = np.nonzero(keep)
    nzat[at[ii], at[jj]] = True
    dmax = ds[0][nzat].max()
    d2y = np.sqrt(ds[0] ** 2 - d[..., 1] ** 2 + dy2 ** 2)
    assert ds[1][nzat].min() > dmax + margin and d2y[nzat].min() > dmax + margin, \
        f"supercell too small: coupled pairs up to {dmax:.2f} A, next images {ds[1][nzat].min():.2f} / {d2y[nzat].min():.2f} A"
    hr = {}
    Ro = Rbest[at][:, at]
    for R in (-1, 0, 1):
        m = keep & (Ro == R)
        hr[(R, 0)] = (np.where(m, H, 0.0), np.where(m, S, 0.0))
    nsx = int(z["nsc"][0])
    ax = Lx / nsx
    xf = z["pos_flat"][:, 0]
    z["xflat"] = xf + Lx * np.round((pos[:, 0] - xf) / Lx)                 # flat x of the GPAW home image
    pc = np.floor(z["xflat"] / ax + 1e-6).astype(int)
    a_, b_ = np.nonzero(nzat)
    z["cell_range"] = int(np.abs(pc[b_] + nsx * Rbest[a_, b_] - pc[a_]).max())
    z["split"] = dict(dmax=float(dmax), next_x=float(ds[1][nzat].min()), next_y=float(d2y[nzat].min()))
    return hr


def bloch(hr, k):
    """H(k), S(k) = sum_R exp(2 pi i k.R) hr[R] (physical convention); k in reciprocal cell units.
    Same eigenvalues as GPAW's H(k)."""
    H = sum(np.exp(2j * np.pi * (k[0] * r[0] + k[1] * r[1])) * h for r, (h, s) in hr.items())
    S = sum(np.exp(2j * np.pi * (k[0] * r[0] + k[1] * r[1])) * s for r, (h, s) in hr.items())
    return H, S


def supercell(hr, nao, ncell_x, ny, geo=None):
    """Sparse H, S of ncell_x x ny cells (x open, y periodic, folded at k_y = 0).
    Orbital order: cell (ix, iy) row major, then the cell's orbitals. Couplings (cell i, cell j):
    <i| H |j> = hr[R_j - R_i]  (physical convention of `load_hr`, checked in `check_convention`).
    geo: optional bond-length disorder dict(pos0, cell, norb, disp (ncell, nat, 3), eta, Eref): an element
    between atoms a != b at displaced distance d (pristine d0) becomes
        H_ab -> Eref S_ab + (d0 / d)^eta (H_ab - Eref S_ab),       S_ab unchanged,
    i.e. the hopping measured from the reference energy Eref scales (energy-zero independent)."""
    rows, cols, hv, sv = [], [], [], []
    cells = [(ix, iy) for ix in range(ncell_x) for iy in range(ny)]
    cid = {c: n for n, c in enumerate(cells)}
    mu = np.arange(nao)
    nz = {r: np.nonzero((h != 0) | (s != 0)) for r, (h, s) in hr.items()}
    if geo is not None:
        at = np.repeat(np.arange(len(geo["norb"])), geo["norb"])
        pos0, A = geo["pos0"], geo["cell"]
    for (ix, iy), n in cid.items():
        for (rx, ry), (h, s) in hr.items():
            jx = ix + rx
            if not 0 <= jx < ncell_x:
                continue
            m = cid[(jx, (iy + ry) % ny)]
            i, j = nz[(rx, ry)]
            hij, sij = h[i, j], s[i, j]
            if geo is not None:
                a, b = at[i], at[j]
                d0v = pos0[b] + rx * A[0] + ry * A[1] - pos0[a]
                if "Ly" in geo:                                    # nearest y image (k_y = 0 fold)
                    d0v[:, 1] -= geo["Ly"] * np.round(d0v[:, 1] / geo["Ly"])
                d0 = np.linalg.norm(d0v, axis=1)
                d = np.linalg.norm(d0v + geo["disp"][m, b] - geo["disp"][n, a], axis=1)
                f = np.where(d0 > 1e-6, (d0 / np.maximum(d, 1e-6)) ** geo["eta"], 1.0)
                hij = geo["Eref"] * sij + f * (hij - geo["Eref"] * sij)
            rows.append(n * nao + i); cols.append(m * nao + j); hv.append(hij); sv.append(sij)
    N = len(cells) * nao
    r, c = np.concatenate(rows), np.concatenate(cols)
    H = sp.csr_matrix((np.concatenate(hv), (r, c)), shape=(N, N))
    S = sp.csr_matrix((np.concatenate(sv), (r, c)), shape=(N, N))
    H.eliminate_zeros(); S.eliminate_zeros()
    return H, S, cells


def check_convention(z, hr, tol=1e-8):
    """(1) Bloch sums of hr reproduce the GPAW eigenvalues on the mesh (only checks the transform; aliasing
    is checked in `dft` from the outer R shell). (2) Physical sign of R: with the right convention, nonzero
    overlaps connect atoms at |tau_nu + R A - tau_mu| shorter than the transposed |tau_nu - R A - tau_mu|
    (2H-MoS2 has no inversion centre, and the positions must be GPAW's wrapped ones). Per species pair,
    every nonzero overlap must also be shorter than the closest vanishing one (radial cutoffs).
    Returns (eigenvalue error, max distance with the rule, with the transpose, separation ok)."""
    err = 0.0
    for k, e in zip(z["kpts"][::7], z["eps"][::7]):
        H, S = bloch(hr, k)
        err = max(err, np.abs(sla.eigvalsh(H, S)[:len(e)] - e).max())
    pos, cell, norb = z["pos"], z["cell"], z["norb"]
    atom = np.repeat(np.arange(len(norb)), norb)
    dmax = [0.0, 0.0]
    sep = {}
    sym = z["symbols"]
    for (rx, ry), (h, s) in hr.items():
        shift = rx * cell[0] + ry * cell[1]
        for c, sg in enumerate((1, -1)):
            mu, nu = np.nonzero(np.abs(s) > tol)
            if len(mu):
                d = np.linalg.norm(pos[atom[nu]] + sg * shift - pos[atom[mu]], axis=1)
                dmax[c] = max(dmax[c], d.max())
        for a in range(len(norb)):
            for b in range(len(norb)):
                d = np.linalg.norm(pos[b] + shift - pos[a])
                e = sep.setdefault("".join(sorted((sym[a], sym[b]))), [0.0, np.inf])
                if np.abs(s[np.ix_(atom == a, atom == b)]).max() > tol:
                    e[0] = max(e[0], d)
                else:
                    e[1] = min(e[1], d)
    ok = all(lo <= hi + 1e-6 for lo, hi in sep.values())
    return err, dmax[0], dmax[1], ok


def lead_bands(H00, H01, S00, S01, nk=41):
    ks = np.linspace(0, np.pi, nk)
    H0, H1, S0, S1 = (M.toarray() for M in (H00, H01, S00, S01))
    return ks, np.array([sla.eigvalsh(H0 + H1 * np.exp(1j * k) + H1.T * np.exp(-1j * k),
                                      S0 + S1 * np.exp(1j * k) + S1.T * np.exp(-1j * k)) for k in ks])


def gaussian_field(x, y, x0, x1, Ly, rms, corr, rng, h=0.5):
    """Zero-mean Gaussian random field V(x, y), periodic in y (period Ly), autocorrelation
    rms^2 exp(-r^2 / corr^2) (white noise smoothed with a Gaussian of sigma = corr / 2), sampled at (x, y)."""
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator
    if rms == 0:
        return np.zeros(len(x))
    pad = 2 * corr
    nx = int(np.ceil((x1 - x0 + 2 * pad) / h)) + 1
    ny = max(int(np.round(Ly / h)), 1)
    gx, gy = x0 - pad + h * np.arange(nx), Ly / ny * np.arange(ny + 1)
    w = rng.standard_normal((nx, ny))
    r = gaussian_filter(w, (corr / 2 / h, corr / 2 / (Ly / ny)), mode=("nearest", "wrap"))
    r = np.c_[r, r[:, :1]]
    v = RegularGridInterpolator((gx, gy), r)(np.c_[x, np.mod(y, Ly)])
    return rms * (v - v.mean()) / v.std()


def dipole_potential(xyz, centers, sign, p, eps, Ly, nimg=6):
    """Electron potential energy [eV] of point dipoles p * sign * z_hat [e A] at `centers` (background
    dielectric constant eps), periodic in y: U(r) = -(e^2/4 pi eps0 eps) p.(r - r_d) / |r - r_d|^3."""
    U = np.zeros(len(xyz))
    for c, s in zip(centers, sign):
        for k in range(-nimg, nimg + 1):
            d = xyz - (c + np.array([0.0, k * Ly, 0.0]))
            U -= KE2 / eps * s * p * d[:, 2] / np.linalg.norm(d, axis=1) ** 3
    return U


def device(hr_path, out, ny=9, length=200.0, nd=None, m=None, vac=0.01, sigma_u=0.04, eta=2.0,
           v_rms=0.05, v_corr=15.0, dip_n=1e13, dip_p=0.5, dip_h=3.0, dip_eps=5.0, dip_sign="random",
           seed=0, cut=1e-6):
    """Rough MoS2 device [lead | nd device slabs | lead] with three disorder types (device slabs only):
      bond-length disorder: random atomic displacements (rms sigma_u [A] per Cartesian component), hoppings
          scaled by (d0 / d)^eta about Eref = mid-gap (see `supercell`);
      S vacancies: fraction vac of the S atoms removed (orbitals deleted, unrelaxed);
      background potential V (per atom, added as H += (V S + S V) / 2):
          oxide-interface roughness: Gaussian field V(x, y), rms v_rms [eV], correlation length v_corr [A];
          interface dipoles: dip_n [cm^-2] per interface, moment dip_p [e A] along z ('random' sign or 'up'),
          dip_h [A] above the top S plane and below the bottom S plane, background eps = dip_eps;
          the total is shifted to zero mean over the device atoms (no band offset to the leads).
    nd = round(length / slab length) unless given."""
    from .inputs import write_inputs
    z, hr = load_hr(hr_path, cut)
    nao = int(z["norb"].sum())
    cell = z["cell"]
    nnz = sum(((h != 0) | (s != 0)).sum() for h, s in hr.values()) / nao
    if "nsc" in z:                       # corrugated supercell (dft_super): unit = supercell, Gamma folded in y
        sup, ny, cpu = True, 1, int(z["nsc"][0])
        print(f"supercell H: {z['nsc'].tolist()} cells, {nao} orbitals, elements > {cut:g} kept ({nnz:.0f} per orbital); "
              f"coupled pairs up to {z['split']['dmax']:.2f} A, next images {z['split']['next_x']:.2f} (x) / "
              f"{z['split']['next_y']:.2f} (y) A; range {z['cell_range']} primitive cells")
        rx_max, xflat = z["cell_range"], z["xflat"]
    else:                                # primitive 6-atom cell (dft): ny cells along y, folded at k_y = 0
        sup, cpu = False, 1
        e_mesh, d_rule, d_tr, sep_ok = check_convention(z, hr)
        print(f"H(R): {len(hr)} cells, elements > {cut:g} kept ({nnz:.0f} per orbital); Bloch vs GPAW eigenvalues "
              f"{e_mesh:.1e} eV; max distance of nonzero overlaps: rule {d_rule:.2f} A, transposed {d_tr:.2f} A; "
              f"radial cutoffs consistent: {sep_ok}")
        assert e_mesh < 1e-3, "H(R) does not reproduce the GPAW eigenvalues (cut too large?)"
        assert d_rule < d_tr - 0.5 and sep_ok, "H(R) sign convention or atom positions inconsistent"
        rx_max, xflat = max(abs(r[0]) for r in hr), z["pos"][:, 0]
    assert ny % 3 == 0 or sup, "ny must be a multiple of 3 (K valley at k_y = 0)"
    Lu = cell[0, 0]                                          # unit cell length along x
    ax = Lu / cpu                                            # primitive cell length
    m = m or rx_max
    assert m >= rx_max, f"slab of {m} cells shorter than the coupling range ({rx_max} cells)"
    Ls = m * ax                                              # device slab length
    lead_u = 1 if sup else m                                 # lead cell = lead_u units (>= one slab)
    nd_u = (int(round(length / Lu)) if sup else m * int(round(length / Ls))) if nd is None else \
        (nd * m // cpu if sup else nd * m)
    assert (nd_u * cpu) % m == 0, f"device of {nd_u * cpu} primitive cells is not a multiple of the slab ({m})"
    nd = nd_u * cpu // m
    nb = nd + 2
    ncx = nd_u + 2 * lead_u
    Ly = ny * cell[1, 1]
    pos0, sym, norb0 = z["pos"], z["symbols"], z["norb"]
    nat = len(pos0)
    eps = z["eps"]
    Ef = float(z["Ef"])
    Eref = 0.5 * (eps[eps < Ef].max() + eps[eps > Ef].min())
    rng = np.random.default_rng(seed)
    cells = [(ix, iy) for ix in range(ncx) for iy in range(ny)]
    # blocks: lead units -> 0 / nb - 1; device atoms binned by flat x into slabs of m primitive cells
    xg = np.concatenate([xflat + ix * Lu for ix, iy in cells])
    ucell = np.repeat([ix for ix, iy in cells], nat)
    blocks = np.clip(np.floor((xg - lead_u * Lu) / Ls + 1e-6).astype(int) + 1, 1, nb - 2)
    blocks[ucell < lead_u] = 0
    blocks[ucell >= ncx - lead_u] = nb - 1
    indev = (blocks > 0) & (blocks < nb - 1)
    # bond-length disorder
    disp = np.zeros((len(cells) * nat, 3))
    disp[indev] = rng.normal(0.0, sigma_u, (indev.sum(), 3))
    disp = disp.reshape(len(cells), nat, 3)
    t = time.time()
    H, S, _ = supercell(hr, nao, ncx, ny, geo=dict(pos0=pos0, cell=cell, norb=norb0, disp=disp, eta=eta, Eref=Eref,
                                                   Ly=Ly))
    xyz0 = np.array([pos0 + ix * cell[0] + iy * cell[1] for ix, iy in cells]).reshape(-1, 3)
    xyz = xyz0 + disp.reshape(-1, 3)
    labels = np.tile(sym, len(cells))
    norb = np.tile(norb0, len(cells))
    print(f"structure {ncx} x {ny} units ({ncx * cpu} x {ny * (int(z['nsc'][1]) if sup else 1)} cells), {H.shape[0]} "
          f"orbitals, nnz(H) {H.nnz}, {time.time() - t:.0f} s")
    # background potential
    x0, x1 = lead_u * Lu, (ncx - lead_u) * Lu
    Vr = gaussian_field(xyz0[:, 0], xyz0[:, 1], x0, x1, Ly, v_rms, v_corr, rng)
    top = s_layer(xyz0, labels, Ly)
    ndip = rng.poisson(dip_n * 1e-16 * (x1 - x0) * Ly, 2)
    cen, sgn = [], []
    for side, nn in ((1, ndip[0]), (-1, ndip[1])):
        c = np.c_[rng.uniform(x0, x1, nn), rng.uniform(0, Ly, nn)]
        sl = np.where((labels == "S") & (top == (side > 0)))[0]          # dipole plane follows the local S layer
        near = [sl[np.argmin(np.hypot(xyz0[sl, 0] - p[0], (xyz0[sl, 1] - p[1] + Ly / 2) % Ly - Ly / 2))] for p in c]
        c = np.c_[c, xyz0[near, 2] + side * dip_h] if nn else np.zeros((0, 3))
        s = rng.choice([-1.0, 1.0], nn) if dip_sign == "random" else np.full(nn, side * 1.0)
        cen.append(c); sgn.append(s)
    cen, sgn = np.vstack(cen), np.concatenate(sgn)
    Vd = dipole_potential(xyz0, cen, sgn, dip_p, dip_eps, Ly)
    V = np.where(indev, Vr + Vd, 0.0)
    V[indev] -= V[indev].mean()
    Vo = sp.diags(np.repeat(V, norb))
    H = (H + 0.5 * (Vo @ S + S @ Vo)).tocsr()
    print(f"potential: roughness rms {Vr[indev].std() * 1e3:.0f} meV (corr {v_corr:g} A), {len(cen)} dipoles "
          f"({ndip.tolist()} top/bottom, {dip_p:g} e A, {dip_sign}) rms {Vd[indev].std() * 1e3:.0f} meV, "
          f"total rms {V[indev].std() * 1e3:.0f} meV, range [{V.min():.3f}, {V.max():.3f}] eV")
    # S vacancies in the device slabs
    cand = np.where((labels == "S") & indev)[0]
    gone = rng.choice(cand, int(round(vac * len(cand))), replace=False) if vac > 0 else np.array([], int)
    keep_at = np.setdiff1d(np.arange(len(labels)), gone)
    off = np.concatenate([[0], np.cumsum(norb)])
    keep_orb = np.concatenate([np.arange(off[i], off[i + 1]) for i in keep_at])
    H, S = H[keep_orb][:, keep_orb].tocsr(), S[keep_orb][:, keep_orb].tocsr()
    assert abs(H - H.T).max() < 1e-12 and abs(S - S.T).max() < 1e-12
    # bond-length statistics (nearest Mo-S bonds)
    dstat = bond_stats(xyz0, xyz, labels, indev, Ly)
    labels, xyz, norb, blocks, V, disp_at = labels[keep_at], xyz[keep_at], norb[keep_at], blocks[keep_at], V[keep_at], \
        disp.reshape(-1, 3)[keep_at]
    # lead cell = first block (pristine) and its coupling to the next pristine block
    HL, SL_, _ = supercell(hr, nao, 2 * lead_u, ny)
    nl = lead_u * ny * nao
    lead = dict(H00=HL[:nl, :nl], H01=HL[:nl, nl:], S00=SL_[:nl, :nl], S01=SL_[:nl, nl:])
    ob = np.repeat(blocks, norb)
    for b in (0, nb - 1):
        o = np.where(ob == b)[0]
        assert abs(H[o][:, o] - lead["H00"]).max() < 1e-12 and abs(S[o][:, o] - lead["S00"]).max() < 1e-12
    t = time.time()
    ks, E = lead_bands(lead["H00"], lead["H01"], lead["S00"], lead["S01"])
    Ev, Ec = E[E < Eref].max(), E[E > Eref].min()
    print(f"lead cell {nl} orbitals ({lead_u * cpu} cells along x): Ev {Ev:.4f}, Ec {Ec:.4f} eV, gap {Ec - Ev:.3f} eV "
          f"(E_F {Ef:.4f}, Eref {Eref:.4f}), {time.time() - t:.0f} s")
    gp = json.loads(str(z["info"]))
    meta = dict(name=f"MoS2{' corrugated' if sup else ''} {nd * Ls / 10:.1f} x {Ly / 10:.2f} nm, {len(gone)} S vacancies",
                model="GPAW LCAO " + gp["basis"], units=dict(energy="eV", length="Angstrom"), orthogonal=False,
                axis=0, block_length=float(Ls), lead_cell_length=float(lead_u * Lu), nblocks=nb, periodic=dict(axis=1, length=float(Ly)),
                transport=dict(device_blocks=list(range(1, nb - 1)), lead_left_block=0, lead_right_block=nb - 1,
                               lead_cell="pristine slab and its coupling to the next slab (+x)"),
                edges=dict(Ev_lead=float(Ev), Ec_lead=float(Ec), Ef=Ef, Eref=float(Eref)), gpaw=gp,
                disorder=dict(seed=seed, cut=cut,
                              bonds=dict(sigma_u=sigma_u, eta=eta, Eref=float(Eref), **dstat),
                              vacancies=dict(fraction=vac, n=int(len(gone))),
                              potential=dict(v_rms=v_rms, v_corr=v_corr, dip_n_cm2=dip_n, dip_p_eA=dip_p, dip_h=dip_h,
                                             dip_eps=dip_eps, dip_sign=dip_sign, n_dipoles=ndip.tolist(),
                                             rms_total=float(np.std(V[(blocks > 0) & (blocks < nb - 1)])))))
    write_inputs(out, labels, xyz, norb, blocks, H, S, lead=lead, meta=meta)
    np.savez(f"{out}/lead_bands.npz", k=ks, E=E)
    np.savez(f"{out}/disorder.npz", V=V, disp=disp_at, vacancies=xyz0[gone], dipoles=cen, dipole_sign=sgn)
    print(f"wrote {out}: {len(labels)} atoms, {H.shape[0]} orbitals, nnz(H) {H.nnz}, {nd} device slabs of "
          f"{np.bincount(np.repeat(blocks, norb))[1:-1].max()} orbitals at most, lead cells of {nl}")


def s_layer(xyz, labels, Ly):
    """True for S atoms above their nearest Mo (top layer); also valid for a corrugated sheet."""
    from scipy.spatial import cKDTree
    mo = np.where(labels == "Mo")[0]
    p = xyz.copy()
    p[:, 1] = np.mod(p[:, 1], Ly)
    lo, hi = p.min(0) - 1, p.max(0) + 1
    box = np.r_[hi[0] - lo[0] + 100, Ly, hi[2] - lo[2] + 100]
    tree = cKDTree(np.c_[p[mo, 0] - lo[0], p[mo, 1], p[mo, 2] - lo[2]], boxsize=box)
    _, k = tree.query(np.c_[p[:, 0] - lo[0], p[:, 1], p[:, 2] - lo[2]])
    return p[:, 2] > p[mo[k], 2]


def bond_stats(xyz0, xyz, labels, sel, Ly, dmax=2.6):
    """Relative change of the nearest-neighbour Mo-S bond lengths of the atoms in `sel` (minimum image in y)."""
    from scipy.spatial import cKDTree
    mo, s = np.where((labels == "Mo") & sel)[0], np.where(labels == "S")[0]
    tree = cKDTree(np.c_[xyz0[s, 0], np.mod(xyz0[s, 1], Ly), xyz0[s, 2]], boxsize=None)
    rel = []
    for k in (-1, 0, 1):
        q = np.c_[xyz0[mo, 0], np.mod(xyz0[mo, 1], Ly) + k * Ly, xyz0[mo, 2]]
        for i, nb_ in zip(mo, tree.query_ball_point(q, dmax)):
            for j in s[nb_]:
                d0v = xyz0[j] - xyz0[i]
                d0v[1] -= Ly * np.round(d0v[1] / Ly)
                dv = d0v + (xyz[j] - xyz0[j]) - (xyz[i] - xyz0[i])
                rel.append(np.linalg.norm(dv) / np.linalg.norm(d0v) - 1)
    rel = np.array(rel)
    return dict(n_MoS_bonds=int(len(rel)), dMoS_rel_rms=float(rel.std()), dMoS_rel_max=float(np.abs(rel).max()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dft")
    d.add_argument("--out", default="inputs/mos2_hr.npz")
    d.add_argument("--a", type=float, default=3.18)
    d.add_argument("--kx", type=int, default=9)
    d.add_argument("--ky", type=int, default=15)
    d.add_argument("--basis", default="dzp")
    c = sub.add_parser("dft-super", help="corrugated supercell at Gamma (run under mpiexec)")
    c.add_argument("--out", default="inputs/mos2_9x9_hr.npz")
    c.add_argument("--nsc", default="9,9", help="orthorhombic cells along x, y")
    c.add_argument("--rms", type=float, default=1.0, help="corrugation height rms [A]")
    c.add_argument("--corr", type=float, default=12.0, help="corrugation correlation length [A]")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--a", type=float, default=3.18)
    c.add_argument("--h", type=float, default=0.2, help="real-space grid spacing [A]")
    c.add_argument("--basis", default="dzp")
    c.add_argument("--txt", default="-")
    v = sub.add_parser("device")
    v.add_argument("--hr", default="inputs/mos2_hr.npz", help="dft or dft-super output")
    v.add_argument("--ny", type=int, default=9, help="transverse cells (periodic, multiple of 3; dft-super: fixed)")
    v.add_argument("--length", type=float, default=200.0, help="device length [A] (sets --nd)")
    v.add_argument("--nd", type=int, help="device slabs (overrides --length)")
    v.add_argument("--m", type=int, help="transport cells per slab (default: coupling range)")
    v.add_argument("--vac", type=float, default=0.01, help="S vacancy fraction")
    v.add_argument("--sigma-u", type=float, default=0.04, help="atomic displacement rms per component [A]")
    v.add_argument("--eta", type=float, default=2.0, help="hopping scaling exponent (d0/d)^eta")
    v.add_argument("--v-rms", type=float, default=0.05, help="oxide-roughness potential rms [eV]")
    v.add_argument("--v-corr", type=float, default=15.0, help="its correlation length [A]")
    v.add_argument("--dip-n", type=float, default=1e13, help="interface dipole density per interface [cm^-2]")
    v.add_argument("--dip-p", type=float, default=0.5, help="dipole moment [e A]")
    v.add_argument("--dip-h", type=float, default=3.0, help="dipole plane distance from the S plane [A]")
    v.add_argument("--dip-eps", type=float, default=5.0, help="background dielectric constant")
    v.add_argument("--dip-sign", choices=["random", "up"], default="random",
                   help="random +-z, or all pointing away from the sheet")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--out", default="inputs/mos2")
    a = p.parse_args()
    if a.cmd == "dft":
        dft(a.out, a=a.a, kx=a.kx, ky=a.ky, basis=a.basis)
    elif a.cmd == "dft-super":
        dft_super(a.out, nsc=tuple(int(x) for x in a.nsc.split(",")), rms=a.rms, corr=a.corr, seed=a.seed, a=a.a,
                  h=a.h, basis=a.basis, txt=a.txt)
    else:
        device(a.hr, a.out, ny=a.ny, length=a.length, nd=a.nd, m=a.m, vac=a.vac, sigma_u=a.sigma_u, eta=a.eta,
               v_rms=a.v_rms, v_corr=a.v_corr, dip_n=a.dip_n, dip_p=a.dip_p, dip_h=a.dip_h, dip_eps=a.dip_eps,
               dip_sign=a.dip_sign, seed=a.seed)


if __name__ == "__main__":
    main()
