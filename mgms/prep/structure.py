"""Atomic structure of Si nanowire / nanosheet FETs with an amorphous-oxide surrogate.

Reimplementation of the project's qtmode/structure.py (same conventions and parameters).

Geometry (transport along x = [100]):

    [ Au source ] gap [ Si channel + rough interface + SiO2 shell ] gap [ Au drain ]

* Box: n_cells conventional cells along x, nt = (ny, nz) cells across (an int means square).
  half_width = (hy, hz) is the nominal half width of the crystalline Si channel.
* Channel: crystalline diamond Si, ideal positions.
* Interface roughness: each of the four channel faces is shifted by a Gaussian-correlated random
  height field (autocorrelation rms^2 exp(-r^2 / corr^2)). Lattice sites outside the faces
  become oxide Si. With rough_2d=True (default) the height varies along x AND along the face
  (needed for wide sheets); rough_2d=False reproduces the original x-only profiles.
  Channel sites with fewer than 2 channel neighbours are converted to oxide.
* Oxide: randomized continuous-network surrogate built on the Si lattice. Every bond touching
  an oxide Si gets a bridging O, bowed H_BOW = 0.4 A off the bond axis with a random azimuth
  (Si-O-Si ~ 142 deg); oxide Si and O are randomly displaced. Outer surfaces are terminated
  by non-bridging O. The network keeps the Si lattice density, so geometric Si-O distances are
  ~1.24 A; the Hamiltonian maps them onto the physical 1.61 A (uniform length rescaling).
  This is NOT a melt-quench a-SiO2; it gives an irregular, insulating, gap-matched shell.
* Au contacts: fcc Au, one s orbital per atom, same cross-section as the box.
* periodic_x: bonds wrap around in x (ring). Used for periodic supercells / references;
  meta["Lx"] then holds the period, and the Hamiltonian uses minimum-image bond vectors.

Transport blocks are 2 Si conventional cells (10.86 A). Every Si lattice site of a channel
block has a "union site id" (its position inside the 2-cell block).
"""
import numpy as np
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter

A_SI, A_AU = 5.431, 4.078
D_SISI = A_SI * np.sqrt(3) / 4
D_SIO = 1.61                                      # physical Si-O bond length (electronic scaling)
H_BOW = 0.40                                      # O bowing off the Si-Si bond axis
D_SIO_GEOM = float(np.hypot(D_SISI / 2, H_BOW))   # Si-O distance in the compressed network
SI_C, SI_OX, OX, AU = 0, 1, 2, 3
NAMES = ["Si", "Si(ox)", "O", "Au"]
NORB = np.array([5, 4, 4, 1])                     # sp3s*, sp3, sp3, s
VB = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])  # sub0 -> sub1 bonds, units a/4
CELLS_PER_BLOCK = 2
NX4_BLOCK = 4 * CELLS_PER_BLOCK
L_BLOCK = CELLS_PER_BLOCK * A_SI
L_AU_BLOCK = 2 * A_AU


@dataclass
class Structure:
    pos: np.ndarray        # (n, 3) atom positions [A]
    species: np.ndarray    # (n,) SI_C / SI_OX / OX / AU
    bonds: np.ndarray      # (nb, 2) explicit covalent bonds (Si-Si, Si-O)
    block: np.ndarray      # (n,) transport block index
    site: np.ndarray       # (n,) union site id inside its Si block, -1 otherwise
    nblocks: int
    meta: dict = field(default_factory=dict)

    @property
    def norb(self):
        return NORB[self.species]

    @property
    def offsets(self):
        return np.concatenate([[0], np.cumsum(self.norb)])


def _nyz(nt):
    return (nt, nt) if np.isscalar(nt) else tuple(nt)


def _hyz(half_width):
    return (half_width, half_width) if np.isscalar(half_width) else tuple(half_width)


def diamond_sites(nx4, ny4, nz4):
    """Integer coordinates (units a/4) of diamond sites in [0,nx4)x[0,ny4)x[0,nz4)."""
    g = np.mgrid[0:nx4, 0:ny4, 0:nz4].reshape(3, -1).T
    X, Y, Z = g.T
    s0 = (X % 2 == 0) & (Y % 2 == 0) & (Z % 2 == 0) & ((X + Y + Z) % 4 == 0)
    s1 = (X % 2 == 1) & (Y % 2 == 1) & (Z % 2 == 1) & ((X + Y + Z) % 4 == 3)
    m = s0 | s1
    return g[m], s1[m].astype(int)


def union_sites(nt):
    """All diamond sites of one 2-cell block of the box: defines the union index."""
    ny, nz = _nyz(nt)
    c, _ = diamond_sites(NX4_BLOCK, 4 * ny, 4 * nz)
    return c, {tuple(x): i for i, x in enumerate(c)}


class Disorder:
    """Random tables for the oxide network, keyed by lattice position.
    nxkey = period along x in units of a/4 (NX4_BLOCK for a block-periodic replica)."""

    def __init__(self, rng, nxkey, nt, s_si=0.15, s_o=0.05):
        ny, nz = _nyz(nt)
        self.nxkey, self.n4y, self.n4z = nxkey, 4 * ny, 4 * nz
        shp = (nxkey, 4 * ny, 4 * nz)
        self.dsi = rng.normal(0, s_si, shp + (3,))
        self.phi = rng.uniform(0, 2 * np.pi, shp + (4,))
        self.do = rng.normal(0, s_o, shp + (4, 3))

    def key(self, c):
        return (c[0] % self.nxkey, c[1] % self.n4y, c[2] % self.n4z)


def rough_profile(n, dx, rms, corr, rng):
    """Original x-only profile: zero mean, Gaussian autocorrelation (length corr), given rms."""
    if rms == 0 or rng is None:
        return np.zeros(n)
    m = int(4 * corr / dx) + 1
    w = rng.standard_normal(n + 2 * m)
    t = np.arange(-m, m + 1) * dx
    r = np.convolve(w, np.exp(-2 * t**2 / corr**2), "valid")
    return rms * (r - r.mean()) / r.std()


def rough_surface(nx, ns, dx, rms, corr, rng, periodic_x=False, two_d=True):
    """Height field h[x, s] on the a/4 grid, autocorrelation rms^2 exp(-r^2/corr^2).
    Smoothing white noise with exp(-2 r^2/corr^2) (a Gaussian of sigma = corr/2) gives that ACF."""
    if rms == 0 or rng is None:
        return np.zeros((nx, ns))
    if not two_d:
        return np.repeat(rough_profile(nx, dx, rms, corr, rng)[:, None], ns, 1)
    sig = corr / 2 / dx
    m = int(4 * sig) + 1
    px = 0 if periodic_x else m
    w = rng.standard_normal((nx + 2 * px, ns + 2 * m))
    r = gaussian_filter(w, sig, mode="wrap")[px:px + nx, m:m + ns]
    return rms * (r - r.mean()) / r.std()


def _perp_basis(u):
    ref = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(u, ref)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def _bridging_O(pA, pB, phi, dO):
    u = pB - pA
    u = u / np.linalg.norm(u)
    e1, e2 = _perp_basis(u)
    return 0.5 * (pA + pB) + H_BOW * (np.cos(phi) * e1 + np.sin(phi) * e2) + dO


def build_si_region(n_cells, nt, half_width, disorder, rough_rng=None, rough_rms=0.0,
                    rough_corr=10.0, periodic_x=False, rough_2d=True, rough_window=None):
    """Si channel + oxide shell occupying an n_cells x ny x nz box of the Si lattice."""
    ny, nz = _nyz(nt)
    hy, hz = _hyz(half_width)
    n4y, n4z, nx4 = 4 * ny, 4 * nz, 4 * n_cells
    a4 = A_SI / 4
    coords, sub = diamond_sites(nx4, n4y, n4z)
    index = {tuple(c): i for i, c in enumerate(coords)}
    xyz = coords * a4
    yc, zc = (n4y - 1) / 2 * a4, (n4z - 1) / 2 * a4
    # faces: 0 = y-low, 1 = y-high (vary with x, z); 2 = z-low, 3 = z-high (vary with x, y)
    prof = [rough_surface(nx4, n4z if f < 2 else n4y, a4, rough_rms, rough_corr, rough_rng,
                          periodic_x, rough_2d) for f in range(4)]
    if rough_window is not None:                 # e.g. roughness only inside the device
        prof = [p_ * rough_window[:, None] for p_ in prof]
    X, Y, Z = coords.T
    ch = ((xyz[:, 1] >= yc - hy + prof[0][X, Z]) & (xyz[:, 1] <= yc + hy + prof[1][X, Z]) &
          (xyz[:, 2] >= zc - hz + prof[2][X, Y]) & (xyz[:, 2] <= zc + hz + prof[3][X, Y]))

    nb = -np.ones((len(coords), 4), int)
    wrap = np.zeros((len(coords), 4))           # x shift (A) of the bonded periodic image
    for i, (c, s) in enumerate(zip(coords, sub)):
        sgn = 1 if s == 0 else -1
        for k in range(4):
            cn = c + sgn * VB[k]
            if periodic_x:
                wrap[i, k] = (cn[0] - cn[0] % nx4) * a4
                cn[0] %= nx4
            nb[i, k] = index.get(tuple(cn), -1)
    valid = nb >= 0
    while True:  # remove channel sites with < 2 channel neighbours
        nch = (ch[np.where(valid, nb, 0)] & valid).sum(1)
        bad = ch & (nch < 2)
        if not bad.any():
            break
        ch[bad] = False

    keys = (X % disorder.nxkey, Y, Z)
    pos_si = xyz.astype(float).copy()
    pos_si[~ch] += disorder.dsi[keys][~ch]

    pos, spec, bonds = list(pos_si), list(np.where(ch, SI_C, SI_OX)), []

    def add_O(p):
        pos.append(p)
        spec.append(OX)
        return len(pos) - 1

    def nonbridging(i, direction, c_partner, k, c_key):
        d = direction.astype(float)
        if c_partner[0] < 0 or c_partner[0] >= nx4:   # x-end face: keep O lateral (away from Au)
            d[0] = 0.0
        d /= np.linalg.norm(d)
        o = add_O(pos_si[i] + D_SIO_GEOM * d + disorder.do[disorder.key(c_key)][k])
        bonds.append((i, o))

    for i in np.where(sub == 0)[0]:
        c0 = coords[i]
        for k in range(4):
            j = nb[i, k]
            if j >= 0:
                if ch[i] and ch[j]:
                    bonds.append((i, j))
                else:
                    kk = disorder.key(c0)
                    pB = pos_si[j] + np.array([wrap[i, k], 0, 0])
                    o = add_O(_bridging_O(pos_si[i], pB, disorder.phi[kk][k], disorder.do[kk][k]))
                    bonds += [(i, o), (o, j)]
            elif not ch[i]:
                nonbridging(i, VB[k], c0 + VB[k], k, c0)
    for i in np.where(sub == 1)[0]:
        for k in range(4):
            if nb[i, k] < 0 and not ch[i]:
                cp = coords[i] - VB[k]
                nonbridging(i, -VB[k], cp, k, cp)

    pos = np.array(pos)
    if periodic_x:
        pos[:, 0] %= nx4 * a4
    else:
        pos[:, 0] = np.clip(pos[:, 0], 0.0, (nx4 - 1) * a4)   # keep the end faces flat (contacts)
    return dict(pos=pos, species=np.array(spec), bonds=np.array(bonds), coords=coords,
                channel=ch, n_si=len(coords), profiles=prof, center=(yc, zc))


def au_block(nt, x0):
    """Two fcc Au cells along x (4 atomic layers) covering the box cross-section."""
    ny, nz = _nyz(nt)
    ngy = int(np.floor((4 * ny - 1) * A_SI / 4 / (A_AU / 2) + 1e-9)) + 1
    ngz = int(np.floor((4 * nz - 1) * A_SI / 4 / (A_AU / 2) + 1e-9)) + 1
    pts = [(x0 + m * A_AU / 2, iy * A_AU / 2, iz * A_AU / 2)
           for m in range(4) for iy in range(ngy) for iz in range(ngz) if (iy + iz + m) % 2 == 0]
    return np.array(pts)


def _si_block_and_site(coords, nt):
    _, umap = union_sites(nt)
    blk = coords[:, 0] // NX4_BLOCK
    site = np.array([umap[(c[0] % NX4_BLOCK, c[1], c[2])] for c in coords])
    return blk, site


def build_device(n_cells=8, nt=4, half_width=6.5, n_au_blocks=2, rough_rms=1.5,
                 rough_corr=8.0, seed=0, d_gap=2.2, rough_2d=True):
    """Au | Si channel (rough) + oxide | Au. n_cells must be a multiple of CELLS_PER_BLOCK."""
    assert n_cells % CELLS_PER_BLOCK == 0
    rng = np.random.default_rng(seed)
    dis = Disorder(rng, 4 * n_cells, nt)
    reg = build_si_region(n_cells, nt, half_width, dis, rng, rough_rms, rough_corr,
                          rough_2d=rough_2d)
    nsi_blocks = n_cells // CELLS_PER_BLOCK
    x_end = (4 * n_cells - 1) * A_SI / 4

    left = [au_block(nt, -d_gap - (4 * n_au_blocks - 1) * A_AU / 2 + 4 * k * A_AU / 2)
            for k in range(n_au_blocks)]
    right = [au_block(nt, x_end + d_gap + 4 * k * A_AU / 2) for k in range(n_au_blocks)]

    nL = sum(len(b) for b in left)
    pos = np.vstack(left + [reg["pos"]] + right)
    n_reg = len(reg["pos"])
    species = np.concatenate([np.full(nL, AU), reg["species"], np.full(len(pos) - nL - n_reg, AU)])
    bonds = reg["bonds"] + nL

    block = np.empty(len(pos), int)
    site = -np.ones(len(pos), int)
    off = 0
    for k, b in enumerate(left):
        block[off:off + len(b)] = k
        off += len(b)
    blk_si, site_si = _si_block_and_site(reg["coords"], nt)
    block[nL:nL + reg["n_si"]] = n_au_blocks + blk_si
    site[nL:nL + reg["n_si"]] = site_si
    xo = reg["pos"][reg["n_si"]:, 0]
    block[nL + reg["n_si"]:nL + n_reg] = n_au_blocks + np.clip(np.floor(xo / L_BLOCK).astype(int),
                                                               0, nsi_blocks - 1)
    off = nL + n_reg
    for k, b in enumerate(right):
        block[off:off + len(b)] = n_au_blocks + nsi_blocks + k
        off += len(b)

    meta = dict(n_cells=n_cells, nt=nt, half_width=half_width, n_au_blocks=n_au_blocks,
                nsi_blocks=nsi_blocks, rough_rms=rough_rms, rough_corr=rough_corr, seed=seed,
                si_blocks=list(range(n_au_blocks, n_au_blocks + nsi_blocks)),
                channel_fraction=reg["channel"].mean(), profiles=reg["profiles"],
                center=reg["center"])
    return Structure(pos, species, bonds, block, site, 2 * n_au_blocks + nsi_blocks, meta)


def build_ring(n_cells=8, nt=4, half_width=6.5, rough_rms=1.5, rough_corr=8.0, seed=0,
               rough_2d=True):
    """Rough channel + oxide, periodic along x with period n_cells * a (no contacts).
    Roughness and oxide disorder are periodic with the ring, so the supercell is seamless."""
    assert n_cells % CELLS_PER_BLOCK == 0
    rng = np.random.default_rng(seed)
    dis = Disorder(rng, 4 * n_cells, nt)
    reg = build_si_region(n_cells, nt, half_width, dis, rng, rough_rms, rough_corr,
                          periodic_x=True, rough_2d=rough_2d)
    nblk = n_cells // CELLS_PER_BLOCK
    blk, site = _si_block_and_site(reg["coords"], nt)
    n = len(reg["pos"])
    block = np.empty(n, int)
    block[:reg["n_si"]] = blk
    block[reg["n_si"]:] = np.floor(reg["pos"][reg["n_si"]:, 0] / L_BLOCK).astype(int) % nblk
    s = -np.ones(n, int)
    s[:reg["n_si"]] = site
    meta = dict(n_cells=n_cells, nt=nt, half_width=half_width, rough_rms=rough_rms,
                rough_corr=rough_corr, seed=seed, Lx=n_cells * A_SI, profiles=reg["profiles"],
                channel_fraction=reg["channel"].mean(), center=reg["center"])
    return Structure(reg["pos"], reg["species"], reg["bonds"], block, s, nblk, meta)


def build_periodic_reference(nt=4, half_width=6.5, seed=1000, n_periods=3):
    """Flat-interface channel with an oxide that repeats every block (block-periodic replica)."""
    rng = np.random.default_rng(seed)
    dis = Disorder(rng, NX4_BLOCK, nt)
    n_cells = n_periods * CELLS_PER_BLOCK
    reg = build_si_region(n_cells, nt, half_width, dis, None, 0.0, periodic_x=True)
    blk, site = _si_block_and_site(reg["coords"], nt)
    n = len(reg["pos"])
    block = np.empty(n, int)
    block[:reg["n_si"]] = blk
    block[reg["n_si"]:] = np.floor(reg["pos"][reg["n_si"]:, 0] / L_BLOCK).astype(int) % n_periods
    s = -np.ones(n, int)
    s[:reg["n_si"]] = site
    return Structure(reg["pos"], reg["species"], reg["bonds"], block, s, n_periods,
                     dict(nt=nt, half_width=half_width, n_periods=n_periods,
                          Lx=n_cells * A_SI, center=reg["center"]))


def au_lead_chain(nt):
    """Two consecutive Au blocks, for the lead H00 / H01."""
    b0, b1 = au_block(nt, 0.0), au_block(nt, L_AU_BLOCK)
    pos = np.vstack([b0, b1])
    n = len(b0)
    return Structure(pos, np.full(2 * n, AU), np.zeros((0, 2), int),
                     np.repeat([0, 1], n), -np.ones(2 * n, int), 2, {})


def _canonical_order(pos, species, block, bonds, x0_of_block):
    """Reorder atoms block by block by (species, block-local position), so that translated
    copies of a block (periodic leads) have identical orbital ordering."""
    loc = pos.copy()
    loc[:, 0] -= x0_of_block[block]
    key = np.lexsort((np.round(loc[:, 2], 3), np.round(loc[:, 1], 3), np.round(loc[:, 0], 3),
                      species, block))
    inv = np.empty_like(key)
    inv[key] = np.arange(len(key))
    return key, inv[bonds]


def build_open_device(n_dev_blocks=4, n_lead_blocks=3, nt=(12, 8), half_width=(25.0, 15.0),
                      rough_rms=1.5, rough_corr=8.0, seed=7, lead_seed=1000, rough_2d=True):
    """Open transport structure  [L0 .. L_{n-1} | D_1 .. D_n | R_0 .. R_{n-1}]  (blocks of 2 cells).
    Lead blocks: flat interface, oxide disorder repeating every block (one template for both
    leads), so all lead blocks are exact translated copies (except the two outermost, which carry
    the x-end terminations). Device blocks: rough interface and random oxide.
    Atom order is canonical per block. meta: lead/device block ids, x0 of each block."""
    nb = 2 * n_lead_blocks + n_dev_blocks
    n_cells = nb * CELLS_PER_BLOCK
    nx4 = 4 * n_cells
    rng = np.random.default_rng(seed)
    dis = Disorder(rng, nx4, nt)
    tmpl = Disorder(np.random.default_rng(lead_seed), NX4_BLOCK, nt)
    xl = np.arange(nx4)
    lead_x = (xl < NX4_BLOCK * n_lead_blocks) | (xl >= NX4_BLOCK * (n_lead_blocks + n_dev_blocks))
    for name in ("dsi", "phi", "do"):
        a, t = getattr(dis, name), getattr(tmpl, name)
        a[lead_x] = t[xl[lead_x] % NX4_BLOCK]
    window = (~lead_x).astype(float)
    reg = build_si_region(n_cells, nt, half_width, dis, rng, rough_rms, rough_corr,
                          rough_2d=rough_2d, rough_window=window)
    pos, spc, bonds = reg["pos"], reg["species"], reg["bonds"]
    n_si = reg["n_si"]
    block = np.empty(len(pos), int)
    block[:n_si] = reg["coords"][:, 0] // NX4_BLOCK
    block[n_si:] = np.clip(np.floor(pos[n_si:, 0] / L_BLOCK).astype(int), 0, nb - 1)
    x0 = np.arange(nb) * L_BLOCK
    order, bonds = _canonical_order(pos, spc, block, bonds, x0)
    pos, spc, block = pos[order], spc[order], block[order]
    lead_L = list(range(n_lead_blocks))
    dev = list(range(n_lead_blocks, n_lead_blocks + n_dev_blocks))
    lead_R = list(range(n_lead_blocks + n_dev_blocks, nb))
    meta = dict(nt=nt, half_width=half_width, rough_rms=rough_rms, rough_corr=rough_corr,
                seed=seed, lead_seed=lead_seed, Lx=None, center=reg["center"], x0=x0,
                lead_L=lead_L, dev=dev, lead_R=lead_R, profiles=reg["profiles"])
    return Structure(pos, spc, bonds, block, -np.ones(len(pos), int), nb, meta)
