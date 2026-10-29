"""Tight-binding Hamiltonian.

Reimplementation of the project's qtmode/hamiltonian.py (same parameters), with vectorized
assembly and minimum-image bonds for structures periodic in x (meta["Lx"]).

* Si: Vogl-Hjalmarson-Dow sp3s* (J. Phys. Chem. Solids 44, 365 (1983)), nearest neighbours,
  converted to Slater-Koster form, Harrison d^-2 scaling.
* Si-O: Harrison universal parameters (V = eta * hbar^2/(m d^2)) for sp3 Si and O, with the
  network Si-O length rescaled to 1.61 A, and onsite shifts of oxide Si and O fitted so that a
  bulk version of the oxide network has VB / CB offsets to Si of 4.4 / 3.2 eV (fit_oxide_offsets).
  The s* orbital of channel Si does not couple to O.
* Au: one s orbital, nearest-neighbour fcc hopping; Au couples to any atom within 3 A
  through Harrison ss-sigma and sp-sigma.
Energies in eV, bulk Si valence-band top = 0.
"""
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from scipy.spatial import cKDTree
from .structure import (SI_C, SI_OX, OX, AU, NORB, D_SISI, A_SI, A_AU, D_SIO, D_SIO_GEOM,
                        _bridging_O)

HB2M = 7.62  # hbar^2/m in eV A^2


@dataclass
class TBParams:
    # Vogl sp3s* Si
    Es: float = -4.2
    Ep: float = 1.715
    Esx: float = 6.685
    ss: float = -8.3 / 4
    sp: float = 5.7292 * np.sqrt(3) / 4
    sxp: float = 5.3749 * np.sqrt(3) / 4
    pps: float = 2.71625
    ppp: float = -0.715
    # oxide: Harrison atomic levels shifted to the Vogl reference (+8.235 eV) plus fitted shifts
    EsO: float = -29.14 + 8.235
    EpO: float = -14.13 + 8.235
    dSiox: float = 0.0     # onsite shift of oxide Si (fitted)
    dO: float = 0.0        # onsite shift of O (fitted)
    # Au
    E_au: float = 2.0
    t_au: float = -0.8
    au_cut: float = 3.0

    def onsite(self, s):
        if s == SI_C:
            return np.array([self.Es, self.Ep, self.Ep, self.Ep, self.Esx])
        if s == SI_OX:
            return np.array([self.Es, self.Ep, self.Ep, self.Ep]) + self.dSiox
        if s == OX:
            return np.array([self.EsO, self.EpO, self.EpO, self.EpO]) + self.dO
        return np.array([self.E_au])


def _sk_params(sA, sB, d, P):
    """(ss, sp, ps, pps, ppp, s*p, ps*) arrays for bond lengths d."""
    pair = {sA, sB}
    z = np.zeros_like(d)
    if sA == SI_C and sB == SI_C:
        f = (D_SISI / d) ** 2
        return P.ss * f, P.sp * f, P.sp * f, P.pps * f, P.ppp * f, P.sxp * f, P.sxp * f
    if AU in pair:
        if sA == AU and sB == AU:
            return P.t_au * (A_AU / np.sqrt(2) / d) ** 2, z, z, z, z, z, z
        k = HB2M / d**2
        return -1.32 * k, 1.42 * k, 1.42 * k, z, z, z, z
    if OX in pair and pair != {OX}:
        k = HB2M / (d * D_SIO / D_SIO_GEOM) ** 2      # network length rescaled to 1.61 A
        return -1.32 * k, 1.42 * k, 1.42 * k, 2.22 * k, -0.63 * k, z, z
    raise ValueError(f"no hopping defined for species pair {sA},{sB}")


def sk_blocks(sA, sB, dvec, P):
    """<orbitals of A | H | orbitals of B> for many bonds; dvec (nb, 3) = r_B - r_A."""
    dvec = np.atleast_2d(dvec)
    d = np.linalg.norm(dvec, axis=1)
    l = dvec / d[:, None]
    ss, sp_, ps, pps, ppp, xp, px = _sk_params(sA, sB, d, P)
    nA, nB = NORB[sA], NORB[sB]
    M = np.zeros((len(d), nA, nB))
    M[:, 0, 0] = ss
    if nB >= 4:
        M[:, 0, 1:4] = l * sp_[:, None]
    if nA >= 4:
        M[:, 1:4, 0] = -l * ps[:, None]
    if nA >= 4 and nB >= 4:
        M[:, 1:4, 1:4] = (l[:, :, None] * l[:, None, :] * (pps - ppp)[:, None, None]
                          + np.eye(3)[None] * ppp[:, None, None])
    if nA == 5 and nB >= 4:
        M[:, 4, 1:4] = l * xp[:, None]
    if nB == 5 and nA >= 4:
        M[:, 1:4, 4] = -l * px[:, None]
    return M


def sk_block(sA, sB, dvec, P):
    return sk_blocks(sA, sB, np.asarray(dvec, float)[None], P)[0]


def interaction_pairs(st, P):
    pairs = st.bonds.reshape(-1, 2)
    if (st.species == AU).any():
        pr = cKDTree(st.pos).query_pairs(P.au_cut, output_type="ndarray")
        keep = (st.species[pr] == AU).any(1)
        pairs = np.vstack([pairs, pr[keep]])
    return pairs


def bond_vectors(st, i, j):
    dv = st.pos[j] - st.pos[i]
    Lx = st.meta.get("Lx") if st.meta else None
    if Lx:
        dv[:, 0] -= Lx * np.round(dv[:, 0] / Lx)
    return dv


def build_H(st, P):
    """Sparse real-symmetric H (Gamma point if the structure is periodic in x)."""
    off = st.offsets
    n = off[-1]
    rows, cols, vals = [], [], []
    for s in np.unique(st.species):
        at = np.where(st.species == s)[0]
        o = P.onsite(s)
        idx = (off[at][:, None] + np.arange(len(o))[None]).ravel()
        rows.append(idx); cols.append(idx); vals.append(np.tile(o, len(at)))
    pairs = interaction_pairs(st, P)
    spA, spB = st.species[pairs[:, 0]], st.species[pairs[:, 1]]
    for a in np.unique(spA):
        for b in np.unique(spB):
            m = (spA == a) & (spB == b)
            if not m.any():
                continue
            i, j = pairs[m, 0], pairs[m, 1]
            M = sk_blocks(a, b, bond_vectors(st, i, j), P)
            r = off[i][:, None, None] + np.arange(NORB[a])[None, :, None]
            c = off[j][:, None, None] + np.arange(NORB[b])[None, None, :]
            R, C = np.broadcast_arrays(r, c)
            rows += [R.ravel(), C.ravel()]
            cols += [C.ravel(), R.ravel()]
            vals += [M.ravel(), M.ravel()]
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    H.eliminate_zeros()
    return H


def orbital_info(st):
    """Per-orbital atom index, species and local orbital number."""
    norb = st.norb
    atom = np.repeat(np.arange(len(st.species)), norb)
    local = np.concatenate([np.arange(k) for k in norb])
    return atom, st.species[atom], local


# ---------------------------------------------------------------------------------------
# Bulk checks and oxide offset fit
# ---------------------------------------------------------------------------------------
def _bloch(onsite, hops, k):
    """onsite: list of diag arrays per atom; hops: list of (i, j, M, dvec)."""
    off = np.concatenate([[0], np.cumsum([len(o) for o in onsite])])
    H = np.zeros((off[-1], off[-1]), complex)
    for i, o in enumerate(onsite):
        H[off[i]:off[i + 1], off[i]:off[i + 1]] += np.diag(o)
    for i, j, M, dv in hops:
        ph = np.exp(1j * k @ dv)
        H[off[i]:off[i + 1], off[j]:off[j + 1]] += M * ph
        H[off[j]:off[j + 1], off[i]:off[i + 1]] += M.T * np.conj(ph)
    return H


def bulk_si_bands(P, ks):
    """Primitive 2-atom diamond cell; ks in 1/A."""
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * A_SI / 4
    hops = [(0, 1, sk_block(SI_C, SI_C, d, P), d) for d in v]
    on = [P.onsite(SI_C)] * 2
    return np.array([np.linalg.eigvalsh(_bloch(on, hops, k)) for k in ks])


def _oxide_cell(P):
    """Conventional cubic cell of the ideal (undisordered) oxide network: 8 Si + 16 bowed O."""
    a = A_SI
    fcc = np.array([[0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0]])
    s0 = fcc * a / 4
    s1 = (fcc + 1) * a / 4
    si = np.vstack([s0, s1])
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * a / 4
    on = [P.onsite(SI_OX)] * 8
    hops = []
    nO = 0
    for i in range(4):
        for k in range(4):
            pB = si[i] + v[k]
            frac = np.round((pB - s1) / a, 6)
            j = 4 + int(np.where(np.all(np.abs(frac - np.round(frac)) < 1e-6, axis=1))[0][0])
            pO = _bridging_O(si[i], pB, 0.5 * np.pi * k, np.zeros(3))
            o = 8 + nO
            nO += 1
            on.append(P.onsite(OX))
            hops.append((i, o, sk_block(SI_OX, OX, pO - si[i], P), pO - si[i]))
            hops.append((o, j, sk_block(OX, SI_OX, pB - pO, P), pB - pO))
    return on, hops


def bulk_oxide_edges(P, nk=6):
    on, hops = _oxide_cell(P)
    a = A_SI
    g = (np.arange(nk) + 0.5) / nk * 2 * np.pi / a - np.pi / a
    ks = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
    ks = np.vstack([ks, np.zeros(3)])
    E = np.array([np.linalg.eigvalsh(_bloch(on, hops, k)) for k in ks])
    nocc = (8 * 4 + 16 * 6) // 2      # electrons: Si 4 each, O 6 each; 2 per state
    return E[:, :nocc].max(), E[:, nocc:].min()


def si_edges(P):
    a = A_SI
    ks = np.linspace(0, 1, 401)[:, None] * np.array([2 * np.pi / a, 0, 0])
    E = bulk_si_bands(P, ks)
    return E[:, :4].max(), E[:, 4:].min(), ks[np.argmin(E[:, 4]), 0] / (2 * np.pi / a)


def fit_oxide_offsets(P, vbo=4.4, cbo=3.2):
    """Fit dSiox, dO so the bulk oxide network has the given VB/CB offsets to Si."""
    from scipy.optimize import least_squares
    vbm, cbm, _ = si_edges(P)

    def res(x):
        P.dSiox, P.dO = x
        v, c = bulk_oxide_edges(P)
        return [v - (vbm - vbo), c - (cbm + cbo)]

    sol = least_squares(res, [0.0, 0.0])
    P.dSiox, P.dO = sol.x
    v, c = bulk_oxide_edges(P)
    return dict(dSiox=P.dSiox, dO=P.dO, ox_vbm=v, ox_cbm=c, si_vbm=vbm, si_cbm=cbm,
                ox_gap=c - v, residual=np.abs(sol.fun).max())
