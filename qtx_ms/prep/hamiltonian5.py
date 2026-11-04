"""sp3d5s* Hamiltonian for the Si/SiO2 structures of qtx_ms.prep.structure (drop-in for hamiltonian.build_H).

* Channel Si (SI_C): sp3d5s*, 10 orbitals, Boykin-Klimeck-Oyafuso PRB 69, 115201 (2004)
  (qtx_ms.prep.sp3d5s), onsite energies shifted so that the bulk Si valence-band top is 0 eV
  (same reference as the sp3s* model). Si-Si integrals scaled as (d0/d)^2 (Harrison); in the
  generated structures all Si-Si bonds have the ideal length, so this is inactive there.
* Oxide Si (SI_OX) and O: the sp3 Harrison oxide of qtx_ms.prep.hamiltonian, unchanged, with the
  onsite shifts dSiox, dO refitted so the bulk oxide network has VB / CB offsets of 4.4 / 3.2 eV
  to the sp3d5s* Si band edges.
* Interface Si-O bonds (SI_C - OX): Harrison s and p integrals between the s, p orbitals of
  channel Si and O; the d and s* orbitals of channel Si do not couple to O (as s* in sp3s*).
* Optional spin-orbit (spin=True): on the p orbitals of channel Si only (Delta/3 of the Si set);
  the oxide is spin-degenerate.
Orbital order on channel Si: s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2, s*.
"""
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field

from . import sp3d5s as T5
from .hamiltonian import TBParams, bond_vectors, bulk_oxide_edges, HB2M
from .structure import SI_C, SI_OX, OX, AU, D_SISI, D_SIO, D_SIO_GEOM

NORB5 = np.array([10, 4, 4, 1])


@dataclass
class TB5Params:
    si: dict = field(default_factory=lambda: dict(T5.BOYKIN_SI))
    a_si: float = T5.A_SI
    shift: float = 0.0               # added to all channel-Si onsites so bulk VBM = 0 (set by align)
    ox: TBParams = field(default_factory=TBParams)   # oxide Si / O parameters and fitted shifts
    vbo: float = 4.4
    cbo: float = 3.2

    def onsite(self, s):
        if s == SI_C:
            return np.diag(T5.onsite(self.si)) + self.shift
        return self.ox.onsite(s)


def si_edges5(P, nk=801, spin=False):
    """Bulk Si VBM (Gamma) and CBM (Delta line) of the unshifted sp3d5s* model."""
    a = P.a_si
    ev = lambda k: np.linalg.eigvalsh(T5.bulk_hk(np.asarray(k, float), P.si, a, spin=spin))
    nv = 8 if spin else 4
    vbm = ev([0, 0, 0])[nv - 1]
    ks = np.linspace(0.6, 1.0, nk) * 2 * np.pi / a
    cb = np.array([ev([k, 0, 0])[nv] for k in ks])
    return vbm, cb.min(), ks[cb.argmin()] * a / (2 * np.pi)


def align(P, spin=False):
    """Shift channel Si so that VBM = 0; refit the oxide shifts to the new Si band edges."""
    from scipy.optimize import least_squares
    vbm, cbm, kmin = si_edges5(P, spin=spin)
    P.shift = -vbm
    vb_t, cb_t = -P.vbo, (cbm - vbm) + P.cbo

    def res(x):
        P.ox.dSiox, P.ox.dO = x
        v, c = bulk_oxide_edges(P.ox)
        return [v - vb_t, c - cb_t]

    sol = least_squares(res, [P.ox.dSiox, P.ox.dO])
    P.ox.dSiox, P.ox.dO = sol.x
    v, c = bulk_oxide_edges(P.ox)
    return dict(si_vbm=0.0, si_cbm=cbm - vbm, si_kmin=kmin, dSiox=P.ox.dSiox, dO=P.ox.dO,
                ox_vbm=v, ox_cbm=c, ox_gap=c - v, residual=float(np.abs(sol.fun).max()))


def _harrison_sp(dvec, d):
    """4 x 4 Harrison s,p block <A|H|B> for Si-O bonds (network length rescaled to 1.61 A)."""
    k = HB2M / (d * D_SIO / D_SIO_GEOM) ** 2
    l = dvec / d
    M = np.zeros((4, 4))
    M[0, 0] = -1.32 * k
    M[0, 1:] = 1.42 * k * l
    M[1:, 0] = -1.42 * k * l
    M[1:, 1:] = np.outer(l, l) * (2.22 * k - (-0.63 * k)) + np.eye(3) * (-0.63 * k)
    return M


_SK_CACHE = {}


def _si_si(dvec, P):
    """Cached Si-Si block (bond directions repeat: 4 in the ideal lattice)."""
    key = (id(P.si), tuple(np.round(dvec, 5)))
    M = _SK_CACHE.get(key)
    if M is None:
        M = _SK_CACHE[key] = T5.sk_block(dvec, P.si)
    return M


def hop_block(sA, sB, dvec, P):
    d = float(np.linalg.norm(dvec))
    if sA == SI_C and sB == SI_C:
        return _si_si(dvec, P) * (D_SISI / d) ** 2
    pair = {sA, sB}
    if OX in pair and pair != {OX}:
        M4 = _harrison_sp(dvec, d)
        M = np.zeros((NORB5[sA], NORB5[sB]))
        M[:4, :4] = M4                       # s, p of channel Si (first 4 orbitals) or oxide Si
        return M
    raise ValueError(f"no hopping defined for species pair {sA},{sB}")


def offsets5(st):
    return np.concatenate([[0], np.cumsum(NORB5[st.species])])


def build_H5(st, P, spin=False):
    """Sparse H in the sp3d5s* basis (Gamma point for x-periodic structures). spin=True doubles
    the basis (all up, then all down) and adds spin-orbit on channel-Si p orbitals."""
    off = offsets5(st); n = off[-1]
    rows, cols, vals = [], [], []
    for s in np.unique(st.species):
        at = np.where(st.species == s)[0]; o = P.onsite(s)
        idx = (off[at][:, None] + np.arange(len(o))[None]).ravel()
        rows.append(idx); cols.append(idx); vals.append(np.tile(o, len(at)))
    pairs = st.bonds.reshape(-1, 2)
    dv = bond_vectors(st, pairs[:, 0], pairs[:, 1])
    spA, spB = st.species[pairs[:, 0]], st.species[pairs[:, 1]]
    for (i, j), d, a, b in zip(pairs, dv, spA, spB):
        M = hop_block(a, b, d, P)
        r, c = np.nonzero(M)
        rows += [off[i] + r, off[j] + c]; cols += [off[j] + c, off[i] + r]; vals += [M[r, c], M[r, c]]
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    H.eliminate_zeros()
    if not spin:
        return H
    H2 = sp.kron(sp.identity(2), H, format="csr").astype(complex)
    so = T5.so_block(P.si)                      # 20 x 20: (10 up, 10 down) on one atom
    si = np.where(st.species == SI_C)[0]
    loc = np.r_[0:10]
    r_, c_ = np.nonzero(so)
    R, C, V = [], [], []
    for a in si:
        g = np.r_[off[a] + loc, n + off[a] + loc]
        R.append(g[r_]); C.append(g[c_]); V.append(so[r_, c_])
    Hso = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(2 * n, 2 * n))
    return (H2 + Hso).tocsr()


def orbital_info5(st):
    norb = NORB5[st.species]
    atom = np.repeat(np.arange(len(st.species)), norb)
    local = np.concatenate([np.arange(k) for k in norb])
    return atom, st.species[atom], local
