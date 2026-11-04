"""Tight-binding model switch for the whole pipeline.

    QT_MODEL=sp3s     (default) Vogl sp3s* Si, 5 orbitals per channel Si   (qtx_ms.prep.hamiltonian)
    QT_MODEL=sp3d5s   Boykin sp3d5s* Si, 10 orbitals per channel Si        (qtx_ms.prep.hamiltonian5)

select() sets the orbital-count table in qtx_ms.prep.structure in place, so Structure.norb / .offsets,
partition() and every block index follow the chosen model. Results go to separate folders
(results_transport + suffix()) so the two models never share caches.

Energy windows of the scripts were written for sp3s* (lead conduction edge 1.2051 eV, bulk CBM
1.1714 eV). energy_shift() maps them: window edges tied to the device conduction edge move with
the lead edge of the chosen model, the lower core-window edge with the bulk CBM.
"""
import os
import json
import numpy as np
from pathlib import Path
from . import structure

EC_LEAD_SP3S = 1.2051     # sp3s* lead conduction edge (3 x 5 nm flat-interface lead)
XSEC = os.environ.get("XSEC", "3x5")                 # cross-section: 3x5 (default) | 3x15 | 5x10 (W x T reversed: 10 nm wide, 5 nm thick)
_CFG = {"3x5": dict(nt=(12, 8), half_width=(25.0, 15.0)), "3x15": dict(nt=(30, 8), half_width=(75.0, 15.0)),
        "5x10": dict(nt=(21, 12), half_width=(50.0, 25.0))}
EB_SP3S = 1.1714          # sp3s* bulk Si CBM
_NORB = {"sp3s": [5, 4, 4, 1], "sp3d5s": [10, 4, 4, 1]}
MODEL = None


def select(name=None):
    global MODEL
    name = name or os.environ.get("QT_MODEL", "sp3s")
    if name not in _NORB:
        raise ValueError(f"unknown QT_MODEL {name}")
    structure.NORB[:] = _NORB[name]
    MODEL = name
    return name


def n_si():
    return int(structure.NORB[structure.SI_C])


def suffix():
    return ("" if MODEL == "sp3s" else f"_{MODEL}") + ("" if XSEC == "3x5" else f"_{XSEC}")


def device_cfg(xsec=None):
    """build_open_device arguments of a cross-section (default: env XSEC; same recipe, seeds and roughness)."""
    return dict(n_dev_blocks=4, n_lead_blocks=3, rough_rms=1.5, rough_corr=8.0, seed=7, lead_seed=1000,
                **_CFG[xsec or XSEC])


def reference_edge():
    """True when the energy settings apply unshifted (sp3s*, 3 x 5 nm: lead edge EC_LEAD_SP3S)."""
    return MODEL == "sp3s" and XSEC == "3x5"


def make_params():
    if MODEL == "sp3s":
        from .hamiltonian import TBParams, fit_oxide_offsets
        P = TBParams(); fit_oxide_offsets(P); return P
    from .hamiltonian5 import TB5Params, align
    P = TB5Params(); align(P); return P


def build_H(st, P):
    if MODEL == "sp3s":
        from .hamiltonian import build_H as b
        return b(st, P)
    from .hamiltonian5 import build_H5
    return build_H5(st, P)


def bulk_cbm(P):
    if MODEL == "sp3s":
        from .hamiltonian import si_edges
        return si_edges(P)[1]
    from .hamiltonian5 import si_edges5
    v, c, _ = si_edges5(P)
    return c - v


def lead_edge(lead, nk=9, k_extra=8, cache=None):
    """Conduction edge of a periodic lead (cell H00, coupling H01): min over k in [0, pi] of the
    lowest eigenvalue above mid-gap, by shift-invert at each k. Cached in `cache` (json)."""
    if cache is not None and Path(cache).exists():
        return json.load(open(cache))["Ec"]
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    H00, H01 = lead.H00.tocsc(), lead.H01.tocsc()
    n = H00.shape[0]; sig = 0.6
    def lowest(k):
        Hk = (H00 + H01 * np.exp(1j * k) + H01.T * np.exp(-1j * k)).astype(complex).tocsc()
        lu = spla.splu((Hk - sig * sp.identity(n, format="csc")).tocsc())
        op = spla.LinearOperator((n, n), matvec=lu.solve, dtype=complex)
        k_ = k_extra
        while True:                             # wide sheets: the nearest k_extra levels can all be valence
            w = spla.eigsh(Hk, k=k_, sigma=sig, which="LM", OPinv=op, tol=1e-9, return_eigenvectors=False)
            w = np.sort(w.real)
            if (w > sig).any():
                return w[w > sig].min()
            k_ *= 2
    ks = np.linspace(0, np.pi, nk); e = np.array([lowest(k) for k in ks])
    i = int(np.argmin(e))                       # refine around the minimum (golden section)
    a, b = ks[max(i - 1, 0)], ks[min(i + 1, nk - 1)]
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a); fc, fd = lowest(c), lowest(d)
    for _ in range(12):
        if fc < fd: b, d, fd = d, c, fc; c = b - g * (b - a); fc = lowest(c)
        else: a, c, fc = c, d, fd; d = a + g * (b - a); fd = lowest(d)
    Ec = float(min(fc, fd, e.min()))
    if cache is not None:
        json.dump(dict(Ec=Ec, k=[float(x) for x in ks], E=e.tolist()), open(cache, "w"))
    return Ec


def lead_cell(st, H, P=None):
    """Periodic lead cell (H00, H01) for the open device st (blocks L0..L_{n-1} | D | R0..R_{n-1}), in the
    orbital order of R1 = the block the device couples to on the right (and L1 on the left).

    The lead block next to the device (R0, L_{n-1}) is not always a translated copy of the next one: atoms at
    the device face follow the rough device (3 x 15, 5 x 10). R1 and L1 are interior, but R2 / L0 carry the
    x-end terminations, so H01 is taken from a build with one more lead block, where R1 -> R2 are both
    interior. If R0 is itself a translated copy of R1 (3 x 5), the original cell H(R0, R0), H(R0, R1) is
    returned unchanged (bit-identical to the pre-2026-10-23 pipeline). Checked: L1, R1 of st and R1, R2 of
    that build are exact translated copies (positions and species in canonical order), and H(R1, R1) agrees
    between the two builds."""
    from .structure import build_open_device
    m = st.meta
    cfg = dict(n_dev_blocks=len(m["dev"]), n_lead_blocks=len(m["lead_L"]), nt=m["nt"], half_width=m["half_width"],
               rough_rms=m["rough_rms"], rough_corr=m["rough_corr"], seed=m["seed"], lead_seed=m["lead_seed"])
    nl, nb = cfg["n_lead_blocks"], st.nblocks
    bL1, bR1 = 1, nb - nl + 1

    def same(sa, ba, sb, bb):
        ma, mb = sa.block == ba, sb.block == bb
        return bool(ma.sum() == mb.sum() and np.array_equal(sa.species[ma], sb.species[mb]) and
                    np.abs(sa.pos[ma][:, 0] - sa.meta["x0"][ba] - sb.pos[mb][:, 0] + sb.meta["x0"][bb]).max() < 1e-8 and
                    np.abs(sa.pos[ma][:, 1:] - sb.pos[mb][:, 1:]).max() < 1e-8)
    assert same(st, bL1, st, bR1), "lead blocks L1 and R1 differ"
    o = np.repeat(st.block, st.norb)
    R0, R1 = np.where(o == bR1 - 1)[0], np.where(o == bR1)[0]
    if same(st, bR1 - 1, st, bR1):                               # R0 periodic (3 x 5): the original cell, exactly
        H00 = H[R0][:, R0].tocsr()
        assert abs(H00 - H[R1][:, R1]).max() < 1e-10
        return H00, H[R0][:, R1].tocsr()
    cfg4 = dict(cfg, n_lead_blocks=nl + 1)
    st4 = build_open_device(**cfg4)
    b1, b2 = st4.nblocks - nl, st4.nblocks - nl + 1             # R1, R2 of the longer build
    assert same(st, bR1, st4, b1) and same(st4, b1, st4, b2), "no periodic interior lead cell"
    H4 = build_H(st4, make_params() if P is None else P).tocsr()
    o4 = np.repeat(st4.block, st4.norb)
    I1, I2 = np.where(o4 == b1)[0], np.where(o4 == b2)[0]
    H00 = H[R1][:, R1].tocsr()
    assert abs(H00 - H4[I1][:, I1]).max() < 1e-12 and abs(H00 - H4[I2][:, I2]).max() < 1e-12
    return H00, H4[I1][:, I2].tocsr()


def energy_shift(Ec_lead, Eb):
    """(dE_edge, dE_bulk): add dE_edge to energies tied to the device edge, dE_bulk to the lower
    core-window edge, to move the sp3s* settings to the chosen model."""
    return Ec_lead - EC_LEAD_SP3S, Eb - EB_SP3S


select()
