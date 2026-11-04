"""Small synthetic transport inputs for the unit tests (generic input-set dicts, as read_inputs returns).

strip(): cubic grid of sites (spacing 2 A), `no` orbitals per site, nearest-neighbour H and S.
Transport blocks are 3 x-planes, so the lead cell has an interior plane (folded by FoldedLead).
Blocks: [lead | device ... | lead]; lead blocks are copies of the pristine cell, device blocks carry
random on-site disorder. complex_gauge=True applies D = diag(exp(i phi(y, z))) per orbital
(H -> D H D^H, S -> D S D^H): complex Hermitian matrices with the same transmission.
"""
import numpy as np
import scipy.sparse as sp
import pytest


def gauge_phase(xyz, no):
    """exp(i phi) per orbital; phi depends on (y, z) and the orbital only, so lead cells stay periodic."""
    phi = np.repeat(1.3 * xyz[:, 1] + 0.7 * xyz[:, 2] ** 2, no) + np.tile(np.arange(no) * 0.5, len(xyz))
    return np.exp(1j * phi)


def strip(nd=3, ny=3, nz=2, no=2, a=2.0, nonorth=True, disorder=0.4, seed=0, complex_gauge=False):
    rng = np.random.default_rng(seed)
    e0 = np.diag(np.linspace(0.0, 1.5, no))
    t = -np.eye(no) + 0.2 * (lambda M: M + M.T)(rng.normal(size=(no, no)))        # NN hopping block
    s1 = (0.08 * np.eye(no) + 0.02 * (lambda M: M + M.T)(rng.normal(size=(no, no)))) if nonorth else np.zeros((no, no))
    npl = 3                                         # x-planes per block
    nb = nd + 2
    nx = nb * npl
    sites = np.array([(ix, iy, iz) for ix in range(nx) for iy in range(ny) for iz in range(nz)])
    xyz = sites * a
    blocks = sites[:, 0] // npl
    ns = len(sites)
    idx = {tuple(s): i for i, s in enumerate(sites)}
    dis = np.zeros(ns)
    dev = (blocks > 0) & (blocks < nb - 1)
    dis[dev] = disorder * rng.uniform(-1, 1, dev.sum())
    H = sp.lil_matrix((ns * no, ns * no))
    S = sp.lil_matrix((ns * no, ns * no))
    for i, s in enumerate(sites):
        o = slice(i * no, (i + 1) * no)
        H[o, o] = e0 + dis[i] * np.eye(no)
        S[o, o] = np.eye(no)
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            j = idx.get(tuple(s + d))
            if j is None:
                continue
            p = slice(j * no, (j + 1) * no)
            H[o, p] = t; H[p, o] = t.T
            S[o, p] = s1; S[p, o] = s1.T
    H, S = H.tocsr(), S.tocsr()
    if complex_gauge:
        D = sp.diags(gauge_phase(xyz, no))
        H, S = (D @ H @ D.conj().T).tocsr(), (D @ S @ D.conj().T).tocsr()
    lead = _lead_cell(e0, t, s1, ny, nz, no, a, npl, complex_gauge)
    meta = dict(name="strip", units=dict(energy="eV", length="Angstrom"), axis=0, nblocks=nb,
                transport=dict(device_blocks=list(range(1, nb - 1)), lead_left_block=0, lead_right_block=nb - 1),
                edges=dict(Ec_lead=0.0))
    return dict(labels=np.array(["X"] * ns), xyz=xyz.astype(float), norb=np.full(ns, no), blocks=blocks,
                H=H, S=S, lead=lead, meta=meta)


def _lead_cell(e0, t, s1, ny, nz, no, a, npl, complex_gauge):
    """H00, S00 of one pristine cell and H01, S01 to the next, in the same orbital order as the blocks."""
    two = strip_pristine(ny, nz, no, a, npl, e0, t, s1, complex_gauge)
    n = two["n"]
    H, S = two["H"], two["S"]
    return dict(H00=H[:n, :n].tocsr(), H01=H[:n, n:].tocsr(), S00=S[:n, :n].tocsr(), S01=S[:n, n:].tocsr())


def strip_pristine(ny, nz, no, a, npl, e0, t, s1, complex_gauge):
    sites = np.array([(ix, iy, iz) for ix in range(2 * npl) for iy in range(ny) for iz in range(nz)])
    idx = {tuple(s): i for i, s in enumerate(sites)}
    ns = len(sites)
    H = sp.lil_matrix((ns * no, ns * no), dtype=complex)
    S = sp.lil_matrix((ns * no, ns * no), dtype=complex)
    for i, s in enumerate(sites):
        o = slice(i * no, (i + 1) * no)
        H[o, o] = e0; S[o, o] = np.eye(no)
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            j = idx.get(tuple(s + d))
            if j is None:
                continue
            p = slice(j * no, (j + 1) * no)
            H[o, p] = t; H[p, o] = t.T; S[o, p] = s1; S[p, o] = s1.T
    H, S = H.tocsr(), S.tocsr()
    if complex_gauge:
        D = sp.diags(gauge_phase(sites * a, no))
        H, S = D @ H @ D.conj().T, D @ S @ D.conj().T
    else:
        H, S = H.real, S.real
    return dict(H=sp.csr_matrix(H), S=sp.csr_matrix(S), n=ns * no // 2)


def dense_T(inp, E, eta=1e-6):
    """Caroli T(E) on the device with Sigma from the unfolded cell (qtx_ms.abtb.sancho_rubio)."""
    from qtx_ms.abtb import sancho_rubio
    m = inp["meta"]["transport"]
    ob = np.repeat(inp["blocks"], inp["norb"])
    dev = np.concatenate([np.where(ob == b)[0] for b in m["device_blocks"]])
    L, R = np.where(ob == m["lead_left_block"])[0], np.where(ob == m["lead_right_block"])[0]
    z = E + 1j * eta
    A = (z * inp["S"] - inp["H"]).toarray()
    ld = inp["lead"]
    A00 = (z * ld["S00"] - ld["H00"]).toarray()
    A01 = (z * ld["S01"] - ld["H01"]).toarray()
    A10 = (z * ld["S01"].conj().T - ld["H01"].conj().T).toarray()
    gL = sancho_rubio(A00, a=A10, b=A01)
    gR = sancho_rubio(A00, a=A01, b=A10)
    SL = A[np.ix_(dev, L)] @ gL @ A[np.ix_(L, dev)]
    SR = A[np.ix_(dev, R)] @ gR @ A[np.ix_(R, dev)]
    Ad = (E * inp["S"] - inp["H"]).toarray()[np.ix_(dev, dev)] - SL - SR
    G = np.linalg.inv(Ad)
    GL, GR = 1j * (SL - SL.conj().T), 1j * (SR - SR.conj().T)
    return float(np.real(np.trace(GR @ G @ GL @ G.conj().T))), SL, SR, dev


@pytest.fixture
def real_strip():
    return strip()


@pytest.fixture
def complex_strip():
    return strip(complex_gauge=True)
