"""Where the energy-independent setup of obc_beyn goes at 12 nm (user, 2026-11-25): the steps of BeynLead.__init__ and BeynLead._basis one by
one (left side, one rank, 8 threads as in the runs), and real / GPU variants of the expensive ones.
    python -u prof_beyn.py STORE PREP"""
import sys, time
import numpy as np
import scipy.sparse as sp
from qtx_ms import devstore, gpu
from qtx_ms import msbasis as mb
from qtx_ms import obc_beyn as ob

gpu.enable()
cp = gpu.cp


class T:
    def __init__(self, name): self.name = name
    def __enter__(self): gpu.sync(); self.t = time.time()
    def __exit__(self, *a):
        gpu.sync(); print(f"{time.time() - self.t:8.1f} s  {self.name}", flush=True)


inp, dev = devstore.load(sys.argv[1], source=sys.argv[2])
m, ld = inp["meta"], inp["lead"]
tr, obk = m["transport"], np.repeat(inp["blocks"], inp["norb"])
s = int(m.get("cells_per_block") or m.get("source", {}).get("cells_per_block", 1))
orbs = np.where(obk == tr["lead_left_block"])[0]
print("lead cell", ld["H00"].shape, "sections", s, "nnz", {k: ld[k].nnz for k in ("H00", "H01", "S00", "S01")},
      "dtype", {k: str(ld[k].dtype) for k in ("H00", "S01")}, flush=True)
with T("lead_cell_perm"):
    p = mb.lead_cell_perm(dict(inp, meta=dict(m, lead_cell_length=m["block_length"])), orbs, s)
p = np.arange(ld["H00"].shape[0]) if p is None else p
n = len(p)
mm = n // s
print("--- BeynLead.__init__, step by step")
with T("toarray + complex, 4 matrices"):
    D = {k: np.asarray(sp.csr_matrix(ld[k]).toarray(), complex) for k in ("H00", "H01", "S00", "S01")}
print("imaginary part", {k: float(np.abs(D[k].imag).max()) for k in D}, flush=True)
with T("H10 = H01^H (2)"):
    M10 = {q: D[q + "01"].conj().T for q in "HS"}
layer = {}
for q in "HS":
    with T(f"permuted copies (np.ix_, 3 of {n}^2), {q}"):
        trip = tuple(X[np.ix_(p, p)] for X in (M10[q], D[q + "00"], D[q + "01"]))
    with T(f"_layer, {q}"):
        layer[q] = ob._layer(*trip, s) if s > 1 else list(trip)
    del trip
with T("probing block Y, m x m"):
    (np.random.default_rng(42).random((mm, mm)) + 1j * np.random.default_rng(43).random((mm, mm)))
print("--- BeynLead._basis, step by step")
with T("_blocks_of (2)"):
    Hij, Sij = (ob._blocks_of(layer[q], s, np, "ij") for q in "HS")
with T("support: 4 abs sums"):
    rows = np.where((np.abs(Hij).sum(1) + np.abs(Sij).sum(1)) > 0)[0]
    cols = np.where((np.abs(Hij).sum(0) + np.abs(Sij).sum(0)) > 0)[0]
with T("submatrices (2)"):
    Hs, Ss = Hij[np.ix_(rows, cols)], Sij[np.ix_(rows, cols)]
del Hij, Sij
nH, nS = np.linalg.norm(Hs), np.linalg.norm(Ss)
with T("hstack"):
    X = np.hstack([Hs / nH, Ss / nS])
print("SVD input", X.shape, X.dtype, "support", len(rows), len(cols), flush=True)
with T("SVD copy to GPU"):
    Xd = cp.asarray(X)
with T("SVD complex128 (gesvd, GPU)"):
    U, sv, _ = cp.linalg.svd(Xd, full_matrices=False)
sv = gpu.asnumpy(sv)
rk = int((sv > 1e-14 * sv[0]).sum())
with T("U to host"):
    Uc = gpu.asnumpy(U[:, :rk])
del U
print("rank", rk, flush=True)
with T("SVD real float64 (GPU)"):
    Ur_, svr, _ = cp.linalg.svd(Xd.real, full_matrices=False)
svr = gpu.asnumpy(svr)
print("rank real", int((svr > 1e-14 * svr[0]).sum()), "max sv diff", float(np.abs(sv - svr).max() / sv[0]), flush=True)
del Ur_
with T("QR of the transpose (GPU, real) + SVD of R"):
    Rf = cp.linalg.qr(Xd.real.T, mode="r")
    _, svq, _ = cp.linalg.svd(Rf.T, full_matrices=False)
print("rank QR", int((gpu.asnumpy(svq) > 1e-14 * float(svq[0])).sum()), flush=True)
del Rf, svq, Xd
gpu.free()
Ur2 = Uc                                                              # same size as Ur: rr = rc up to a few
with T("K = Uc^H Hs Ur (host, complex, OMP threads of the run)"):
    K = Uc.conj().T @ Hs @ Ur2[:len(cols)] if len(cols) == len(rows) else Uc.conj().T @ Hs
with T("K on GPU, complex"):
    Uc_d, Hs_d = cp.asarray(Uc), cp.asarray(Hs)
    Kd = Uc_d.conj().T @ Hs_d @ Uc_d
with T("reconstruction error, one of two (host, complex)"):
    err = np.abs(Uc @ K @ Ur2.conj().T - Hs).max()
with T("reconstruction error, GPU"):
    cp.abs(Uc_d @ Kd @ Uc_d.conj().T - Hs_d).max().item()
del Kd, Uc_d, Hs_d
gpu.free()
print("--- whole calls (as the run)")
gpu.free()
with T("BeynLead()"):
    bl = ob.BeynLead(ld["H00"], ld["H01"], ld["S00"], ld["S01"], s, p, sides="L", layers_on="host", fp_store="host", m0=max(64, mm // 2))
with T("BeynLead._basis('L')"):
    bl._basis("L")
