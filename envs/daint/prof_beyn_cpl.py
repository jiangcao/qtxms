"""The rest of the "leads" time of hier.obc._beyn at 12 nm: faces, the couplings of the store (dense, memory mapped) to sparse. python -u prof_beyn_cpl.py STORE PREP"""
import sys, time
import numpy as np
import scipy.sparse as sp
from qtx_ms import devstore

t = time.time()
def tick(name):
    global t
    print(f"{time.time() - t:8.1f} s  {name}", flush=True)
    t = time.time()


inp, dev = devstore.load(sys.argv[1], source=sys.argv[2])
tick("devstore.load")
ld = inp["lead"]
for k in ("H00", "H01", "S00", "S01"):
    print(k, type(ld[k]).__name__)
P01 = abs(sp.csr_matrix(ld["H01"])) + abs(sp.csr_matrix(ld["S01"]))
Rf, Lf = (np.unique(q) for q in P01.nonzero())
face = np.union1d(Rf, Lf)
tick(f"faces ({len(face)})")
n = ld["H00"].shape[0]
for side, cpl in (("L", dev.cL), ("R", dev.cR)):
    for q, c in enumerate(cpl):
        a = np.asarray(c)
        print(side, q, a.shape, a.dtype, f"{a.nbytes / 2**30:.2f} GiB")
        float(a[::64].sum())                                          # touch some pages only
        b = np.array(a)
        tick("read into memory")
        M = sp.csr_matrix(b)
        tick(f"csr_matrix, nnz {M.nnz}")
        E = sp.csr_matrix((np.ones(len(face)), (np.asarray(face), np.arange(len(face)))), shape=(n, len(face)))
        F = (E @ M if q < 2 else M @ E.T).tocsr()
        tick("expand to the lead cell")
