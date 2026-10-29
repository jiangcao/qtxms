"""Stand-in for pypardiso when MKL is not available (e.g. Apple Silicon): same calls, SciPy SuperLU.
Only the subset used in qtmode: PyPardisoSolver(mtype=11).factorize / solve / free_memory."""
import numpy as np
import scipy.sparse.linalg as spla


class PyPardisoSolver:
    def __init__(self, mtype=11):
        self.lu = None

    def factorize(self, A):
        self.lu = spla.splu(A.tocsc(), permc_spec="COLAMD")

    def solve(self, A, B):
        if self.lu is None:
            self.factorize(A)
        B = np.asarray(B.toarray() if hasattr(B, "toarray") else B, dtype=float)
        return self.lu.solve(B)

    def free_memory(self, everything=True):
        self.lu = None
