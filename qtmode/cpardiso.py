"""Minimal ctypes interface to MKL PARDISO for complex sparse systems (mtype 6: complex symmetric, upper triangle; mtype 13: complex unsymmetric with matching + scaling)."""
import os
import sys
import ctypes
import numpy as np
import scipy.sparse as sp

_MKL = [os.environ.get("PYPARDISO_MKL_RT"), os.path.join(sys.prefix, "lib", "libmkl_rt.so.3"),   # env, uv venv (PyPI mkl)
        "/usr/local/lib/libmkl_rt.so.3"]                                                         # cloud box
_lib = ctypes.CDLL(next(p for p in _MKL if p and os.path.exists(p)))
_p = _lib.pardiso
_i32 = ctypes.POINTER(ctypes.c_int32)
_p.argtypes = [ctypes.POINTER(ctypes.c_int64), _i32, _i32, _i32, _i32, _i32, ctypes.c_void_p,
               _i32, _i32, _i32, _i32, _i32, _i32, ctypes.c_void_p, ctypes.c_void_p, _i32]
_p.restype = None


class CPardiso:
    def __init__(self, A, mtype=6):
        A = sp.csr_matrix(A, dtype=np.complex128)
        if mtype == 6:                       # complex symmetric: upper triangle only
            A = sp.triu(A, format="csr")
        A.sort_indices()
        self.A = A
        self.n = A.shape[0]
        self.mtype = mtype
        self.pt = np.zeros(64, np.int64)
        self.iparm = np.zeros(64, np.int32)
        self.iparm[0] = 1          # user iparm
        self.iparm[1] = 2          # nested dissection (METIS)
        self.iparm[7] = 2          # iterative refinement steps
        self.iparm[9] = 8 if mtype == 6 else 13      # pivot perturbation
        self.iparm[10] = 0 if mtype == 6 else 1      # scaling
        self.iparm[12] = 0 if mtype == 6 else 1      # weighted matching
        self.iparm[17] = -1        # report nnz(factor)
        self.iparm[18] = -1        # report MFlop
        self.iparm[34] = 1         # zero-based indexing
        self.ia = A.indptr.astype(np.int32)
        self.ja = A.indices.astype(np.int32)
        self.data = np.ascontiguousarray(A.data)
        self._call(12, np.zeros((self.n, 1), complex))

    def _call(self, phase, b):
        b = np.asfortranarray(b, dtype=np.complex128)
        x = np.zeros_like(b)
        err = ctypes.c_int32(0)
        nrhs = 1 if b.ndim == 1 else b.shape[1]
        _p(self.pt.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.byref(ctypes.c_int32(1)),
           ctypes.byref(ctypes.c_int32(1)), ctypes.byref(ctypes.c_int32(self.mtype)),
           ctypes.byref(ctypes.c_int32(phase)), ctypes.byref(ctypes.c_int32(self.n)),
           self.data.ctypes.data, self.ia.ctypes.data_as(_i32), self.ja.ctypes.data_as(_i32),
           np.zeros(1, np.int32).ctypes.data_as(_i32), ctypes.byref(ctypes.c_int32(nrhs)),
           self.iparm.ctypes.data_as(_i32), ctypes.byref(ctypes.c_int32(0)), b.ctypes.data,
           x.ctypes.data, ctypes.byref(err))
        if err.value != 0:
            raise RuntimeError(f"PARDISO error {err.value} in phase {phase}")
        return x

    def solve(self, B):
        return self._call(33, B)

    def free(self):
        self._call(-1, np.zeros((self.n, 1), complex))
