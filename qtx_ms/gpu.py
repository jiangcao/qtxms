# qtx_ms twin of release/qtx_modespace/src/qtx_modespace/gpu.py: keep the two files identical below this line
"""Optional GPU backend (cupy) for the dense parts: lead Sigma (interior fold + Sancho decimation), the local density
matrices of level 1 (dense Schur / full LU per contour point), the level-2 slab recursion (g_diag), the exact QTBM
solve (block-tridiagonal LU of the device), the CAP-Schwarz window LUs and the FGMRES loop.

Switch: --gpu of the drivers (cli, run_ms, exact_T) or QTX_MODESPACE_GPU=1. Off by
default; everything then runs on numpy / scipy exactly as before. Needs cupy (the `gpu` extra, cupy-cuda12x).

Rules followed by the ports:
  * results that leave a module (Sigma, rho, eigenvectors, psi, T) are numpy arrays; device arrays stay inside
  * dense blocks are built on the device from the host CSR matrix (csr_dense): no multi-GB host -> device copies
  * LU + solve instead of explicit inverses where only solves are needed (cupy's inv costs ~3 LUs)
  * no fork after CUDA is initialized: in GPU mode the contour points and Schwarz windows run one after another in
    one process (one 8k-17k LU already fills the card)
"""
import os
import numpy as np
import scipy.linalg as sla

cp = None                       # the cupy module once enable() succeeded
on = False


def enable(flag=True, device=None):
    """Turn the GPU backend on (flag True) or off. Raises RuntimeError when cupy or a CUDA device is missing."""
    global cp, on
    if not flag:
        on = False
        return False
    try:
        import cupy
        import cupyx.scipy.linalg                                    # noqa: F401  (lu_factor / lu_solve)
        import cupyx.scipy.sparse                                    # noqa: F401
        if device is not None:
            cupy.cuda.Device(int(device)).use()
        cupy.abs(cupy.ones(1))                                       # fails here without a working driver
    except Exception as e:                                          # noqa: BLE001
        raise RuntimeError(f"GPU backend: cupy is not usable ({e}); install the gpu extra (cupy-cuda12x)") from e
    cp, on = cupy, True
    return True


def from_env():
    """enable() when QTX_MODESPACE_GPU is set to 1 / true / yes."""
    if os.environ.get("QTX_MODESPACE_GPU", "").lower() in ("1", "true", "yes", "on"):
        return enable()
    return False


def describe():
    if not on:
        return "CPU (numpy / scipy)"
    d = cp.cuda.Device()
    p = cp.cuda.runtime.getDeviceProperties(d.id)
    free, total = d.mem_info
    return f"GPU {d.id} {p['name'].decode()} ({total / 2**30:.1f} GB, {free / 2**30:.1f} GB free), cupy {cp.__version__}"


def xp_of(a):
    """numpy or cupy, whichever owns a."""
    return cp.get_array_module(a) if on else np


def asarray(a, dtype=None):
    """a on the device (GPU mode) or as a numpy array."""
    if on:
        return cp.asarray(a, dtype=dtype)
    return np.asarray(a, dtype=dtype)


def asnumpy(a):
    if on and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def pinned(a):
    """Copy of a (numpy or cupy array) in page-locked host memory, as a numpy array (plain numpy in CPU mode).
    Use it for host arrays that are copied to the device repeatedly: pageable copies go through a single-threaded CPU
    staging buffer (GH200: 4.4 GB/s host -> device, 26 GB/s back), pinned ones run at bus speed (~350 / ~300 GB/s)."""
    if not on:
        return np.asarray(a)
    import cupyx
    order = "F" if a.flags.f_contiguous and not a.flags.c_contiguous else "C"
    h = cupyx.empty_pinned(a.shape, a.dtype, order=order)
    if isinstance(a, cp.ndarray):
        a.get(out=h)
    else:
        np.copyto(h, a)
    return h


def csr(A, dtype=None):
    """Host sparse matrix -> cupyx CSR on the device (dtype: one of float32/64, complex64/128)."""
    import scipy.sparse as sp
    import cupyx.scipy.sparse as csp
    A = sp.csr_matrix(A)
    if dtype is not None:
        A = A.astype(dtype)
    elif A.dtype not in (np.float32, np.float64, np.complex64, np.complex128):
        A = A.astype(np.complex128 if np.iscomplexobj(A.data) else np.float64)
    return csp.csr_matrix(A)


def csr_dense(A, dtype=None):
    """Host sparse matrix -> dense device array (only the CSR arrays cross the bus)."""
    return csr(A, dtype).toarray()


def lu_factor(a):
    if on and isinstance(a, cp.ndarray):
        import cupyx.scipy.linalg as cla
        return cla.lu_factor(a, overwrite_a=True, check_finite=False)
    return sla.lu_factor(a, overwrite_a=True, check_finite=False)


def lu_solve(lu, b, trans=0):
    if on and isinstance(lu[0], cp.ndarray):
        import cupyx.scipy.linalg as cla
        return cla.lu_solve(lu, b, trans=trans, check_finite=False)
    return sla.lu_solve(lu, b, trans=trans, check_finite=False)


def inv(a):
    return xp_of(a).linalg.inv(a)


def solve(a, b):
    xp = xp_of(a)
    if xp is np:
        return sla.solve(a, b, check_finite=False)
    return xp.linalg.solve(a, b)


def sync():
    if on:
        cp.cuda.Stream.null.synchronize()


def free():
    """Return cached device blocks to the driver (between large phases)."""
    if on:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def mem_used():
    """Bytes held by cupy's memory pool (0 on CPU)."""
    return int(cp.get_default_memory_pool().used_bytes()) if on else 0


_peak = [0, False]                                                   # peak of the live bytes, allocator installed


def track_peak():
    """Record the peak of the live bytes of cupy's memory pool (an allocator that looks at the pool after every
    allocation). No-op on the CPU."""
    if not on or _peak[1]:
        return
    pool = cp.get_default_memory_pool()

    def alloc(size):
        m = pool.malloc(size)
        u = pool.used_bytes()
        if u > _peak[0]:
            _peak[0] = u
        return m
    cp.cuda.set_allocator(alloc)
    _peak[0], _peak[1] = int(pool.used_bytes()), True


def mem_mark():
    """Memory since the last mark, in bytes: live (arrays alive now), peak (highest live since the last mark; needs
    track_peak), pool (held by cupy, live + cached), free (driver free: other users of a shared GPU count).
    None on the CPU. Resets the peak."""
    if not on:
        return None
    pool = cp.get_default_memory_pool()
    live = int(pool.used_bytes())
    out = dict(live=live, peak=max(_peak[0], live), pool=int(pool.total_bytes()),
               free=int(cp.cuda.Device().mem_info[0]))
    _peak[0] = live
    return out
