"""
Python bindings for C++ causal discovery kernels
==================================================

Accelerated NOTEARS and Granger via ctypes.
Auto-fallback to Python implementations if library not available.

Usage:
    from fusionmind4.realtime.causal_bindings import fast_notears, fast_granger

    W = fast_notears(data, lambda1=0.05, w_threshold=0.10)
    gc = fast_granger(data, max_lag=5, alpha=0.05)

Author: Dr. Mladen Mešter, dr.med., March 2026
"""

import ctypes
import numpy as np
import os
import time
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Load shared library
# ---------------------------------------------------------------------------

_LIB_NAME = "libfusionmind_causal.so"
_LIB_PATHS = [
    os.path.join(os.path.dirname(__file__), "cpp", _LIB_NAME),
    os.path.join(os.path.dirname(__file__), _LIB_NAME),
    _LIB_NAME,
]

_lib = None
for path in _LIB_PATHS:
    if os.path.exists(path):
        try:
            _lib = ctypes.CDLL(path)
            break
        except OSError:
            pass

CAUSAL_CPP_AVAILABLE = _lib is not None

# ---------------------------------------------------------------------------
# Setup C function signatures
# ---------------------------------------------------------------------------

if _lib:
    _c_double_p = ctypes.POINTER(ctypes.c_double)

    _lib.fm_notears.argtypes = [
        _c_double_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int,
        _c_double_p,
    ]
    _lib.fm_notears.restype = ctypes.c_double

    _lib.fm_notears_bootstrap.argtypes = [
        _c_double_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_int, _c_double_p,
    ]
    _lib.fm_notears_bootstrap.restype = None

    _lib.fm_granger_all_pairs.argtypes = [
        _c_double_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_double,
        _c_double_p, _c_double_p,
    ]
    _lib.fm_granger_all_pairs.restype = None

    _lib.fm_h_acyclicity.argtypes = [_c_double_p, ctypes.c_int]
    _lib.fm_h_acyclicity.restype = ctypes.c_double

    _lib.fm_expm.argtypes = [_c_double_p, ctypes.c_int, _c_double_p]
    _lib.fm_expm.restype = None

    # Optional hardware control functions (may not be present in all builds)
    for _fname, _at, _rt in [
        ('fm_pin_to_core', [ctypes.c_int], ctypes.c_int),
        ('fm_set_threads', [ctypes.c_int, ctypes.c_int], None),
        ('fm_capabilities', [], ctypes.c_char_p),
        ('fm_get_max_threads', [], ctypes.c_int),
    ]:
        try:
            fn = getattr(_lib, _fname)
            fn.argtypes = _at
            fn.restype = _rt
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# High-level Python API
# ---------------------------------------------------------------------------

def _as_c_double(arr: np.ndarray):
    """Convert to contiguous float64 and return ctypes pointer."""
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), arr


def fast_notears(data: np.ndarray,
                 lambda1: float = 0.05,
                 w_threshold: float = 0.10,
                 max_outer: int = 50,
                 max_inner: int = 30) -> np.ndarray:
    """Run NOTEARS DAG learning using C++ kernel.

    Args:
        data: (n_samples, n_vars) observational data
        lambda1: L1 regularization
        w_threshold: Edge pruning threshold
        max_outer: Augmented Lagrangian outer iterations
        max_inner: Proximal gradient inner iterations

    Returns:
        W: (n_vars, n_vars) weighted adjacency matrix (DAG)
    """
    if not CAUSAL_CPP_AVAILABLE:
        # Fallback to Python
        from ..discovery.notears import NOTEARSDiscovery
        nt = NOTEARSDiscovery(lambda1=lambda1, w_threshold=w_threshold,
                              max_iter=max_outer)
        return nt.fit(data)

    n, d = data.shape
    X_ptr, X_arr = _as_c_double(data)

    W_out = np.zeros((d, d), dtype=np.float64, order='C')
    W_ptr, _ = _as_c_double(W_out)

    _lib.fm_notears(X_ptr, n, d,
                    lambda1, w_threshold,
                    max_outer, max_inner,
                    W_ptr)

    return W_out


def fast_notears_bootstrap(data: np.ndarray,
                           n_bootstrap: int = 15,
                           lambda1: float = 0.05,
                           w_threshold: float = 0.10,
                           seed: int = 42) -> np.ndarray:
    """Run NOTEARS bootstrap for edge stability using C++ kernel.

    Args:
        data: (n_samples, n_vars) data
        n_bootstrap: Number of bootstrap rounds
        lambda1: Base L1 regularization
        w_threshold: Base edge threshold
        seed: Random seed

    Returns:
        stability: (n_vars, n_vars) fraction of bootstraps with edge present
    """
    if not CAUSAL_CPP_AVAILABLE:
        from ..discovery.notears import NOTEARSDiscovery
        nt = NOTEARSDiscovery(lambda1=lambda1, w_threshold=w_threshold)
        rng = np.random.RandomState(seed)
        return nt.fit_bootstrap(data, n_bootstrap, rng)

    n, d = data.shape
    X_ptr, _ = _as_c_double(data)

    stab = np.zeros((d, d), dtype=np.float64, order='C')
    stab_ptr, _ = _as_c_double(stab)

    _lib.fm_notears_bootstrap(X_ptr, n, d,
                              n_bootstrap,
                              lambda1, w_threshold,
                              seed, stab_ptr)
    return stab


def fast_granger(data: np.ndarray,
                 max_lag: int = 5,
                 alpha: float = 0.05,
                 return_pvalues: bool = False) -> np.ndarray:
    """Run Granger causality for all pairs using C++ kernel.

    Args:
        data: (n_samples, n_vars) time series data
        max_lag: Maximum lag to test
        alpha: Significance level (Bonferroni-corrected internally)
        return_pvalues: If True, return (gc_matrix, pval_matrix)

    Returns:
        gc_matrix: (n_vars, n_vars) binary causality matrix
        pval_matrix: (optional) (n_vars, n_vars) p-value matrix
    """
    if not CAUSAL_CPP_AVAILABLE:
        from ..discovery.granger import GrangerCausalityTest
        gc = GrangerCausalityTest(max_lag=max_lag, alpha=alpha, bonferroni=True)
        gc_mat = gc.test_all_pairs(data)
        if return_pvalues:
            pval_mat = gc.test_all_pairs_pvalues(data)
            return gc_mat, pval_mat
        return gc_mat

    n, d = data.shape
    data_ptr, _ = _as_c_double(data)

    gc_out = np.zeros((d, d), dtype=np.float64, order='C')
    gc_ptr, _ = _as_c_double(gc_out)

    if return_pvalues:
        pval_out = np.ones((d, d), dtype=np.float64, order='C')
        pval_ptr, _ = _as_c_double(pval_out)
    else:
        pval_ptr = ctypes.cast(ctypes.c_void_p(0), ctypes.POINTER(ctypes.c_double))
        pval_out = None

    _lib.fm_granger_all_pairs(data_ptr, n, d,
                              max_lag, alpha,
                              gc_ptr, pval_ptr)

    if return_pvalues:
        return gc_out, pval_out
    return gc_out


def fast_h_acyclicity(W: np.ndarray) -> float:
    """Compute h(W) = tr(exp(W∘W)) - d using C++ kernel."""
    if not CAUSAL_CPP_AVAILABLE:
        from scipy.linalg import expm
        M = W * W
        return float(np.trace(expm(M)) - W.shape[0])

    d = W.shape[0]
    W_ptr, _ = _as_c_double(W)
    return float(_lib.fm_h_acyclicity(W_ptr, d))


def fast_expm(M: np.ndarray) -> np.ndarray:
    """Compute matrix exponential using AVX-512 Taylor series."""
    if not CAUSAL_CPP_AVAILABLE:
        from scipy.linalg import expm
        return expm(M)

    d = M.shape[0]
    M_ptr, _ = _as_c_double(M)
    E = np.zeros((d, d), dtype=np.float64, order='C')
    E_ptr, _ = _as_c_double(E)
    _lib.fm_expm(M_ptr, d, E_ptr)
    return E


def pin_to_core(core_id: int) -> int:
    """Pin current thread to a specific CPU core for real-time determinism."""
    if CAUSAL_CPP_AVAILABLE and hasattr(_lib, 'fm_pin_to_core'):
        try:
            return int(_lib.fm_pin_to_core(core_id))
        except (AttributeError, OSError):
            pass
    return -1


def set_threads(n_threads: int = 0, pin_start: int = -1):
    """Set OpenMP thread count and optional core pinning."""
    if CAUSAL_CPP_AVAILABLE and hasattr(_lib, 'fm_set_threads'):
        try:
            _lib.fm_set_threads(n_threads, pin_start)
        except (AttributeError, OSError):
            pass


def get_capabilities() -> dict:
    """Query engine SIMD and threading capabilities."""
    info = {
        'cpp_available': CAUSAL_CPP_AVAILABLE,
        'backend': 'scalar',
        'max_threads': 1,
    }
    if CAUSAL_CPP_AVAILABLE:
        try:
            info['backend'] = _lib.fm_capabilities().decode()
        except (AttributeError, OSError):
            info['backend'] = 'avx512+openmp (no query fn)'
        try:
            info['max_threads'] = int(_lib.fm_get_max_threads())
        except (AttributeError, OSError):
            import os
            info['max_threads'] = os.cpu_count() or 1
    return info


# ---------------------------------------------------------------------------
# Benchmark utility
# ---------------------------------------------------------------------------

def benchmark_causal_kernels(data: np.ndarray, n_rounds: int = 5) -> dict:
    """Benchmark C++ vs Python causal discovery speed."""
    n, d = data.shape

    results = {}

    # -- NOTEARS --
    # Python
    from ..discovery.notears import NOTEARSDiscovery
    nt = NOTEARSDiscovery(lambda1=0.05, w_threshold=0.10)
    t0 = time.perf_counter()
    for _ in range(n_rounds):
        nt.fit(data)
    py_notears = (time.perf_counter() - t0) / n_rounds * 1000

    # C++
    t0 = time.perf_counter()
    for _ in range(n_rounds):
        fast_notears(data)
    cpp_notears = (time.perf_counter() - t0) / n_rounds * 1000

    results['notears'] = {
        'python_ms': py_notears,
        'cpp_ms': cpp_notears,
        'speedup': py_notears / max(cpp_notears, 0.001),
    }

    # -- Granger --
    from ..discovery.granger import GrangerCausalityTest
    gc = GrangerCausalityTest(max_lag=5, alpha=0.05, bonferroni=True)
    t0 = time.perf_counter()
    for _ in range(n_rounds):
        gc.test_all_pairs(data)
    py_granger = (time.perf_counter() - t0) / n_rounds * 1000

    t0 = time.perf_counter()
    for _ in range(n_rounds):
        fast_granger(data)
    cpp_granger = (time.perf_counter() - t0) / n_rounds * 1000

    results['granger'] = {
        'python_ms': py_granger,
        'cpp_ms': cpp_granger,
        'speedup': py_granger / max(cpp_granger, 0.001),
    }

    # -- Bootstrap NOTEARS --
    rng = np.random.RandomState(42)
    t0 = time.perf_counter()
    nt.fit_bootstrap(data, n_bootstrap=5, rng=rng)
    py_boot = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    fast_notears_bootstrap(data, n_bootstrap=5)
    cpp_boot = (time.perf_counter() - t0) * 1000

    results['bootstrap_5x'] = {
        'python_ms': py_boot,
        'cpp_ms': cpp_boot,
        'speedup': py_boot / max(cpp_boot, 0.001),
    }

    # -- Matrix exponential --
    M = np.random.randn(d, d) * 0.3
    from scipy.linalg import expm as scipy_expm
    t0 = time.perf_counter()
    for _ in range(100):
        scipy_expm(M * M)
    py_expm = (time.perf_counter() - t0) / 100 * 1000

    t0 = time.perf_counter()
    for _ in range(100):
        fast_expm(M * M)
    cpp_expm = (time.perf_counter() - t0) / 100 * 1000

    results['expm_9x9'] = {
        'python_ms': py_expm,
        'cpp_ms': cpp_expm,
        'speedup': py_expm / max(cpp_expm, 0.001),
    }

    return results
