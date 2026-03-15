"""
Python bindings for the C++ FusionMind Real-Time Engine
========================================================

Uses ctypes for zero-overhead FFI.  Auto-detects and loads the
shared library; falls back to pure Python if not compiled.

Usage:
    from fusionmind4.realtime.fast_bindings import FastEngine

    engine = FastEngine(n_vars=9)
    engine.load_ml_model(trees, means, stds, threshold)
    engine.load_causal_model(effects, boundaries)

    result = engine.predict(values, timestamp)
    print(f"Prob: {result.fused_prob:.3f}, Latency: {result.total_latency_ns:.0f} ns")

Author: Dr. Mladen Mešter, dr.med., March 2026
"""

import ctypes
import ctypes.util
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Locate shared library
# ---------------------------------------------------------------------------

_LIB_NAME = "libfusionmind_rt.so"
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

CPP_AVAILABLE = _lib is not None


# ---------------------------------------------------------------------------
# C struct mirrors
# ---------------------------------------------------------------------------

class CDualPredictionResult(ctypes.Structure):
    _fields_ = [
        ("ml_prob",           ctypes.c_float),
        ("causal_prob",       ctypes.c_float),
        ("fused_prob",        ctypes.c_float),
        ("fused_threat",      ctypes.c_int),
        ("simpsons_detected", ctypes.c_int),
        ("ml_latency_ns",     ctypes.c_float),
        ("causal_latency_ns", ctypes.c_float),
        ("total_latency_ns",  ctypes.c_float),
        ("ttd_ms",            ctypes.c_float),
    ]


@dataclass
class PredictionResult:
    ml_prob: float
    causal_prob: float
    fused_prob: float
    fused_threat: int  # 0=SAFE, 1=WATCH, 2=WARNING, 3=CRITICAL, 4=IMMINENT
    simpsons_detected: bool
    ml_latency_ns: float
    causal_latency_ns: float
    total_latency_ns: float
    ttd_ms: float

    @property
    def threat_name(self) -> str:
        names = {0: 'SAFE', 1: 'WATCH', 2: 'WARNING',
                 3: 'CRITICAL', 4: 'IMMINENT'}
        return names.get(self.fused_threat, 'UNKNOWN')

    @property
    def total_latency_us(self) -> float:
        return self.total_latency_ns / 1000.0


@dataclass
class BenchmarkResult:
    mean_ns: float
    p99_ns: float
    max_ns: float
    n_iterations: int

    @property
    def mean_us(self) -> float:
        return self.mean_ns / 1000.0

    @property
    def p99_us(self) -> float:
        return self.p99_ns / 1000.0


# ---------------------------------------------------------------------------
# Setup C function signatures
# ---------------------------------------------------------------------------

if _lib:
    _lib.fm_init.argtypes = [ctypes.c_int]
    _lib.fm_init.restype = ctypes.c_int

    _lib.fm_set_var_indices.argtypes = [ctypes.c_int] * 9
    _lib.fm_set_var_indices.restype = None

    _lib.fm_load_ml_model.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
    ]
    _lib.fm_load_ml_model.restype = ctypes.c_int

    _lib.fm_load_causal_model.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    _lib.fm_load_causal_model.restype = ctypes.c_int

    _lib.fm_set_weights.argtypes = [ctypes.c_float] * 4
    _lib.fm_set_weights.restype = None

    _lib.fm_predict.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_double,
        ctypes.POINTER(CDualPredictionResult),
    ]
    _lib.fm_predict.restype = ctypes.c_int

    _lib.fm_predict_ml_only.argtypes = [ctypes.POINTER(ctypes.c_float)]
    _lib.fm_predict_ml_only.restype = ctypes.c_float

    _lib.fm_predict_causal_only.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    _lib.fm_predict_causal_only.restype = ctypes.c_float

    _lib.fm_predict_batch.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    _lib.fm_predict_batch.restype = None

    _lib.fm_extract_features.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_float),
    ]
    _lib.fm_extract_features.restype = ctypes.c_int

    _lib.fm_reset.argtypes = []
    _lib.fm_reset.restype = None

    _lib.fm_benchmark.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _lib.fm_benchmark.restype = None

    _lib.fm_version.argtypes = []
    _lib.fm_version.restype = ctypes.c_char_p


# ---------------------------------------------------------------------------
# High-level Python API
# ---------------------------------------------------------------------------

class FastEngine:
    """High-performance real-time inference engine.

    Wraps the C++ engine via ctypes.  Falls back to pure Python
    if the shared library is not available.

    Typical latency:
      C++:    1–5 μs per dual prediction
      Python: 50–500 μs per dual prediction
    """

    def __init__(self, n_vars: int, var_names: Optional[List[str]] = None):
        self.n_vars = n_vars
        self.var_names = var_names or [f"v{i}" for i in range(n_vars)]
        self.use_cpp = CPP_AVAILABLE
        self._initialized = False

        if self.use_cpp:
            rc = _lib.fm_init(n_vars)
            if rc != 0:
                raise RuntimeError("Failed to initialize C++ engine")
            self._setup_var_indices()
            self._initialized = True
        else:
            # Fallback: import Python predictor
            from .predictor import (
                FastMLPredictor as _PyML,
                CausalDisruptionPredictor as _PyCausal,
            )
            self._py_ml = None
            self._py_causal = None
            self._initialized = True

    def _setup_var_indices(self):
        """Map variable names to indices for the C++ feature engine."""
        idx = {v: i for i, v in enumerate(self.var_names)}

        def _get(name, *alts):
            if name in idx:
                return idx[name]
            for a in alts:
                if a in idx:
                    return idx[a]
            return -1

        _lib.fm_set_var_indices(
            _get('betaN', 'βN'),
            _get('ne', 'ne_core'),
            _get('Ip'),
            _get('q95', 'q'),
            _get('P_rad'),
            _get('P_NBI'),
            _get('li'),
            _get('MHD_amp', 'MHD'),
            _get('ne_core', 'ne'),
        )

    # -- Model loading ------------------------------------------------------

    def load_ml_model_from_python(self, py_predictor):
        """Load ML model from a trained Python FastMLPredictor."""
        if not py_predictor._fitted:
            raise ValueError("Python ML predictor is not fitted")

        if self.use_cpp:
            trees = py_predictor._trees
            n_trees = len(trees)
            n_feat = len(py_predictor._feature_names)

            feat_arr = (ctypes.c_int * n_trees)(
                *[t['feature'] for t in trees])
            split_arr = (ctypes.c_float * n_trees)(
                *[t['split'] for t in trees])
            left_arr = (ctypes.c_float * n_trees)(
                *[t['value_left'] for t in trees])
            right_arr = (ctypes.c_float * n_trees)(
                *[t['value_right'] for t in trees])
            lr_arr = (ctypes.c_float * n_trees)(
                *[t['lr'] for t in trees])

            means = np.ascontiguousarray(
                py_predictor._means, dtype=np.float32)
            stds = np.ascontiguousarray(
                py_predictor._stds, dtype=np.float32)

            _lib.fm_load_ml_model(
                n_trees, n_feat,
                feat_arr, split_arr, left_arr, right_arr, lr_arr,
                means.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                stds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                py_predictor._threshold,
            )
        else:
            self._py_ml = py_predictor

    def load_causal_model_from_python(self, py_causal):
        """Load causal model from a trained Python CausalDisruptionPredictor."""
        if not py_causal._fitted:
            raise ValueError("Python causal predictor is not fitted")

        if self.use_cpp:
            n = py_causal.n_vars
            effects = np.zeros(n, dtype=np.float32)
            has_bound = np.zeros(n, dtype=np.int32)
            bound_low = np.zeros(n, dtype=np.float32)
            bound_high = np.zeros(n, dtype=np.float32)

            for j, vj in enumerate(py_causal.var_names):
                effects[j] = py_causal._disruption_coeffs.get(vj, 0.0)
                if vj in py_causal.causal_boundaries:
                    has_bound[j] = 1
                    bound_low[j], bound_high[j] = py_causal.causal_boundaries[vj]

            _lib.fm_load_causal_model(
                n,
                effects.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                has_bound.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                bound_low.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                bound_high.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        else:
            self._py_causal = py_causal

    def set_weights(self, w_ml: float = 0.35, w_causal: float = 0.65,
                    w_ml_simpson: float = 0.15,
                    w_causal_simpson: float = 0.85):
        if self.use_cpp:
            _lib.fm_set_weights(w_ml, w_causal, w_ml_simpson, w_causal_simpson)
        else:
            self._w_ml = w_ml
            self._w_causal = w_causal

    # -- Inference ----------------------------------------------------------

    def predict(self, values: np.ndarray,
                timestamp_s: float = 0.0) -> PredictionResult:
        """Run dual-mode prediction.

        Args:
            values: array of shape (n_vars,) — current plasma state
            timestamp_s: seconds since start of discharge

        Returns:
            PredictionResult with all metrics and latency
        """
        if self.use_cpp:
            vals = np.ascontiguousarray(values, dtype=np.float32)
            result = CDualPredictionResult()
            _lib.fm_predict(
                vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_double(timestamp_s),
                ctypes.byref(result),
            )
            return PredictionResult(
                ml_prob=result.ml_prob,
                causal_prob=result.causal_prob,
                fused_prob=result.fused_prob,
                fused_threat=result.fused_threat,
                simpsons_detected=bool(result.simpsons_detected),
                ml_latency_ns=result.ml_latency_ns,
                causal_latency_ns=result.causal_latency_ns,
                total_latency_ns=result.total_latency_ns,
                ttd_ms=result.ttd_ms,
            )
        else:
            return self._predict_python(values, timestamp_s)

    def predict_ml_only(self, features: np.ndarray) -> float:
        """Fast ML-only prediction (< 1 μs in C++)."""
        if self.use_cpp:
            f = np.ascontiguousarray(features, dtype=np.float32)
            return float(_lib.fm_predict_ml_only(
                f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ))
        elif self._py_ml is not None:
            feat_dict = {self._py_ml._feature_names[i]: float(features[i])
                         for i in range(len(features))
                         if i < len(self._py_ml._feature_names)}
            return self._py_ml.predict(feat_dict).disruption_probability
        return 0.5

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch ML prediction."""
        n = X.shape[0]
        if self.use_cpp:
            X_f = np.ascontiguousarray(X, dtype=np.float32)
            out = np.zeros(n, dtype=np.float32)
            _lib.fm_predict_batch(
                X_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            return out.astype(np.float64)
        elif self._py_ml is not None:
            return self._py_ml.predict_batch(X)
        return np.full(n, 0.5)

    def reset(self):
        """Reset internal state (history buffer)."""
        if self.use_cpp:
            _lib.fm_reset()

    # -- Benchmarking -------------------------------------------------------

    def benchmark(self, sample_values: np.ndarray,
                  n_iterations: int = 10000) -> BenchmarkResult:
        """Run latency benchmark."""
        if self.use_cpp:
            vals = np.ascontiguousarray(sample_values, dtype=np.float32)
            mean_ns = ctypes.c_double()
            p99_ns = ctypes.c_double()
            max_ns = ctypes.c_double()

            _lib.fm_benchmark(
                vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n_iterations,
                ctypes.byref(mean_ns),
                ctypes.byref(p99_ns),
                ctypes.byref(max_ns),
            )
            return BenchmarkResult(
                mean_ns=mean_ns.value,
                p99_ns=p99_ns.value,
                max_ns=max_ns.value,
                n_iterations=n_iterations,
            )
        else:
            # Python benchmark
            latencies = []
            for i in range(n_iterations):
                t0 = time.perf_counter()
                self.predict(sample_values, i * 0.001)
                latencies.append((time.perf_counter() - t0) * 1e9)
            lats = np.array(latencies)
            return BenchmarkResult(
                mean_ns=float(np.mean(lats)),
                p99_ns=float(np.percentile(lats, 99)),
                max_ns=float(np.max(lats)),
                n_iterations=n_iterations,
            )

    # -- Info ---------------------------------------------------------------

    @property
    def version(self) -> str:
        if self.use_cpp:
            return _lib.fm_version().decode()
        return "FusionMind-RT 4.5.0 (Python fallback)"

    @property
    def backend(self) -> str:
        return "C++" if self.use_cpp else "Python"

    # -- Private Python fallback --------------------------------------------

    def _predict_python(self, values: np.ndarray,
                        timestamp_s: float) -> PredictionResult:
        """Pure Python fallback prediction."""
        t0 = time.perf_counter()

        ml_prob = 0.5
        if self._py_ml is not None:
            feat_dict = {v: float(values[i])
                         for i, v in enumerate(self.var_names)
                         if i < len(values)}
            ml_result = self._py_ml.predict(feat_dict)
            ml_prob = ml_result.disruption_probability

        t_ml = time.perf_counter()

        causal_prob = 0.5
        simpsons = False
        if self._py_causal is not None:
            feat_dict = {v: float(values[i])
                         for i, v in enumerate(self.var_names)
                         if i < len(values)}
            causal_result = self._py_causal.predict(feat_dict)
            causal_prob = causal_result.disruption_probability
            simpsons = getattr(causal_result, '_simpsons', False)

        t_causal = time.perf_counter()

        w_ml = getattr(self, '_w_ml', 0.35)
        w_causal = getattr(self, '_w_causal', 0.65)
        fused = w_ml * ml_prob + w_causal * causal_prob

        if ml_prob > 0.95 or causal_prob > 0.95:
            fused = max(fused, 0.95)

        ttd = 800 * (1 - fused) if fused > 0.3 else 1e6
        threat = 4 if fused > 0.9 and ttd < 50 else \
                 3 if fused > 0.7 and ttd < 200 else \
                 2 if fused > 0.5 else \
                 1 if fused > 0.3 else 0

        return PredictionResult(
            ml_prob=ml_prob,
            causal_prob=causal_prob,
            fused_prob=fused,
            fused_threat=threat,
            simpsons_detected=simpsons,
            ml_latency_ns=(t_ml - t0) * 1e9,
            causal_latency_ns=(t_causal - t_ml) * 1e9,
            total_latency_ns=(t_causal - t0) * 1e9,
            ttd_ms=ttd,
        )
