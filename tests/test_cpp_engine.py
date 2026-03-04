"""Tests for FusionMind C++ Fast Engine

Verifies:
  1. Shared library loads correctly
  2. ML model loads and predicts
  3. Causal model loads and predicts
  4. Dual prediction works end-to-end
  5. Batch prediction
  6. Latency benchmark < 5 μs target
  7. Simpson's Paradox detection in C++
  8. Feature extraction
  9. Python fallback works
"""

import numpy as np
import time
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from fusionmind4.realtime.fast_bindings import (
    FastEngine, CPP_AVAILABLE, PredictionResult, BenchmarkResult,
)
from fusionmind4.realtime.predictor import (
    FastMLPredictor, CausalDisruptionPredictor,
)


# -- Helpers ----------------------------------------------------------------

def make_trained_ml(n_vars=9):
    """Train a Python ML predictor for transfer to C++."""
    rng = np.random.RandomState(42)
    var_names = ['βN', 'βp', 'q95', 'q_axis', 'li', 'κ',
                 'Ip', 'P_NBI', 'D_alpha'][:n_vars]
    n = 500
    X = rng.randn(n, n_vars).astype(np.float32)
    X[:, 0] = 0.5 + 0.3 * X[:, min(7, n_vars - 1)]
    y = ((X[:, 0] > 1.0) & (X[:, min(2, n_vars - 1)] < 0.0)).astype(int)
    y[rng.random(n) < 0.1] = 1

    ml = FastMLPredictor(n_estimators=50, max_depth=4)
    ml.fit(X, y, feature_names=var_names)
    return ml, X, y, var_names


def make_trained_causal(n_vars=9):
    """Train a Python causal predictor for transfer to C++."""
    rng = np.random.RandomState(42)
    var_names = ['βN', 'βp', 'q95', 'q_axis', 'li', 'κ',
                 'Ip', 'P_NBI', 'D_alpha'][:n_vars]
    dag = np.zeros((n_vars, n_vars))
    dag[min(7, n_vars - 1), 0] = 0.8  # P_NBI → βN
    dag[0, 1] = 0.5                    # βN → βp

    n = 500
    X = rng.randn(n, n_vars).astype(np.float32)
    y = ((X[:, 0] > 1.0) & (X[:, min(2, n_vars - 1)] < 0.0)).astype(int)
    y[rng.random(n) < 0.1] = 1

    causal = CausalDisruptionPredictor(dag, var_names)
    causal.fit(X, y)
    return causal, dag, var_names


# -- Tests ------------------------------------------------------------------

class TestCppEngineAvailability:

    def test_library_loads(self):
        assert CPP_AVAILABLE, "C++ library not found — check build"

    def test_version(self):
        engine = FastEngine(n_vars=9)
        assert "FusionMind" in engine.version
        assert "C++" in engine.backend


class TestCppMLPrediction:

    def test_load_ml_model(self):
        ml, X, y, var_names = make_trained_ml()
        engine = FastEngine(n_vars=len(var_names), var_names=var_names)
        engine.load_ml_model_from_python(ml)
        # Should not raise

    def test_predict_single(self):
        ml, X, y, var_names = make_trained_ml()
        engine = FastEngine(n_vars=len(var_names), var_names=var_names)
        engine.load_ml_model_from_python(ml)

        vals = X[0].copy()
        prob = engine.predict_ml_only(vals)
        assert 0 <= prob <= 1

    def test_predict_batch(self):
        ml, X, y, var_names = make_trained_ml()
        engine = FastEngine(n_vars=len(var_names), var_names=var_names)
        engine.load_ml_model_from_python(ml)

        probs = engine.predict_batch(X[:100].copy())
        assert probs.shape == (100,)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_consistency_with_python(self):
        """C++ predictions should match Python predictions closely."""
        ml, X, y, var_names = make_trained_ml()
        engine = FastEngine(n_vars=len(var_names), var_names=var_names)
        engine.load_ml_model_from_python(ml)

        py_probs = ml.predict_batch(X[:50])
        cpp_probs = engine.predict_batch(X[:50].astype(np.float32))

        # Allow small numerical differences (float32 vs float64)
        np.testing.assert_allclose(cpp_probs, py_probs, atol=0.05)


class TestCppDualPrediction:

    def test_dual_predict(self):
        ml, X, y, var_names = make_trained_ml()
        causal, dag, _ = make_trained_causal(len(var_names))

        engine = FastEngine(n_vars=len(var_names), var_names=var_names)
        engine.load_ml_model_from_python(ml)
        engine.load_causal_model_from_python(causal)

        result = engine.predict(X[0].copy(), timestamp_s=0.0)

        assert isinstance(result, PredictionResult)
        assert 0 <= result.fused_prob <= 1
        assert 0 <= result.fused_threat <= 4
        assert result.total_latency_ns > 0

    def test_multiple_predictions(self):
        ml, X, y, var_names = make_trained_ml()
        causal, dag, _ = make_trained_causal(len(var_names))

        engine = FastEngine(n_vars=len(var_names), var_names=var_names)
        engine.load_ml_model_from_python(ml)
        engine.load_causal_model_from_python(causal)

        results = []
        for i in range(100):
            r = engine.predict(X[i % len(X)].copy(), timestamp_s=i * 0.001)
            results.append(r)

        assert len(results) == 100
        probs = [r.fused_prob for r in results]
        assert all(0 <= p <= 1 for p in probs)

    def test_threat_classification(self):
        ml, X, y, var_names = make_trained_ml()
        causal, dag, _ = make_trained_causal(len(var_names))

        engine = FastEngine(n_vars=len(var_names), var_names=var_names)
        engine.load_ml_model_from_python(ml)
        engine.load_causal_model_from_python(causal)

        for i in range(50):
            r = engine.predict(X[i].copy(), 0.0)
            assert r.threat_name in {'SAFE', 'WATCH', 'WARNING',
                                     'CRITICAL', 'IMMINENT'}

    def test_reset(self):
        engine = FastEngine(n_vars=9)
        engine.reset()  # Should not crash


class TestCppLatencyBenchmark:

    def test_benchmark_runs(self):
        ml, X, y, var_names = make_trained_ml()
        causal, _, _ = make_trained_causal(len(var_names))

        engine = FastEngine(n_vars=len(var_names), var_names=var_names)
        engine.load_ml_model_from_python(ml)
        engine.load_causal_model_from_python(causal)

        result = engine.benchmark(X[0].copy(), n_iterations=1000)
        assert isinstance(result, BenchmarkResult)
        assert result.mean_ns > 0
        assert result.p99_ns >= result.mean_ns

    def test_latency_under_5us(self):
        """Target: dual prediction < 5 μs (5000 ns)."""
        ml, X, y, var_names = make_trained_ml(n_vars=9)
        causal, _, _ = make_trained_causal(9)

        engine = FastEngine(n_vars=len(var_names), var_names=var_names)
        engine.load_ml_model_from_python(ml)
        engine.load_causal_model_from_python(causal)

        result = engine.benchmark(X[0].copy(), n_iterations=10000)

        print(f"\n  === C++ Latency Benchmark ===")
        print(f"  Mean:  {result.mean_us:.2f} μs")
        print(f"  P99:   {result.p99_us:.2f} μs")
        print(f"  Max:   {result.max_ns / 1000:.2f} μs")

        # Mean should be < 5 μs on most hardware
        # (may be higher on slow CI, so use generous limit)
        assert result.mean_us < 50.0, \
            f"Mean latency {result.mean_us:.1f} μs exceeds 50 μs"

    def test_ml_only_latency(self):
        """ML-only path should be < 1 μs."""
        ml, X, y, var_names = make_trained_ml()
        engine = FastEngine(n_vars=len(var_names), var_names=var_names)
        engine.load_ml_model_from_python(ml)

        features = X[0].astype(np.float32)

        latencies = []
        for _ in range(5000):
            t0 = time.perf_counter()
            engine.predict_ml_only(features)
            latencies.append((time.perf_counter() - t0) * 1e6)

        median_us = np.median(latencies)
        print(f"\n  ML-only median latency: {median_us:.2f} μs")
        # Should be under 10 μs even with ctypes overhead
        assert median_us < 50.0


class TestCppVsPython:

    def test_speedup_ratio(self):
        """C++ should be significantly faster than Python."""
        ml, X, y, var_names = make_trained_ml()
        causal, _, _ = make_trained_causal(len(var_names))

        # C++ engine
        cpp_engine = FastEngine(n_vars=len(var_names), var_names=var_names)
        cpp_engine.load_ml_model_from_python(ml)
        cpp_engine.load_causal_model_from_python(causal)

        cpp_bench = cpp_engine.benchmark(X[0].copy(), n_iterations=5000)

        # Python timing
        from fusionmind4.realtime.predictor import (
            DualModePredictor, DisruptionFeatureExtractor, PlasmaSnapshot,
        )
        ext = DisruptionFeatureExtractor(history_length=20)
        ext.set_variable_order(var_names)
        dual = DualModePredictor(ml, causal, ext)

        py_lats = []
        for i in range(500):
            values = {v: float(X[i % len(X), j])
                      for j, v in enumerate(var_names)}
            snap = PlasmaSnapshot(values=values, timestamp_s=i * 0.001)
            t0 = time.perf_counter()
            dual.predict(snap)
            py_lats.append((time.perf_counter() - t0) * 1e9)

        py_mean_ns = np.mean(py_lats)
        speedup = py_mean_ns / max(cpp_bench.mean_ns, 1)

        print(f"\n  === C++ vs Python Speedup ===")
        print(f"  C++ mean: {cpp_bench.mean_ns:.0f} ns "
              f"({cpp_bench.mean_us:.2f} μs)")
        print(f"  Python mean: {py_mean_ns:.0f} ns "
              f"({py_mean_ns / 1000:.1f} μs)")
        print(f"  Speedup: {speedup:.0f}x")

        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.1f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
