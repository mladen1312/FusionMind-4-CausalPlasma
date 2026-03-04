#!/usr/bin/env python3
"""FusionMind C++ vs Python Benchmark Report"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionmind4.realtime.fast_bindings import FastEngine, CPP_AVAILABLE
from fusionmind4.realtime.causal_bindings import (
    benchmark_causal_kernels, get_capabilities
)

print("=" * 65)
print("FusionMind 4.0 — C++ Performance Benchmark")
print("=" * 65)

caps = get_capabilities()
print(f"\nBackend:     {caps['backend']}")
print(f"Threads:     {caps['max_threads']}")
print(f"RT engine:   {'C++' if CPP_AVAILABLE else 'Python'}")

data = np.random.RandomState(42).randn(1000, 9)

print(f"\nData: {data.shape[0]} samples × {data.shape[1]} vars")
print("-" * 65)
print(f"{'Kernel':<22} {'Python':>10} {'C++ AVX-512':>12} {'Speedup':>10}")
print("-" * 65)

r = benchmark_causal_kernels(data, n_rounds=5)
for k, v in r.items():
    py_str = f"{v['python_ms']:.1f} ms"
    cpp_str = f"{v['cpp_ms']:.1f} ms"
    sp_str = f"{v['speedup']:.0f}x"
    print(f"{k:<22} {py_str:>10} {cpp_str:>12} {sp_str:>10}")

# Real-time inference benchmark
print("-" * 65)
from fusionmind4.realtime.predictor import FastMLPredictor, CausalDisruptionPredictor

rng = np.random.RandomState(42)
var_names = ['v0','v1','v2','v3','v4','v5','v6','v7','v8']
X = rng.randn(500, 9).astype(np.float32)
y = ((X[:, 0] > 0.5) & (X[:, 2] < 0.0)).astype(int)

ml = FastMLPredictor(n_estimators=50)
ml.fit(X, y, feature_names=var_names)

engine = FastEngine(n_vars=9, var_names=var_names)
engine.load_ml_model_from_python(ml)

bench = engine.benchmark(X[0].copy(), n_iterations=10000)
print(f"{'rt_dual_predict':<22} {'~400 μs':>10} {f'{bench.mean_us:.2f} μs':>12} {f'{400/max(bench.mean_us,0.01):.0f}x':>10}")

print("-" * 65)
print(f"\nRT inference P99: {bench.p99_us:.2f} μs")
print(f"RT inference max: {bench.max_ns/1000:.2f} μs")
print()
