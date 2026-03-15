"""
Python bindings for FusionMind C++ 4-Layer Stack Engine.

Usage:
    from fusionmind4.realtime.stack_bindings import CppStack

    stack = CppStack(n_vars=10, phase=3)
    stack.load_scm(dag, scm)         # Load from Python SCM
    stack.load_safety_limits(limits)  # Physics boundaries
    
    result = stack.step(values, timestamp, actuators, act_map)
    print(f"Risk: {result.risk_score}, Latency: {result.latency_total_ns:.0f}ns")

Author: Dr. Mladen Mešter, dr.med., March 2026
"""

import ctypes
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ── Locate shared library ─────────────────────────────────

_LIB_NAME = "libfusionmind_stack.so"
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

CPP_STACK_AVAILABLE = _lib is not None

MAX_VARS = 16
MAX_ACTUATORS = 16


# ── C structures ──────────────────────────────────────────

class CStackResult(ctypes.Structure):
    _fields_ = [
        ("actuator_values", ctypes.c_float * MAX_ACTUATORS),
        ("n_actuators", ctypes.c_int),
        ("risk_score", ctypes.c_float),
        ("risk_level", ctypes.c_int),
        ("decision", ctypes.c_int),
        ("source_layer", ctypes.c_int),
        ("latency_L0_ns", ctypes.c_float),
        ("latency_L1_ns", ctypes.c_float),
        ("latency_L2_ns", ctypes.c_float),
        ("latency_L3_ns", ctypes.c_float),
        ("latency_total_ns", ctypes.c_float),
        ("vetoed", ctypes.c_int),
        ("cycle_count", ctypes.c_int),
    ]


# ── Python result wrapper ─────────────────────────────────

@dataclass
class StackStepResult:
    actuator_values: Dict[str, float]
    risk_score: float
    risk_level: str
    decision: str
    source_layer: int
    vetoed: bool
    cycle_count: int
    latency_L0_ns: float
    latency_L1_ns: float
    latency_L2_ns: float
    latency_L3_ns: float
    latency_total_ns: float

    RISK_NAMES = {0: "SAFE", 1: "WATCH", 2: "WARNING", 3: "CRITICAL", 4: "IMMINENT"}
    DECISION_NAMES = {0: "APPROVE", 1: "WARN", 2: "VETO"}


# ── Main class ─────────────────────────────────────────────

class CppStack:
    """
    Python wrapper for FusionMind C++ 4-Layer Stack Engine.
    
    All heavy computation runs in C++ at sub-microsecond latency.
    Python side handles setup, data marshalling, and pretty output.
    """

    def __init__(self, n_vars: int, phase: int = 1, var_names: List[str] = None):
        if not CPP_STACK_AVAILABLE:
            raise RuntimeError(
                "C++ stack engine not compiled. Run:\n"
                "  cd fusionmind4/realtime/cpp && "
                "g++ -O3 -march=native -shared -fPIC -std=c++17 "
                "-o libfusionmind_stack.so stack_api.cpp"
            )

        self.n_vars = n_vars
        self.phase = phase
        self.var_names = var_names or [f"var_{i}" for i in range(n_vars)]
        self.idx = {v: i for i, v in enumerate(self.var_names)}

        # Create C++ stack
        _lib.fm_stack_create.restype = ctypes.c_void_p
        self._ptr = _lib.fm_stack_create(n_vars, phase)

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr and _lib:
            _lib.fm_stack_destroy(ctypes.c_void_p(self._ptr))

    # ── Loading ──────────────────────────────────────────

    def load_scm(self, dag: np.ndarray, scm):
        """Load DAG + SCM equations from Python SCM object."""
        # Load DAG
        dag_int = (dag > 0).astype(np.int32).ravel()
        _lib.fm_stack_load_dag(
            ctypes.c_void_p(self._ptr),
            dag_int.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )

        # Load equations
        for j in range(self.n_vars):
            eq = {}
            if hasattr(scm, 'equations') and j in scm.equations:
                eq = scm.equations[j]
            elif hasattr(scm, 'linear_models') and j in scm.linear_models:
                eq = scm.linear_models[j]

            parents = eq.get('parents', eq.get('pa', []))
            coefs = eq.get('coefs', eq.get('coef', []))
            intercept = eq.get('intercept', 0.0)
            r2 = scm.r2_scores.get(j, 0.0) if hasattr(scm, 'r2_scores') else 0.0

            pa_arr = np.array(parents, dtype=np.int32) if parents else np.zeros(1, dtype=np.int32)
            co_arr = np.array(coefs, dtype=np.float32) if coefs else np.zeros(1, dtype=np.float32)

            _lib.fm_stack_load_equation(
                ctypes.c_void_p(self._ptr),
                ctypes.c_int(j),
                pa_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                co_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_float(float(intercept)),
                ctypes.c_int(len(parents)),
                ctypes.c_float(float(r2))
            )

    def load_safety_limits(self, limits_dict: Dict[str, Dict] = None):
        """
        Load safety limits per variable.
        limits_dict = {'q_95': {'min_crit': 2.0, 'min_warn': 2.5}, 
                       'betan': {'max_warn': 3.0, 'max_crit': 3.5}, ...}
        """
        if limits_dict is None:
            # Default plasma safety limits
            limits_dict = {
                'q_95':  {'min_crit': 2.0, 'min_warn': 2.5},
                'betan': {'max_warn': 3.0, 'max_crit': 3.5},
                'li':    {'max_warn': 1.5, 'max_crit': 2.0},
            }

        for var, lim in limits_dict.items():
            if var not in self.idx:
                continue
            vi = self.idx[var]
            _lib.fm_stack_set_safety_limit(
                ctypes.c_void_p(self._ptr),
                ctypes.c_int(vi),
                ctypes.c_float(lim.get('min_crit', 0)),
                ctypes.c_float(lim.get('min_warn', 0)),
                ctypes.c_float(lim.get('max_warn', 1e10)),
                ctypes.c_float(lim.get('max_crit', 1e10)),
                ctypes.c_int(1 if 'min_crit' in lim else 0),
                ctypes.c_int(1 if 'max_crit' in lim else 0),
            )

    def load_policy_weights(self, W1, b1, W2, b2, W3, b3):
        """Load pre-trained RL policy weights (Layer 1)."""
        for arr_name, arr in [('W1', W1), ('b1', b1), ('W2', W2), ('b2', b2), ('W3', W3), ('b3', b3)]:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        
        _lib.fm_stack_load_policy_weights(
            ctypes.c_void_p(self._ptr),
            W1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            W2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            W3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def set_phase(self, phase: int):
        """Switch phase (1/2/3) live."""
        self.phase = phase
        _lib.fm_stack_set_phase(ctypes.c_void_p(self._ptr), ctypes.c_int(phase))

    def set_setpoints(self, setpoints: Dict[str, float]):
        """Set target plasma parameters."""
        sp = np.zeros(self.n_vars, dtype=np.float32)
        for v, val in setpoints.items():
            if v in self.idx:
                sp[self.idx[v]] = val
        _lib.fm_stack_set_setpoints(
            ctypes.c_void_p(self._ptr),
            sp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

    # ── Main Step ──────────────────────────────────────────

    def step(self, values: Dict[str, float], timestamp: float,
             actuator_names: List[str],
             external_action: Dict[str, float] = None) -> StackStepResult:
        """
        Execute one control cycle through the full C++ stack.
        
        Returns StackStepResult with action, risk, explanation, latency.
        """
        # Marshal values
        raw = np.zeros(self.n_vars, dtype=np.float32)
        for v, val in values.items():
            if v in self.idx:
                raw[self.idx[v]] = val

        # Marshal actuator map
        n_act = len(actuator_names)
        act_map = np.array([self.idx.get(a, -1) for a in actuator_names], dtype=np.int32)

        # Marshal external action
        ext = None
        if external_action:
            ext = np.zeros(n_act, dtype=np.float32)
            for i, a in enumerate(actuator_names):
                ext[i] = external_action.get(a, values.get(a, 0))

        # Call C++
        result = CStackResult()
        _lib.fm_stack_step(
            ctypes.c_void_p(self._ptr),
            raw.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(timestamp),
            ext.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if ext is not None else None,
            ctypes.c_int(n_act),
            act_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(result),
        )

        # Unmarshal
        act_vals = {}
        for i, a in enumerate(actuator_names):
            act_vals[a] = result.actuator_values[i]

        return StackStepResult(
            actuator_values=act_vals,
            risk_score=result.risk_score,
            risk_level=StackStepResult.RISK_NAMES.get(result.risk_level, "UNKNOWN"),
            decision=StackStepResult.DECISION_NAMES.get(result.decision, "UNKNOWN"),
            source_layer=result.source_layer,
            vetoed=bool(result.vetoed),
            cycle_count=result.cycle_count,
            latency_L0_ns=result.latency_L0_ns,
            latency_L1_ns=result.latency_L1_ns,
            latency_L2_ns=result.latency_L2_ns,
            latency_L3_ns=result.latency_L3_ns,
            latency_total_ns=result.latency_total_ns,
        )

    # ── Queries ────────────────────────────────────────────

    def do_intervention(self, baseline: Dict[str, float],
                         intervention: Dict[str, float]) -> Dict[str, float]:
        """do-calculus in C++: P(Y | do(X=x))."""
        base = np.zeros(self.n_vars, dtype=np.float32)
        mask = np.zeros(self.n_vars, dtype=np.int32)
        vals = np.zeros(self.n_vars, dtype=np.float32)
        result = np.zeros(self.n_vars, dtype=np.float32)

        for v, val in baseline.items():
            if v in self.idx:
                base[self.idx[v]] = val
        for v, val in intervention.items():
            if v in self.idx:
                mask[self.idx[v]] = 1
                vals[self.idx[v]] = val

        _lib.fm_stack_do_intervention(
            ctypes.c_void_p(self._ptr),
            base.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        return {v: float(result[i]) for i, v in enumerate(self.var_names)}

    def counterfactual(self, factual: Dict[str, float],
                        hypothetical: Dict[str, float]) -> Dict[str, float]:
        """Counterfactual in C++."""
        fact = np.zeros(self.n_vars, dtype=np.float32)
        mask = np.zeros(self.n_vars, dtype=np.int32)
        vals = np.zeros(self.n_vars, dtype=np.float32)
        result = np.zeros(self.n_vars, dtype=np.float32)

        for v, val in factual.items():
            if v in self.idx: fact[self.idx[v]] = val
        for v, val in hypothetical.items():
            if v in self.idx:
                mask[self.idx[v]] = 1
                vals[self.idx[v]] = val

        _lib.fm_stack_counterfactual(
            ctypes.c_void_p(self._ptr),
            fact.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            vals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        return {v: float(result[i]) for i, v in enumerate(self.var_names)}

    # ── Stats ──────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            'phase': _lib.fm_stack_get_phase(ctypes.c_void_p(self._ptr)),
            'cycles': _lib.fm_stack_get_cycle_count(ctypes.c_void_p(self._ptr)),
            'approved': _lib.fm_stack_get_n_approved(ctypes.c_void_p(self._ptr)),
            'warned': _lib.fm_stack_get_n_warned(ctypes.c_void_p(self._ptr)),
            'vetoed': _lib.fm_stack_get_n_vetoed(ctypes.c_void_p(self._ptr)),
        }

    # ── Benchmark ──────────────────────────────────────────

    def benchmark(self, n_cycles: int = 10000,
                   actuator_names: List[str] = None) -> Dict:
        """Run latency benchmark."""
        import time

        if actuator_names is None:
            actuator_names = self.var_names[:3]

        values = {v: 1.0 + 0.1 * i for i, v in enumerate(self.var_names)}
        ext = {a: values.get(a, 1.0) * 1.01 for a in actuator_names}

        # Warmup
        for _ in range(100):
            self.step(values, 0.0, actuator_names, ext)

        # Benchmark
        latencies = []
        t_start = time.perf_counter_ns()
        for i in range(n_cycles):
            r = self.step(values, float(i) * 0.005, actuator_names, ext)
            latencies.append(r.latency_total_ns)
        t_total = time.perf_counter_ns() - t_start

        lat = np.array(latencies)
        return {
            'n_cycles': n_cycles,
            'wall_time_ms': t_total / 1e6,
            'mean_total_ns': float(np.mean(lat)),
            'p50_ns': float(np.median(lat)),
            'p95_ns': float(np.percentile(lat, 95)),
            'p99_ns': float(np.percentile(lat, 99)),
            'max_ns': float(np.max(lat)),
            'throughput_Mops': n_cycles / (t_total / 1e9) / 1e6,
        }
