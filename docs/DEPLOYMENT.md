# Deployment Guide

## Prerequisites

- Python 3.10+
- g++ with C++17 support (for C++ engine)
- NumPy, SciPy, scikit-learn

## Installation

```bash
git clone https://github.com/mladen1312/FusionMind-4-CausalPlasma.git
cd FusionMind-4-CausalPlasma

# Core install
pip install -e .

# Full install (with ML dependencies)
pip install -e ".[full]"

# Development (with pytest)
pip install -e ".[full,dev]"

# Real data validation (S3 access to FAIR-MAST)
pip install -e ".[full,real-data]"
```

## C++ Compilation

Required for production latency (sub-microsecond):

```bash
cd fusionmind4/realtime/cpp

# Stack engine (primary — 4-layer controller)
g++ -O3 -march=native -shared -fPIC -std=c++17 -o libfusionmind_stack.so stack_api.cpp

# ML prediction engine (GBM stump ensemble)
g++ -O3 -march=native -shared -fPIC -std=c++17 -o libfusionmind_rt.so fast_engine_api.cpp

# Causal discovery kernels (NOTEARS, Granger, bootstrap)
g++ -O3 -march=native -shared -fPIC -std=c++17 -o libfusionmind_causal.so causal_kernels.cpp

# Verify
python -c "from fusionmind4.realtime.stack_bindings import CPP_STACK_AVAILABLE; print(f'C++ Stack: {CPP_STACK_AVAILABLE}')"
```

## Verification

```bash
# Full test suite
FM_SKIP_S3=1 pytest tests/ -v

# Quick smoke test
python examples/quickstart.py
```

## Deployment by Phase

### Phase 1: Wrapper (Minimal Risk)

Deploy FusionMind alongside your existing PCS (Plasma Control System).
FusionMind reads sensor data and RL actions, provides causal explanations
and safety vetoes. No modifications to existing control loop required.

```
┌──────────────────────────────────────┐
│  Existing PCS                        │
│  ┌──────────┐    ┌────────────────┐  │
│  │ Sensors  │───→│ Existing RL    │──┼──→ Actuators
│  └──────────┘    └────────────────┘  │
│       │                │             │
│       │    ┌───────────▼──────────┐  │
│       └───→│ FusionMind (L0+L3)  │  │ ← READ-ONLY + VETO
│            │ - Causal explanation │  │
│            │ - Risk assessment    │  │
│            │ - Disruption alert   │  │
│            └─────────────────────┘  │
└──────────────────────────────────────┘
```

Integration: FusionMind receives a copy of sensor data and RL output.
If risk > threshold, FusionMind sends VETO + safe alternative to PCS.

### Phase 2: Hybrid (Strategic Control)

FusionMind takes over strategic decisions (setpoints, profile targets).
Existing RL handles fine-grained actuator commands within FusionMind's
setpoint boundaries.

### Phase 3: Full Stack (Autonomous)

FusionMind controls everything: strategy (L2), tactics (L1), safety (L3).
External RL is decommissioned.

## Configuration for Specific Tokamaks

### MAST / MAST-U (Spherical Tokamak)

```python
var_names = ['betan', 'betap', 'q_95', 'q_axis', 'elongation',
             'li', 'wplasmd', 'betat', 'Ip', 'Prad']

limits = SafetyLimits(
    q95_min=1.5,        # Spherical tokamaks operate at lower q
    q95_warning=2.0,
    betan_max=6.0,      # Higher beta limit for STs
    betan_warning=5.0,
    li_max=2.5,
)
```

### ITER / Conventional Tokamak

```python
var_names = ['betan', 'betap', 'q_95', 'q_axis', 'elongation',
             'li', 'wmhd', 'betat', 'Ip', 'Prad',
             'ne_bar', 'Te_core', 'P_NBI', 'P_ECRH']

limits = SafetyLimits(
    q95_min=2.0,
    q95_warning=3.0,
    betan_max=2.5,      # Conservative for ITER
    betan_warning=2.0,
    li_max=1.2,
)
```

## Monitoring

```python
stats = stack.get_stats()
print(f"Phase: {stats['phase']}")
print(f"Cycles: {stats['cycles']}")
print(f"Veto rate: {stats['safety']['veto_rate']:.1%}")
print(f"Avg risk: {stats['safety']['avg_risk']:.3f}")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CPP_STACK_AVAILABLE = False` | Recompile: `g++ -O3 -march=native -shared -fPIC -std=c++17 -o libfusionmind_stack.so stack_api.cpp` |
| `ImportError: libfusionmind_causal.so` | Run compilation commands above |
| MLX tests skipped | Expected on non-Apple hardware |
| S3 download fails | Set `FM_SKIP_S3=1` for CI or check network access to `s3.echo.stfc.ac.uk` |
| High veto rate | Check safety limits — may be too conservative for your tokamak |
| `set_targets` raises RuntimeError | Only available in Phase 2+ — call `upgrade_phase(Phase.PHASE_2)` first |
