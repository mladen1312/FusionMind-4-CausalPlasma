# FusionMind 4.0 — System Architecture

## Overview

FusionMind is a 4-layer causal AI control stack for tokamak plasma. Each layer
has a distinct role, runs at a specific timescale, and can be deployed independently.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: Causal Safety Monitor              [C++ + Python] │
│  Latency: ~100ns | Role: Veto/approve/explain every action  │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2: Causal Strategic Controller        [C++ + Python] │
│  Latency: ~700ns | Role: do-calculus planning, setpoints    │
├─────────────────────────────────────────────────────────────┤
│  LAYER 1: Tactical RL Controller             [C++]          │
│  Latency: ~200ns | Role: MLP policy, actuator commands      │
├─────────────────────────────────────────────────────────────┤
│  LAYER 0: Real-Time Engine                   [C++]          │
│  Latency: ~100ns | Role: Features, risk, rate limiting      │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Plasma Diagnostics (Thomson, ECE, Magnetics)
    │
    ▼  (5ms diagnostic cycle)
Layer 0: Extract features, compute rates dx/dt, fast risk score
    │
    ▼  (~100ns)
Layer 1: MLP forward pass → raw actuator commands
    │  OR external RL action arrives here
    ▼  (~200ns)
Layer 2: Test action via do-calculus against SCM
    │  Predict outcome, compare to setpoints
    ▼  (~700ns)
Layer 3: Causal safety evaluation
    │  Risk assessment → APPROVE / WARN / VETO
    ▼  (~100ns)
Actuators (coils, heating, gas puff)
    Total: < 2μs from measurement to command
```

## Module Map

```
fusionmind4/
├── control/                    # CONTROL LAYERS
│   ├── stack.py                #   FusionMindStack — unified 4-layer Python interface
│   ├── causal_controller.py    #   FusionMindController — 3-mode wrapper (legacy)
│   ├── scm.py                  #   PlasmaSCM — linear structural causal model
│   └── interventions.py        #   InterventionEngine — do/counterfactual
│
├── discovery/                  # CAUSAL DISCOVERY (offline)
│   ├── ensemble.py             #   EnsembleCPDE — main discovery pipeline
│   ├── notears.py              #   NOTEARS with h(W)=tr(e^{W∘W})-d
│   ├── granger.py              #   Granger causality (conditional, spectral)
│   ├── temporal.py             #   Temporal Granger with selective conditioning
│   ├── pc.py                   #   PC algorithm (stable + Meek rules R1-R4)
│   ├── physics.py              #   Physics priors + PINN validation
│   ├── nonlinear_scm.py        #   GradientBoosting SCM (96.7% βN R²)
│   └── interventional.py       #   Interventional scoring
│
├── realtime/                   # C++ ENGINE + BINDINGS
│   ├── cpp/
│   │   ├── stack_engine.hpp    #   ★ Complete 4-layer C++ engine
│   │   ├── stack_api.cpp       #   C API for ctypes
│   │   ├── fast_engine.hpp     #   ML prediction engine (GBM stumps)
│   │   ├── fast_engine_api.cpp #   C API for ML engine
│   │   └── causal_kernels.cpp  #   NOTEARS/Granger/bootstrap in C++
│   ├── stack_bindings.py       #   CppStack — Python ↔ C++ bridge
│   ├── fast_bindings.py        #   FastEngine bindings
│   ├── causal_bindings.py      #   Causal kernel bindings
│   ├── predictor.py            #   DualPredictor (ML + causal)
│   ├── streaming.py            #   Real-time data interface
│   └── control_bridge.py       #   Actuator command bridge
│
├── learning/                   # REINFORCEMENT LEARNING
│   ├── neural_scm.py           #   Neural network SCM equations
│   ├── causal_rl_hybrid.py     #   CausalShieldRL (PPO + causal reward)
│   ├── gym_plasma_env.py       #   OpenAI Gym environment
│   └── causal_reward.py        #   Causal reward shaping
│
├── experiment/                 # ACTIVE EXPERIMENT DESIGN (PF5)
├── foundation/                 # UNIVERSAL PLASMA FOUNDATION MODEL (PF3)
├── reconstruction/             # DIFFUSION 3D RECONSTRUCTION (PF4)
├── copilot/                    # LLM CAUSAL QUERY INTERFACE (PF8)
│   ├── causal_context.py       #   DAG context for LLM
│   └── query_engine.py         #   Query classification (HR/EN)
├── mlx_backend/                # APPLE SILICON ACCELERATION
└── utils/                      # Shared utilities
```

## C++ Engine Architecture

The C++ engine (`stack_engine.hpp`) is a single-header library with zero heap
allocation in the hot path. All data structures use fixed-size arrays with
cache-line alignment (`alignas(64)`).

### Memory Layout

```
FusionMindStack_CPP (total ~300KB):
├── Layer0_Engine
│   ├── values[16]           float32 × 16 = 64B
│   ├── prev_values[16]      float32 × 16 = 64B
│   ├── rates[16]            float32 × 16 = 64B
│   └── rate_history[16][64] float32 × 1024 = 4KB
├── Layer1_Policy
│   ├── W1[32][128]          float32 × 4096 = 16KB
│   ├── W2[128][128]         float32 × 16384 = 64KB
│   └── W3[128][16]          float32 × 2048 = 8KB
├── Layer2_SCM
│   ├── equations[16]        SCMEquation × 16 = 1KB
│   ├── topo_order[16]       int32 × 16 = 64B
│   └── dag[16][16]          int32 × 256 = 1KB
└── Layer3_Safety
    ├── limits[16]           VarSafetyLimit × 16 = 384B
    └── counters             3 × int32 = 12B
```

### Timing

Measured via `rdtsc` instruction on x86-64 (3GHz reference):

| Layer | Operation | Cycles | Time |
|-------|-----------|--------|------|
| L0 | Feature extract + rates | ~300 | 100ns |
| L1 | MLP forward (20→64→64→10) | ~600 | 200ns |
| L2 | do-intervention (10 vars) | ~900 | 300ns |
| L2 | batch_do (11 candidates) | ~10K | 3.3μs |
| L3 | Risk assess + decision | ~300 | 100ns |
| **Total** | **Phase 1 (L0+L3)** | **~600** | **83ns** |
| **Total** | **Phase 3 (all layers)** | **~2100** | **705ns** |

### Compilation

```bash
cd fusionmind4/realtime/cpp

# Stack engine (primary)
g++ -O3 -march=native -shared -fPIC -std=c++17 -o libfusionmind_stack.so stack_api.cpp

# ML prediction engine
g++ -O3 -march=native -shared -fPIC -std=c++17 -o libfusionmind_rt.so fast_engine_api.cpp

# Causal discovery kernels
g++ -O3 -march=native -shared -fPIC -std=c++17 -o libfusionmind_causal.so causal_kernels.cpp
```

## Deployment Phases

| Phase | Layers Active | Use Case | Risk to Customer |
|-------|---------------|----------|------------------|
| 1 | L0 + L3 | Safety wrapper over existing RL | Zero — read-only + veto |
| 2 | L0 + L2 + L3 | Causal strategic + external tactical | Low — strategic only |
| 3 | L0 + L1 + L2 + L3 | Full autonomous control | Medium — full control |

Phase upgrade is live via `stack.set_phase(N)` or `stack.upgrade_phase(Phase.PHASE_N)`.
No restart, no reconfiguration. DAG and SCM state are preserved.

## Patent Coverage

| Layer | Patent Family | Description |
|-------|---------------|-------------|
| L0 | PF6 | Integrated real-time causal control system |
| L1 | PF7 | CausalShield-RL — causal reward shaping |
| L2 | PF1 + PF2 | CPDE (causal discovery) + CPC (counterfactual controller) |
| L3 | PF2 + PF6 | Counterfactual safety + integrated system |
| Cross-device | PF3 | UPFM — dimensionless tokenization for transfer |
| Experiment | PF5 | AEDE — active experiment design |
