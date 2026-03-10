# FusionMind 4.0 — Algorithm Architecture

## Codebase: 28K lines, 8 patent families, 310 tests passing

## Verified Results (reproducible, seed=42)

| Algorithm | Source | AUC (MAST) | AUC (C-Mod) | Features |
|-----------|--------|------------|-------------|----------|
| **GBT 63f + margins + aug** | engine.py Track B | **0.979 ± 0.011** | — | 63 physics |
| Track D causal (li deep) | engine.py Track D | 0.950 | — | 12 on primary driver |
| Physics: max(li) | manual | 0.908 | — | 0 params |
| Physics: f_GW formula | manual | — | 0.978 | 0 params |
| v1.1 GRU timepoint | temporal_gru.py | 0.842 | 0.842 | 78 temporal |
| CPDE DAG F1 | ensemble.py | 88.9% | — | — |
| Nonlinear SCM R² | nonlinear_scm.py | 96.7% | — | — |

## Module Architecture

```
fusionmind4/
├── discovery/          PF1: CPDE — 5-algorithm ensemble causal discovery
│   ├── notears.py      NOTEARS + DYNOTEARS (augmented Lagrangian DAG)
│   ├── pc.py           PC algorithm + Meek R1-R4 + stable variant
│   ├── granger.py      Granger + spectral + conditional
│   ├── ensemble.py     EnsembleCPDE: weighted fusion + bootstrap
│   ├── physics.py      Physics priors (actuator exogeneity)
│   └── nonlinear_scm.py GBT structural equations
│
├── control/            PF2: CPC — Counterfactual Controller
│   ├── scm.py          Pearl's SCM + do-calculus
│   ├── causal_controller.py  3-mode CausalRL
│   ├── stack.py        4-layer unified control stack
│   ├── dynamic_overseer.py   Mimosa multi-track arbitrator
│   └── temporal_gru.py GRU sequence predictor
│
├── predictor/          NEW: Unified 6-track predictor
│   └── engine.py       CausalDisruptionPredictor
│
├── foundation/         PF3: UPFM — Universal Plasma Foundation Model
├── reconstruction/     PF4: D3R — Diffusion 3D Reconstruction
├── experiment/         PF5: AEDE — Active Experiment Design
├── learning/           PF7: Neural SCM + Causal RL + Gym
├── realtime/           PF6/PF8: C++ engine + control bridge
└── utils/              Simulators + plasma variable definitions
```

## CausalDisruptionPredictor: 6 Parallel Tracks

```
  Track A: PHYSICS MARGINS      9f   always on, 0 parameters
           distance-to-limit for each stability mechanism
           
  Track B: SHOT-LEVEL STATS    63f   ≥3 signals, GBT
           mean/std/max + margins + interactions + temporal shape
           
  Track C: TRAJECTORY          32f   ≥3 signals, GBT
           shot split into thirds, compare end vs start
           
  Track D: CAUSAL DRIVER       12f   auto-detected, GBT
           deep analysis of primary mechanism (li or f_GW)
           
  Track E: RATE EXTREMES       40f   ≥3 signals, GBT
           max rate, late volatility — precursor detection
           
  Track F: PAIRWISE            15f   ≥4 signals, GBT
           li×βN, li/q95, Hugill space — stability boundaries
           
  META: LogReg on out-of-fold track predictions
```

## What's Unique vs Competitors

| Feature | FusionMind | CCNN | FRNN | Random Forest |
|---------|-----------|------|------|---------------|
| Causal discovery (DAG) | ✓ CPDE | ✗ | ✗ | ✗ |
| do-calculus interventions | ✓ SCM | ✗ | ✗ | ✗ |
| Counterfactual "what if" | ✓ PF2 | ✗ | sensitivity only | ✗ |
| Physics stability margins | ✓ Track A | ✗ | ✗ | ✗ |
| Auto device adaptation | ✓ SIGNAL_ALIASES | ✗ | manual | manual |
| Mechanism identification | ✓ Track D | ✗ | ✗ | ✗ |
| Cross-device transfer | ✓ UPFM design | ✗ | few-shot | ✗ |
| Interpretable "WHY" | ✓ causal path | ✗ | partial | feature importance |
