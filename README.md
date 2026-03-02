# FusionMind 4.0 — Causal AI for Fusion Plasma Control

> **The first application of Pearl's causal inference framework to tokamak plasma physics.**

[![Tests](https://img.shields.io/badge/tests-56%20passed-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![License](https://img.shields.io/badge/license-Proprietary-red)](LICENSE)
[![Patent Families](https://img.shields.io/badge/patents-7%20families-orange)](#patent-portfolio)
[![Real Data](https://img.shields.io/badge/validated-MAST%20%7C%20Alcator%20C--Mod-purple)](#real-data-validation)

---

## Why This Matters

Every existing fusion AI system — DeepMind/CFS (TORAX), KSTAR RL, Princeton Diag2Diag, TokaMind — operates at **Pearl's Ladder Level 1** (correlation). They learn statistical patterns (`P(Y|X)`) but cannot answer:

- *"What happens to Te if we **set** P_ECRH to 10 MW?"* → **Intervention** `P(Y|do(X))`
- *"What **would have** happened if we had increased current instead?"* → **Counterfactual**

**FusionMind 4.0 operates at Levels 2–3**, enabling true causal reasoning for plasma control. This is validated on **real tokamak data** from MAST and Alcator C-Mod.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FusionMind 4.0 System                        │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   CPDE   │  │   CPC    │  │   UPFM   │  │   D3R    │       │
│  │  (PF1)   │→ │  (PF2)   │  │  (PF3)   │  │  (PF4)   │       │
│  │ Causal   │  │ Counter- │  │ Foundation│  │ Diffusion│       │
│  │ Discovery│  │ factual  │  │  Model   │  │  Recon   │       │
│  └────┬─────┘  └────┬─────┘  └──────────┘  └──────────┘       │
│       │              │                                          │
│  ┌────▼──────────────▼─────────────────────────────────┐       │
│  │         CausalShield-RL (PF7) — Hybrid Agent        │       │
│  │  ┌───────────┐  ┌───────────┐  ┌─────────────────┐ │       │
│  │  │ NeuralSCM │  │  Causal   │  │   PPO Policy    │ │       │
│  │  │ World     │  │  Reward   │  │   with Causal   │ │       │
│  │  │ Model     │  │  Shaping  │  │   Constraints   │ │       │
│  │  └───────────┘  └───────────┘  └─────────────────┘ │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  ┌──────────┐  ┌──────────┐                                    │
│  │   AEDE   │  │  FM3Lite │                                    │
│  │  (PF5)   │  │ Physics  │                                    │
│  │ Experiment│  │ Engine   │                                    │
│  │  Design  │  │          │                                    │
│  └──────────┘  └──────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Results

### Synthetic Data (FM3-Lite Physics Engine)

| Metric | Value | Details |
|--------|-------|---------|
| CPDE F1 Score | **79.2%** | 21/25 true positives, 14 variables |
| CPDE Precision | 84.0% | Ensemble: NOTEARS + Granger + PC |
| PINN Physics Checks | **8/8** | Energy, momentum, Maxwell, MHD |
| OOD Noise Robustness | 93% @ 5% | Out-of-distribution validation |
| CPC Interventions | **All correct** | do-calculus validated |
| UPFM Cross-Device CV | 0.267 | 6 tokamaks (ITER, JET, DIII-D, EAST, AUG, KSTAR) |
| D3R Compression | 156:1 | Conditional denoising diffusion |

### Real Tokamak Data

| Dataset | Metric | Value |
|---------|--------|-------|
| **MAST** (UKAEA) | CPDE F1 | **91.9%** (8 shots, 625 timepoints) |
| **MAST** | Precision / Recall | 89.5% / 94.4% |
| **MAST** | Sign Accuracy | 100% (all edges physically correct) |
| **MAST** | Cross-Shot Robustness | F1 = 88.2% ± 4.4% |
| **Alcator C-Mod** (MIT) | Density Limit AUC | **0.974** (264K timepoints, 2333 shots) |
| **Alcator C-Mod** | Simpson's Paradox | Detected ✓ (ρ: +0.53 → +0.02) |
| **Alcator C-Mod** | vs Greenwald Fraction | AUC 0.974 vs 0.946 |

---

## Installation

### Requirements

- Python ≥ 3.10
- NumPy ≥ 1.24, SciPy ≥ 1.11, NetworkX ≥ 3.0

### Quick Install

```bash
git clone https://github.com/mladen1312/FusionMind-4-CausalPlasma.git
cd FusionMind-4-CausalPlasma
pip install -e ".[full,dev]"
```

### Minimal Install (core only)

```bash
pip install -e .
```

### Verify Installation

```bash
pytest tests/ -v
# Expected: 56 passed
```

---

## Quick Start

### Run Full Validation Pipeline

```bash
python scripts/run_full_validation.py      # PF1-PF5 on synthetic data
python scripts/run_fair_mast.py            # PF1 on real MAST data
python scripts/run_real_data.py            # PF1+PF3 on Alcator C-Mod
python scripts/train_causal_rl.py          # PF7 CausalShield-RL training
```

### Minimal Code Example

```python
from fusionmind4.discovery import EnsembleCPDE
from fusionmind4.control import PlasmaSCM, InterventionEngine
from fusionmind4.utils import FM3LitePhysicsEngine

# 1. Generate physics-consistent synthetic data
engine = FM3LitePhysicsEngine(n_samples=10000, seed=42)
data, interventions = engine.generate()

# 2. Discover causal graph (PF1)
cpde = EnsembleCPDE(config={"n_bootstrap": 10, "threshold": 0.32})
results = cpde.discover(data, interventional_data=interventions)
print(f"F1: {results['f1']:.1%}, Physics: {results['physics_passed']}/{results['physics_total']}")

# 3. Build structural causal model (PF2)
scm = PlasmaSCM()
scm.fit(results['dag'], data)

# 4. Answer causal questions
ie = InterventionEngine(scm)
# "What happens to Te if we SET P_ECRH = 10 MW?"
result = ie.do_intervention({'P_ECRH': 10.0})
print(f"Predicted Te: {result['Te']:.2f} keV")
```

### CausalShield-RL Example

```python
from fusionmind4.learning import CausalRLHybrid

hybrid = CausalRLHybrid()
hybrid.discover_causal_graph(data, interventions)
hybrid.fit_world_model(data)
hybrid.train(n_episodes=300)

# Act with causal explanation
result = hybrid.act_with_explanation(observation)
print(f"Action: {result['action']}")
print(f"Reasoning: {result['explanation']}")
```

---

## Patent Portfolio (7 Families)

| ID | Name | Module | Novelty | Status |
|----|------|--------|---------|--------|
| PF1 | Causal Plasma Discovery Engine (CPDE) | `fusionmind4/discovery/` | 9/10 | PoC validated (synthetic + real) |
| PF2 | Counterfactual Plasma Controller (CPC) | `fusionmind4/control/` | 10/10 | PoC validated |
| PF3 | Universal Plasma Foundation Model (UPFM) | `fusionmind4/foundation/` | 8/10 | PoC validated (6 devices) |
| PF4 | Diffusion-Based 3D Reconstruction (D3R) | `fusionmind4/reconstruction/` | 8/10 | PoC validated (156:1) |
| PF5 | Active Experiment Design Engine (AEDE) | `fusionmind4/experiments/` | 7/10 | PoC validated |
| PF6 | Integrated Causal Plasma System | System-level | 8/10 | Architecture defined |
| **PF7** | **CausalShield-RL** | `fusionmind4/learning/` | **9/10** | **PoC validated** |

---

## Competitive Landscape

| Feature | DeepMind/CFS | KSTAR | Princeton | TokaMind | **FusionMind 4.0** |
|---------|:---:|:---:|:---:|:---:|:---:|
| Pearl's Ladder Level | 1 | 1 | 1 | 1 | **2–3** |
| Causal Discovery | ✗ | ✗ | ✗ | ✗ | **✓** |
| do-Calculus | ✗ | ✗ | ✗ | ✗ | **✓** |
| Counterfactual Reasoning | ✗ | ✗ | ✗ | ✗ | **✓** |
| Causal RL | ✗ | RL only | ✗ | ✗ | **✓** |
| Explainable Decisions | ✗ | ✗ | ✗ | ✗ | **✓** |
| Simpson's Paradox Safe | Vulnerable | Vulnerable | Vulnerable | Vulnerable | **Prevented** |
| Real Data Validated | ✓ | ✓ | ✓ | ✓ (MAST only) | **✓ (MAST + C-Mod)** |
| Cross-Device Transfer | ✗ | ✗ | ✗ | ✗ | **✓** |

---

## Repository Structure

```
FusionMind-4-CausalPlasma/
│
├── fusionmind4/                    # Main package (5,321 lines)
│   ├── discovery/                  # PF1: Causal discovery
│   │   ├── ensemble.py             #   Bayesian ensemble fusion (9-step pipeline)
│   │   ├── notears.py              #   NOTEARS/DYNOTEARS DAG learning
│   │   ├── granger.py              #   Physics-aware Granger causality
│   │   ├── pc.py                   #   PC algorithm (conditional independence)
│   │   ├── physics.py              #   Physics priors + PINN validation
│   │   └── interventional.py       #   Interventional scoring
│   ├── control/                    # PF2: Counterfactual controller
│   │   ├── controller.py           #   Main controller (action, explain, safety)
│   │   ├── scm.py                  #   Structural Causal Model (Pearl's SCM)
│   │   └── interventions.py        #   do-calculus + counterfactual engines
│   ├── foundation/                 # PF3: Universal foundation model
│   │   └── core.py                 #   Dimensionless tokenization (βn,ν*,ρ*,q95,H98)
│   ├── reconstruction/             # PF4: Diffusion 3D reconstruction
│   │   └── core.py                 #   Conditional denoising diffusion model
│   ├── experiments/                # PF5: Active experiment design
│   │   └── core.py                 #   Bootstrap uncertainty + experiment ranking
│   ├── learning/                   # PF7: CausalShield-RL
│   │   ├── causal_rl_hybrid.py     #   Full hybrid agent (CPDE→SCM→PPO)
│   │   ├── neural_scm.py           #   Learnable neural SCM world model
│   │   ├── gym_plasma_env.py       #   Gym environment wrapping FM3Lite
│   │   └── causal_reward.py        #   Causal reward shaping
│   └── utils/                      # Shared utilities
│       ├── fm3lite.py              #   FM3-Lite physics simulation engine
│       ├── plasma_vars.py          #   14 plasma variables + ground truth DAG
│       └── simulator.py            #   Advanced physics simulator
│
├── scripts/                        # Runnable scripts
│   ├── run_full_validation.py      #   Validate all PFs on synthetic data
│   ├── run_fair_mast.py            #   Real MAST data validation
│   ├── run_real_data.py            #   Alcator C-Mod validation
│   └── train_causal_rl.py          #   Train CausalShield-RL agent
│
├── tests/                          # Test suite (56 tests)
│   ├── test_cpde.py                #   23 tests: discovery, control, foundation
│   └── test_learning.py            #   33 tests: neural SCM, Gym, RL, reward
│
├── dashboards/                     # Interactive React visualizations
│   ├── CPDE_v3_2_Results.jsx       #   CPDE real-data results dashboard
│   ├── FM4_Investor_Dashboard.jsx  #   Investor-facing overview
│   ├── FM4_Competitive_Analysis.jsx#   Competitive landscape analysis
│   └── FM4_CausalShieldRL_Dashboard.jsx
│
├── examples/quickstart.py          # Minimal working example
├── docs/                           # Detailed documentation
├── pyproject.toml                  # Package configuration
├── requirements.txt                # Dependencies
├── CHANGELOG.md                    # Version history
├── CITATION.cff                    # Academic citation
├── CONTRIBUTING.md                 # Contribution guidelines
└── LICENSE                         # Proprietary license
```

---

## Real Data Validation

### FAIR-MAST Database (UKAEA)

Accessed via [FAIR-MAST](https://fair.mast.ukaea.uk/) GraphQL API and S3 zarr storage (`s3.echo.stfc.ac.uk`). Eight shots from the MAST spherical tokamak were analyzed across 10 plasma variables (Ip, PNBI, ne, Te, βN, q95, Wmhd, Prad, κ, li).

CPDE discovered 11 causal edges with **91.9% F1 score**, 100% sign accuracy, and cross-shot robustness of F1 = 88.2% ± 4.4%. Key physics captured: Ip → q95 (inverse), PNBI → Wmhd, PNBI → βN, ne·Te → Prad.

See [`docs/REAL_DATA_VALIDATION.md`](docs/REAL_DATA_VALIDATION.md) for full methodology and results.

### MIT PSFC Open Density Limit Database (Alcator C-Mod)

264,385 timepoints from 2,333 discharges. UPFM dimensionless tokenization achieved AUC = 0.974 for density limit prediction, beating Greenwald fraction alone (AUC = 0.946). **Simpson's Paradox detected**: density-disruption correlation drops from +0.53 to +0.02 when conditioning on plasma current — demonstrating why causal reasoning is essential.

---

## Testing

```bash
pytest tests/ -v                                     # Run all 56 tests
pytest tests/ --cov=fusionmind4 --cov-report=term    # With coverage
pytest tests/test_cpde.py -v                         # Discovery + control
pytest tests/test_learning.py -v                     # Neural SCM + RL
```

---

## Citation

```bibtex
@software{mester2026fusionmind,
  author    = {Mester, Mladen},
  title     = {{FusionMind 4.0: Causal Inference for Fusion Plasma Control}},
  year      = {2026},
  url       = {https://github.com/mladen1312/FusionMind-4-CausalPlasma},
  note      = {Patent portfolio PF1--PF7 pending}
}
```

---

## Author

**Dr. Mladen Mester** — Scientist and inventor · March 2026

---

*CONFIDENTIAL — Patent filing in progress (PF1–PF7)*
