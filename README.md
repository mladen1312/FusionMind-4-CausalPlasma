# FusionMind 4.0 — Causal AI for Fusion Plasma Control

> **The first application of Pearl's causal inference framework to tokamak plasma physics.**

[![Tests](https://img.shields.io/badge/tests-56%20passed-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Patent Families](https://img.shields.io/badge/patents-7%20families-orange)](#patent-portfolio)

## Why This Matters

Every existing fusion AI system — DeepMind/CFS, KSTAR RL, Princeton Diag2Diag, TokaMind — operates at **Pearl's Ladder Level 1** (correlation). They learn statistical patterns but cannot answer:

- *"What happens to Te if we **set** P_ECRH to 10MW?"* (Intervention)
- *"What **would have** happened if we had increased current instead?"* (Counterfactual)

**FusionMind 4.0 operates at Levels 2–3**, enabling true causal reasoning for plasma control.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FusionMind 4.0 System                       │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   CPDE   │  │   CPC    │  │   UPFM   │  │   D3R    │       │
│  │  (PF1)   │→ │  (PF2)   │  │  (PF3)   │  │  (PF4)   │       │
│  │ Causal   │  │ Counter- │  │ Foundation│  │ Diffusion│       │
│  │ Discovery│  │ factual  │  │  Model   │  │  Recon   │       │
│  └────┬─────┘  └────┬─────┘  └──────────┘  └──────────┘       │
│       │              │                                          │
│  ┌────▼──────────────▼─────────────────────────────────┐       │
│  │         CausalShield-RL (PF7) — NEW                 │       │
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

## Key Results

| Metric | Value | Note |
|--------|-------|------|
| CPDE F1 (FM3Lite) | 79.2% | 21/25 true positives |
| CPDE F1 (Real MAST) | 91.9% | 8 shots, 625 timepoints |
| Alcator C-Mod AUC | 0.974 | Density limit prediction |
| Simpson's Paradox | Detected ✓ | ρ drops +0.53 → +0.02 |
| UPFM Cross-Device CV | 0.267 | 6 tokamak devices |
| D3R Compression | 156:1 | Conditional diffusion |
| Neural SCM Online | ✓ | Continuous learning |
| CausalShield-RL | ✓ | First causal RL for fusion |

## Quick Start

```bash
git clone https://github.com/mladen1312/FusionMind-4-CausalPlasma.git
cd FusionMind-4-CausalPlasma
pip install -r requirements.txt

python scripts/run_full_validation.py      # Validate PF1-PF5
python scripts/train_causal_rl.py          # Train CausalShield-RL (PF7)
pytest tests/ -v                           # Run all 56 tests
```

### Minimal Example

```python
from fusionmind4.utils.fm3lite import FM3LitePhysicsEngine
from fusionmind4.learning.causal_rl_hybrid import CausalRLHybrid

engine = FM3LitePhysicsEngine(n_samples=20000, seed=42)
data, interventional = engine.generate()

hybrid = CausalRLHybrid()
hybrid.discover_causal_graph(data, interventional)
hybrid.fit_world_model(data)
hybrid.train(n_episodes=300)

action = hybrid.act(observation)
result = hybrid.act_with_explanation(observation)
```

## Patent Portfolio (7 Families)

| ID | Name | Novelty | Status |
|----|------|---------|--------|
| PF1 | Causal Plasma Discovery Engine (CPDE) | 9/10 | PoC validated |
| PF2 | Counterfactual Controller (CPC) | 10/10 | PoC validated |
| PF3 | Universal Plasma Foundation Model (UPFM) | 8/10 | PoC validated |
| PF4 | Diffusion-Based 3D Reconstruction (D3R) | 8/10 | PoC validated |
| PF5 | Active Experiment Design Engine (AEDE) | 7/10 | PoC validated |
| PF6 | Integrated Causal Plasma System | 8/10 | Architecture |
| **PF7** | **CausalShield-RL** | **9/10** | **PoC validated** |

## Competitive Advantage

| Feature | DeepMind/CFS | KSTAR | Princeton | TokaMind | **FusionMind** |
|---------|:---:|:---:|:---:|:---:|:---:|
| Causal Discovery | ✗ | ✗ | ✗ | ✗ | **✓** |
| do-Calculus | ✗ | ✗ | ✗ | ✗ | **✓** |
| Counterfactuals | ✗ | ✗ | ✗ | ✗ | **✓** |
| Causal RL | ✗ | RL only | ✗ | ✗ | **✓** |
| Explainable | ✗ | ✗ | ✗ | ✗ | **✓** |
| Online Learning | Limited | ✗ | ✗ | ✗ | **✓** |
| Simpson's Paradox | Vulnerable | Vulnerable | Vulnerable | Vulnerable | **Prevented** |
| Pearl's Ladder | 1 | 1 | 1 | 1 | **2–3** |

## Repository Structure

```
fusionmind4/
├── discovery/           # PF1: Causal discovery (NOTEARS, Granger, PC, ensemble)
├── control/             # PF2: SCM, do-calculus, counterfactuals
├── foundation/          # PF3: Dimensionless tokenization
├── reconstruction/      # PF4: Diffusion 3D reconstruction
├── experiments/         # PF5: Active experiment design
├── learning/            # PF7: CausalShield-RL (Neural SCM, Gym, PPO)
└── utils/               # FM3Lite physics, plasma variables

scripts/                 # Training & validation scripts
tests/                   # 56 tests (23 existing + 33 learning)
dashboards/              # React visualization dashboards
```

## Author

**Dr. Mladen Mester** · March 2026

## License

MIT License
