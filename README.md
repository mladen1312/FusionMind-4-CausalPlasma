<p align="center">
  <strong>FusionMind 4.0</strong><br/>
  <em>Causal AI for Fusion Plasma Control</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CPDE_F1-88.9%25-22C55E?style=flat-square" alt="F1"/>
  <img src="https://img.shields.io/badge/Physics-10%2F10-3B82F6?style=flat-square" alt="Physics"/>
  <img src="https://img.shields.io/badge/Tests-23%2F23-22C55E?style=flat-square" alt="Tests"/>
  <img src="https://img.shields.io/badge/Patent_Families-PF1--PF6-F59E0B?style=flat-square" alt="Patents"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-306998?style=flat-square" alt="Python"/>
</p>

---

## Overview

**FusionMind 4.0** is the first application of Pearl's causal inference framework (do-calculus) to tokamak plasma physics. While all existing fusion AI systems operate at Pearl's Ladder Level 1 (association/prediction), FusionMind 4.0 operates at **Levels 2–3** — enabling intervention reasoning and counterfactual analysis.

### The Gap

Every major fusion AI system — DeepMind/CFS, KSTAR RL, Princeton Diag2Diag, TokaMind — is purely correlational. They predict *what will happen* but cannot answer:

- **Intervention**: *What happens if I increase NBI power by 20%?*
- **Counterfactual**: *Would this disruption have occurred without the density ramp?*
- **Confounding**: *Is high density causing disruptions, or is Ip the hidden confounder?*

Our Simpson's Paradox detection on real Alcator C-Mod data shows the power of causal reasoning: the apparent density→disruption correlation (+0.53) vanishes after conditioning on plasma current (+0.02).

---

## Patent Portfolio (5 Inventions)

| # | Name | Module | Key Innovation | Novelty |
|---|------|--------|----------------|---------|
| PF1 | Causal Plasma Discovery Engine (CPDE v3.2) | `fusionmind4/discovery/` | 5-algorithm ensemble: NOTEARS + Granger + PC + Interventional + Physics | 9/10 |
| PF2 | Counterfactual Plasma Controller (CPC) | `fusionmind4/control/` | SCM with do-calculus + counterfactual reasoning + causal path tracing | 10/10 |
| PF3 | Universal Plasma Foundation Model (UPFM) | `fusionmind4/foundation/` | Dimensionless tokenization (βN, ν*, ρ*, q95, H98) for cross-device transfer | 8/10 |
| PF4 | Diffusion-Based 3D Reconstruction (D3R) | `fusionmind4/reconstruction/` | Score-based diffusion for sparse→full plasma state reconstruction | 7/10 |
| PF5 | Active Experiment Design Engine (AEDE) | `fusionmind4/experiments/` | Optimal experiment selection via causal uncertainty reduction | 8/10 |

---

## CPDE v3.2 Results

| Metric | v1.0 | v2.1 | **v3.2** |
|--------|------|------|----------|
| F1 Score | 52.8% | 79.2% | **88.9%** |
| Precision | 56.0% | 75.0% | **92.3%** |
| Recall | 50.0% | 84.0% | **85.7%** |
| SHD | 25 | 10 | **6** |
| Physics | 5/5 | 8/8 | **10/10** |
| TP/FP | 14/11 | 21/7 | **24/2** |
| OOD @5% noise | — | — | **88.9%** |

### CPDE Pipeline (5 Algorithms)

| Method | Weight | Description |
|--------|--------|-------------|
| NOTEARS | 30% | Structural DAG learning with L1 regularization |
| Granger Causality | 22% | Temporal causation with lag selection + Bonferroni |
| PC Algorithm | 18% | Constraint-based discovery with v-structures |
| Interventional Scoring | — | do-calculus validation for actuator edges |
| Physics Priors | 30% | Domain knowledge from plasma physics (21 known edges) |

Plus: Bootstrap CI (15 iterations), adaptive thresholding, actuator-skip indirect detection, DAG enforcement.

---

## CPC Features (Patent Family PF2)

- **do-calculus interventions**: `P(Y|do(X=x))` — "what state results if I SET X=x?"
- **Counterfactual reasoning**: `P(Y_x|X=x',Y=y)` — "what WOULD have happened?"
- **Causal path tracing**: Explain WHY an action affects a target via DAG paths
- **Retrospective analysis**: Post-hoc "would alternative action have been better?"
- **Disruption avoidance**: Emergency safety intervention targeting βN, q, MHD
- **Optimal control**: Find actuator settings to achieve target via causal reasoning

---

## Quick Start

```bash
pip install -e .
```

```python
from fusionmind4.discovery import EnsembleCPDE
from fusionmind4.utils import FM3LitePhysicsEngine

# Generate data & discover causal structure
engine = FM3LitePhysicsEngine(n_samples=20000, seed=42)
data, interventions = engine.generate()
cpde = EnsembleCPDE(config={"n_bootstrap": 15, "threshold": 0.32})
results = cpde.discover(data, interventional_data=interventions)

print(f"F1: {results['f1']:.1%}  Physics: {results['physics_passed']}/{results['physics_total']}")
```

```python
from fusionmind4.control import PlasmaSCM, InterventionEngine, CounterfactualEngine

# Build SCM from discovered DAG
scm = PlasmaSCM(var_names, results['dag'])
scm.fit(data)

# What happens if we increase NBI power?
engine = InterventionEngine(scm)
result = engine.do({"P_NBI": 1.5}, current_state)

# What WOULD have happened without the density ramp?
cf = CounterfactualEngine(scm)
result = cf.counterfactual(observed_state, {"gas_puff": 0.3})
```

---

## Validation Scripts

```bash
# Full CPDE validation (F1=88.9%)
python scripts/run_full_validation.py --n_samples 20000 --bootstrap 15 --ood

# Real data pipeline (Alcator C-Mod)
python scripts/run_real_data.py

# Run all tests
python -m pytest tests/ -v
```

---

## Real Data Validation

Validated on **MIT PSFC Alcator C-Mod** Open Density Limit Database:

- 264,385 timepoints across 1,876 plasma shots
- Density limit prediction AUC: **0.974** (vs Greenwald fraction ~0.85)
- **Simpson's Paradox detected**: ne↔disruption drops from +0.53 to +0.02 after Ip conditioning

---

## Project Structure

```
fusionmind4/
├── fusionmind4/
│   ├── discovery/           # PF1: CPDE v3.2 — 5 causal discovery algorithms
│   │   ├── ensemble.py      # Orchestrator: fusion + DAG enforcement
│   │   ├── notears.py       # NOTEARS with bootstrap
│   │   ├── granger.py       # Granger causality with Bonferroni
│   │   ├── pc.py            # PC algorithm with v-structures
│   │   ├── interventional.py # do-calculus validation
│   │   └── physics.py       # Physics priors + 10 validation checks
│   ├── control/             # PF2: CPC — Counterfactual Controller
│   │   ├── scm.py           # Pearl's Structural Causal Model
│   │   ├── interventions.py # do-calculus + counterfactual engines
│   │   └── controller.py    # Full controller with safety + explanation
│   ├── foundation/          # PF3: UPFM — Cross-device foundation model
│   │   └── core.py          # Dimensionless tokenizer + transfer learning
│   ├── reconstruction/      # PF4: D3R — Diffusion 3D reconstruction
│   ├── experiments/         # PF5: AEDE — Active experiment design
│   └── utils/
│       ├── plasma_vars.py   # 14 variables, 28 ground-truth edges
│       └── fm3lite.py       # Causally-faithful physics simulator
├── tests/test_cpde.py       # 23 tests (all passing)
├── scripts/
│   ├── run_full_validation.py
│   └── run_real_data.py     # Alcator C-Mod pipeline
├── dashboards/              # Interactive React visualizations
│   ├── CPDE_v3_2_Results.jsx
│   └── FM4_Competitive_Analysis.jsx
├── pyproject.toml
└── README.md
```

---

## Competitive Landscape

| System | Causal | Cross-Device | Foundation Model |
|--------|--------|--------------|-----------------|
| **FusionMind 4.0** | **✓ Level 2-3** | **✓ 6 devices** | **✓ Physics tokens** |
| DeepMind/CFS | ✗ RL | ✗ TCV | ✗ |
| KSTAR RL | ✗ RL | ✗ DIII-D | ✗ |
| TokaMind | ✗ | ✗ MAST | ✓ DCT tokens |
| Diag2Diag | ✗ | Partial | ✗ |

**No competitor uses causal inference.**

---

## Citation

```bibtex
@software{mester2026fusionmind,
  title={FusionMind 4.0: Causal AI for Fusion Plasma Control},
  author={Mester, Mladen},
  year={2026},
  url={https://github.com/mladen1312/FusionMind-4-CausalPlasma}
}
```

## License

Proprietary. Patent pending (PF1–PF6). Contact for licensing inquiries.

---

<p align="center">
  <em>Dr. Mladen Mester · March 2026</em><br/>
  <em>First-ever application of Pearl's causal inference framework to tokamak plasma dynamics</em>
</p>
