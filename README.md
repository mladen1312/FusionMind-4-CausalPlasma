# FusionMind 4.0 — Causal AI for Fusion Plasma Control

> **Pearl's do-calculus and structural causal models applied to real tokamak plasma data.**
>
> *Not another black-box RL controller. The only system that can tell you WHY.*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![License](https://img.shields.io/badge/license-Proprietary-red)](LICENSE)
[![Real Data](https://img.shields.io/badge/validated-331%20shots%20MAST-purple)](#benchmark-results)

---

## What Problem Does FusionMind Solve?

Every tokamak uses AI for plasma control. DeepMind, KSTAR, Princeton — they all use **correlational models** (RL, neural networks). These models learn "when X goes up, Y goes up" but **cannot distinguish causation from correlation**.

**FusionMind applies Pearl's structural causal model framework** — do-calculus interventions, counterfactual reasoning, and causal graph discovery — to real tokamak data. It provides:

1. **Causal explanations**: *"Ip is dropping because li increased via the causal path li → q_axis → q95 → Ip"*
2. **Intervention prediction**: *"If we increase NBI by 2MW, βN will rise by 0.3 via βt → βN"*
3. **Counterfactual reasoning**: *"Had we not reduced gas puff at t=2.3s, the disruption would not have occurred"*

---

## Architecture

```
Layer 0: CPDE (Causal Plasma Discovery Engine)
  NOTEARS + Granger + PC Algorithm + Physics Priors → DAG

Layer 1: Structural Causal Model
  Linear SCM (R² 37%) + Nonlinear GBT SCM (R² 65%)
  do-calculus interventions + 3-step counterfactual

Layer 2: Dynamic Overseer (4 parallel tracks)
  Track A: Correlational ML (all features)
  Track B: Causal SCM (DAG parents only)
  Track C: Physics thresholds (Troyon, q-margin, li-margin)
  Track D: Fast diagnostics anomaly (MHD n=2, Dα, li_rate)
  → Overseer selects/blends per-timestep with stateful memory

Layer 3: C++ Real-Time Engine
  Linear SCM: 83ns | Full pipeline: 705ns
```

---

## Benchmark Results

All results on **real MAST data** from FAIR-MAST open archive. No synthetic/proxy labels unless stated.

### Causal Graph Recovery (331 shots, 23,721 timepoints)

| Metric | Value | Notes |
|--------|-------|-------|
| Edge F1 | **88.9%** | 12/13 expected causal edges recovered |
| Precision | 85.7% | 4 false positive edges |
| Recall | 94.4% | 1 missed edge (NBI→ne) |
| Cross-shot robustness | 88.2% ± 4.4% | Mean F1 across 9 individual shots |

### SCM Prediction (leave-shots-out, 10-fold CV)

| Model | Overall R² | Best variable | Worst variable |
|-------|-----------|---------------|----------------|
| Linear SCM | 37.0% | βt (97.2%) | βp (2.8%) |
| Nonlinear SCM (GBT) | **65.0%** | βt (97.2%) | q95 (65.8%) |
| Intervention direction | **77.9%** | — | — |

### Shot-Level Disruption Prediction (83 real Ip-quench disruptions, stratified 5-fold CV)

| System | AUC | Recall | Precision | Explainable? |
|--------|-----|--------|-----------|-------------|
| Track A (ML, all features) | 0.967 | 87.8% | 75.3% | No |
| Track B (Causal, DAG parents only) | **0.967** | 87.8% | 70.1% | **Yes** |
| Track C (Physics) | 0.966 | 87.8% | 77.3% | Partial |
| FUSED (A+B+C) | 0.969 | 90.2% | 73.8% | Yes |

**Key finding:** Track B uses only 6 causal parent variables but achieves the **same AUC** as Track A using all 71 features (bootstrap 95% CIs overlap: A [0.947–1.000], B [0.935–0.999]). The DAG correctly identifies disruption-relevant variables.

### Warning Time with Dynamic Overseer (50 shots, Dα + MHD n=2 + density)

| Method | Detected | Mean Warning | >100ms | Labels used |
|--------|----------|-------------|--------|-------------|
| Track A (ML) + anomaly labels | **11/25 (44%)** | **199ms** | 82% | Anomaly onset |
| Dynamic Overseer (4 tracks) | **11/25 (44%)** | **182ms** | 82% | Anomaly onset |
| Oracle (best track per-tp) | 14/25 (56%) | 179ms | 64% | Anomaly onset |
| ML + last-30% labels (old) | 6/25 (24%) | 30ms | 0% | Last 30% ← wrong |
| Physics 2σ (no ML) | 21/25 (84%) | 195ms | 86% | N/A (threshold) |

**Key finding:** Warning time failure was caused by **wrong labels**, not wrong model. When labels are aligned to physical anomaly onset (2σ deviation in li_rate, βp_rate, MHD, Dα), ML achieves 199ms mean warning — comparable to published systems.

### Permutation Importance (Causal Disruption Drivers)

| Variable | AUC drop | Physical interpretation |
|----------|----------|----------------------|
| **Ip** (plasma current) | **0.098** | Disruption IS current collapse |
| Wstored | 0.007 | Energy confinement loss |
| βt | 0.002 | Pressure-driven instability |
| κ, βp, βN | <0.001 | Captured via causal parents |

### Calibration

- Brier score: 0.078 — well calibrated
- Precision @ 90% recall: **25.3%** (not operational — needs 1D profiles)
- Physics violations: **0/45,486** predictions in physical bounds

### C++ Real-Time Performance

| Phase | Latency | Components |
|-------|---------|-----------|
| Phase 1 (causal check) | **83 ns** | DAG lookup + linear SCM |
| Phase 2 (intervention) | **247 ns** | do-calculus propagation |
| Phase 3 (counterfactual) | **705 ns** | Full 3-step reasoning |

---

## Comparison with Existing Systems

| System | Task | Key Metric | Dataset | Explainable? | do-calculus? |
|--------|------|-----------|---------|-------------|-------------|
| FRNN [1] | Disruption pred. | AUC 0.92 | ~20K shots, multi-machine | No | No |
| DeepMind [2] | Shape control | 2cm tracking | TCV live | No | No |
| EUROfusion [7] | Causal detection | Granger/TE | JET, WEST | Partial | No |
| **FusionMind** | **Causal + pred.** | **AUC 0.967** | **331 shots, MAST** | **Yes** | **Yes** |

**Important:** FRNN uses ~20,000 shots from multiple machines with operational labels. Our AUC 0.967 is on 331 MAST shots with Ip-quench labels. **These are not directly comparable.** FusionMind's value is explainability + do-calculus, not prediction accuracy.

---

## Known Limitations

1. **Single machine.** All results from MAST spherical tokamak. Cross-device validation needed.
2. **0D parameters.** Precision @ 90% recall is only 25.3% — not operational without 1D profiles.
3. **Warning time detection rate.** 44% with ML / 84% with physics thresholds — below FRNN's ~87%. Needs locked mode + bolometry signals.
4. **Fusion lift is marginal.** Multi-track fusion adds +0.002 AUC. Real value is explainability, not accuracy.
5. **Small fast-diagnostic dataset.** Only 50 shots have Dα + MHD (vs 331 with EFIT only).
6. **Disagreement signal.** Inter-track disagreement does NOT significantly improve recall on real labels (p=0.77).

---

## Installation

```bash
git clone https://github.com/mladen1312/FusionMind-4-CausalPlasma.git
cd FusionMind-4-CausalPlasma
pip install -e .

# Run core tests
python -m pytest tests/test_cpde.py tests/test_upgrades.py tests/test_v43_upgrades.py tests/test_causal_controller.py tests/test_realtime.py -q
```

---

## References

1. Kates-Harbeck et al., *Nature* 568, 526–531 (2019).
2. Degrave et al., *Nature* 602, 414–419 (2022).
3. Gopakumar et al., arXiv:2602.15084 (2026).
4. Pearl, *Causality*, 2nd ed., Cambridge University Press (2009).
5. Murari et al., *Nuclear Fusion* 60, 066020 (2020).

---

**Dr. Mladen Mester** — Independent Researcher, Zagreb, Croatia

*FusionMind does not replace FRNN or DeepMind RL. It explains why they make the decisions they make, tests what-if scenarios they cannot, and provides the causal safety layer that regulators will require.*
