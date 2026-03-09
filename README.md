# FusionMind 4.0 — Causal AI for Fusion Plasma Control

> **Pearl's do-calculus applied to real tokamak disruption prediction.**
> **C-Mod AUC=0.928 (5-seed avg) — beats GPT-2 (0.840) and RF (0.832).**

[![Tests](https://img.shields.io/badge/tests-310%20passed-brightgreen)]()
[![C-Mod AUC](https://img.shields.io/badge/C--Mod%20AUC-0.928-blue)]()
[![MAST shots](https://img.shields.io/badge/MAST%20shots-2941-orange)]()

---

## DisruptionBench Results

| Model | Machine | AUC | Note |
|-------|---------|-----|------|
| CCNN (Spangher 2025) | C-Mod | **0.974** | Published SOTA |
| FusionMind v4.2 best seed | C-Mod | 0.955 | Competitive |
| **FusionMind v4.2 (5-seed avg)** | **C-Mod** | **0.928 ± 0.04** | **Beats GPT-2, RF, HDL** |
| GPT-2 (Spangher 2025) | C-Mod | 0.840 | |
| RF (Rea 2018) | C-Mod | 0.832 | |
| HDL (Zhu 2020) | C-Mod | 0.780 | |
| FusionMind v4.2 | MAST | 0.649 | First-ever benchmark |
| FRNN (Kates-Harbeck 2019) | DIII-D | ~0.97 | |

> **⚠️ Previous versions claimed AUC=0.986. This was a cherry-picked seed.
> Honest 5-seed average is 0.928. See RESULTS.md for full correction.**

### Why We're Competitive (Not SOTA)

A simple GRU with **48 physics-informed features** beats GPT-2 (1.6B params)
because the features encode the *causal mechanism* of disruption:

1. **Greenwald fraction** `f_GW = ne / n_GW` — density limit physics
2. **Simpson's Paradox** — `ne/Ip` (density conditioned on plasma current)
3. **Multi-scale temporal diffs** — `smooth(x, 3tp) - smooth(x, 7tp)`
4. **f_GW margin** — distance from rolling maximum

### MAST: First-Ever Benchmark

- **2,941 shots** (448 disrupted) from FAIR-MAST public data
- **714 ms-precision disruption times** from operator comments
- AUC = 0.649 — MAST (spherical tokamak) is harder than conventional

### DIII-D / EAST

DisruptionBench data requires MDSplus credentials (not public).
Contact: `disruption-py@mit.edu`

---

## Architecture

```
Layer 0: CPDE — Causal Plasma Discovery Engine
  NOTEARS DAG + Granger + PC algorithm → causal graph

Layer 1: SCM — Structural Causal Models
  do-calculus, counterfactual reasoning

Layer 2: Device-Specific Prediction
  Physics-informed GRU per tokamak

Layer 3: C++ Real-Time Engine (<100μs)
```

### Causal Discovery Results

| Metric | MAST | C-Mod |
|--------|------|-------|
| DAG edges recovered | 17/18 (F1=88.9%) | — |
| Simpson's Paradox | — | ρ +0.53 → +0.02 |
| SCM R² | Linear 37%, GBT 65% | — |

---

## Datasets

| Data | Shots | Source |
|------|-------|--------|
| C-Mod density limit | 2,333 (78d) | MIT PSFC Open Data |
| MAST Level 2 | 2,941 (448d) | FAIR-MAST S3 (public) |
| MAST ops log | 15,969 comments | FAIR-MAST GraphQL |
| MAST disruption times | 714 (ms precision) | Parsed from ops comments |

## Quick Start

```bash
git clone https://github.com/mladen1312/FusionMind-4-CausalPlasma
cd FusionMind-4-CausalPlasma
pip install -e .
pytest tests/ --ignore=tests/test_integration.py -q  # 310 pass
```

## Tests

310 passed, 0 failed, 25 skipped (MLX on Linux)

## Author

**Dr. Mladen Mester** — Zagreb, Croatia

## License

MIT
