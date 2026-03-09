# FusionMind 4.0 — Honest Results (v4.2 CORRECTED)

## ⚠️ CORRECTION: Previous AUC=0.986 was cherry-picked

Previous commits claimed C-Mod AUC=0.986. This was from a **single lucky seed**
with a specific model architecture. Honest multi-seed verification shows:

| Seed | AUC | TPR@5% |
|------|-----|--------|
| 0 | 0.943 | 62% |
| 1 | 0.942 | 58% |
| 2 | 0.955 | 73% |
| 3 | 0.947 | 69% |
| 4 | 0.855 | 46% |
| **Mean** | **0.928 ± 0.040** | **62%** |

With only 26 test disrupted shots, one misclassification = ±4% TPR,
and AUC swings ±0.04 between seeds.

## Corrected Comparison

| Model | Machine | AUC | Status |
|-------|---------|-----|--------|
| CCNN many-shot (Spangher 2025) | C-Mod | **0.974** | Published SOTA |
| FusionMind v4.2 best seed | C-Mod | 0.955 | Competitive |
| **FusionMind v4.2 (5-seed avg)** | **C-Mod** | **0.928 ± 0.04** | **Beats GPT-2, RF, HDL** |
| GPT-2 (Spangher 2025) | C-Mod | 0.840 | Published |
| RF (Rea 2018) | C-Mod | 0.832 | Published |
| HDL (Zhu 2020) | C-Mod | 0.780 | Published |
| FusionMind v4.2 | MAST | 0.649 | First ever |
| FRNN (Kates-Harbeck 2019) | DIII-D | ~0.97 | Published |

**We beat GPT-2, RF, and HDL but do NOT beat CCNN.**

## What Works

### Physics-Informed Features (+11% AUC vs raw signals)

Our causal analysis identified key features that improve prediction:

1. **Greenwald fraction** (f_GW = ne / n_GW) — density limit physics
2. **ne/Ip ratio** — Simpson's Paradox correction
3. **Multi-scale temporal diffs** — dynamics at 30ms, 70ms, 150ms scales
4. **f_GW rolling max margin** — distance from peak

48 features from 6 base variables → AUC=0.928 (vs ~0.82 with raw signals)

### Causal Discovery (validated on real data)

| Metric | MAST (9 shots) | C-Mod (2333 shots) |
|--------|----------------|---------------------|
| DAG edges recovered | 17/18 (F1=88.9%) | — |
| Simpson's Paradox | — | ρ drops +0.53 → +0.02 |

### MAST: Largest Public Dataset

- **2,941 shots** (448 disrupted + 2,493 clean)
- **714 ms-precision disruption times** parsed from operator comments
- **15,969 operator comments** from FAIR-MAST GraphQL API
- AUC = 0.649 (first-ever MAST benchmark)

## Why Only 26 Test Disrupted

C-Mod density limit database has 78 disrupted shots total.
With 2/3 train split, only 26 go to test. This means:

- TPR CI = ±15-20% (not publishable as definitive)
- AUC variance = ±0.04 across seeds
- Need 200+ disrupted for CI ±5%

## Datasets on GitHub

| File | Description |
|------|-------------|
| `data/cmod/cmod_density_limit.npz` | 2333 C-Mod shots |
| `data/mast/mast_level2_2941shots.npz` | 2941 MAST shots |
| `data/mast/mast_disruption_times.json` | 714 disruption times |
| `data/mast/mast_ops_log.json` | 15,969 operator comments |

## DIII-D / EAST Testing

DisruptionBench DIII-D/EAST data requires MDSplus credentials
from General Atomics and ASIPP. Not publicly downloadable.
Contact: `disruption-py@mit.edu`

## Tests

310 passed, 0 failed, 25 skipped
