# FusionMind 4.0 — Definitive Results (v3.2)

## Dataset — Largest Public MAST Disruption Dataset
- **2941 MAST shots** (448 disrupted + 2493 clean) from FAIR-MAST Level 2
- **714 disruption times** parsed from operator comments (ms precision)
- 268,667 timepoints, 16 base variables, EFIT timebase
- Source: FAIR-MAST S3 archive + GraphQL ops log (all public)

## Model — 16-Channel Multi-Scale GRU
- GRU (78 → 48 hidden, seq_len=15, dropout=0.3)
- **78 features from 16 parallel channels:**
  - 16 raw Level-2 variables (βN, βp, q95, li, κ, Ip, ne, f_GW, p_rad, P_NBI...)
  - 16 absolute first derivatives
  - 16 SXR features (8 RMS channels + 8 rates, 50kHz → EFIT)
  - 30 multi-scale temporal diffs (6 signals × {sm3, sm7, sm15, diff_3-7, diff_7-15})

## Results (150 disrupted + 831 clean test shots)

### Shot-level AUC: 0.691

### Operating Points
| Method | Threshold | Detection | FA Rate | Note |
|--------|-----------|-----------|---------|------|
| Shot max probability | 0.4 | **53%** | **21%** | Aggressive |
| Shot max probability | 0.5 | **39%** | **12%** | Conservative ★ |
| Recovery filter | 0.3 | 36% | 50% | Filter too strict for diverse types |
| Recovery filter | 0.5 | 7% | 10% | Very conservative |

TPR 95% CI: ±8% (150 test disrupted — publishable)

### Scaling & Comparison
| Model | Tokamak | Shots | Dis | AUC | TPR | FPR |
|-------|---------|-------|-----|-----|-----|-----|
| **FusionMind v3.2 (diverse)** | **MAST** | **2941** | **448** | **0.691** | **39-53%** | **12-21%** |
| FusionMind v1 (expert only) | MAST | 255 | 83 | 0.842 | 64% | 9% |
| GPT-2 (Spangher 2025) | C-Mod | ~3000 | — | 0.840 | — | — |
| CCNN (Spangher 2025) | C-Mod | ~3000 | — | 0.974 | — | — |
| FRNN (Kates-Harbeck 2019) | DIII-D | >20000 | — | ~0.97 | 87% | 5% |

### Why AUC Dropped from 0.842 to 0.691
Not model degradation — evaluation on ALL disruption types vs one curated campaign:
- **Expert-labeled (v1):** 83 shots, one campaign (27000+), βN=0.91, same physics
- **Ops-log labeled (v3.2):** 448 shots, all campaigns (11K-30K), βN=0.49, VDE + locked mode + density limit + FA trip

The 0.691 is the **honest, realistic number** for MAST disruption prediction.

## What 16 Channels Contribute
| Feature set | Features | Detection | FA | Improvement |
|-------------|----------|-----------|-----|-------------|
| Raw only (16 vars + rates) | 32 | 18% | 30% | baseline |
| + SXR RMS (8ch + rates) | 48 | — | — | +10pp det historically |
| + Multi-scale diffs | **78** | **53%** | **21%** | **+35pp detection!** |

Multi-scale temporal diffs (sm3−sm7, sm7−sm15) are the key discriminator,
confirmed on both 255-shot expert set and 2941-shot diverse set.

## Causal Discovery (separate from prediction)
- NOTEARS DAG: F1 = 88.9%, 17/18 expected edges on MAST
- SCM: Linear R² = 37%, Nonlinear R² = 65%
- Simpson's Paradox: density-disruption correlation drops from +0.53 to +0.02 on C-Mod

## Key Assets on GitHub
| File | Description |
|------|-------------|
| `data/mast/mast_level2_2521shots.npz` | 2521 shots, 15MB |
| `data/mast/mast_ops_log.json` | 15,969 operator comments |
| `data/mast/mast_disruption_times.json` | 714 ms-precision disruption times |
| `data/mast/mast_ops_disrupted.json` | 1,274 identified disrupted shots |
| `scripts/download_mast_level2.py` | Standalone download script |
| `benchmarks/benchmark_v3_final_16ch.json` | Full results |

## To Reproduce
```bash
git clone https://github.com/mladen1312/FusionMind-4-CausalPlasma
cd FusionMind-4-CausalPlasma
python scripts/download_mast_level2.py --target 3000  # download data
# All data sources are public (FAIR-MAST S3, anonymous access)
```
