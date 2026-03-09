# FusionMind 4.0 — Definitive Results (v3.3)

## Dataset — Largest Public MAST Analysis
- **2,584 MAST shots** downloaded from FAIR-MAST Level 2 S3
- **15,969 operator comments** from FAIR-MAST GraphQL API
- **714 disruption times** parsed to millisecond precision
- **2,268 disruptions classified** by type (VDE/locked mode/IRE/density limit/equipment)

| Data | Count | Source |
|------|-------|--------|
| Shots downloaded | 2,584 | FAIR-MAST S3 (of 11,573 available) |
| Expert-labeled disrupted | 83 | disruption-py (shots 27000-30443) |
| Ops-log disrupted | 1,191 | Operator comments (shots 11779-30443) |
| Disruption times (ms precision) | 714 | Parsed from comments |
| Disruption types classified | 2,268 | NLP on operator comments |

## Validated Results

### Best Model: Expert-labeled disruptions (83d, same campaign)
| Threshold | Detection | FA Rate | Test set | CI |
|-----------|-----------|---------|----------|-----|
| 0.3 | **64% ± 11%** | **9% ± 1%** | 28d + 223c | FA ±3% |
| 0.5 | **55% ± 9%** | **6% ± 1%** | 28d + 223c | FA ±2% |

Model: GRU(hid=96, sl=30), 78 features (16 raw + 16 rates + 30 multi-scale + 16 SXR)

### 16-Channel Model: All disruption types (444d, diverse campaigns)
| Threshold | Detection | FA Rate | Test set |
|-----------|-----------|---------|----------|
| 0.3 | 32% | 100% | 148d + 714c |
| 0.5 | 13% | 4.5% | 148d + 714c |

val_AUC: 0.646 (barely above random)

## Key Finding: Expert vs Ops-Log Disruptions Are Different Populations

| Property | Expert (83d) | Ops-log (361d) |
|----------|-------------|----------------|
| Shot range | 27000-30443 | 11779-30443 |
| Median βN | 0.91 | 0.49 |
| Median shot length | 126 tp | 71 tp |
| val_AUC | 0.978 | 0.646 |
| Detection @th=0.3 | 64% | 14-32% |

## Disruption Type Classification

| Type | Count | % | Predictable? |
|------|-------|---|-------------|
| IRE | 895 | 39.5% | Yes |
| Generic disruption | 470 | 20.7% | Yes |
| NBI issue | 216 | 9.5% | No (equipment) |
| Locked mode | 209 | 9.2% | Yes |
| VDE | 180 | 7.9% | Yes |
| Early termination | 112 | 4.9% | No (operational) |
| Density limit | 61 | 2.7% | Yes |
| Other | 125 | 5.5% | Mixed |

**80% of MAST disruptions are plasma instabilities (theoretically predictable).**
But even these give AUC=0.651 with 0D signals — the challenge is label quality
and cross-campaign variability, not disruption type.

## What Works (proven)
1. Recovery filter: FA 70% → 30% (consistent)
2. SXR RMS 50kHz: +10pp detection (consistent)  
3. Multi-scale temporal diffs: +5pp det, −5pp FA (consistent)
4. All 16 channels × multi-scale: same as 6-channel subset (96 feat ≈ 78 feat)

## What Does NOT Work
- Ops-log labels without expert onset timing
- Cross-campaign model transfer (11K shots ≠ 27K shots)
- SDS, TTD+uncertainty, Thomson profiles, physics margins
- More than 96 features (overfitting)
- Auto-finding disrupted shots from Ip behavior

## Honest Assessment
FusionMind achieves **64% detection @ 9% FA** on well-characterized,
same-campaign MAST disruptions with 0D+SXR signals. This is competitive
with published results given the data volume (83 disrupted shots).

Scaling to diverse, multi-campaign disruptions requires either:
1. Expert disruption timing for all campaigns (MAST ops database)
2. DIII-D data via disruption-py (proper labels for 2000+ disruptions)
3. Campaign-specific normalization and onset detection

## DisruptionBench Comparison
| Model | Tokamak | Shots | AUC |
|-------|---------|-------|-----|
| FusionMind | MAST (expert) | 255 | 0.842 |
| FusionMind | MAST (diverse) | 2,584 | 0.646 |
| FRNN | DIII-D | 20,000+ | ~0.97 |
| GPT-2 | C-Mod | ~3,000 | 0.84 |

## Repository
- `data/mast/mast_level2_2521shots.npz` — 2,521 shots, 15MB
- `data/mast/mast_ops_log.json` — 15,969 operator comments  
- `data/mast/mast_disruption_times.json` — 714 ms-precision times
- `data/mast/mast_disruption_types.json` — 2,268 classified disruptions
- `scripts/download_mast_level2.py` — standalone download script
- `benchmarks/` — all benchmark results with full reproducibility
