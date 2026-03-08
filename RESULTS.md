# FusionMind 4.0 — MAST Dataset & Results (v2.2)

## Dataset — Largest Public MAST Disruption Dataset
- **2405 MAST shots** (83 disrupted + 2322 clean) from FAIR-MAST Level 2
- 229,255 timepoints, 16 variables, EFIT timebase (~10ms)
- 1410/2179 shots downloaded from disrupted range (27000-30443)
- Source: FAIR-MAST S3 (anonymous public access)

## Best Validated Results (78 features, recovery filter)
| Threshold | Detection | FA Rate | Test set | CI |
|-----------|-----------|---------|----------|-----|
| 0.3 | **64%** | **9%** | 28d + 223c | Det ±19%, FA ±3% |
| 0.5 | **55%** | **6%** | 28d + 223c | Det ±18%, FA ±2% |

Note: FA measured on 777-shot homogeneous subset. On diverse 2259 shots,
FA rises to 16% (multi-campaign variability). True FA is 9-16%.

## What Works (proven)
1. Recovery filter: FA 70%→30%
2. SXR RMS (50kHz): +10pp detection
3. Multi-scale temporal diffs: +5pp det, −5pp FA

## Key Finding: Disrupted Shots Cannot Be Auto-Detected
- MAST disrupted/clean have identical Ip end behavior
- No disruption metadata in FAIR-MAST zarr files
- 83 expert-labeled disrupted is the hard limit
- GRU self-labeling has 49% FP rate → unreliable

## Bottleneck
- **Detection CI: ±19%** (only 28 test disrupted)
- FA CI: ±2% (sufficient with 726+ clean test)
- Need: 225+ total disrupted (±8% CI) → DIII-D access

## Comparison to Literature
| Model | Tokamak | Shots | Det | FA | AUC |
|-------|---------|-------|-----|-----|-----|
| FusionMind | MAST | 2405 | 64% | 9-16% | 0.842 |
| FRNN | DIII-D | 20,000+ | 87% | 5% | ~0.97 |

## To Continue Downloading
```bash
python scripts/download_mast_level2.py --target 5000
```
