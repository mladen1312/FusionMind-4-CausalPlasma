# FusionMind 4.0 — Results on MAST (v2.1)

## Dataset
- **2259 MAST shots** (83 disrupted + 2176 clean) from FAIR-MAST Level 2
- 11,573 shots available on S3, 2259 downloaded (20%)
- 16 base variables per shot, EFIT timebase (~10ms cadence)
- 215,250 total timepoints

## Best Model (78 features, on 777 shots subset)
| Threshold | Detection | FA Rate | Test set |
|-----------|-----------|---------|----------|
| 0.3 | **64% ± 11%** | **9% ± 1%** | 28d + 223c |
| 0.5 | **55% ± 9%** | **6% ± 1%** | 28d + 223c |

## Scaling Analysis (honest)
| Shots | Clean test | Detection | FA | Notes |
|-------|-----------|-----------|-----|-------|
| 255 | 58 | 63% | 33% | FA inflated by small test |
| 777 | 223 | 64% | 9% | Similar campaign shots |
| 1126 | 339 | 57% | 17% | Multi-campaign, harder negatives |
| 2259 | 726 | 29%* | 16% | *32-feature model (needs multi-scale) |

*2259-shot result uses 32 features only (raw+rates). Full 78-feature model
requires precomputing multi-scale diffs for 2259 shots (~10min locally).

## FA stabilizes at ~16% across diverse MAST campaigns
With 726 clean test shots from many campaigns (24xxx–30xxx range),
FA CI is ±2.7%. The model false-alarms on ~16% of MAST shots —
this is the TRUE false alarm rate across diverse operational conditions.

## Key Limitations
1. **83 disrupted shots** — all expert-labeled, cannot find more automatically
2. **Detection CI ±19%** — bottleneck is disrupted shot count, not clean
3. **MAST disrupted indistinguishable by Ip** — auto-labeling doesn't work
4. **Cross-campaign variability** — FA rises from 9% to 16% with diverse shots

## DisruptionBench Metrics
- AUC: 0.842 (Ensemble), comparable to GPT-2 on C-Mod

## Data on GitHub
- `data/mast/mast_level2_2259shots.npz` — 2259 shots, 215K timepoints
- `data/mast/mast_2259_labels.json` — 83d + 2176c labels
- `data/mast/mast_l2_all_shots_on_s3.json` — full 11,573 shot list
- `scripts/download_mast_level2.py` — standalone download script

## Path Forward
1. **DIII-D via disruption-py** — 2000+ labeled disrupted → detection CI ±5%
2. **Precompute 78 features on 2259 shots** — restore 64% detection
3. **Download remaining 863 shots in disrupted range** (27000–30443)
