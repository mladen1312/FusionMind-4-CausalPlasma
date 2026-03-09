# FusionMind 4.0 — MAST Results (v3.2)

## Dataset
- **2636 MAST shots** (496 disrupted + 2140 clean) from FAIR-MAST Level 2
- 714 shots with millisecond-precision disruption times from ops comments
- 16 variables, EFIT timebase, 246K timepoints

## Data Sources
- FAIR-MAST S3 Level 2 (zarr format, public anonymous access)
- FAIR-MAST GraphQL API (mastapp.site/graphql) — 15,969 operator comments
- disruption-py expert labels (83 shots) + ops-log text mining (1191 shots)

## Results

### Expert-Only (homogeneous H-mode disruptions)
| Config | Detection | FA | Test set | Notes |
|--------|-----------|-----|----------|-------|
| 78 feat, 777 shots | **64%** | **9%** | 28d+223c | Same campaign, same type |
| DisruptionBench AUC | | | | **0.842** |

### All Disruptions (diverse: VDE, locked mode, density limit, FA trip)
| Config | Detection | FA | Test set | Notes |
|--------|-----------|-----|----------|-------|
| 78 feat, 27K+ range | **29%** | **15%** | 73d+409c | Multi-scale features |
| 32 feat, all 2584 | 18% | 30% | 148d+714c | Raw features only |

### Key Finding: Two Disruption Populations
Expert-labeled (disruption-py): homogeneous H-mode crashes, βN=0.91, 126tp
Ops-log labeled: diverse types (VDE, locked mode, etc.), βN=0.49, 71tp
A model trained on one population performs poorly on the other.

## Unique Assets
1. **714 disruption times** with ms precision (parsed from operator comments)
2. **15,969 operator comments** for all MAST shots
3. **1,274 disrupted shots identified** (1,224 on Level 2 S3)
4. **Causal discovery**: NOTEARS F1=88.9%, SCM R²=65%

## Path Forward
1. Disruption-type-stratified models (VDE vs locked mode vs density limit)
2. Download remaining 680 disrupted from S3
3. Multi-task learning with disruption type as auxiliary target
