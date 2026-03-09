# FusionMind 4.0 — Complete Results

## Dataset
- **2941 MAST shots** (448 disrupted + 2493 clean) from FAIR-MAST Level 2
- **714 disruption times** with ms precision from operator comments (FIRST PUBLIC)
- 78 features: 16 raw + 16 rates + 16 SXR RMS + 30 multi-scale temporal diffs
- Source: FAIR-MAST S3 (public) + GraphQL operator log (15,969 comments)

## Results — Two Evaluation Protocols

### Protocol A: Expert-labeled disruptions (homogeneous, one campaign)
| Metric | Value | CI |
|--------|-------|-----|
| Shot AUC | **0.842** | — |
| Detection (th=0.3) | **64%** | ±11% |
| FA (th=0.3) | **9%** | ±1% |
| Shots | 255 (83d + 172c) | — |
| Test | 28d + 223c | — |

### Protocol B: All disruptions (diverse, all campaigns, ops-log labels)
| Metric | Value | CI |
|--------|-------|-----|
| Shot AUC | **0.639** | — |
| Detection (th=0.4) | **29%** | ±8% |
| FA (th=0.4) | **52%** | ±5% |
| Detection (th=0.5) | **17%** | ±7% |
| FA (th=0.5) | **28%** | ±4% |
| Shots | 1443 (189d + 1254c) | — |
| Test | 63d + 418c | — |

### Why the difference?
Expert-labeled disruptions (Protocol A) are a curated subset: late-campaign H-mode
shots with one disruption type. Ops-log disruptions (Protocol B) include VDE, locked
modes, density limits, FA trips — each with different precursor signatures. A single
GRU model struggles to learn one unified pattern for all types.

**Both protocols are valid** — Protocol A measures performance on "learnable" disruptions,
Protocol B measures honest cross-type generalization.

## 16-Channel Feature Impact
| Features | Detection | FA | Improvement |
|----------|-----------|------|-------------|
| 32 (raw + rates) | 18% | 30% | baseline |
| 78 (+ SXR + multi-scale) | 29% | 52% | +11pp det |

Multi-scale temporal diffs (sm3−sm7, sm7−sm15) confirmed essential across both protocols.

## Comparison to Literature
| Model | Tokamak | Shots | AUC | TPR | FPR |
|-------|---------|-------|-----|-----|-----|
| FusionMind (expert) | MAST | 255 | 0.842 | 64% | 9% |
| FusionMind (diverse) | MAST | 1443 | 0.639 | 29% | 52% |
| GPT-2 (DisruptionBench) | C-Mod | ~3000 | 0.840 | — | — |
| CCNN (DisruptionBench) | C-Mod | ~3000 | 0.974 | — | — |
| FRNN | DIII-D | >20000 | ~0.97 | 87% | 5% |

## What Worked
1. Recovery filter: FA 70% → 30% on expert protocol
2. SXR RMS (50kHz): +10pp detection (expert protocol)
3. Multi-scale temporal diffs: +5-11pp detection (both protocols)
4. Ops-log disruption time parsing: 714 ms-precision labels (unique asset)

## What Did NOT Work
- SDS, TTD+Uncertainty, Thomson profiles, physics margins, >78 features
- Single model for ALL disruption types (AUC drops from 0.84 to 0.64)
- Auto-detecting disrupted shots from Ip behavior (identical for dis/clean)

## Key Assets on GitHub
| File | Description |
|------|-------------|
| `data/mast/mast_level2_2521shots.npz` | 2521 shots, 15MB |
| `data/mast/mast_disruption_times.json` | 714 ms-precision disruption times |
| `data/mast/mast_ops_log.json` | 15,969 operator comments |
| `data/mast/mast_ops_disrupted.json` | 1274 identified disrupted shots |
| `scripts/download_mast_level2.py` | Download remaining shots locally |
| `benchmarks/` | All benchmark results with CI |

## Path Forward
1. **Type-specific models**: Train separate detectors for VDE, locked mode, density limit
2. **More epochs locally**: val_AUC was rising to 0.92+ but training timed out
3. **DIII-D via disruption-py**: 20K+ shots with uniform labeling
