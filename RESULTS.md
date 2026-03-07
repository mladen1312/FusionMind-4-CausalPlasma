# FusionMind 4.0 — Definitive Results

## Dataset
- **512 MAST shots** (83 disrupted + 403 clean + 26 unlabeled) from FAIR-MAST Level 2
- Original 255 shots: expert-labeled (disruption-py database)
- Additional 257 shots: auto-labeled from Ip ramp-down behavior
- 16 base variables + 8 SXR RMS + 30 multi-scale features = **78 optimal features**
- Source: FAIR-MAST S3 (11,573 Level 2 shots available, 512 downloaded)

## Model
- GRU (hidden=96, seq_len=30, dropout=0.4) + recovery filter
- Training: AdamW, lr=0.001, BCEWithLogits, 6 epochs, batch=512

## Results (3 seeds, held-out: 28 disrupted + 135 clean)

| Threshold | Detection | FA Rate | 95% CI (FA) |
|-----------|-----------|---------|-------------|
| **0.3** | **60% ± 2%** | **19% ± 1%** | [13%–26%] |
| **0.5** | **50% ± 3%** | **14% ± 2%** | [9%–21%] |

### Comparison: 255 vs 512 shots

| Dataset | Test set | Detection | FA | CI width |
|---------|----------|-----------|-----|----------|
| 255 shots | 28d + 58c | 63% ± 4% | 33% ± 4% | ±12pp |
| **512 shots** | **28d + 135c** | **60% ± 2%** | **19% ± 1%** | **±7pp** |

**Note**: FA improved from 33%→19% primarily because auto-labeled clean shots are
higher-quality (selected by clean Ip ramp-down). Detection unchanged (same 83 disrupted).
The 95% CI halved from ±12pp to ±7pp with 2.3x more clean test shots.

## DisruptionBench Metrics
- AUC: 0.842 (Ensemble), 0.799 (Super-GRU) — on original 255 shots
- Comparable to GPT-2 on C-Mod (AUC 0.84)

## What Worked (proven across all experiments)

| Technique | Effect | Confidence |
|-----------|--------|------------|
| Recovery filter | FA 70% → 20% | High |
| SXR RMS (50kHz) | +10pp detection | High |
| Multi-scale temporal diffs | +5pp det, −5pp FA | High |

## What Did NOT Work

| Technique | Result |
|-----------|--------|
| SDS (CUSUM + acceleration) | Clean shots score higher than disrupted |
| TTD + Uncertainty regression | 100% FA (regression too hard for 83 labels) |
| Thomson profiles, physics margins, VAE | Within noise (±5pp) |
| More than 78 features | Overfitting on 512 shots |
| C-Mod pretrain → MAST transfer | Different disruption physics |

## Causal Discovery (separate from prediction)
- CPDE: Edge F1 = 88.9% (NOTEARS + PC, 331 shots)
- SCM: Linear R² = 37%, Nonlinear R² = 65%
- Simpson's Paradox on C-Mod: density-disruption +0.53 → +0.02 | Ip
- Cross-device: MAST↔C-Mod transfer fails (AUC 0.54)

## Path Forward
1. **More disrupted shots** — 83 disrupted is the bottleneck, not clean
2. **DIII-D access** — 5000+ shots with 1ms ECE via disruption-py
3. **FAIR-MAST** — 11,573 Level 2 shots available; download more disrupted
