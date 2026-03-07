# FusionMind 4.0 — Definitive Results on MAST

## Dataset
- **565 MAST shots** (83 disrupted + 456 clean + 26 uncertain) from FAIR-MAST Level 2
- Original 255 shots with expert labels (disruption-py) + 310 auto-labeled clean shots
- Auto-labeling: Ip rampdown shape (>12 EFIT steps of gradual decay = clean)
- 11,573 Level 2 shots available on FAIR-MAST S3; sampled 600, downloaded 565
- 16 base variables + 8 SXR RMS + 30 multi-scale temporal features = **78 optimal features**

## Model
- GRU (hidden=96, seq_len=30, dropout=0.4) + recovery filter
- Recovery filter: persist=2 timepoints, window=6tp, probability drop threshold=0.7
- Training: AdamW, lr=0.001, pos_weight=auto, 6 epochs, batch=512

## Definitive Results (3 seeds, held-out: 28 disrupted + 152 clean)

| Threshold | Detection | FA Rate | CI |
|-----------|-----------|---------|-----|
| **0.3** | **65% ± 4%** | **22% ± 2%** | 3 seeds |
| **0.5** | **51% ± 2%** | **15% ± 1%** | 3 seeds |

val_AUC: 0.987 ± 0.007

## Scaling Effect (more clean shots = lower FA)

| Dataset | Test split | Detection | FA Rate |
|---------|-----------|-----------|---------|
| 255 shots (28d+58c) | 28d+58c | 63% ± 2% | 34% ± 4% |
| **565 shots (28d+152c)** | **28d+152c** | **65% ± 4%** | **22% ± 2%** |

Adding 310 clean shots reduced FA from 34% to 22% (−12pp) while keeping detection stable.
The GRU learns a better decision boundary with more negative examples.

## DisruptionBench Metrics (shot-level)
- **AUC: 0.842** (Ensemble), 0.799 (Super-GRU)
- TPR: 92.9%, FPR: 24.4%

## What Worked

| Technique | Effect | Confidence |
|-----------|--------|------------|
| Recovery filter | FA 70% → 22% | High |
| SXR RMS (50kHz) | +10pp detection | High |
| Multi-scale temporal diffs | +5pp det, −5pp FA | Medium |
| **More clean shots (310→456)** | **FA −12pp** | **High** |

## What Did NOT Work

Thomson profiles, physics margins, VAE/PCA anomaly, SDS (CUSUM+accel),
TTD+uncertainty regression, >78 features, C-Mod pretrain, UltraView,
5-branch ensemble — all within ±5pp noise on MAST.

## Ceiling and Path Forward

Detection is capped at ~65% because we only have **83 disrupted shots**.
More clean shots reduce FA but cannot improve detection.
Next steps:
1. **More disrupted shots** — DIII-D via disruption-py (5000+, ~20% disrupted)
2. **More MAST disrupted** — FAIR-MAST has 11K+ shots; mining for disruptions
3. **1ms ECE profiles** — needed for >80% detection on any tokamak

## Causal Discovery (separate from prediction)
- CPDE NOTEARS: F1 = 88.9% edge recovery (331 shots)
- SCM: Linear R² = 37%, Nonlinear R² = 65%
- Simpson's Paradox on C-Mod: density-disruption +0.53 → +0.02 | Ip
