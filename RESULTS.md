# FusionMind 4.0 — Definitive Results on MAST

## Dataset
- **205 MAST shots** (83 disrupted + 122 clean) from FAIR-MAST Level 2
- **16 base variables**: βN, βp, βt, q95, κ, li, Wmhd, q_axis, a, δ_lower, δ_upper, Ip, ne_line, f_GW, p_rad, P_NBI
- **SXR RMS**: 18 channels at 50kHz, RMS per EFIT window (8 features)
- **Multi-scale diffs**: 6 key signals × 5 temporal views (30 features)
- **Total optimal features**: 78

## Model
- GRU (hidden=96, seq_len=30, dropout=0.4)
- Recovery filter (persist=2, window=6 timepoints, drop_threshold=0.7)
- Training: AdamW, lr=0.001, batch=512, 6 epochs, BCEWithLogits(pos_weight=auto)

## Results (6 random seeds, held-out: 28 disrupted + 41 clean)

| Threshold | Detection | FA Rate | Warning Time |
|-----------|-----------|---------|--------------|
| 0.3 | **70% ± 5%** | **34% ± 3%** | ~140ms |
| 0.5 | **62% ± 3%** | **30% ± 2%** | ~130ms |

Ranges: Detection [64%–79%], FA [27%–39%], val_AUC 0.978 ± 0.010

## DisruptionBench Metrics (shot-level)
- **AUC: 0.842** (Ensemble), 0.799 (Super-GRU)
- TPR: 92.9%, FPR: 24.4% (any-alarm-in-shot criterion)
- Comparable to GPT-2 on C-Mod (AUC 0.84) despite MAST being harder

## What Worked (proven across all seeds)

| Technique | Contribution | Evidence |
|-----------|-------------|----------|
| Recovery filter | FA 70% → 30% | Consistent across all configs |
| SXR RMS (50kHz) | +10pp detection | Consistent across all configs |
| Multi-scale diff (sm3−sm7, sm7−sm15) | +5pp det, −5pp FA | Consistent |

## What Did NOT Work (within noise ±5pp)

| Technique | Result | Why |
|-----------|--------|-----|
| Thomson profiles (120ch) | ≈0 | 40% coverage, 5ms cadence too slow |
| Physics margins (Greenwald, Troyon) | ≈0 | MAST not near density limit |
| VAE/PCA anomaly detection | ≈0 | Disrupted ≈ clean reconstruction |
| >78 features (any combination) | Overfitting | 20K params / 10K samples |
| C-Mod pretrain → MAST fine-tune | ±5pp | Different disruption physics |
| 5-branch parallel ensemble | ±5pp | Redundant with multi-scale |
| UltraView (velocity, acceleration) | ±5pp | Marginal over multi-scale diff |

## Ceiling Analysis

With 205 shots (~10K training sequences) and GRU(hid=96, ~20K parameters):
- **78 features** is the optimal complexity (proven by progressive branch addition)
- More features → overfitting (118 feat: detection drops 7pp)
- More branches → diminishing returns (5 branches worse than 3)
- **The ceiling is the data, not the model architecture**

## Comparison to Literature

| Model | Tokamak | Shots | Detection | FA | AUC |
|-------|---------|-------|-----------|-----|-----|
| **FusionMind (ours)** | **MAST** | **205** | **70%** | **34%** | **0.842** |
| FRNN | DIII-D | 20,000+ | 87% | 5% | ~0.97 |
| CCNN (DisruptionBench) | C-Mod | ~3,000 | — | — | 0.974 |
| GPT-2 (DisruptionBench) | C-Mod | ~3,000 | — | — | 0.84 |

## Path Forward
1. **DIII-D access** (via disruption-py) — 5000+ shots with 1ms ECE profiles
2. **500+ MAST shots** — FAIR-MAST has 30K+ available
3. **1D profile diagnostics** at fast cadence (not Thomson at 5ms)

## Causal Discovery (Separate from Prediction)
- NOTEARS DAG: F1 = 88.9% edge recovery (331 shots, leave-shots-out)
- SCM: Linear R² = 37%, Nonlinear R² = 65%
- Simpson's Paradox detected on C-Mod density-disruption correlation
- Zero-shot cross-device transfer fails (AUC 0.54) — different causal structures
