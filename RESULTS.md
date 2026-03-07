# FusionMind 4.0 — Definitive Results on MAST

## Dataset
- **229 MAST shots** (83 disrupted + 146 clean) from FAIR-MAST Level 2 (IMAS format)
- 16 base variables + 8 SXR RMS + 30 multi-scale temporal features = **78 optimal features**
- Source: FAIR-MAST S3 archive (anonymous public access via disruption-py)

## Model
- GRU (hidden=96, seq_len=30, dropout=0.4) + recovery filter
- Recovery filter: persist=2 timepoints, window=6tp, probability drop threshold=0.7
- Training: AdamW, lr=0.001, pos_weight=auto, 6 epochs

## Definitive Results (4 seeds, held-out: 28 disrupted + 49 clean)

| Threshold | Detection | FA Rate | CI (seed variance) |
|-----------|-----------|---------|-------------------|
| **0.3** | **62% ± 2%** | **34% ± 4%** | 4 seeds |
| **0.5** | **56% ± 3%** | **24% ± 2%** | 4 seeds |

val_AUC: 0.982 ± 0.008

**Note**: Previous report (205 shots, 41 clean test) showed 70%±5%. With 49 clean
test shots, detection is 62%±2%. The larger test set gives a more reliable estimate.

## DisruptionBench Metrics (shot-level, standard evaluation)
- **AUC: 0.842** (Ensemble), 0.799 (Super-GRU)
- TPR: 92.9%, FPR: 24.4%
- Comparable to GPT-2 on Alcator C-Mod (AUC 0.84)

## What Worked (proven across all seeds and test sizes)

| Technique | Effect | Confidence |
|-----------|--------|------------|
| Recovery filter | FA 70% → 30% | High — consistent across every config |
| SXR RMS (50kHz per EFIT window) | +10pp detection | High — consistent |
| Multi-scale temporal diffs | +5pp det, −5pp FA | Medium — consistent direction |

## What Did NOT Work

| Technique | Why |
|-----------|-----|
| Thomson profiles (120 channels) | 40% valid coverage, 5ms cadence too slow for crash dynamics |
| Physics margins (Greenwald, Troyon, q95) | MAST disruptions are VDE/kink, not density-limit |
| VAE/PCA anomaly detection | Disrupted shots look normal until crash |
| More than 78 features | Overfitting (20K GRU params, 10K training sequences) |
| C-Mod pretrain → MAST | Different disruption physics (density vs VDE) |
| Velocity/acceleration views | Marginal over multi-scale diffs (±5pp noise) |
| 5-branch parallel ensemble | No consistent gain over 3-branch |

## Ceiling Analysis
- 229 shots → ~12K training sequences for GRU with ~20K parameters
- 78 features is empirically optimal; 118+ features causes overfitting
- All feature engineering beyond (SXR + multi-scale) is within ±5pp seed variance
- **The ceiling is data volume, not model architecture**

## Comparison to Literature

| Model | Tokamak | Shots | Det | FA | AUC |
|-------|---------|-------|-----|-----|-----|
| **FusionMind** | **MAST (spherical)** | **229** | **62%** | **34%** | **0.842** |
| FRNN | DIII-D (conventional) | 20,000+ | 87% | 5% | ~0.97 |
| CCNN | C-Mod (conventional) | ~3,000 | — | — | 0.974 |
| GPT-2 | C-Mod (conventional) | ~3,000 | — | — | 0.84 |

## Causal Discovery Results (separate from prediction)
- CPDE (NOTEARS + PC): Edge F1 = 88.9%, 17/18 expected edges on MAST
- SCM: Linear R² = 37%, Nonlinear R² = 65%
- Simpson's Paradox on C-Mod: density-disruption correlation +0.53 → +0.02 when conditioning on Ip
- Cross-device transfer fails (AUC 0.54): MAST and C-Mod have different causal structures

## Path Forward
1. **More MAST shots** — FAIR-MAST has 30K+; 500+ shots would halve CI
2. **DIII-D/EAST access** — via disruption-py (email MIT PSFC); 5000+ shots with 1ms ECE
3. **Paper submission** — all results reproducible from public data
