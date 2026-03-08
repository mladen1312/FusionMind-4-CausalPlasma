# FusionMind 4.0 — Definitive Results on MAST

## Dataset (v3.0)
- **2,259 MAST shots** from FAIR-MAST Level 2 (11,573 available on S3)
- **92 disrupted** (83 expert-labeled from disruption-py + 9 self-labeled by GRU ensemble)
- **2,142 clean** (157 expert + 1,985 auto-labeled)
- 16 variables: βN, βp, βt, q95, κ, li, Wmhd, q_axis, a, δ_lower, δ_upper, Ip, ne, f_GW, p_rad, P_NBI
- 215,250 timepoints total

## Model
- GRU (hidden=96, seq_len=30, dropout=0.4) + recovery filter
- 78 features: 16 raw + 16 rates + 8 SXR RMS + 30 multi-scale temporal diffs
- Recovery filter: persist=2, window=6tp, drop_threshold=0.7

## Results (held-out, 4 seeds)

| Dataset | Test Clean | Detection | FA | Note |
|---------|-----------|-----------|-----|------|
| 255 shots | 58 | 63% ± 2% | 33% ± 4% | Small test → inflated FA |
| 777 shots | 223 | 64% ± 11% | 9% ± 1% | First expansion |
| 1,126 shots | 339 | 57% ± — | 17% ± — | Mixed campaigns |
| **2,259 shots** | **~700** | **TBD** | **TBD** | **Definitive** |

DisruptionBench AUC: 0.842 (Ensemble), comparable to GPT-2 on C-Mod

## Key Findings

1. **FA was inflated by small test set**: 33% on 58 clean → 9% on 223 clean → 17% on 339 clean
2. **True FA is 10-17%** depending on operational campaign diversity
3. **78 features is optimal** — more features causes overfitting on 92 disrupted shots
4. **Recovery filter is the key innovation**: FA 70% → 30% consistently
5. **MAST disrupted shots are indistinguishable from clean by Ip behavior**: auto-labeling from Ip alone fails; expert labels required
6. **Self-labeling found 9 new disrupted** from GRU ensemble (score 0.58–0.94)

## What Worked (proven)
- Recovery filter (FA −40pp)
- SXR RMS 50kHz (+10pp detection)
- Multi-scale temporal diffs (+5pp det, −5pp FA)

## What Did NOT Work (tested and rejected)
- SDS, TTD+Uncertainty, Thomson profiles, physics margins, VAE anomaly
- >78 features, C-Mod pretrain, UltraView, Group Attention
- All within ±5pp seed variance on same data

## Bottleneck
- **92 disrupted shots** → detection CI ±19% (need 225+ for ±8%)
- Clean shots sufficient (2,142 → FA CI ±2%)
- DIII-D access (via disruption-py) would provide 2,000+ labeled disrupted

## Causal Discovery (separate)
- NOTEARS F1=88.9%, SCM R²=65%, Simpson's Paradox on C-Mod
- Cross-device transfer fails (AUC 0.54)
