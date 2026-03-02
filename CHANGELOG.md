# Changelog

All notable changes to FusionMind 4.0 are documented here.

## [4.1.0] — 2026-03-02

### Added
- **CausalShield-RL (PF7)**: Complete causal RL hybrid agent
  - Neural SCM world model with online learning (`fusionmind4/learning/neural_scm.py`)
  - Gym-compatible plasma environment (`fusionmind4/learning/gym_plasma_env.py`)
  - Causal reward shaping (`fusionmind4/learning/causal_reward.py`)
  - PPO policy with causal constraints (`fusionmind4/learning/causal_rl_hybrid.py`)
- 33 new tests for learning module (total: 56 tests)
- CausalShield-RL dashboard (`dashboards/FM4_CausalShieldRL_Dashboard.jsx`)
- `scripts/train_causal_rl.py` — full training pipeline
- Comprehensive documentation (`docs/`)
- `CITATION.cff` for academic citation
- `CONTRIBUTING.md`
- GitHub Actions CI workflow

### Changed
- Updated README with real data results, architecture diagram, competitive table
- Updated `pyproject.toml` with proper classifiers and URLs
- Expanded `requirements.txt` with networkx dependency

## [4.0.2] — 2026-03-02

### Added
- **FAIR-MAST real data validation**: CPDE v3.2 on 8 MAST shots
  - F1 = 91.9%, precision 89.5%, recall 94.4%
  - 100% sign accuracy on all discovered edges
  - Cross-shot robustness: F1 = 88.2% ± 4.4%
- D3R diffusion reconstruction validation (156:1 compression)
- `scripts/run_fair_mast.py` — MAST data access and validation
- Investor integration dashboard (`dashboards/FM4_Investor_Dashboard.jsx`)
- Competitive analysis dashboard (`dashboards/FM4_Competitive_Analysis.jsx`)

## [4.0.1] — 2026-03-01

### Added
- **Alcator C-Mod real data validation**: 264K timepoints, 2333 shots
  - Density limit AUC = 0.974 (vs Greenwald 0.946)
  - Simpson's Paradox detected (ρ: +0.53 → +0.02)
- D3R diffusion proof-of-concept (130:1 compression on C-Mod geometry)
- UPFM cross-device validation (6 tokamaks, CV = 0.267)
- `scripts/run_real_data.py` — Alcator C-Mod pipeline

## [4.0.0] — 2026-02-28

### Added
- **CPDE v3.2**: 9-step ensemble causal discovery pipeline
  - NOTEARS + DYNOTEARS + Granger + PC + physics priors
  - Bayesian fusion with triple-tier thresholding
  - F1 = 79.2% on FM3-Lite (14 variables, 23 edges)
  - 8/8 PINN physics checks
- **CPC v1.0**: Structural causal model with do-calculus
  - Interventional queries validated
  - Counterfactual reasoning
  - Causal path tracing
- **UPFM**: Dimensionless tokenization (βn, ν*, ρ*, q95, H98)
- **D3R**: Conditional denoising diffusion for 3D reconstruction
- **AEDE**: Active experiment design with bootstrap uncertainty
- FM3-Lite physics simulation engine
- 23 tests (all passing)
- Quickstart example

## [3.0.0] — 2025-08-01

### Foundation
- FusionMind 3.0 monolithic implementation
- Advanced plasma physics simulator
- Transformer + GNN architecture
- Safety wrapper system
- Training pipeline with mixed precision
