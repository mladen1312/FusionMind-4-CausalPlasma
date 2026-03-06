# Errata: FusionMind Paper v1 → v2

All corrections based on independent review by Grok, Claude, and Perplexity (March 2026).

## Critical Corrections

### 1. Reference [1] — WRONG: "DeepMind/CFS (TORAX-RL)"
- **Error:** TORAX is a differentiable plasma simulator, not an RL controller. CFS was not a partner.
- **Correction:** "DeepMind RL controller for TCV" (Degrave et al., Nature 602, 2022). Collaboration was with EPFL.

### 2. Reference [2] — WRONG: "KSTAR LSTM-based predictor"
- **Error:** Kates-Harbeck et al. (Nature 568, 2019) describes FRNN on DIII-D/JET, not KSTAR LSTM.
- **Correction:** Separated FRNN [2] from KSTAR work [4] (Zheng et al., 2023).

### 3. Reference [4] — WRONG: "arXiv:2403.xxxxx (2024)"
- **Error:** Placeholder arXiv number, wrong year.
- **Correction:** TokaMind is arXiv:2602.15084 (2026), by IBM Research Europe and UKAEA.

### 4. Table 1 — WRONG: βt listed as "Root node" with R²=99%
- **Error:** Paper v1 described βt as root node but reported R²=99%. This is a contradiction.
- **Actual finding:** In our DAG, βt is a LEAF node with parents (βN, Wstored, Prad). R²=99% is correct for a leaf with strong parents. Paper v1 had the DAG description wrong.
- **Note added:** βt as leaf (not root) reflects statistical signal direction in MAST data, which may reverse the physical causation direction.

### 5. AUC = 1.000 — MISLEADING
- **Error:** Reported as "disruption detection" AUC, implying detection of actual disruptions.
- **Correction:** Explicitly labeled as "disruption risk PROXY detection." Labels are physics-based thresholds (percentiles), not actual disruption events. Added leave-one-discharge-out validation (min AUC = 0.989). Added caveat that this is an upper bound and real disruption prediction performance is unknown.

## Significant Corrections

### 6. "First application" claim — OVERSTATED
- **Error:** "No prior work has applied this framework to real tokamak plasma data."
- **Correction:** Acknowledged Murari et al. (transfer entropy on JET), Rossi et al. (EUROfusion causality project). Qualified claim to "first system implementing the full Pearl do-calculus framework with interventional and counterfactual inference."

### 7. IAEA safety standards — OVERSTATED
- **Error:** Implied IAEA has binding AI requirements for fusion.
- **Correction:** IAEA develops recommendations, not binding standards, for AI in fusion specifically. Referenced EU AI Act Article 13 instead.

### 8. C++ latency explanation — MISSING
- **Error:** Claimed sub-microsecond for "do-calculus + GradientBoosting inference" without explaining that C++ uses LINEAR SCM, not GBM.
- **Correction:** Added explicit note that C++ engine uses linear SCM for real-time inference. GBM used offline for model fitting and CV evaluation. Added CPU specification (AMD EPYC 7763).

### 9. Shot count — AMBIGUOUS
- **Error:** "shots 27000–27014 and 30400–30450" implies 66 shots but only 44 used.
- **Correction:** Explained that remaining shots were excluded due to missing diagnostics, failed equilibrium reconstruction, or insufficient valid data.

### 10. li R² = 0% (linear) — UNEXPLAINED
- **Correction:** li has one parent (Ip) in the DAG. Cross-validated GBM R² is actually negative (−15.0% ± 30.6%), indicating overfitting on this weak relationship. Fitted R² of 73.3% is reported with caution.

## Minor Corrections

### 11. βN = f(βt, Ip) — OVERSIMPLIFIED
- **Correction:** Stated full relation βN = βt · aB₀/Ip where a is minor radius and B₀ is toroidal field.

### 12. Intervention accuracy 76.9% — UNDEREXPLAINED
- **Correction:** Added discussion of three contributing factors: linear SCM backbone limitation, non-monotonic relationships, and confounding in observational test data.

### 13. Simpson's Paradox data source — IMPRECISE
- **Correction:** Referenced as "Alcator C-Mod disruption database from MIT PSFC" with citation to Greenwald density limit review.

### 14. Added references
- [9] Murari et al. (2020) — transfer entropy on JET
- [10] Rossi et al. (2023) — EUROfusion causality detection
- [11] EU AI Act (2024)
- [12] Pau et al. (2019) — ML for disruption warnings
- [14] FAIR-MAST archive
- [15] Greenwald density limits review

## Acknowledgment

Paper v1 was AI-assisted (generated with Claude). All corrections, validations, and scientific content verified by Dr. Mladen Mester. The software, benchmarks, and validation results are real and reproducible (297 tests passing, open-source at github.com/mladen1312/FusionMind-4-CausalPlasma).
