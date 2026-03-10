# FusionMind 4.0 — Causal AI for Tokamak Disruption Prediction

**The first disruption prediction system built on Pearl's causal inference framework.**

While all existing fusion AI (CCNN, FRNN, GPT-2, Random Forest) learns *correlations* between plasma signals and disruptions, FusionMind discovers the *causal mechanism* — which variable actually **causes** the disruption on each specific machine — then builds a predictor focused on that mechanism.

```
  MAST (spherical tokamak):  AUC = 0.979 ± 0.011  (5-fold CV, 83 disrupted, verified)
  C-Mod (density-limit):     AUC = 0.978           (physics formula, 0 parameters)
  Causal discovery F1:       88.9%                  (17/18 edges on real MAST data)
```

**Author:** Dr. Mladen Mester, Zagreb, Croatia  
**Repo:** https://github.com/mladen1312/FusionMind-4-CausalPlasma

---

## Why FusionMind Exists

Every disruption predictor today — CCNN, FRNN, Random Forest, GPT-2 — treats disruption prediction as pattern matching: feed N signals into a model, get a probability out. None of them ask **why** the disruption happens. None of them identify which signal **causes** the disruption on this specific machine. And none of them can answer "what intervention would have prevented it?"

FusionMind changes this with three capabilities no competitor has:

1. **Causal Discovery**: CPDE identifies the causal DAG from data → finds that `li` drives disruptions on MAST, `f_GW` drives them on C-Mod
2. **Counterfactual Reasoning**: SCM answers "what if we had reduced heating power by 10%?" → P(disruption) drops from 0.87 to 0.23
3. **Physics-Informed Prediction**: stability margins (distance to each limit) give a 0-parameter predictor that matches ML

---

## Our Advantages

### 1. Knowing the CAUSE Beats Fitting Correlations

CPDE identifies the **one signal** that matters most on each machine:

| Machine | Primary Driver | Single-Variable AUC | 78-Feature GRU AUC |
|---------|---------------|--------------------|--------------------|
| MAST (spherical) | **li** (internal inductance) | 0.908 | 0.842 |
| C-Mod (conventional) | **f_GW** (Greenwald fraction) | 0.985 | 0.894 |

One physics variable, zero parameters, no training — and it beats a 78-feature neural network. This is the strongest validation of the causal approach.

### 2. Stability Margins Normalize Across Disruption Types

Instead of raw signal values, we compute distance-to-limit for each mechanism:

```
margin_li   = 1 - max(li) / li_limit       → 0.05 means 5% from kink limit
margin_q95  = 1 - q95_crit / min(q95)      → 0.10 means 10% from q-limit
margin_βN   = 1 - max(βN) / βN_limit       → 0.70 means far from beta limit
```

This puts density-limit, kink, beta-limit, and radiation-collapse disruptions on the **same [0,1] scale**. The model learns "approaching ANY limit is dangerous" instead of memorizing separate patterns.

### 3. Six Parallel Analysis Tracks

Each track answers a different question about the plasma:

| Track | Question | Features | Verified AUC |
|-------|----------|----------|-------------|
| **A: Physics Margins** | How close to each stability limit? | 9 | 0.905 |
| **B: Shot Statistics** | What do aggregate signal stats show? | 63 | **0.979** |
| **C: Trajectory** | How does the plasma evolve over time? | 32 | ~0.94 |
| **D: Causal Driver** | What does the PRIMARY cause signal show? | 12 | 0.950 |
| **E: Rate Extremes** | Are there sudden changes (precursors)? | 40 | ~0.93 |
| **F: Pairwise** | Which signal PAIRS indicate instability? | 15 | ~0.91 |

Tracks auto-activate based on available signals. Meta-learner combines them.

### 4. Auto-Configuration Per Machine

```python
predictor = CausalDisruptionPredictor.from_data(
    data, shot_ids, variables, disrupted_set,
    machine_type=MachineType.SPHERICAL
)
results = predictor.evaluate_cv(n_folds=5)
```

- `SIGNAL_ALIASES` maps 16 canonical names to machine-specific variable names
- `StabilityLimits` adjusts per machine type automatically
- Tracks degrade gracefully when signals are missing (C-Mod: 6 vars → 4 tracks active)

### 5. Every Prediction Comes with "WHY"

```python
prob, explanation = predictor.predict_shot(signals)
# explanation = {
#   "closest_limit": "li",
#   "min_margin": 0.05,
#   "primary_driver": "li",
#   "margins": {"li": 0.05, "q95": 0.42, "betan": 0.71, "fGW": 0.83},
#   "n_stressed": 1,
# }
```

For ITER regulatory compliance: "disruption predicted because internal inductance reached 95% of kink stability limit."

---

## Verified Results

Reproducible: `python scripts/reproduce_all_results.py` (seed=42, ~40s)

### Progression on MAST (same data, each step verified)

| Step | AUC | Δ over v1.1 | Method |
|------|-----|-------------|--------|
| v1.1 GRU (78 timepoint feat) | 0.842 | baseline | Per-timepoint neural network |
| Physics: max(li) | 0.908 | +0.066 | 0 parameters, deterministic |
| GBT 8 physics stats | 0.918 | +0.076 | max(li), min(q95), max(βN)... |
| GBT 40 features | 0.961 | +0.119 | All signal stats + margins |
| **GBT 63f + margins + augmentation** | **0.979** | **+0.137** | + interactions + temporal shape + 4× aug |

### Comparison with Literature

| Model | Machine | AUC | Causal? | Interpretable? |
|-------|---------|-----|---------|----------------|
| **FusionMind GBT+margins** | **MAST** | **0.979** | **Yes** | **Yes** |
| CCNN (Spangher 2025) | C-Mod | 0.974 | No | No |
| FRNN (Kates-Harbeck 2019) | DIII-D | ~0.97 | No | Sensitivity only |
| HDL (Zhu 2020) | Multi | 0.920 | No | No |
| GPT-2 (Spangher 2025) | C-Mod | 0.840 | No | Attention maps |
| RF (Rea 2018) | C-Mod | 0.832 | No | Feature importance |

**⚠️ Numbers are NOT directly comparable** — different machines, different datasets, different disruption types. See `VERIFICATION.md`.

### What We Can Fairly Claim

- **First causal disruption prediction system** — no other system uses Pearl's do-calculus for fusion
- **Physics (0 params) matches ML** — single causal variable competitive with multi-feature neural networks
- **CPDE finds the right mechanism** — li for spherical tokamaks, f_GW for conventional, confirmed by physics
- **+0.137 AUC improvement** on MAST through physics-informed feature engineering (same data, v1.1 → v4.0)
- **Simpson's Paradox detected** — density-disruption correlation drops from +0.53 to +0.02 on C-Mod when conditioning on Ip

### What We Cannot Claim (Yet)

- "We beat CCNN" — different machines, different data
- C-Mod AUC=0.978 is publishable — density-limit disruptions are trivially separable
- Results generalize to ITER — needs cross-device validation on DIII-D/JET

---

## Codebase: 28K Lines, 8 Patent Families

```
fusionmind4/
├── discovery/           PF1: CPDE — Causal Plasma Discovery Engine
│   ├── notears.py       NOTEARS + DYNOTEARS (augmented Lagrangian DAG constraint)
│   ├── pc.py            PC algorithm + Meek rules R1-R4, stable variant
│   ├── granger.py       Granger + spectral + conditional causality
│   ├── ensemble.py      EnsembleCPDE: 5-algorithm fusion + bootstrap CI
│   ├── nonlinear_scm.py GBT structural equations (R² = 96.7% on MAST)
│   └── physics.py       Physics-constrained priors (actuator exogeneity)
│
├── control/             PF2: CPC — Counterfactual Plasma Controller
│   ├── scm.py           Pearl's SCM + do-calculus + counterfactuals
│   ├── causal_controller.py  3-mode controller (wrapper/hybrid/advisor)
│   ├── stack.py         4-layer unified control stack (1030 lines)
│   ├── dynamic_overseer.py   Mimosa-style multi-track arbitrator
│   └── temporal_gru.py  GRU sequence predictor (v1.1 baseline: AUC=0.842)
│
├── predictor/           Unified Multi-Track Disruption Predictor
│   └── engine.py        CausalDisruptionPredictor — 6 tracks, auto-config
│
├── foundation/          PF3: UPFM — Universal Plasma Foundation Model
│   └── core.py          Dimensionless tokenization for cross-device transfer
│
├── reconstruction/      PF4: D3R — Diffusion 3D Plasma Reconstruction
│   └── core.py          MHD-constrained denoising diffusion (PoC)
│
├── experiment/          PF5: AEDE — Active Experiment Design Engine
│   └── aede.py          Bayesian optimal experiment selection
│
├── learning/            PF7: Causal RL Integration
│   ├── neural_scm.py    Differentiable SCM (NumPy, no PyTorch needed)
│   ├── causal_rl_hybrid.py  PPO + causal reward shaping
│   └── gym_plasma_env.py    OpenAI Gym plasma environment
│
├── realtime/            PF6/PF8: Real-Time Engine
│   ├── predictor.py     Dual-mode ML + causal predictor
│   ├── fast_bindings.py C++ AVX-512 bindings (0.27μs on synthetic)
│   └── control_bridge.py Real-time control interface
│
└── utils/               FM3-Lite simulator, plasma variable definitions
```

### Patent Families

| PF | Name | Novelty | Module |
|----|------|---------|--------|
| PF1 | CPDE — Causal Plasma Discovery Engine | 9/10 | discovery/ |
| PF2 | CPC — Counterfactual Plasma Controller | 10/10 | control/ |
| PF3 | UPFM — Universal Plasma Foundation Model | 8/10 | foundation/ |
| PF4 | D3R — Diffusion 3D Reconstruction | 7/10 | reconstruction/ |
| PF5 | AEDE — Active Experiment Design Engine | 8/10 | experiment/ |
| PF6 | Integrated System | 7/10 | realtime/ |
| PF7 | CausalShield-RL | 8/10 | learning/ |
| PF8 | LLM Copilot | 6/10 | copilot/ |

---

## Data (included in repo)

| Dataset | Shots | Disrupted | Variables | Source |
|---------|-------|-----------|-----------|--------|
| MAST Level 2 | 2941 | 83 expert-labeled | 16 EFIT equilibrium | FAIR-MAST S3 (public) |
| C-Mod Density Limit | 2333 | 78 | 6 equilibrium | MIT PSFC Open DB |
| MAST Ops Log | 15,969 entries | 1,274 disrupted identified | — | FAIR-MAST GraphQL |
| MAST Disruption Times | — | 714 with ms precision | — | Parsed from operator comments |

**Data sources are publicly verifiable:**
- MAST: `s3://mast/level1/shots/` via `https://s3.echo.stfc.ac.uk`
- MAST GraphQL: `https://mastapp.site/graphql`
- C-Mod: MIT PSFC disruption-py package

---

## Quick Start

### Reproduce all verified results
```bash
git clone https://github.com/mladen1312/FusionMind-4-CausalPlasma.git
cd FusionMind-4-CausalPlasma
pip install numpy scikit-learn scipy
python scripts/reproduce_all_results.py
```
Output: 6 tests, all verified, ~40 seconds. See `VERIFICATION.md` for details.

### Run the full predictor
```python
from fusionmind4.predictor.engine import run_mast, run_cmod

mast_results = run_mast()   # MAST: 6 tracks, auto-configured
cmod_results = run_cmod()   # C-Mod: 4 tracks (fewer signals)
```

### Run causal discovery
```python
from fusionmind4.discovery.ensemble import EnsembleCPDE

cpde = EnsembleCPDE()
result = cpde.discover(data, var_names=variable_names)
print(f"DAG F1: {result['metrics']['f1']:.1%}")
print(f"Edges: {result['edge_details']}")
```

### Run tests
```bash
python -m pytest tests/ -x -q
# 310 passed, 0 failed, 25 skipped
```

---

## Technical Details

### Why GBT on Shot-Level Features Beats GRU on Timepoints

The v1.1 GRU (AUC=0.842) processes per-timepoint windows. The GBT (AUC=0.979) processes per-shot aggregated statistics. Why the 0.137 AUC gap?

1. **MAST shots are short** (~100 timepoints). GRU needs long sequences to learn temporal context — MAST doesn't provide enough
2. **Shot-level stats capture the trajectory** — mean, std, max, late_mean, trend tell you whether li is rising, how high it gets, how variable it is. The GRU has to learn this from raw sequences
3. **Physics features encode domain knowledge** — stability margins, cross-variable interactions (li × βN, li/q95) give the model the right representation. The GRU must discover these patterns from data
4. **Augmentation compensates for small N** — 83 disrupted → 4× = 332 effective. The GRU can't easily augment sequences while preserving temporal structure

### Key Innovation: Stability Margins

Normalizing `1 - value/limit` instead of using raw values is the single most impactful feature engineering step. It transforms the problem from "learn N separate disruption patterns" to "learn one pattern: margin approaching zero."

### Counterfactual Augmentation

For each disrupted shot, we create 4 copies with 5% Gaussian noise on all features. This is justified as a counterfactual: "what if the same disruption happened with slightly different conditions?" It increased AUC from 0.971 to 0.979.

---

## Files for Verification

| File | Purpose |
|------|---------|
| `scripts/reproduce_all_results.py` | Reproduces all 6 claimed numbers from raw data |
| `VERIFICATION.md` | Step-by-step guide with expected outputs |
| `RESULTS.md` | Complete results with caveats |
| `ARCHITECTURE.md` | Full algorithm comparison |
| `benchmarks/best_model_mast_v2.json` | AUC=0.979 detailed results |
| `benchmarks/physics_vs_ml.json` | Physics vs ML comparison |
| `benchmarks/cmod_honest_assessment.json` | Why C-Mod ≠ DisruptionBench |

---

## References

- Spangher et al. (2025) "DisruptionBench and Complementary New Models" — J. Fusion Energy 44:26
- Kates-Harbeck et al. (2019) "Predicting disruptive instabilities through deep learning" — Nature 568:526
- Rea et al. (2018) "Disruption prediction investigations using ML" — PPCF 60:084008
- Zhu et al. (2020) "Hybrid deep learning architecture" — Nucl. Fusion 61:026607
- Zheng et al. (2018) "DAGs with NO TEARS" — NeurIPS
- Pearl (2009) "Causality: Models, Reasoning and Inference" — Cambridge University Press

## License

MIT
