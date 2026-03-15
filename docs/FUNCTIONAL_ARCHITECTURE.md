# FusionMind 4.0 — Functional Architecture
# How everything works, function by function

## ══════════════════════════════════════════════════════
## PIPELINE 1: DISRUPTION PREDICTION (main product)
## ══════════════════════════════════════════════════════
##
## Entry points:
##   scripts/predict_production.py --shot 27095    → single shot
##   scripts/predict_production.py --output p.json → batch
##   from fusionmind4.predictor.engine import run_mast
##
## Data flow:
##
##   .npz file (16 vars × N timepoints)
##        │
##        ▼
##   resolve_signals(variable_names)          [engine.py:138]
##   │  Maps machine-specific names to canonical:
##   │  "li" ← "li" | "internal_inductance"
##   │  "q95" ← "q95" | "safety_factor_95"
##   │  Returns: {"li": col_idx, "q95": col_idx, ...}
##   │
##   ▼
##   detect_machine_type(variable_names)      [predict.py:85]
##   │  Heuristic: has betan + greenwald_den → spherical
##   │  Returns: ("spherical", LIMITS_SPHERICAL)
##   │
##   ▼
##   StabilityLimits.for_machine(type)        [engine.py:86]
##   │  SPHERICAL:    li_max=2.0  q95_min=1.5  βN_max=6.0  f_GW_max=1.5
##   │  CONVENTIONAL: li_max=1.5  q95_min=2.0  βN_max=3.5  f_GW_max=1.0
##   │
##   ▼
##   FOR EACH SHOT:
##   │
##   ├─── TrackA_PhysicsMargins.build_features(signals)    [engine.py:248]
##   │    │  margin_li   = clip(1 - max(li)/li_limit, -1, 1)
##   │    │  margin_q95  = clip(1 - q95_min/min(q95), -1, 1)
##   │    │  margin_bN   = clip(1 - max(bN)/bN_limit, -1, 1)
##   │    │  margin_fGW  = clip(1 - max(fGW)/fGW_limit, -1, 1)
##   │    │  margin_bp   = clip(1 - max(bp)/bp_limit, -1, 1)
##   │    │  + min_margin, n_stressed, closest_limit
##   │    └──→ 9 features [ALWAYS ON, 0 parameters, AUC=0.905]
##   │
##   ├─── TrackB_ShotStats.build_features(signals)         [engine.py:278]
##   │    │  FOR EACH of 8 signals (li, q95, bN, bp, fGW, Prad, Wmhd, Ip):
##   │    │    mean(s), std(s), max(s)
##   │    │    late_mean = mean(s[-30%:])
##   │    │    trend = late_mean - mean(s[:30%])
##   │    │    max_rate = max(|diff(s)|)
##   │    │  = 48 features
##   │    │  + 5 margins + min_margin + n_stressed = 7
##   │    │  + 4 interactions: li×bN, li/q95, std(li)×std(q95), fGW×li
##   │    │  + 4 temporal shape: late_li/early_li, late_li-early_li, etc.
##   │    └──→ 63 features [GBT 100 trees, AUC=0.979 ★ BEST]
##   │
##   ├─── TrackC_Trajectory.build_features(signals)        [engine.py:344]
##   │    │  Split shot into thirds (early, mid, late)
##   │    │  FOR EACH signal:
##   │    │    end/start ratio, volatility change, max shift
##   │    │    late_mean / early_mean
##   │    └──→ 32 features [AUC ~0.94]
##   │
##   ├─── TrackD_CausalMechanism.build_features(signals)   [engine.py:429]
##   │    │  PRIMARY DRIVER auto-detected:
##   │    │    identify_from_data() ranks each variable by single-var AUC
##   │    │    MAST → li (AUC=0.908), C-Mod → f_GW (AUC=0.985)
##   │    │  12 deep features on primary driver ONLY:
##   │    │    max, min, std, range, late_max, late_mean, late_rate
##   │    │    skewness, kurtosis, percentiles, crossing_count
##   │    └──→ 12 features [AUC=0.950]
##   │
##   ├─── TrackE_RateExtremes.build_features(signals)      [engine.py:480]
##   │    │  FOR EACH signal:
##   │    │    max_rate = max(|diff(s)|)
##   │    │    late_mean_rate = mean(|diff(s[-30%:])|)
##   │    │    late_volatility = std(diff(s[-30%:]))
##   │    │    max_acceleration = max(|diff(diff(s))|)
##   │    │    late_max_rate
##   │    └──→ 40 features [precursor detection, AUC ~0.93]
##   │
##   ├─── TrackF_Pairwise.build_features(signals)          [engine.py:523]
##   │    │  Signal pair interactions (stability boundaries):
##   │    │    li × bN         (Troyon stability space)
##   │    │    li / q95        (internal kink parameter)
##   │    │    bN / q95        (normalized stability)
##   │    │    fGW × q95       (density-q coupling)
##   │    │    li × fGW        (Hugill diagram)
##   │    │    max versions, late versions, products
##   │    └──→ 15 features [AUC ~0.91]
##   │
##   ▼
##   TRAINING (per track):
##   │  Augmentation: 4× disrupted copies with 5% Gaussian noise
##   │  GBT: GradientBoostingClassifier(n_estimators=100, max_depth=4,
##   │        subsample=0.8, min_samples_leaf=3, random_state=42)
##   │  5-fold StratifiedGroupKFold (groups=shot_ids)
##   │
##   ▼
##   MetaLearner.fit(track_outputs, labels)   [engine.py:556]
##   │  Collects out-of-fold predictions from each track
##   │  LogisticRegression on stacked track predictions
##   │  Weighted alternative: α_A × P_A + α_B × P_B + ... (learned weights)
##   │
##   ▼
##   OUTPUT per shot:
##     {
##       "disruption_probability": 0.961,
##       "risk_level": "CRITICAL",          # assign_risk()
##       "time_to_disruption_ms": 142,      # estimate_ttd_from_margin()
##       "uncertainty": 0.054,              # estimate_uncertainty() [GBT staged]
##       "closest_limit": "li",
##       "margins": {"li": 0.05, "q95": 0.42, ...},
##       "explanation": "li at 95% of kink limit",
##       "active_tracks": ["physics","stats","trajectory","causal","rates","pairwise"],
##       "recommendation": "ALARM"
##     }


## ══════════════════════════════════════════════════════
## PIPELINE 2: CAUSAL DISCOVERY (CPDE — PF1)
## ══════════════════════════════════════════════════════
##
## Entry: EnsembleCPDE().discover(data, var_names=names)
##
##   data: (n_timepoints, n_vars) matrix
##        │
##        ▼
##   STEP 1: NOTEARSDiscovery.fit_bootstrap(data, n=15)  [notears.py:151]
##   │  FOR EACH bootstrap sample:
##   │    Subsample rows with replacement
##   │    Minimize: ½n ||X - XW||²_F + λ₁||W||₁
##   │    Subject to: h(W) = tr(e^{W∘W}) - d = 0  (DAG constraint)
##   │    Augmented Lagrangian: L = loss + λ₁||W||₁ + α·h(W) + ½ρ·h(W)²
##   │    Solver: L-BFGS-B inner loop, α/ρ update outer loop
##   │    Threshold edges at |w| < 0.10
##   │  Result: stability matrix [0,1] — how often each edge appears
##   │
##   ▼
##   STEP 2: GrangerCausalityTest.test_all_pairs(data)    [granger.py:36]
##   │  FOR EACH pair (X, Y):
##   │    BIC lag selection: find optimal lag p in [1, max_lag]
##   │    Restricted model: Y_t = Σ a_k Y_{t-k} + ε
##   │    Unrestricted model: Y_t = Σ a_k Y_{t-k} + Σ b_k X_{t-k} + ε
##   │    F-statistic: F = (RSS_r - RSS_u) / (RSS_u / (n - 2p - 1)) × p
##   │    p-value from scipy.stats.f.sf(F, p, n-2p-1)
##   │    Bonferroni correction: α' = α / n_pairs
##   │  Result: binary matrix [0/1] — significant Granger causality
##   │
##   ▼
##   STEP 3: PCAlgorithm.fit_bootstrap(data, n=15)        [pc.py:232]
##   │  Discover skeleton via conditional independence:
##   │    FOR EACH pair (i, j):
##   │      FOR conditioning set size 0, 1, 2, ...:
##   │        Fisher z-test: z = ½ ln((1+r)/(1-r)) × √(n-|S|-3)
##   │        If p > α: remove edge, record separation set
##   │  Orient v-structures: i → k ← j (if k not in sep(i,j))
##   │  Apply Meek rules R1-R4 for maximal orientation:
##   │    R1: i → j — k  becomes i → j → k
##   │    R2: i → k → j and i — j  becomes i → j
##   │    R3: i — k → j and i — l → j and k ≠ l  becomes i → j
##   │    R4: i — k → l → j and i — j  becomes i → j
##   │  Result: stability matrix from bootstrap [0,1]
##   │
##   ▼
##   STEP 4: InterventionalScorer.score(data)              [interventional.py:24]
##   │  FOR EACH actuator × target pair:
##   │    Split data into high/low actuator groups
##   │    Cohen's d = (mean_high - mean_low) / pooled_std
##   │    If |d| > 0.3: mark edge as causal
##   │  Result: interventional score matrix
##   │
##   ▼
##   STEP 5: Ensemble fusion                               [ensemble.py:118-180]
##   │  combined = notears_wt × NOTEARS_stability
##   │           + granger_wt × Granger_matrix
##   │           + pc_wt × PC_stability
##   │           + physics_wt × physics_prior_matrix
##   │
##   │  physics_prior_matrix:                              [physics.py:43]
##   │    Known edges: P_NBI → Ti (heating), gas → ne (fueling),
##   │                 Ip → q (current), ne+Te → βN (pressure)
##   │    Actuator exogeneity: no incoming edges to actuators
##   │
##   │  Threshold at 0.32 → binary DAG
##   │  Enforce acyclicity: if cycles remain, remove weakest edge
##   │  Remove indirect edges: if A→B→C and A→C, remove A→C
##   │
##   ▼
##   OUTPUT:
##     {
##       "dag": np.ndarray (n_vars × n_vars),  # adjacency matrix
##       "metrics": {"precision": 0.895, "recall": 0.944, "f1": 0.889},
##       "physics_checks": {"actuator_exogeneity": True, "no_cycles": True, ...},
##       "edge_details": [{"from": "li", "to": "disruption", "weight": 0.87}, ...]
##     }


## ══════════════════════════════════════════════════════
## PIPELINE 3: COUNTERFACTUAL REASONING (CPC — PF2)
## ══════════════════════════════════════════════════════
##
## Entry: PlasmaSCM.fit(data) → scm.predict() / InterventionEngine.do()
##
##   DAG from CPDE
##        │
##        ▼
##   PlasmaSCM.fit(data)                                   [scm.py:62]
##   │  FOR EACH variable in topological order:
##   │    parents = variables with DAG edge pointing to this var
##   │    Linear regression: X_j = Σ β_k × PA_k + intercept + noise
##   │    Store: StructuralEquation(variable, parents, coefficients, noise_std)
##   │  Result: fitted SCM with one equation per variable
##   │
##   ├── predict(values)                                    [scm.py:134]
##   │   │  Given partial observation, predict remaining variables
##   │   │  Forward pass through topological order
##   │   └──→ Dict[str, float] of all variable values
##   │
##   ├── InterventionEngine.do(interventions)               [interventions.py:61]
##   │   │  do(X=x): force variable X to value x
##   │   │  Removes all incoming edges to X (Pearl's do-operator)
##   │   │  Forward propagate through SCM to compute effects
##   │   │  average_causal_effect(): ACE over range of interventions
##   │   │  find_optimal_intervention(): minimize risk via L-BFGS-B
##   │   └──→ InterventionResult(predicted_state, causal_path, ACE)
##   │
##   └── CounterfactualEngine.counterfactual(factual, intervention) [interventions.py:188]
##       │  3-step process:
##       │  1. Abduction: infer noise U from observed factual data
##       │     U_j = X_j_observed - f_j(PA_j_observed)
##       │  2. Action: apply intervention (replace structural equation)
##       │  3. Prediction: forward pass with new equation + original noise
##       └──→ CounterfactualResult(factual, counterfactual, difference)
##
##
## NonlinearPlasmaSCM (upgrade)                             [nonlinear_scm.py:39]
##   │  Same structure but GradientBoosting per equation
##   │  fit(): GBT predicts X_j from parents (R² = 96.7% on MAST)
##   │  Linear backbone kept for counterfactual noise extraction
##   │  cross_validate(): 5-fold CV R² per variable
##   └──→ Higher accuracy, same counterfactual capability


## ══════════════════════════════════════════════════════
## PIPELINE 4: PRODUCTION INFERENCE
## ══════════════════════════════════════════════════════
##
## Entry: python scripts/predict_production.py --data X --labels Y --shot Z
##
##   1. Load .npz + disruption_info.json
##   2. resolve_signals() + detect_machine_type()
##   3. build_shot_features() for all shots  [from predict.py — 63f]
##   4. Augment disrupted 4×, train GBT(100 trees)
##   5. predict_shot_full(shot_data):
##      ├── build_shot_features() → 63 features + physics explanation
##      ├── GBT.predict_proba() → P(disruption)
##      ├── estimate_uncertainty() → std of staged tree predictions
##      ├── estimate_ttd_from_margin() → TTD from margin slope extrapolation
##      ├── assign_risk() → CRITICAL/HIGH/MEDIUM/LOW/SAFE
##      └── active tracks list
##   6. Output: JSON with all fields + inference time (3ms/shot)


## ══════════════════════════════════════════════════════
## PIPELINE 5: AGPI SOFT GATING (cross-device)
## ══════════════════════════════════════════════════════
##
## Entry: build_agpi_features(li, q95, bN, fGW, prad, pin, wmhd, n30, A)
##
##   agpi_weight(q95_mean, aspect_ratio)                    [agpi.py:38]
##   │  base = σ(4.2 - q95_mean)        # sigmoid threshold
##   │  geometry = min(3.5 / A, 2.0)    # aspect ratio factor
##   │  weight = clip(base × geometry, 0.05, 1.0)
##   │
##   │  MAST (q95=7, A=1.3):   weight = 0.115  (FM3 physics ~off)
##   │  DIII-D (q95=3, A=2.5): weight = 1.000  (FM3 physics full)
##   │  ITER (q95=3, A=3.1):   weight = 0.868  (FM3 physics on)
##   │
##   ▼
##   build_fm3_physics_features()                           [fm3_physics.py:77]
##   │  6 feature groups × machine-specific thresholds:
##   │
##   │  1. Rational q-surface proximity (tearing mode risk):
##   │     |q95 - 2.0|, |q95 - 1.5|, closest_rational, 1/(closest+0.1)
##   │
##   │  2. Shape-corrected Troyon limit:
##   │     βN_crit = C_T × li, troyon_margin = 1 - βN/βN_crit
##   │
##   │  3. li dynamics (internal kink precursor):
##   │     li_rate, li_acceleration, li/q95, li_rising_fast
##   │
##   │  4. Radiation fraction:
##   │     P_rad/P_input, late radiation fraction, above threshold
##   │
##   │  5. Confinement degradation:
##   │     W_drop = 1 - W_late/W_max, τ_E degradation
##   │
##   │  6. Multi-mechanism stress count:
##   │     n_stressed = count(li>thr, q<thr, βN>thr, fGW>thr, rad>thr, W_drop>thr)
##   │
##   ▼
##   weighted_features = raw_features × agpi_weight
##   └──→ 22 features, soft-weighted by machine geometry


## ══════════════════════════════════════════════════════
## PIPELINE 6: DYNAMIC OVERSEER (designed, not tested)
## ══════════════════════════════════════════════════════
##
## Entry: DynamicOverseer().decide(tracks)
##
##   4 tracks run in parallel:
##   │  Track A: ML (GBT on all features)     → prob, confidence
##   │  Track B: Causal (SCM parents only)     → prob, confidence
##   │  Track C: Physics (margin thresholds)   → prob, confidence
##   │  Track D: Fast (MHD amplitude, rates)   → prob, confidence
##   │
##   ▼
##   DynamicOverseer.decide()                               [dynamic_overseer.py:78]
##   │  disagreement = max(probs) - min(probs)
##   │
##   │  IF disagreement < 0.20:
##   │    final = weighted_mean(probs, confidences)
##   │    → All tracks agree → high confidence
##   │
##   │  IF disagreement ≥ 0.20:
##   │    Physics tracks (C, D) get PRIORITY
##   │    If physics says dangerous → trust physics
##   │    If ML says dangerous but physics doesn't → check confounders
##   │    → Simpson's Paradox protection
##   │
##   │  History smoothing: rolling mean over last N decisions
##   │
##   └──→ OverseerDecision(final_prob, best_track, warning_level)


## ══════════════════════════════════════════════════════
## PIPELINE 7: CONTROL STACK (designed, not tested)
## ══════════════════════════════════════════════════════
##
## 4-layer stack, customer enables by phase:
##
##   Layer 3: CausalSafetyMonitor [ALWAYS ON]     [causal_controller.py:192]
##   │  evaluate_action(state, proposed_action):
##   │    1. Get causal parents of each affected variable
##   │    2. Predict intervention effect via SCM
##   │    3. Check if predicted state violates any safety limit
##   │    4. Compute disruption risk from causal path
##   │    5. IF risk > threshold: VETO action
##   │    6. Return: approved/modified action + explanation
##   │
##   Layer 2: Strategic Controller [PHASE 2+]      [controller.py:74]
##   │  compute_action(current_state, targets):
##   │    Generate candidate actions
##   │    FOR EACH candidate:
##   │      Predict outcome via SCM.do(action)
##   │      Score vs targets
##   │    Select best action
##   │    Explain causal path from action to outcome
##   │
##   Layer 1: Tactical RL [PHASE 3]                [stack.py:236]
##   │  PPO policy network: obs → action
##   │  Causal reward shaping from Layer 2 setpoints
##   │  Constrained by Layer 3 safety limits
##   │
##   Layer 0: C++ Realtime Engine [ALWAYS ON]      [stack.py:160]
##   │  extract_features(): raw diagnostics → plasma state
##   │  fast_risk_score(): AVX-512 margin computation (0.27μs)
##   │  apply_rate_limits(): smooth actuator commands


## ══════════════════════════════════════════════════════
## PIPELINE 8: ADVANCED MODULES (activate conditionally)
## ══════════════════════════════════════════════════════
##
## Each module: check_activation(data_info) → (bool, reason)
##
## A. Deep Learning Track                           [deep_learning.py]
##    Condition: ≥200 disrupted + PyTorch + GPU
##    Models: GRU(≥200d) → CNN(≥500d) → Transformer(≥1000d)
##    Each: train on rolling windows → per-window P(disruption)
##    Shot-level: max(window_probs) + mean + late_mean
##    └──→ 3-9 features for meta-learner
##
## B. PINO                                          [pino.py]
##    Condition: 1D profiles (Te(r), ne(r)) ≥1kHz ≥100 shots
##    Architecture: SpectralConv1d (Fourier space) × 4 layers
##    Physics: energy_conservation + diffusive_transport + positivity
##    Disruption = anomalous PDE residual (z-score vs clean shots)
##    └──→ 6 features (residual stats) for meta-learner
##
## C. Self-Supervised Pretraining                   [self_supervised.py]
##    Condition: ≥1M timepoints
##    3 pretext tasks:
##      MSP: mask 15% signals, predict from embedding
##      CTL: same-shot windows = positive pairs (InfoNCE)
##      NSP: predict next timestep from current window
##    Encoder: Conv1D stack → global pool → embedding
##    └──→ 4×emb_dim features per shot for meta-learner
##
## D. PINN+TGN                                     [pinn_tgn.py]
##    Mode A (now): nodes = variables, edges = causal DAG
##    Mode B (future): nodes = radial zones, edges = transport
##    Message passing: MSG(h_i, h_j, e_ij) → AGG → UPDATE
##    Temporal attention over graph snapshots
##    Physics: transport residual, energy balance
##    └──→ 12 features for meta-learner


## ══════════════════════════════════════════════════════
## DATA SOURCES AND FORMATS
## ══════════════════════════════════════════════════════
##
## MAST Level 2 (.npz):
##   data: float32 (268667, 16) — 16 EFIT signals at ~10ms
##   shot_ids: int (268667,) — which shot each timepoint belongs to
##   variables: ['betan','betap','betat','q95','elongation','li',
##               'wmhd','q_axis','minor_radius','tribot','tritop',
##               'Ip','ne_line','greenwald_den','p_rad','p_nbi']
##
## C-Mod Density Limit (.npz):
##   data: float32 (264385, 6)
##   variables: ['density','elongation','minor_radius',
##               'plasma_current','toroidal_B_field','triangularity']
##
## Disruption labels (.json):
##   {"disrupted": [27000, 27030, ...], "clean": [11766, 11767, ...]}
##
## FM3-Lite simulator:
##   14 variables with KNOWN causal DAG (48 edges)
##   Used for: CPDE validation, SCM testing
##   generate() → (n_samples, 14) observational data
