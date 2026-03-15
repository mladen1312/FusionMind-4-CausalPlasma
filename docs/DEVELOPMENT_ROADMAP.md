# FusionMind 4.0 — Development Roadmap

**From offline prediction to real-time tokamak control.**

Last updated: March 15, 2026
Author: Dr. Mladen Mešter, dr.med.

---

## Current State (March 2026)

### What Works (verified on real data, reproducible seed=42)

| Capability | Result | Evidence |
|---|---|---|
| Shot-level disruption prediction | AUC = 0.979 ± 0.011 | MAST 2,941 shots, 5-fold CV |
| Causal DAG discovery | F1 = 88.9% | 17/18 edges on real MAST |
| Physics formula (0 params) | AUC = 0.908 (li), 0.978 (f_GW) | Deterministic, no training |
| Production inference | 3 ms/shot | predict_production.py |
| Streaming inference | 165 μs/cycle (P99: 229 μs) | StreamingPredictor, real MAST replay |
| NX-MIMOSA cold-start | AUC = 0.977 | Zero domain engineering |
| Simpson's Paradox detection | r = +0.53 → +0.02 | C-Mod density conditioned on Ip |

### What Is Designed But Not Tested on Real Data

| Module | Lines | Status |
|---|---|---|
| Control stack (4-layer) | 1,030 | Architecture complete |
| Dynamic overseer | 191 | Mimosa arbitrator |
| C++ AVX-512 engine | 501 | 0.27 μs on synthetic |
| CausalShield-RL (PPO) | 1,955 | Gym environment ready |
| UPFM cross-device | 460 | PoC CV=0.267 |
| D3R diffusion | 366 | PoC 130:1 compression |
| LLM Copilot | 670 | Query engine |
| MLX backend | 1,615 | Apple Silicon |

### What We Do Not Have

- Cross-device validation (DIII-D access pending)
- Real-time closed-loop control
- PCS integration (ITER CODAC / DIII-D PCS / KSTAR EPICS)
- Hardware-in-the-loop testing
- Regulatory qualification

### Codebase: 33,032 lines, 50 modules, 8 patent families

---

## Tested and Rejected (honest log)

These ideas were independently tested on real MAST data and found to not improve over the baseline. They are documented here to prevent re-testing.

| Idea | Source | Test Result | Why It Failed |
|---|---|---|---|
| Radar matched filter (LFM chirp) | Grok suggestion | AUC +0.0006 vs baseline | Equivalent to our trend feature on short MAST shots |
| Welch PSD for tearing modes | Grok suggestion | Cannot resolve 0.03 Hz on 1-second shots | Need 30+ seconds for low-freq spectral analysis |
| CA-CFAR adaptive threshold | Grok suggestion | Not applicable to shot-level classification | CFAR is for streaming, not batch |
| DMD as Track H (standalone) | Combustion instability analogy | AUC = 0.956 (-0.023 vs baseline) | Too few modes from 100 timepoints |
| DMD as Track E upgrade | Combustion instability analogy | AUC +0.001 combined with baseline | Same info as max_rate/trend, more complex |
| NX-MIMOSA combined with baseline | NX-MIMOSA repo | AUC = 0.964 (-0.016 vs baseline) | Overfitting: 83 disrupted cannot support 160+ features |
| FM3 physics features on MAST | FM3 codebase | AUC 0.972 (worse than 0.979 without) | MAST q95~7, far from rational surfaces |

### Techniques Deferred (not rejected — waiting for data)

| Technique | Waiting For | Expected Benefit |
|---|---|---|
| DMD mode decomposition | DIII-D shots (500+ tp) | Tearing mode frequency identification on long shots |
| Deep Learning (GRU/CNN/Transformer) | ≥200 disrupted shots | Neural temporal patterns |
| Self-Supervised Pretraining | ≥1M timepoints | Representation learning |
| PINO (Neural Operator) | 1D profile data ≥1kHz | PDE residual anomaly detection |
| UPFM cross-device transfer | ≥3 tokamak datasets | Dimensionless tokenization |

---

## Development Phases

### Phase 1: Streaming Prediction ✅ DONE

**Completed March 2026.**

Deliverable: `fusionmind4/realtime/streaming_predictor.py` (871 lines)

| Component | Latency | State/signal | Function |
|---|---|---|---|
| RollingBuffer | O(N_sig) | 100 tp circular | Welford incremental stats |
| StreamingIPDA | <1 μs | 1 float (r) | Bayesian P(precursor exists) |
| StreamingIMM | <1 μs | 1 float (μ) | Stable vs unstable regime |
| StreamingPHANTOM | <0.1 μs | stateless | σ(margin) feasibility |
| AlarmStateMachine | instant | 1 state + counter | SAFE→MONITOR→ALERT→ALARM→MITIGATE |
| TTDEstimator | <1 μs | 20 margin history | Slope extrapolation → time-to-disruption |

Multi-track fusion:
- Physics priority when P_physics > 0.7 (safety-critical)
- GBT priority when trained model available
- NX-MIMOSA (IPDA+IMM) primary for cold-start (no GBT)

Validated: 165 μs avg, 229 μs P99 on real MAST replay (10× under 2 ms budget).

---

### Phase 2: Overseer Calibration + DIII-D Validation (April–June 2026)

**Goal:** Calibrate alarm thresholds on real disruption outcomes. Validate cross-device on DIII-D.

#### 2a. Alarm Threshold Calibration (April 2026)

- Replay ALL 83 disrupted + 2,858 clean MAST shots through StreamingPredictor
- Measure: true positive rate, false alarm rate, warning time
- Tune alarm hysteresis thresholds (currently hardcoded)
- Optimize GBT feature recompute frequency (every N cycles)
- Target: >90% disrupted detected at ALERT, <5% false alarm on clean

#### 2b. DIII-D Data Access (pending — David Pace routing)

- Contact Form submitted March 14, 2026 ("Control of Damaging Transients")
- Research team evaluating → David Pace handles User Agreement
- Follow-up reminder: March 22, 2026
- Parallel channels: MIT PSFC (disruption-py), ITPA database

#### 2c. DIII-D Validation (May–June 2026, conditional on data access)

- Run CPDE on DIII-D: discover causal mechanism (expect q95 rational surface proximity)
- Run 6-track predictor: validate cross-device AUC
- AGPI soft gate validation: weight should increase to ~1.0 on DIII-D (A~2.5, q95~3)
- FM3 physics features: rational-q proximity should now help (unlike MAST)
- Test DMD: DIII-D shots are 5× longer → DMD may work here
- Test NX-MIMOSA cold-start: predict DIII-D WITHOUT any DIII-D training data

#### 2d. MAST Ops-Log Expansion (April 2026)

- Currently: 83 expert-labeled disrupted shots
- Available: 448+ disrupted shots from ops-log mining (already have disruption_times.json)
- Download Level 1 data for these shots → 5× more training data
- Re-train GBT → expect AUC improvement past 0.979 ceiling
- Activate Deep Learning track (GRU needs ≥200 disrupted)

Deliverables:
- Calibrated alarm thresholds
- DIII-D causal DAG + AUC
- Cross-device comparison paper data
- 448-shot MAST dataset

---

### Phase 3: Causal Control — From Prediction to Intervention (July–October 2026)

**Goal:** Answer "what should we DO?" not just "disruption coming."

#### 3a. Real-Time SCM Intervention Queries

Current SCM (scm.py) works offline. Upgrade to real-time:

```
StreamingPredictor detects: P(disruption) = 0.85, cause = li rising

SCM query (< 1ms):
  P(disruption | current)              = 0.85
  P(disruption | do(NBI_power -= 20%)) = 0.42
  P(disruption | do(gas_puff += 30%))  = 0.61
  P(disruption | do(ECRH_aim = 20°))   = 0.35  ← BEST

  RECOMMEND: redirect ECRH to θ=20°
```

#### 3b. CausalSafetyMonitor Integration

Activate control/causal_controller.py Layer 3 (safety monitor):
- Every proposed action → predict via SCM → check all limits
- If predicted state violates ANY limit → VETO + explain why
- Example: "NBI reduction would drop q95 below 2.0 → VETO"

#### 3c. Intervention Planning Table

Pre-compute a lookup table of interventions for common scenarios:

| Disruption Cause | Primary Actuator | Secondary | Expected ΔP |
|---|---|---|---|
| li rising (kink) | Reduce NBI power | Adjust ECRH aim | -0.30 to -0.50 |
| f_GW rising (density limit) | Reduce gas puff | Increase pumping | -0.40 to -0.60 |
| βN rising (beta limit) | Reduce heating | Reduce Ip | -0.20 to -0.40 |
| q95 dropping (MHD) | Increase Ip | Adjust shaping | -0.25 to -0.45 |
| Radiation collapse | Impurity injection | Controlled shutdown | -0.10 (mitigate) |

#### 3d. Output Format

```python
class ControlRecommendation:
    action: str           # "reduce_nbi_power"
    magnitude: float      # -20%
    expected_dp: float    # -0.43 (P drops by this much)
    confidence: float     # 0.72 (from SCM uncertainty)
    safety_check: bool    # True (no limits violated)
    explanation: str      # "li is primary cause; NBI→li→disruption path"
    alternative: str      # "redirect ECRH to θ=20° (ΔP=-0.35)"
```

Deliverables:
- Real-time SCM query (<1 ms per intervention)
- Safety veto system
- Intervention lookup table
- Control recommendation API

---

### Phase 4: PCS Integration (November 2026 – April 2027)

**Goal:** Connect FusionMind to a real plasma control system.

#### 4a. Interface Standards

| System | Protocol | Latency Requirement |
|---|---|---|
| ITER CODAC | EPICS + SDN | 10 ms cycle |
| DIII-D PCS | Custom TCP | ~1 ms cycle |
| KSTAR PCS | EPICS | 4 ms cycle |
| MAST-U PCS | EPICS | 10 ms cycle |

#### 4b. C++ Real-Time Engine

Upgrade existing `realtime/cpp/fast_engine.hpp`:
- Current: AVX-512 margin computation (0.27 μs synthetic)
- Add: GBT inference in C++ (scikit-learn → ONNX → C++)
- Add: IPDA/IMM update loop in C++
- Add: Alarm state machine in C++
- Target: full pipeline <100 μs in C++

#### 4c. Actuator Mapping Layer

Abstract actuator commands → hardware-specific commands:

```
FusionMind output:        PCS command:
  gas_puff: -20%    →      GASystem::setPressure(ch3, 0.8 * current)
  nbi_power: -15%   →      NBIController::setBeamEnergy(beam2, 0.85 * P)
  ecrh_aim: 20°     →      ECRHLauncher::setAngle(gyro1, 20.0)
  alarm: MITIGATE    →      DMS::triggerMGI(valve_all)  # Massive Gas Injection
```

#### 4d. Safety Interlocks

- Hardware watchdog timer: if FusionMind stops responding → PCS takes over
- Rate limiters: max actuator change per cycle (prevent oscillation)
- Absolute limits: hardcoded in C++, cannot be overridden by ML
- Redundancy: physics track (Track A) runs independently of ML tracks

Deliverables:
- C++ real-time engine (<100 μs)
- EPICS/PCS adapter
- Actuator mapping configuration
- Safety interlock system

---

### Phase 5: Tokamak Validation (May 2027 – February 2028)

**Goal:** Demonstrate on a real tokamak.

#### 5a. Shadow Mode (3 months)

Run alongside existing PCS, no control authority:
- FusionMind predicts, PCS controls
- Log: all predictions + actual outcomes
- Measure: TPR, FPR, warning time, explanation quality
- Requirement: >95% TPR at <2% FPR before advancing

#### 5b. Advisory Mode (3 months)

Display recommendations to operator:
- "FusionMind suggests: reduce gas 20% within 200 ms"
- Operator decides whether to act
- Track: operator agreement rate, outcome when followed vs ignored
- Requirement: operator agrees >70% of the time before advancing

#### 5c. Closed-Loop Control (6 months)

FusionMind controls actuators directly:
- Phase 5c-i: Non-critical only (gas puff)
- Phase 5c-ii: Add NBI modulation
- Phase 5c-iii: Full actuator suite (gas + NBI + ECRH + shaping coils)
- Each phase requires safety review and approval

Target tokamaks (in order of accessibility):
1. MAST-U (UKAEA) — already have FAIR-MAST data, EPICS-based PCS
2. DIII-D (GA) — data access in progress, strong disruption research group
3. KSTAR (KFE) — DeepMind precedent for AI control
4. ITER (ITER Organization) — ultimate target, strictest regulation

---

## Data Acquisition Pipeline

| Dataset | Status | Shots | Next Step |
|---|---|---|---|
| MAST Level 2 (expert) | ✅ Done | 2,941 (83 dis) | — |
| MAST Ops-Log (mined) | 📋 Ready | ~15,000 (448+ dis) | Download Level 1 data |
| C-Mod Density Limit | ✅ Done | 2,333 (78 dis) | — |
| DIII-D EFIT | ⏳ Pending | ~5,000 target | David Pace → research team |
| MIT PSFC disruption-py | ⏳ Draft email | 1,150 DIII-D | Send draft from Gmail |
| ITPA Disruption DB | ⏳ Draft email | 1,150 DIII-D | Send draft from Gmail |
| JET (EUROfusion) | 🔜 Future | ~30,000 | After DIII-D validation |
| EAST (ASIPP) | 🔜 Future | ~10,000 | After paper publication |

---

## Patent Filing Timeline

| Patent Family | Priority | Status | Target Date |
|---|---|---|---|
| PF1 — CPDE | HIGHEST | Ready to file | April 2026 |
| PF2 — CPC | HIGHEST | Ready to file | April 2026 |
| PF3 — UPFM | Medium | Needs more data | After DIII-D |
| PF4 — D3R | Medium | PoC only | Q3 2026 |
| PF5 — AEDE | Medium | PoC only | Q3 2026 |
| PF6 — Integrated | Low | Needs PCS work | 2027 |
| PF7 — CausalShield-RL | Medium | Design complete | Q4 2026 |
| PF8 — LLM Copilot | Low | Design only | 2027 |

Strategy: US Provisional → PCT for PF1+PF2 first (highest novelty, most time-sensitive vs DeepMind/CFS competition).

---

## Publication Plan

### Paper 1: Causal Disruption Prediction (target: June 2026)

- Target journal: Nuclear Fusion
- Title: "Causal disruption prediction outperforms deep learning with zero parameters"
- Content: MAST 0.979, C-Mod 0.978, Simpson's Paradox, mechanism identification
- Key claim: first application of Pearl's causal framework to fusion disruptions

### Paper 2: Cross-Device Causal Generalization (target: Q4 2026)

- Target journal: Plasma Physics and Controlled Fusion
- Content: MAST + C-Mod + DIII-D, AGPI soft gate, machine-specific mechanisms
- Requires: DIII-D data access

### Paper 3: Real-Time Causal Control (target: 2027)

- Target journal: Nature Machine Intelligence or Nuclear Fusion
- Content: StreamingPredictor + SCM interventions + shadow mode results
- Requires: Phase 5a completion

---

## Budget and Resources

### Current (solo researcher, zero cost)

- Compute: MacBook + free cloud CI
- Data: all public (FAIR-MAST, MIT PSFC)
- IP: BSL-1.1 license, patents pending

### Phase 2-3 (~$15K)

- Patent filing: $5-10K (US Provisional × 2)
- Cloud compute: $2-3K (DIII-D dataset processing, GRU training)
- Travel: $2-3K (1 conference for visibility)

### Phase 4-5 (~$200-500K, requires funding)

- 1-2 engineers: $150-300K/year
- Tokamak access agreement: $0 (collaborative, not commercial)
- Hardware (if FPGA needed): $20-50K
- Safety certification: $30-50K

### Revenue model

- BSL-1.1: research free, production requires license
- Target: CFS/TAE/fusion startups for production license ($1-5M/year)
- ITER: regulatory-compliant causal safety layer (long-term, high value)

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| DIII-D data access denied | Low | High | MIT PSFC + ITPA as alternatives |
| DeepMind publishes causal fusion AI | Medium | High | File PF1+PF2 patents NOW, publish Paper 1 |
| 83 disrupted shots insufficient for advancement | High | Medium | MAST ops-log → 448+ shots; DIII-D → thousands |
| Streaming predictor too slow for DIII-D PCS | Low | Medium | C++ engine already benchmarked at 0.27 μs |
| SCM interventions physically incorrect | Medium | High | CausalSafetyMonitor veto + shadow mode validation |
| Overfitting on MAST, fails on other machines | Medium | High | NX-MIMOSA cold-start fallback (0.977 AUC no tuning) |
| Patent prior art from TokaMind/CFS | Low | Medium | Our causal approach is fundamentally different from RL |

---

## Contacts

| Person | Role | Status |
|---|---|---|
| David Pace | DIII-D Deputy Director | Engaged, routing to research team |
| Alba (DIII-D) | Coordinator | CC'd on David's emails |
| Cristina Rea | MIT PSFC, disruption-py | Email draft ready |
| Gabriele Trevisan | MIT PSFC | Email draft ready (CC) |
| Eidietis | ITPA Disruption DB | Email draft ready |

---

## Success Metrics

| Milestone | Metric | Target Date |
|---|---|---|
| DIII-D data access | User Agreement signed | June 2026 |
| Cross-device AUC | >0.95 on DIII-D | August 2026 |
| Paper 1 submitted | Nuclear Fusion | June 2026 |
| PF1+PF2 filed | US Provisional | April 2026 |
| Real-time demo | Shadow mode on MAST-U or DIII-D | Q2 2027 |
| First customer | CFS or TAE evaluation | Q4 2027 |
| ITER qualification | Regulatory submission | 2028+ |
