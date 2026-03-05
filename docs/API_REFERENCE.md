# API Reference

## Core Classes

### FusionMindStack

**Module:** `fusionmind4.control.stack`

The unified 4-layer control interface. Primary class for all deployments.

```python
FusionMindStack(dag, scm, var_names, config=StackConfig())
FusionMindStack.from_data(data, var_names, phase=Phase.PHASE_1)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `step(state, actuators, external_action=None)` | `ActionCommand` | Execute one control cycle |
| `evaluate_external_action(state, action)` | `ActionCommand` | Evaluate external RL action (Phase 1) |
| `set_targets(targets)` | None | Set plasma targets (Phase 2+) |
| `upgrade_phase(new_phase)` | `str` | Live phase upgrade |
| `explain_state(state)` | `Dict` | Causal analysis of current state |
| `explain_disruption(pre, post)` | `Dict` | Post-mortem root cause analysis |
| `predict_intervention(state, intervention)` | `Dict` | do-calculus: P(Y\|do(X=x)) |
| `counterfactual(state, hypothetical)` | `Dict` | What would have happened if... |
| `get_stats()` | `Dict` | Performance statistics |

### CppStack

**Module:** `fusionmind4.realtime.stack_bindings`

C++ engine wrapper for sub-microsecond latency.

```python
CppStack(n_vars, phase=1, var_names=None)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `load_scm(dag, scm)` | None | Load DAG + SCM from Python objects |
| `load_safety_limits(limits_dict)` | None | Configure physics boundaries |
| `load_policy_weights(W1,b1,W2,b2,W3,b3)` | None | Load RL policy (Layer 1) |
| `set_phase(phase)` | None | Switch phase live |
| `set_setpoints(targets)` | None | Set target parameters |
| `step(values, timestamp, actuators, external_action=None)` | `StackStepResult` | Execute one C++ cycle |
| `do_intervention(baseline, intervention)` | `Dict` | C++ do-calculus (~300ns) |
| `counterfactual(factual, hypothetical)` | `Dict` | C++ counterfactual (~400ns) |
| `benchmark(n_cycles=10000)` | `Dict` | Latency benchmark |
| `get_stats()` | `Dict` | Cycle/veto/approve counts |

### NonlinearPlasmaSCM

**Module:** `fusionmind4.discovery.nonlinear_scm`

GradientBoosting structural causal model. Higher R² than linear SCM.

```python
NonlinearPlasmaSCM(dag, var_names, n_estimators=100, max_depth=3)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(X)` | None | Fit nonlinear equations from data |
| `predict(X)` | `ndarray` | Predict all variables from parents |
| `do(interventions, baseline)` | `ndarray` | do-calculus intervention |
| `counterfactual(factual, interventions)` | `ndarray` | Counterfactual query |
| `cross_validate(X, n_folds=5)` | `Dict` | Cross-validated R² per variable |
| `summary()` | `str` | Human-readable model summary |

### EnsembleCPDE

**Module:** `fusionmind4.discovery.ensemble`

Ensemble causal discovery pipeline (NOTEARS + Granger + PC + Physics).

```python
EnsembleCPDE(config=None)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `discover(data, interventional_data=None)` | `Dict` | Full pipeline: DAG + metrics + edge details |

### PlasmaSCM

**Module:** `fusionmind4.control.scm`

Linear structural causal model (fast, analytical counterfactuals).

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(dag, data, var_names)` | None | Fit linear equations |
| `predict(obs)` | `ndarray` | Predict from parents |
| `do_intervention(obs, interventions)` | `ndarray` | do-calculus |
| `counterfactual_query(factual, intervention)` | `ndarray` | Counterfactual |

### NeuralSCM

**Module:** `fusionmind4.learning.neural_scm`

Neural network structural equations (learnable nonlinear SCM).

### CausalRLHybrid

**Module:** `fusionmind4.learning.causal_rl_hybrid`

PPO agent with causal reward shaping and SCM world model.

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(data)` | None | Train world model + policy |
| `act(state)` | `Dict` | Select action with causal explanation |
| `online_update(transition)` | None | Update from new observation |
| `summary()` | `Dict` | Model statistics |

---

## Data Structures

### PlasmaState

```python
PlasmaState(values: Dict[str, float], timestamp: float, shot_id: int = 0)
```

### ActionCommand

```python
ActionCommand(
    actuator_values: Dict[str, float],
    source_layer: int,           # 0=L0, 1=L1, 2=L2, 3=L3
    risk_score: float,           # 0.0 - 1.0
    vetoed: bool,
    explanation: str,
    causal_paths: List[str],
    counterfactual: str,
)
```

### Phase

```python
Phase.PHASE_1  # Wrapper (L0+L3)
Phase.PHASE_2  # Hybrid (L0+L2+L3)
Phase.PHASE_3  # Full Stack (L0+L1+L2+L3)
```

### SafetyLimits

```python
SafetyLimits(
    q95_min=2.0, q95_warning=2.5,
    betan_max=3.5, betan_warning=3.0,
    li_max=2.0, li_warning=1.5,
    max_rate_of_change=0.2,
)
```

---

## Discovery Algorithms

| Module | Algorithm | Description |
|--------|-----------|-------------|
| `discovery.notears` | NOTEARS | DAG learning with h(W)=tr(e^{W∘W})-d constraint |
| `discovery.granger` | Granger Causality | Temporal causality (conditional, spectral, BIC lag) |
| `discovery.temporal` | Temporal Granger | Selective conditioning (top-K confounders) |
| `discovery.pc` | PC Algorithm | Constraint-based (stable variant + Meek R1-R4) |
| `discovery.physics` | Physics Priors | Known tokamak relationships + PINN validation |
| `discovery.interventional` | Interventional Scoring | do-calculus validation of edges |

---

## Copilot

**Module:** `fusionmind4.copilot`

Natural language interface for causal queries (supports Croatian and English).

```python
from fusionmind4.copilot import CausalCopilot

copilot = CausalCopilot(dag, scm, var_names)
response = copilot.query("What happens if I increase Ip by 10%?")
response = copilot.query("Zašto je došlo do disrupcije?")
```
