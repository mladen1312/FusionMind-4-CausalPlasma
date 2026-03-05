# 4-Layer Stack Engine — User Guide

## Quick Start

```python
from fusionmind4.control.stack import FusionMindStack, Phase, StackConfig, PlasmaState

# Build from raw data (auto-runs CPDE + fits SCM)
stack = FusionMindStack.from_data(plasma_data, var_names, phase=Phase.PHASE_1)

# Or build manually with pre-computed DAG/SCM
stack = FusionMindStack(dag, scm, var_names, StackConfig(phase=Phase.PHASE_1))
```

## Phase 1: Wrapper (Safety Layer)

Wraps an external RL controller (DeepMind, KSTAR, custom). FusionMind evaluates
every proposed action through the causal model, explains it, and optionally vetoes.

```python
stack = FusionMindStack(dag, scm, var_names, StackConfig(phase=Phase.PHASE_1))

# Every control cycle:
state = PlasmaState(values={'betan': 2.0, 'q_95': 4.0, ...}, timestamp=t)
external_action = my_rl_controller.act(state)  # Your existing RL

cmd = stack.evaluate_external_action(state, external_action)

if cmd.vetoed:
    send_to_actuators(cmd.actuator_values)  # Safe alternative
    log(f"VETOED: {cmd.explanation}")
    log(f"Counterfactual: {cmd.counterfactual}")
else:
    send_to_actuators(cmd.actuator_values)  # Original action approved
    log(f"APPROVED (risk={cmd.risk_score:.2f}): {cmd.explanation}")
```

## Phase 2: Hybrid (Causal Strategy + External Tactics)

FusionMind computes strategic setpoints via do-calculus. External RL (or
simple controller) handles fine-grained actuator control.

```python
stack = FusionMindStack(dag, scm, var_names, StackConfig(phase=Phase.PHASE_2))
stack.set_targets({'betan': 2.5, 'q_95': 4.0, 'wplasmd': 50000})

# With external tactical controller:
cmd = stack.step(state, ['Ip', 'Prad'], external_action=my_rl.act(state))

# Without external (FusionMind uses setpoints directly):
cmd = stack.step(state, ['Ip', 'Prad'])
```

## Phase 3: Full Autonomous Control

All four layers active. Layer 2 plans strategy, Layer 1 (built-in RL policy)
executes tactically, Layer 3 monitors safety.

```python
stack = FusionMindStack(dag, scm, var_names, StackConfig(phase=Phase.PHASE_3))
stack.set_targets({'betan': 2.5, 'q_95': 4.0})

cmd = stack.step(state, ['Ip', 'Prad', 'P_heat'])
# cmd.source_layer = 1 (from tactical RL, approved by safety)
```

## Live Phase Upgrade

Switch phases without restart. DAG, SCM, and history are preserved.

```python
stack = FusionMindStack(dag, scm, var_names, StackConfig(phase=Phase.PHASE_1))

# Customer gains confidence...
stack.upgrade_phase(Phase.PHASE_2)
stack.set_targets({'betan': 2.5})

# Even more confidence...
stack.upgrade_phase(Phase.PHASE_3)
```

## C++ Engine (Sub-Microsecond)

For production deployment where Python overhead is unacceptable:

```python
from fusionmind4.realtime.stack_bindings import CppStack

stack = CppStack(n_vars=10, phase=3, var_names=var_names)
stack.load_scm(dag, scm)
stack.load_safety_limits({
    'q_95':  {'min_crit': 2.0, 'min_warn': 2.5},
    'betan': {'max_warn': 3.0, 'max_crit': 3.5},
    'li':    {'max_warn': 1.5, 'max_crit': 2.0},
})
stack.set_setpoints({'betan': 2.5, 'q_95': 4.0})

# Hot path: ~705ns per cycle
result = stack.step(values_dict, timestamp, ['Ip', 'Prad'])
print(result.decision)         # "APPROVE" / "WARN" / "VETO"
print(result.risk_score)       # 0.0 - 1.0
print(result.latency_total_ns) # ~705

# do-calculus in C++ (~300ns)
pred = stack.do_intervention(baseline, {'Ip': 600000})

# Counterfactual in C++ (~400ns)
cf = stack.counterfactual(factual, {'betan': 1.0})

# Latency benchmark
bench = stack.benchmark(n_cycles=50000)
print(f"P50: {bench['p50_ns']:.0f}ns, P95: {bench['p95_ns']:.0f}ns")
```

## Causal Queries (All Phases)

Available regardless of phase:

```python
# "What happens if I set Ip to 600kA?"
pred = stack.predict_intervention(state, {'Ip': 600000})

# "What would have happened if βN had been 1.0?"
cf = stack.counterfactual(state, {'betan': 1.0})

# "Explain the current plasma state causally"
report = stack.explain_state(state)
# → {'risk': 0.15, 'causal_map': {'betan': {'caused_by': ['betat', 'Ip'], ...}}}

# "Why did this disruption happen?"
expl = stack.explain_disruption(pre_state, post_state)
# → {'root_causes': [...], 'counterfactuals': [...], 'recommendations': [...]}

# "What are the system statistics?"
stats = stack.get_stats()
# → {'phase': 'full_stack', 'cycles': 10000, 'safety': {'veto_rate': 0.02, ...}}
```

## Safety Limits Configuration

```python
from fusionmind4.control.stack import SafetyLimits

limits = SafetyLimits(
    q95_min=2.0,              # Below → guaranteed disruption
    q95_warning=2.5,          # Below → warning
    betan_max=3.5,            # Above → Troyon limit
    betan_warning=3.0,        # Above → warning
    li_max=2.0,               # Above → current peaking limit
    li_warning=1.5,
    max_rate_of_change=0.2,   # Max 20% actuator change per cycle
)

stack = FusionMindStack(dag, scm, var_names, StackConfig(
    phase=Phase.PHASE_2,
    safety_limits=limits,
    n_candidates=11,          # Actions tested via do-calculus
    candidate_range=0.15,     # ±15% search range
))
```

## ActionCommand Output

Every `step()` / `evaluate_external_action()` returns an `ActionCommand`:

| Field | Type | Description |
|-------|------|-------------|
| `actuator_values` | `Dict[str, float]` | What to send to actuators |
| `source_layer` | `int` | Which layer produced this (0-3) |
| `risk_score` | `float` | 0.0 (safe) to 1.0 (imminent disruption) |
| `vetoed` | `bool` | Was the original action vetoed? |
| `explanation` | `str` | Human-readable causal explanation |
| `causal_paths` | `List[str]` | e.g. `["Ip → q_95", "li → q_axis"]` |
| `counterfactual` | `str` | What would happen without this action |
| `latency_us` | `float` | Processing time in microseconds |
