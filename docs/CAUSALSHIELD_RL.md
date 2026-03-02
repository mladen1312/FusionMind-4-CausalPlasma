# CausalShield-RL — Causal Reinforcement Learning for Fusion (PF7)

CausalShield-RL is the first hybrid causal RL system for tokamak plasma control. It combines FusionMind's causal discovery (CPDE) and counterfactual reasoning (CPC) with reinforcement learning to produce an agent that is both performant and explainable.

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                   CausalShield-RL                          │
│                                                            │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────┐  │
│  │   CPDE   │────▶│  Neural SCM  │────▶│  PPO Policy  │  │
│  │  Causal  │     │  World Model │     │  with Causal │  │
│  │  Graph   │     │  (learnable) │     │  Constraints │  │
│  └──────────┘     └──────┬───────┘     └──────────────┘  │
│                          │                                 │
│                   ┌──────▼───────┐                         │
│                   │   Causal     │                         │
│                   │   Reward     │                         │
│                   │   Shaping    │                         │
│                   └──────────────┘                         │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Gym Environment (FM3-Lite physics + disruptions)    │ │
│  └──────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

## Components

### 1. Neural SCM World Model

Replaces the linear SCM from CPC with learnable neural equations:

```python
# Each variable has a neural equation:
# X_i = f_θ(parents(X_i)) + ε_i
# where f_θ is a 2-layer MLP with residual connections
```

Key features:
- Online learning via `update(new_data)` — adapts to changing plasma conditions
- Jacobian computation for sensitivity analysis
- do-interventions via parent clamping (same semantics as linear SCM)
- Counterfactual reasoning via abduction-action-prediction

Implementation: `fusionmind4/learning/neural_scm.py`

### 2. Causal Reward Shaping

Standard RL reward + causal bonuses/penalties:

```
R_total = R_base + R_causal_alignment + R_safety - R_forbidden_path

where:
  R_base           = -||state - target||²  (track reference)
  R_causal_alignment = +bonus if action aligns with beneficial causal paths
  R_safety         = -penalty if βN > limit or q95 < threshold
  R_forbidden_path = -penalty if action affects variables via forbidden causal paths
```

Implementation: `fusionmind4/learning/causal_reward.py`

### 3. Gym Plasma Environment

OpenAI Gym-compatible environment wrapping FM3-Lite physics:

- **Observation space**: 14 plasma variables (continuous)
- **Action space**: 4 actuators (P_NBI, P_ECRH, P_ICRH, gas_puff), continuous
- **Episode**: 200 steps (200 ms of plasma evolution)
- **Disruption**: Episode terminates if βN > 4.0 or q95 < 1.5

Implementation: `fusionmind4/learning/gym_plasma_env.py`

### 4. PPO Policy

Proximal Policy Optimization with:
- Actor-critic architecture (separate value and policy heads)
- Causal constraint layer (masks forbidden actuator-variable paths)
- Entropy bonus for exploration
- GAE (Generalized Advantage Estimation)

Implementation: `fusionmind4/learning/causal_rl_hybrid.py`

---

## Training Pipeline

```python
from fusionmind4.learning import CausalRLHybrid

hybrid = CausalRLHybrid()

# Step 1: Discover causal graph from data
hybrid.discover_causal_graph(data, interventional_data)

# Step 2: Fit neural SCM world model
hybrid.fit_world_model(data)

# Step 3: Train RL policy within causal constraints
hybrid.train(n_episodes=300)

# Step 4: Deploy with explanation
result = hybrid.act_with_explanation(observation)
# Returns: action, value, causal_paths, safety_check
```

---

## Why Causal RL > Standard RL for Fusion

| Property | Standard RL (KSTAR) | CausalShield-RL |
|----------|:---:|:---:|
| Learns causal structure | ✗ | ✓ |
| Explainable actions | ✗ | ✓ (causal path tracing) |
| Simpson's Paradox safe | ✗ | ✓ |
| Transfer across devices | Requires retraining | Causal graph transfers |
| Online adaptation | Risky (reward hacking) | Safe (causal constraints) |
| Sample efficiency | Low | Higher (world model) |

---

## Novelty

No existing fusion RL system integrates formal causal inference. KSTAR uses standard PPO, DeepMind/CFS uses model-based RL without causal structure. CausalShield-RL is the first to:

1. Use a learned causal graph as a constraint on the policy
2. Shape rewards using causal path analysis
3. Provide post-hoc explanations via counterfactual reasoning
4. Enable safe online adaptation via Neural SCM updates
