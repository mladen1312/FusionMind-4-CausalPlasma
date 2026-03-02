# CPDE — Causal Plasma Discovery Engine (PF1)

The CPDE is a 9-step ensemble causal discovery pipeline that combines four complementary algorithms with physics constraints to discover the causal structure of tokamak plasmas.

---

## Pipeline Overview

```
Step 1: Physics Priors → Hard mask (52 forbidden) + soft prior (23 expected edges)
Step 2: NOTEARS → Contemporaneous DAG structure learning
Step 3: DYNOTEARS → Lagged temporal causal structure (time-series aware)
Step 4: Granger Causality → Temporal causal links with physics-informed lag selection
Step 5: PC Algorithm → Conditional independence veto (constraint-based)
Step 6: Bayesian Fusion → 4-source ensemble + direction resolution + physics-aware
                          PC veto + triple-tier threshold + multi-method agreement bonus
Step 7: Cycle-Breaking → DAG enforcement preserving physics-supported edges
Step 8: Interventional Scoring → do-calculus validation + data-only edge filtering
Step 9: PINN Validation → 8 physics consistency checks
```

## Step Details

### Step 1: Physics Priors

Two types of prior knowledge injected:

- **Hard mask (52 edges)**: Physically impossible causal paths (e.g., Te cannot cause B0, gas puff cannot directly cause q-profile). These are never discovered regardless of statistical evidence.
- **Soft prior (23 edges)**: Expected causal relationships from plasma physics theory (e.g., PNBI → Te, Ip → q95). These receive a Bayesian bonus during fusion but can be overridden by strong contrary evidence.

Implementation: `fusionmind4/discovery/physics.py` → `PlasmaPhysicsPriors`

### Step 2: NOTEARS

Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning. Solves the continuous optimization problem:

```
min ||X - XW||² + λ₁||W||₁
s.t. tr(e^(W∘W)) - d = 0    (acyclicity constraint)
```

With bootstrap (N=10 by default) for stability estimates.

Implementation: `fusionmind4/discovery/notears.py`

### Step 3: DYNOTEARS

Temporal extension of NOTEARS that learns both contemporaneous (W) and lagged (A) adjacency matrices from time-series data.

### Step 4: Granger Causality

Physics-informed lag selection (lag = 1–5 timesteps), with multivariate conditioning. Tests whether past values of X improve prediction of Y beyond Y's own history.

Implementation: `fusionmind4/discovery/granger.py`

### Step 5: PC Algorithm

Constraint-based causal discovery using conditional independence tests. Acts as a veto: edges passing NOTEARS/Granger but failing conditional independence are penalized.

Implementation: `fusionmind4/discovery/pc.py`

### Step 6: Bayesian Fusion

The key innovation — combining all four algorithms with physics priors:

```
score(i→j) = w_NOTEARS·s_NOTEARS + w_DYNO·s_DYNO + w_Granger·s_Granger
           + physics_bonus(i,j) - pc_penalty(i,j) + agreement_bonus
```

Where:
- `agreement_bonus` = 0.15 if 3+ methods agree
- `physics_bonus` = 0.10 for expected edges
- `pc_penalty` = 0.20 for edges failing conditional independence
- Triple-tier threshold: High (>0.5), Medium (>0.3 + 2 methods), Low (>0.2 + physics)

Implementation: `fusionmind4/discovery/ensemble.py` → `EnsembleCPDE.discover()`

### Step 7: Physics-Aware Cycle-Breaking

If the fused graph contains cycles, edges are removed in order of lowest physics support, preserving physics-critical edges.

### Step 8: Interventional Scoring

Using CPC's do-calculus engine, each discovered edge is validated by simulating `do(X)` interventions and checking whether Y changes as expected.

Implementation: `fusionmind4/discovery/interventional.py`

### Step 9: PINN Validation

Eight physics consistency checks on the final DAG:

1. Energy conservation (heating → stored energy path exists)
2. Momentum conservation
3. Particle conservation (gas puff → density path)
4. Maxwell's equations (current → B-field → q-profile chain)
5. MHD stability (β → MHD path)
6. Ohm's law (current diffusion)
7. Radiation balance (ne, Te → Prad)
8. Actuator exogeneity (actuators have no parents in DAG)

Implementation: `fusionmind4/discovery/physics.py` → `PhysicsValidator`

---

## Performance

### FM3-Lite Synthetic Data (14 variables, 23 ground truth edges)

| Timesteps | F1 | Precision | Recall | TP | FP |
|-----------|-----|-----------|--------|----|----|
| 500 | 73.7% | 93.3% | 60.9% | 14 | 1 |
| 1,000 | 80.0% | 94.1% | 69.6% | 16 | 1 |
| 2,000 | 85.7% | 94.7% | 78.3% | 18 | 1 |

### Real MAST Data (10 variables)

| Metric | Value |
|--------|-------|
| F1 | 91.9% |
| Precision | 89.5% |
| Recall | 94.4% |
| Sign Accuracy | 100% |

---

## Usage

```python
from fusionmind4.discovery import EnsembleCPDE
from fusionmind4.utils import FM3LitePhysicsEngine

engine = FM3LitePhysicsEngine(n_samples=10000, seed=42)
data, interventions = engine.generate()

cpde = EnsembleCPDE(config={
    "n_bootstrap": 10,
    "threshold": 0.32,
    "use_physics_priors": True,
    "use_pc_veto": True,
    "use_interventional": True,
})
results = cpde.discover(data, interventional_data=interventions)

# Access results
dag = results['dag']           # NetworkX DiGraph
f1 = results['f1']            # F1 score vs ground truth
edges = results['edges']       # List of (cause, effect, weight)
physics = results['physics_passed']  # PINN checks passed
```
