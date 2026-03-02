# API Reference — FusionMind 4.0

## Core Modules

### `fusionmind4.discovery` — Causal Discovery (PF1)

#### `EnsembleCPDE`

Main entry point for causal discovery.

```python
from fusionmind4.discovery import EnsembleCPDE

cpde = EnsembleCPDE(config={
    "n_bootstrap": 10,       # Bootstrap iterations for NOTEARS
    "threshold": 0.32,       # Edge inclusion threshold
    "use_physics_priors": True,
    "use_pc_veto": True,
    "use_interventional": True,
})

results = cpde.discover(
    data,                     # np.ndarray (n_samples, n_variables)
    interventional_data=None, # Optional dict of intervention results
    variable_names=None,      # Optional list of variable names
)
# Returns dict: dag, edges, f1, precision, recall, physics_passed, physics_total
```

#### `NOTEARSDiscovery`

```python
from fusionmind4.discovery.notears import NOTEARSDiscovery
notears = NOTEARSDiscovery(lambda1=0.1, max_iter=100)
W = notears.fit(data)                    # Returns adjacency matrix
W, stability = notears.fit_bootstrap(data, n_bootstrap=10)
```

#### `GrangerCausality`

```python
from fusionmind4.discovery.granger import GrangerCausality
gc = GrangerCausality(max_lag=5, alpha=0.05)
G = gc.fit(data)                          # Returns adjacency matrix
```

#### `PCAlgorithm`

```python
from fusionmind4.discovery.pc import PCAlgorithm
pc = PCAlgorithm(alpha=0.05)
skeleton = pc.fit(data)                   # Returns undirected adjacency
```

#### `PlasmaPhysicsPriors` / `PhysicsValidator`

```python
from fusionmind4.discovery.physics import PlasmaPhysicsPriors, PhysicsValidator
priors = PlasmaPhysicsPriors()
mask = priors.get_hard_mask(n_vars)       # Forbidden edges
soft = priors.get_soft_prior(n_vars)      # Expected edges

validator = PhysicsValidator()
passed, total, details = validator.validate(dag, variable_names)
```

---

### `fusionmind4.control` — Counterfactual Controller (PF2)

#### `PlasmaSCM`

```python
from fusionmind4.control import PlasmaSCM
scm = PlasmaSCM()
scm.fit(dag, data)                        # Fit linear SCM to data
prediction = scm.predict(observation)     # Forward prediction
equations = scm.get_equations()           # Human-readable equations
```

#### `InterventionEngine`

```python
from fusionmind4.control import InterventionEngine
ie = InterventionEngine(scm)
result = ie.do_intervention({'P_ECRH': 10.0})    # P(Y|do(X=x))
paths = ie.trace_causal_paths('P_ECRH', 'Te')    # Explain mechanism
```

#### `CounterfactualEngine`

```python
from fusionmind4.control import CounterfactualEngine
ce = CounterfactualEngine(scm)
cf = ce.counterfactual(
    factual={'P_NBI': 5.0, 'Te': 3.2},
    intervention={'P_NBI': 8.0}
)
# "What would Te have been if P_NBI had been 8.0 instead of 5.0?"
```

---

### `fusionmind4.foundation` — Foundation Model (PF3)

#### `DimensionlessTokenizer`

```python
from fusionmind4.foundation import DimensionlessTokenizer
tokenizer = DimensionlessTokenizer()
tokens = tokenizer.tokenize(plasma_state)  # Returns [βn, ν*, ρ*, q95, H98]
consistency = tokenizer.cross_device_cv(multi_device_data)
```

---

### `fusionmind4.learning` — CausalShield-RL (PF7)

#### `CausalRLHybrid`

```python
from fusionmind4.learning import CausalRLHybrid

hybrid = CausalRLHybrid()
hybrid.discover_causal_graph(data, interventional_data)
hybrid.fit_world_model(data)
hybrid.train(n_episodes=300)

action = hybrid.act(observation)                  # Get action
result = hybrid.act_with_explanation(observation)  # Action + explanation
hybrid.online_update(new_data)                    # Online adaptation
summary = hybrid.summary()                         # System status
```

#### `NeuralSCM`

```python
from fusionmind4.learning import NeuralSCM

nscm = NeuralSCM(variable_names, dag)
nscm.fit(data, epochs=100, lr=1e-3)
prediction = nscm.predict(observation)
intervention = nscm.do_intervention(observation, {'P_NBI': 8.0})
jacobian = nscm.jacobian(observation)              # Sensitivity matrix
nscm.update(new_data, epochs=10)                  # Online update
```

---

### `fusionmind4.utils` — Utilities

#### `FM3LitePhysicsEngine`

```python
from fusionmind4.utils import FM3LitePhysicsEngine

engine = FM3LitePhysicsEngine(n_samples=10000, seed=42, noise_level=0.03)
data, interventional = engine.generate()
# data: np.ndarray (n_samples, 14)
# interventional: dict mapping variable names to intervention results
```

#### `PlasmaVariables`

```python
from fusionmind4.utils.plasma_vars import VARIABLE_NAMES, GROUND_TRUTH_EDGES, evaluate_dag
f1, precision, recall, tp, fp, fn = evaluate_dag(discovered_dag)
```
