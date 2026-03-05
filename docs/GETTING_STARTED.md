# Getting Started with FusionMind 4.0

From zero to running causal plasma analysis in 10 minutes.

---

## Step 1: Install

```bash
git clone https://github.com/mladen1312/FusionMind-4-CausalPlasma.git
cd FusionMind-4-CausalPlasma
pip install -e ".[full]"
```

### Compile C++ Engine (optional but recommended — 500x speedup)

```bash
cd fusionmind4/realtime/cpp
g++ -O3 -march=native -shared -fPIC -std=c++17 -o libfusionmind_stack.so stack_api.cpp
g++ -O3 -march=native -shared -fPIC -std=c++17 -o libfusionmind_rt.so fast_engine_api.cpp
cd ../../..
```

### Verify Installation

```bash
python -c "import fusionmind4; print(f'FusionMind {fusionmind4.__version__} installed')"
python -m pytest tests/test_stack.py -q  # Should show 37 passed
```

---

## Step 2: Your First Causal Analysis (Synthetic Data)

```python
from fusionmind4.discovery.ensemble import EnsembleCPDE
from fusionmind4.utils.plasma_vars import FM3LitePhysicsEngine

# Generate realistic synthetic plasma data
engine = FM3LitePhysicsEngine(n_samples=5000, seed=42)
data, interventions = engine.generate()

# Discover causal structure
cpde = EnsembleCPDE(config={"n_bootstrap": 10, "threshold": 0.32})
results = cpde.discover(data, interventional_data=interventions)

print(f"F1: {results['f1']:.1%}")
print(f"Discovered {len(results['edges'])} causal edges")
print(f"Physics checks: {results['physics_passed']}/{results['physics_total']}")
```

---

## Step 3: Real Tokamak Data (FAIR-MAST)

```python
import numpy as np
import s3fs
import zarr

# Connect to UKAEA FAIR-MAST archive (public, no credentials needed)
fs = s3fs.S3FileSystem(anon=True, client_kwargs={
    'endpoint_url': 'https://s3.echo.stfc.ac.uk'})

# Download a shot
shot_id = 30421
store = s3fs.S3Map(root=f'mast/level1/shots/{shot_id}.zarr', s3=fs, check=False)
root = zarr.open(store, mode='r')

# Extract plasma parameters from EFIT equilibrium reconstruction
efm = root['efm']
times = np.array(efm['all_times'])
mask = times > 0.01  # Plasma phase only

data = np.column_stack([
    np.array(efm['betan'])[mask],
    np.array(efm['betap'])[mask],
    np.array(efm['q_95'])[mask],
    np.array(efm['q_axis'])[mask],
    np.array(efm['elongation'])[mask],
    np.array(efm['li'])[mask],
    np.array(efm['wplasmd'])[mask],
    np.array(efm['betat'])[mask],
])

var_names = ['betan', 'betap', 'q_95', 'q_axis', 'elongation', 'li', 'wplasmd', 'betat']

# Remove NaN rows
valid = ~np.any(np.isnan(data) | np.isinf(data), axis=1)
data = data[valid]
print(f"Loaded {len(data)} timepoints from MAST shot {shot_id}")
```

---

## Step 4: Build Causal Model + Controller

```python
from fusionmind4.discovery.nonlinear_scm import NonlinearPlasmaSCM
from fusionmind4.control.stack import FusionMindStack, Phase, StackConfig, PlasmaState

# Quick CPDE (simplified for tutorial)
from fusionmind4.discovery.ensemble import EnsembleCPDE
cpde = EnsembleCPDE()
result = cpde.discover(data)
dag = result['dag']

# Fit nonlinear structural causal model
scm = NonlinearPlasmaSCM(dag, var_names)
scm.fit(data)
print(scm.summary())

# Create the 4-layer control stack
stack = FusionMindStack(dag, scm, var_names, StackConfig(phase=Phase.PHASE_1))
```

---

## Step 5: Use the Controller

### Phase 1: Evaluate External RL Actions (Wrapper Mode)

```python
# Simulate a plasma state
state = PlasmaState(
    values={'betan': 2.0, 'betap': 0.8, 'q_95': 4.0, 'q_axis': 1.5,
            'elongation': 1.7, 'li': 1.0, 'wplasmd': 50000, 'betat': 1.5},
    timestamp=1.0)

# External RL proposes: increase Ip slightly
external_action = {'betan': 2.1}  # RL wants to push beta up

# FusionMind evaluates causally
cmd = stack.evaluate_external_action(state, external_action)
print(f"Decision: {cmd.explanation}")
print(f"Risk: {cmd.risk_score:.3f}")
print(f"Vetoed: {cmd.vetoed}")
```

### Phase 2: Strategic Control (Upgrade Live)

```python
stack.upgrade_phase(Phase.PHASE_2)
stack.set_targets({'betan': 2.5, 'q_95': 4.0})

cmd = stack.step(state, ['betan', 'q_95'])
print(f"Strategy: {cmd.explanation}")
print(f"Setpoints: {cmd.actuator_values}")
```

### Ask "What If?" (Counterfactual)

```python
# What would have happened if beta had been 20% lower?
cf = stack.counterfactual(state, {'betan': 1.6})
print(f"Counterfactual wplasmd: {cf.get('wplasmd', 'N/A')}")

# What happens if I increase li? (do-calculus)
pred = stack.predict_intervention(state, {'li': 1.5})
print(f"do(li=1.5): q_95 becomes {pred.get('q_95', 'N/A'):.2f}")
```

---

## Step 6: C++ Engine (Sub-Microsecond)

```python
from fusionmind4.realtime.stack_bindings import CppStack

# Create C++ stack (100-1000x faster than Python)
cpp = CppStack(8, phase=3, var_names=var_names)
cpp.load_scm(dag, scm)
cpp.load_safety_limits()
cpp.set_setpoints({'betan': 2.5, 'q_95': 4.0})

# Single step: ~700ns
result = cpp.step(state.values, 1.0, ['betan', 'q_95'])
print(f"C++ decision: {result.decision}, latency: {result.latency_total_ns:.0f}ns")

# Benchmark
bench = cpp.benchmark(n_cycles=10000)
print(f"P50: {bench['p50_ns']:.0f}ns, P95: {bench['p95_ns']:.0f}ns")
```

---

## What To Read Next

| Document | What It Covers |
|----------|---------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | 4-layer stack design, data flow |
| [CPDE_PIPELINE.md](docs/CPDE_PIPELINE.md) | How causal discovery works (9-step pipeline) |
| [STACK_ENGINE.md](docs/STACK_ENGINE.md) | Full stack usage guide |
| [BENCHMARKS.md](docs/BENCHMARKS.md) | All validation results on real data |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment guide |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | Complete API documentation |
| [REAL_DATA_VALIDATION.md](docs/REAL_DATA_VALIDATION.md) | MAST + C-Mod validation details |
| [CAUSALSHIELD_RL.md](docs/CAUSALSHIELD_RL.md) | Causal RL integration (PF7) |

---

## Need Help?

- **Issues:** [github.com/mladen1312/FusionMind-4-CausalPlasma/issues](https://github.com/mladen1312/FusionMind-4-CausalPlasma/issues)
- **Author:** Dr. Mladen Mester — mladen@fusionmind.ai
