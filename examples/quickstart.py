#!/usr/bin/env python3
"""FusionMind 4.0 — Quick Start Example."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionmind4.discovery import EnsembleCPDE
from fusionmind4.utils import FM3LitePhysicsEngine

# Generate data
engine = FM3LitePhysicsEngine(n_samples=10000, seed=42)
data, interventions = engine.generate()

# Run CPDE
cpde = EnsembleCPDE(config={"n_bootstrap": 10, "threshold": 0.32})
results = cpde.discover(data, interventional_data=interventions)

print(f"F1: {results['f1']:.1%}")
print(f"Precision: {results['precision']:.1%}")
print(f"Recall: {results['recall']:.1%}")
print(f"Physics: {results['physics_passed']}/{results['physics_total']}")
