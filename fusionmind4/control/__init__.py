"""CPC — Counterfactual Plasma Controller (Patent Family PF2).

Implements Pearl's three levels of causal reasoning:
  Level 1: P(Y|X)        — Association (all competitors)
  Level 2: P(Y|do(X))    — Intervention (FusionMind 4.0)
  Level 3: P(Y_x|X,Y)   — Counterfactual (FusionMind 4.0)
"""
from .scm import PlasmaSCM
from .interventions import InterventionEngine, CounterfactualEngine
from .controller import CounterfactualPlasmaController, ControlDecision

__all__ = [
    "PlasmaSCM",
    "InterventionEngine",
    "CounterfactualEngine",
    "CounterfactualPlasmaController",
    "ControlDecision",
]
