"""CPDE — Causal Plasma Discovery Engine (Patent Family PF1)."""
from .ensemble import EnsembleCPDE
from .notears import NOTEARSDiscovery, DYNOTEARSDiscovery
from .granger import GrangerCausalityTest, ConditionalGrangerTest, SpectralGrangerCausality
from .pc import PCAlgorithm
from .interventional import InterventionalScorer
from .physics import validate_physics, get_physics_prior_matrix

__all__ = [
    "EnsembleCPDE",
    "NOTEARSDiscovery",
    "DYNOTEARSDiscovery",
    "GrangerCausalityTest",
    "ConditionalGrangerTest",
    "SpectralGrangerCausality",
    "PCAlgorithm",
    "InterventionalScorer",
    "validate_physics",
    "get_physics_prior_matrix",
]
