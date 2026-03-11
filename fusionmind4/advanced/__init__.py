"""FusionMind 4.0 Advanced Modules — activate when data conditions are met.

PINO:               Needs 1D+ profile data (Te(r), ne(r)) at ≥1kHz
SelfSupervisedPT:   Needs ≥1M unlabeled timepoints
HybridPINNTGN:      Needs 1D+ profile data + spatial grid

All modules check activation conditions and gracefully skip if not met.
"""
from .pino import PhysicsInformedNeuralOperator
from .self_supervised import SelfSupervisedPretrainer
from .pinn_tgn import HybridPINNTemporalGraphNetwork
