"""FusionMind 4.0 Real-Time Subsystem

Dual-mode disruption prediction + causal control bridge for
live tokamak operation.

Patent Families: PF1 (CPDE), PF2 (CPC), PF7 (CausalShield-RL)

Architecture:
                 ┌─────────────────────────────────────────────┐
  Live           │          Real-Time Pipeline                 │
  Diagnostics ──►│                                             │
  (< 1 ms)       │  ┌──────────┐   ┌──────────┐              │
                 │  │ Fast ML  │   │ Causal   │              │
                 │  │Predictor │   │Predictor │              │
                 │  │ (< 1 ms) │   │ (< 5 ms) │              │
                 │  └────┬─────┘   └────┬─────┘              │
                 │       │              │                      │
                 │       ▼              ▼                      │
                 │  ┌────────────────────────┐                │
                 │  │   Fusion + Arbitrator  │                │
                 │  │   (safety override)    │                │
                 │  └────────────┬───────────┘                │
                 │               │                             │
                 │       ┌───────▼───────┐                    │
                 │       │ Control Bridge│──► Actuator Cmds   │
                 │       │ (do-calculus) │                    │
                 │       └───────────────┘                    │
                 └─────────────────────────────────────────────┘
"""
from .predictor import (
    CausalDisruptionPredictor,
    FastMLPredictor,
    DualModePredictor,
)
from .control_bridge import RealtimeControlBridge
from .streaming import StreamingPlasmaInterface

__all__ = [
    "CausalDisruptionPredictor",
    "FastMLPredictor",
    "DualModePredictor",
    "RealtimeControlBridge",
    "StreamingPlasmaInterface",
]
