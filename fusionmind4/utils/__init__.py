"""Shared utilities for FusionMind 4.0."""
from .plasma_vars import PLASMA_VARS, N_VARS, ACTUATOR_IDS, VAR_NAMES
from .fm3lite import FM3LitePhysicsEngine

__all__ = ["PLASMA_VARS", "N_VARS", "ACTUATOR_IDS", "VAR_NAMES", "FM3LitePhysicsEngine"]
