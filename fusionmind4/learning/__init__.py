"""FusionMind 4.0 Learning Module — Patent Family PF7: CausalShield-RL.

Combines causal inference (PF1-PF2) with reinforcement learning to create
the first explainable, causally-constrained RL controller for tokamak plasma.

Components:
    NeuralSCM        — Differentiable structural causal model (upgrades linear SCM)
    GymPlasmaEnv      — Gymnasium environment wrapping FM3Lite / TORAX
    CausalReward      — Causal reward shaping using discovered DAG
    CausalRLHybrid    — Main integration: CPDE + CPC + PPO with causal safety shield

Author: Dr. Mladen Mešter, dr.med., March 2026
"""

from .neural_scm import NeuralSCM
from .gym_plasma_env import GymPlasmaEnv
from .causal_reward import CausalRewardShaper
from .causal_rl_hybrid import CausalRLHybrid

__all__ = [
    "NeuralSCM",
    "GymPlasmaEnv",
    "CausalRewardShaper",
    "CausalRLHybrid",
]
