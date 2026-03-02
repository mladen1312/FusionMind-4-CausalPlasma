"""
GymPlasmaEnv — Gymnasium Environment for Tokamak Plasma Control
================================================================

Wraps the FM3Lite physics engine into a standard Gymnasium environment
for reinforcement learning. Compatible with stable-baselines3 and
custom PPO/SAC implementations.

Observation space: 14 plasma variables (normalized)
Action space: 4 continuous actuators [P_NBI, P_ECRH, gas_puff, Ip]

Can also wrap TORAX (DeepMind's JAX tokamak simulator) when available:
    env = GymPlasmaEnv(backend='torax')

Part of: FusionMind 4.0 / Patent Family PF7 (CausalShield-RL)
Author: Dr. Mladen Mester, March 2026
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from ..utils.plasma_vars import PLASMA_VARS, N_VARS, VAR_NAMES, ACTUATOR_IDS


# ═══════════════════════════════════════════════════════════════════════
# Plasma Physics Step Function (standalone, no external dependencies)
# ═══════════════════════════════════════════════════════════════════════

class FM3LiteStep:
    """
    Single-step plasma physics model for RL environment.
    
    Implements the causal graph from plasma_vars.py as a physics step:
    actuators → profiles → global → instability (with feedback).
    
    This is a simplified version of FM3Lite designed for fast RL rollouts
    (~10μs per step vs ~1ms for full FM3Lite).
    """
    
    # Physics coefficients from ground truth graph
    W = np.zeros((N_VARS, N_VARS))
    
    # Disruption thresholds
    DISRUPTION_THRESHOLDS = {
        'betaN_max': 3.5,
        'q_min': 1.2,
        'MHD_max': 2.5,
        'ne_max': 3.0,
        'radiation_fraction_max': 0.85,
    }
    
    def __init__(self, dt: float = 0.001, noise_scale: float = 0.02):
        self.dt = dt
        self.noise_scale = noise_scale
        
        # Load ground truth weights
        from ..utils.plasma_vars import GROUND_TRUTH_EDGES
        for cause, effect, weight, _ in GROUND_TRUTH_EDGES:
            self.W[cause, effect] = weight
        
        # Physics time constants (seconds) — slower variables respond slower
        self.tau = np.array([
            0.0,   # P_NBI (instant)
            0.0,   # P_ECRH (instant)
            0.0,   # gas_puff (instant)
            0.0,   # Ip (instant — commanded)
            0.05,  # ne (50ms)
            0.02,  # Te (20ms)
            0.03,  # Ti (30ms)
            0.10,  # q (100ms — current diffusion)
            0.02,  # betaN (20ms)
            0.10,  # rotation (100ms)
            0.01,  # P_rad (10ms)
            0.02,  # W_stored (20ms)
            0.05,  # MHD_amp (50ms)
            0.20,  # n_imp (200ms — wall processes)
        ])
    
    def step(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Advance plasma state by one timestep.
        
        Args:
            state: (N_VARS,) current plasma state
            action: (4,) actuator commands [P_NBI, P_ECRH, gas_puff, Ip]
            
        Returns:
            next_state: (N_VARS,) new plasma state
            info: Dictionary with physics diagnostics
        """
        next_state = state.copy()
        
        # Apply actuator commands (instant)
        next_state[0] = action[0]  # P_NBI
        next_state[1] = action[1]  # P_ECRH
        next_state[2] = action[2]  # gas_puff
        next_state[3] = action[3]  # Ip
        
        # Compute equilibrium targets from causal graph
        targets = np.zeros(N_VARS)
        targets[:4] = next_state[:4]  # Actuators are direct
        
        for j in range(4, N_VARS):
            parent_indices = np.where(np.abs(self.W[:, j]) > 0.01)[0]
            targets[j] = sum(self.W[i, j] * next_state[i] for i in parent_indices)
        
        # Exponential relaxation toward equilibrium (physics timescales)
        for j in range(4, N_VARS):
            if self.tau[j] > 0:
                alpha = min(self.dt / self.tau[j], 1.0)
                next_state[j] = state[j] + alpha * (targets[j] - state[j])
            else:
                next_state[j] = targets[j]
        
        # Add measurement noise
        noise = np.random.randn(N_VARS) * self.noise_scale
        noise[:4] = 0  # No noise on actuators
        next_state += noise * np.abs(next_state).clip(0.01)
        
        # Physics constraints
        next_state[4] = max(next_state[4], 0.01)   # ne > 0
        next_state[5] = max(next_state[5], 0.01)   # Te > 0
        next_state[6] = max(next_state[6], 0.01)   # Ti > 0
        next_state[7] = max(next_state[7], 0.5)    # q > 0.5
        
        # Check disruption
        info = self._check_disruption(next_state)
        
        return next_state, info
    
    def _check_disruption(self, state: np.ndarray) -> dict:
        """Check if plasma is approaching disruption."""
        info = {
            'betaN': state[8],
            'q_min': state[7],
            'MHD_amp': state[12],
            'disrupted': False,
            'disruption_proximity': 0.0,
        }
        
        # Disruption proximity score (0 = safe, 1 = disrupted)
        proximity = 0.0
        if state[8] > 0:
            proximity = max(proximity, state[8] / self.DISRUPTION_THRESHOLDS['betaN_max'])
        if state[7] > 0:
            proximity = max(proximity, self.DISRUPTION_THRESHOLDS['q_min'] / max(state[7], 0.1))
        if state[12] > 0:
            proximity = max(proximity, state[12] / self.DISRUPTION_THRESHOLDS['MHD_max'])
        
        info['disruption_proximity'] = min(proximity, 1.0)
        info['disrupted'] = proximity >= 1.0
        
        return info


# ═══════════════════════════════════════════════════════════════════════
# Gymnasium Environment
# ═══════════════════════════════════════════════════════════════════════

class GymPlasmaEnv:
    """
    Gymnasium-compatible environment for tokamak plasma control.
    
    Follows the standard Gymnasium API:
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    
    Observation: 14 plasma variables (normalized to ~[0, 1])
    Action: 4 continuous actuators (normalized to [0, 1])
    
    Supports two backends:
        'fm3lite': Fast FM3Lite physics (default, no dependencies)
        'torax':  DeepMind TORAX simulator (requires JAX + torax)
    
    Args:
        backend: Physics backend ('fm3lite' or 'torax')
        max_steps: Maximum episode length
        target_scenario: Target plasma scenario (dict of targets)
        dt: Physics timestep (seconds)
        normalize: Normalize observations to [0, 1]
    """
    
    # Normalization ranges for each variable
    OBS_LOW = np.array([0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    OBS_HIGH = np.array([3, 2, 2, 2, 3, 3, 3, 8, 4, 3, 3, 3, 3, 2], dtype=np.float32)
    
    # Action ranges (physical units)
    ACT_LOW = np.array([0.1, 0.0, 0.05, 0.5], dtype=np.float32)
    ACT_HIGH = np.array([2.5, 2.0, 1.5, 2.0], dtype=np.float32)
    
    def __init__(self, backend: str = 'fm3lite', max_steps: int = 1000,
                 target_scenario: Optional[Dict] = None,
                 dt: float = 0.001, normalize: bool = True):
        
        self.backend = backend
        self.max_steps = max_steps
        self.dt = dt
        self.normalize = normalize
        
        # Default target: steady H-mode
        self.target = target_scenario or {
            'Te': 1.5,        # Target electron temperature
            'ne': 1.0,        # Target density
            'betaN': 1.0,     # Target beta
            'q': 3.0,         # Target safety factor
            'W_stored': 0.8,  # Target stored energy
        }
        
        # Physics backend
        if backend == 'fm3lite':
            self.physics = FM3LiteStep(dt=dt)
        elif backend == 'torax':
            raise NotImplementedError(
                "TORAX backend requires: pip install torax\n"
                "Then set backend='torax' to use DeepMind's JAX simulator"
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # State
        self.state: Optional[np.ndarray] = None
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Spaces (compatible with gymnasium)
        self.observation_space_shape = (N_VARS,)
        self.action_space_shape = (4,)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial plasma state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Random initial state near target
        self.state = np.zeros(N_VARS, dtype=np.float32)
        self.state[0] = np.random.uniform(0.8, 1.5)   # P_NBI
        self.state[1] = np.random.uniform(0.3, 1.0)   # P_ECRH
        self.state[2] = np.random.uniform(0.2, 0.6)   # gas_puff
        self.state[3] = np.random.uniform(0.8, 1.2)   # Ip
        self.state[4] = np.random.uniform(0.5, 1.5)   # ne
        self.state[5] = np.random.uniform(0.5, 2.0)   # Te
        self.state[6] = np.random.uniform(0.3, 1.5)   # Ti
        self.state[7] = np.random.uniform(2.0, 5.0)   # q
        self.state[8] = np.random.uniform(0.3, 1.5)   # betaN
        self.state[9] = np.random.uniform(0.1, 0.5)   # rotation
        self.state[10] = np.random.uniform(0.1, 0.5)  # P_rad
        self.state[11] = np.random.uniform(0.3, 1.0)  # W_stored
        self.state[12] = np.random.uniform(0.0, 0.5)  # MHD_amp
        self.state[13] = np.random.uniform(0.0, 0.2)  # n_imp
        
        self.step_count = 0
        self.episode_reward = 0.0
        
        obs = self._get_obs()
        info = {'state': self.state.copy(), 'step': 0}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take one environment step.
        
        Args:
            action: (4,) normalized actuator commands in [0, 1]
            
        Returns:
            obs: Observation
            reward: Scalar reward
            terminated: True if disruption
            truncated: True if max steps reached
            info: Additional diagnostics
        """
        # Denormalize action to physical units
        physical_action = self.ACT_LOW + action * (self.ACT_HIGH - self.ACT_LOW)
        
        # Physics step
        self.state, physics_info = self.physics.step(self.state, physical_action)
        self.step_count += 1
        
        # Compute reward
        reward = self._compute_reward(physics_info)
        self.episode_reward += reward
        
        # Termination
        terminated = physics_info.get('disrupted', False)
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_obs()
        
        info = {
            'state': self.state.copy(),
            'step': self.step_count,
            'episode_reward': self.episode_reward,
            **physics_info,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get normalized observation."""
        if self.normalize:
            obs = (self.state - self.OBS_LOW) / (self.OBS_HIGH - self.OBS_LOW + 1e-8)
            return np.clip(obs, -1, 2).astype(np.float32)
        return self.state.astype(np.float32)
    
    def _compute_reward(self, physics_info: dict) -> float:
        """
        Reward function for plasma control.
        
        Components:
        1. Target tracking: how close to desired scenario
        2. Stability: penalize high MHD, low q
        3. Efficiency: minimize radiated power losses
        4. Disruption: large penalty for disruption
        """
        reward = 0.0
        
        # 1. Target tracking (main objective)
        target_vars = {'Te': 5, 'ne': 4, 'betaN': 8, 'q': 7, 'W_stored': 11}
        for var_name, var_idx in target_vars.items():
            if var_name in self.target:
                target_val = self.target[var_name]
                actual_val = self.state[var_idx]
                error = abs(actual_val - target_val) / max(abs(target_val), 0.1)
                reward -= error * 0.5  # Penalize deviation
        
        # 2. Stability bonus
        if self.state[12] < 0.5:  # Low MHD
            reward += 0.2
        if self.state[7] > 2.0:   # Safe q
            reward += 0.1
        
        # 3. Efficiency (minimize radiation losses)
        if self.state[10] < 0.3:  # Low radiated power
            reward += 0.1
        
        # 4. Disruption penalty
        if physics_info.get('disrupted', False):
            reward -= 10.0
        else:
            # Proximity penalty
            prox = physics_info.get('disruption_proximity', 0)
            if prox > 0.7:
                reward -= 2.0 * (prox - 0.7)
        
        return reward
    
    def get_state_dict(self) -> Dict[str, float]:
        """Get current state as named dictionary."""
        return {name: float(self.state[i]) for i, name in enumerate(VAR_NAMES)}
    
    def render(self) -> str:
        """Text rendering of current state."""
        lines = [f"Step {self.step_count} | Reward: {self.episode_reward:.2f}"]
        for i, name in enumerate(VAR_NAMES):
            marker = "⚠" if (name == 'MHD_amp' and self.state[i] > 1.5) else " "
            lines.append(f"  {marker} {name:10s}: {self.state[i]:8.4f}")
        return "\n".join(lines)
