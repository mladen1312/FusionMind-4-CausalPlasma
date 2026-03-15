"""
CausalRL_MLX — Complete CausalShield-RL Training on Apple Silicon
==================================================================

Orchestrates the full causal RL training pipeline on MLX:

  Shot Data → CPDE → NeuralSCM_MLX (world model)
                          ↓
                    PPO_MLX (policy)  ←  CausalReward
                          ↓
                    GymPlasmaEnv (simulator)

The PPO agent uses the NeuralSCM as a differentiable world model,
enabling model-based planning through backpropagation along causal paths.

Part of: FusionMind 4.0 / Patent Family PF7 (CausalShield-RL)
Author: Dr. Mladen Mešter, dr.med., March 2026
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .neural_scm_mlx import NeuralSCM_MLX
from .policy_mlx import (
    PPOPolicy_MLX, ValueNetwork_MLX, RolloutBuffer_MLX,
    PPOConfig_MLX, ppo_update,
)


@dataclass
class CausalRLConfig:
    """Configuration for CausalShield-RL training."""
    # RL
    ppo: PPOConfig_MLX = None
    n_episodes: int = 200
    max_steps_per_episode: int = 100
    rollout_steps: int = 512

    # Causal reward shaping
    causal_bonus_weight: float = 0.3
    safety_penalty_weight: float = 1.0
    stability_margin_q: float = 1.5
    max_greenwald_fraction: float = 0.9

    # Neural SCM world model
    scm_hidden_dim: int = 32
    scm_fit_epochs: int = 500
    scm_online_epochs: int = 50
    scm_lr: float = 1e-3

    def __post_init__(self):
        if self.ppo is None:
            self.ppo = PPOConfig_MLX()


class CausalRL_MLX:
    """
    CausalShield-RL: Causal RL for tokamak plasma control on MLX.

    Training loop:
    1. Discover causal graph (CPDE — stays NumPy, not bottleneck)
    2. Fit NeuralSCM_MLX as differentiable world model
    3. Collect rollouts in GymPlasmaEnv with causal reward shaping
    4. PPO update on MLX (policy + value on GPU)
    5. Online SCM update with new shot data
    6. Repeat

    Usage:
        from fusionmind4.mlx_backend import CausalRL_MLX

        agent = CausalRL_MLX(variable_names, dag, config)
        agent.fit_world_model(training_data)
        metrics = agent.train(env_fn=make_plasma_env)
    """

    def __init__(self, variable_names: List[str], dag: np.ndarray,
                 config: CausalRLConfig = None):
        self.names = variable_names
        self.n_vars = len(variable_names)
        self.dag = dag
        self.config = config or CausalRLConfig()

        # Identify actuators (exogenous: no parents in DAG)
        self.actuator_indices = [
            i for i in range(self.n_vars)
            if np.sum(np.abs(dag[:, i]) > 0.01) == 0
        ]
        self.actuator_names = [variable_names[i] for i in self.actuator_indices]
        self.n_actuators = len(self.actuator_indices)

        # World model
        self.scm = NeuralSCM_MLX(
            variable_names, dag, hidden_dim=self.config.scm_hidden_dim
        )

        # Policy and value networks
        obs_dim = self.n_vars
        act_dim = self.n_actuators
        self.policy = PPOPolicy_MLX(obs_dim, act_dim, self.config.ppo)
        self.value_fn = ValueNetwork_MLX(obs_dim, self.config.ppo)

        # Rollout buffer
        self.buffer = RolloutBuffer_MLX()

        # Training history
        self.history: List[Dict] = []

    def fit_world_model(self, data: np.ndarray, verbose: bool = True):
        """Fit Neural SCM world model from historical data."""
        if verbose:
            print("\n" + "=" * 60)
            print("CausalShield-RL: Fitting World Model (MLX)")
            print("=" * 60)
        self.scm.fit(
            data, n_epochs=self.config.scm_fit_epochs,
            lr=self.config.scm_lr, verbose=verbose,
        )

    def compute_causal_reward(self, state: Dict[str, float],
                               next_state: Dict[str, float],
                               action: np.ndarray) -> Tuple[float, float, float]:
        """
        Causal reward shaping using do-calculus.

        Returns:
            (base_reward, causal_bonus, safety_penalty)
        """
        # Base reward: plasma performance (βN high, frad low)
        beta_n = next_state.get("beta_n", 0.0)
        f_rad = next_state.get("f_rad", 0.5)
        base_reward = beta_n - 0.5 * f_rad

        # Causal bonus: reward actions whose causal effect aligns with goals
        causal_bonus = 0.0
        intervention = {
            self.actuator_names[i]: float(action[i])
            for i in range(self.n_actuators)
        }
        predicted = self.scm.do_intervention(state, intervention)
        pred_beta = predicted.get("beta_n", 0.0)
        if pred_beta > beta_n:
            causal_bonus = self.config.causal_bonus_weight * (pred_beta - beta_n)

        # Safety penalty: penalize actions that causally lead to instability
        safety_penalty = 0.0
        pred_q = predicted.get("q95", 3.0)
        if pred_q < self.config.stability_margin_q:
            safety_penalty += self.config.safety_penalty_weight * (
                self.config.stability_margin_q - pred_q
            )

        pred_fgw = predicted.get("f_GW", 0.5)
        if pred_fgw > self.config.max_greenwald_fraction:
            safety_penalty += self.config.safety_penalty_weight * (
                pred_fgw - self.config.max_greenwald_fraction
            )

        return base_reward, causal_bonus, safety_penalty

    def train_step(self, env) -> Dict[str, float]:
        """
        Collect rollout + PPO update.

        Args:
            env: Gym-like environment with step(action) → (obs, reward, done, info)

        Returns:
            Training metrics dict
        """
        self.buffer.clear()

        obs = env.reset()
        episode_rewards = []
        ep_reward = 0.0

        for step in range(self.config.rollout_steps):
            # Get action from policy
            action, log_prob = self.policy.sample_action(obs)
            value = self.value_fn.value_np(obs)

            # Environment step
            next_obs, env_reward, done, info = env.step(action)

            # Causal reward shaping
            state_dict = {self.names[i]: float(obs[i]) for i in range(self.n_vars)}
            next_dict = {self.names[i]: float(next_obs[i]) for i in range(self.n_vars)}
            base_r, causal_b, safety_p = self.compute_causal_reward(
                state_dict, next_dict, action
            )
            shaped_reward = base_r + causal_b - safety_p

            self.buffer.add(
                obs, action, shaped_reward, value, log_prob, done,
                causal_bonus=causal_b, safety_penalty=safety_p,
            )

            ep_reward += shaped_reward
            obs = next_obs

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs = env.reset()

        # Compute GAE
        last_value = self.value_fn.value_np(obs)
        self.buffer.compute_gae(
            self.config.ppo.gamma, self.config.ppo.gae_lambda, last_value
        )

        # PPO update on MLX
        update_metrics = ppo_update(
            self.policy, self.value_fn, self.buffer, self.config.ppo
        )

        metrics = {
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "n_episodes": len(episode_rewards),
            "mean_causal_bonus": float(np.mean(self.buffer.causal_bonuses)),
            "mean_safety_penalty": float(np.mean(self.buffer.safety_penalties)),
            **update_metrics,
        }
        self.history.append(metrics)
        return metrics

    def train(self, env, n_iterations: int = None, verbose: bool = True) -> List[Dict]:
        """
        Full training loop.

        Args:
            env: Gym-like plasma environment
            n_iterations: Override config.n_episodes
            verbose: Print progress
        """
        n_iter = n_iterations or self.config.n_episodes
        if verbose:
            print(f"\nCausalShield-RL Training ({n_iter} iterations)")
            print(f"  Policy: {sum(p.size for _, p in self.policy.trainable_parameters()):,} params")
            print(f"  Value:  {sum(p.size for _, p in self.value_fn.trainable_parameters()):,} params")
            print(f"  Device: {mx.default_device()}")

        for i in range(n_iter):
            metrics = self.train_step(env)
            if verbose and (i % 10 == 0 or i == n_iter - 1):
                print(
                    f"  Iter {i:4d}: reward={metrics['mean_reward']:+.3f}  "
                    f"π_loss={metrics.get('policy_loss', 0):.4f}  "
                    f"V_loss={metrics.get('value_loss', 0):.4f}  "
                    f"causal_b={metrics['mean_causal_bonus']:.3f}  "
                    f"safety_p={metrics['mean_safety_penalty']:.3f}"
                )

        return self.history

    def save(self, path: str):
        """Save policy + value + SCM weights."""
        import mlx.core as mx
        weights = {
            "policy": dict(self.policy.parameters()),
            "value": dict(self.value_fn.parameters()),
            "scm": dict(self.scm.parameters()),
        }
        mx.savez(path, **{f"{k}/{kk}": v for k, d in weights.items()
                          for kk, v in d.items()})

    def summary(self) -> str:
        lines = [
            "CausalShield-RL (MLX Backend)",
            "=" * 50,
            f"Device: {mx.default_device()}",
            f"Actuators: {self.actuator_names}",
            f"Variables: {self.n_vars}",
            f"DAG edges: {int(np.sum(np.abs(self.dag) > 0.01))}",
            f"Policy params: {sum(p.size for _, p in self.policy.trainable_parameters()):,}",
            f"Value params: {sum(p.size for _, p in self.value_fn.trainable_parameters()):,}",
            "",
            self.scm.summary(),
        ]
        return "\n".join(lines)
