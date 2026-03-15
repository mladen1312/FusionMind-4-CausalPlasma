#!/usr/bin/env python3
"""
train_causal_rl.py — Train CausalShield-RL on FM3Lite Data
============================================================

Complete pipeline:
1. Generate synthetic plasma data via FM3Lite
2. Run CPDE to discover causal graph
3. Fit Neural SCM as differentiable world model
4. Train PPO agent with causal reward shaping
5. Evaluate and report results

Usage:
    python scripts/train_causal_rl.py
    python scripts/train_causal_rl.py --episodes 1000 --eval-every 100

Part of: FusionMind 4.0 / Patent Family PF7 (CausalShield-RL)
Author: Dr. Mladen Mešter, dr.med., March 2026
"""

import sys
import os
import time
import argparse
import numpy as np

# Ensure fusionmind4 is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusionmind4.utils.fm3lite import FM3LitePhysicsEngine
from fusionmind4.utils.plasma_vars import VAR_NAMES, N_VARS
from fusionmind4.learning.causal_rl_hybrid import CausalRLHybrid


def parse_args():
    parser = argparse.ArgumentParser(description="Train CausalShield-RL")
    parser.add_argument('--episodes', type=int, default=300,
                        help='Number of training episodes')
    parser.add_argument('--rollout-steps', type=int, default=200,
                        help='Steps per episode')
    parser.add_argument('--eval-every', type=int, default=50,
                        help='Evaluate every N episodes')
    parser.add_argument('--samples', type=int, default=20000,
                        help='Number of training samples for CPDE')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--scm-hidden', type=int, default=16,
                        help='Neural SCM hidden dim')
    parser.add_argument('--policy-hidden', type=int, default=64,
                        help='Policy network hidden dim')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    verbose = not args.quiet
    
    print("=" * 70)
    print("  FusionMind 4.0 — CausalShield-RL Training Pipeline")
    print("  Patent Family PF7: Causal Reinforcement Learning")
    print("=" * 70)
    print(f"\n  Episodes: {args.episodes}")
    print(f"  Rollout steps: {args.rollout_steps}")
    print(f"  Seed: {args.seed}")
    
    t_start = time.time()
    
    # ── Step 1: Generate Training Data ──
    print("\n" + "─" * 60)
    print("Step 1: Generating FM3Lite plasma data...")
    print("─" * 60)
    
    engine = FM3LitePhysicsEngine(n_samples=args.samples, seed=args.seed)
    data, interventional_data = engine.generate()
    
    print(f"  Generated {data.shape[0]:,} samples × {data.shape[1]} variables")
    print(f"  Interventional data: {len(interventional_data)} actuators")
    
    # ── Step 2: Create and Run Hybrid ──
    config = {
        'scm_hidden_dim': args.scm_hidden,
        'scm_lr': 1e-3,
        'policy_hidden': args.policy_hidden,
        'gamma': 0.99,
        'lr_policy': 3e-4,
        'causal_bonus_weight': 0.3,
        'forbidden_penalty_weight': 0.5,
        'safety_weight': 1.0,
        'dt': 0.001,
    }
    
    hybrid = CausalRLHybrid(config=config)
    
    # ── Phase 1: Causal Discovery ──
    print("\n" + "─" * 60)
    print("Step 2: CPDE Causal Discovery...")
    print("─" * 60)
    
    cpde_results = hybrid.discover_causal_graph(
        data, interventional_data,
        var_names=VAR_NAMES,
        verbose=verbose
    )
    
    print(f"\n  ✓ DAG: {cpde_results['n_edges']} edges discovered")
    print(f"  ✓ F1={cpde_results['f1']:.3f}, "
          f"Precision={cpde_results['precision']:.3f}, "
          f"Recall={cpde_results['recall']:.3f}")
    
    # ── Phase 2: Neural SCM ──
    print("\n" + "─" * 60)
    print("Step 3: Neural SCM World Model...")
    print("─" * 60)
    
    scm_losses = hybrid.fit_world_model(data, n_epochs=200, verbose=verbose)
    avg_loss = np.mean(list(scm_losses.values()))
    print(f"\n  ✓ Neural SCM fitted, avg MSE: {avg_loss:.6f}")
    
    # Test do-intervention
    test_state = {name: float(data[0, i]) for i, name in enumerate(VAR_NAMES)}
    test_result = hybrid.neural_scm.do_intervention(
        test_state, {'P_ECRH': 1.5}
    )
    print(f"  ✓ do(P_ECRH=1.5) → Te={test_result['Te']:.4f} "
          f"(baseline Te={test_state['Te']:.4f})")
    
    # ── Phase 3: RL Training ──
    print("\n" + "─" * 60)
    print("Step 4: CausalShield-RL Training...")
    print("─" * 60)
    
    history = hybrid.train(
        n_episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        verbose=verbose,
        eval_every=args.eval_every
    )
    
    # ── Phase 4: Evaluation ──
    print("\n" + "─" * 60)
    print("Step 5: Final Evaluation...")
    print("─" * 60)
    
    eval_metrics = hybrid.evaluate(n_episodes=20, verbose=verbose)
    
    # ── Summary ──
    elapsed = time.time() - t_start
    
    print("\n" + "=" * 70)
    print("  CausalShield-RL Training Complete!")
    print("=" * 70)
    print(hybrid.summary())
    print(f"\n  Total training time: {elapsed:.1f}s")
    
    # ── Training curves data ──
    print("\n  Training History (last 50 episodes):")
    last = slice(-50, None)
    print(f"    Avg Reward:     {np.mean(history['episode_rewards'][last]):8.2f}")
    print(f"    Avg Length:      {np.mean(history['episode_lengths'][last]):8.0f}")
    print(f"    Disruption Rate: {np.mean(history['disruption_rate'][last]):8.1%}")
    print(f"    Causal Bonus:    {np.mean(history['causal_bonuses'][last]):8.2f}")
    
    # ── Online Learning Demo ──
    print("\n" + "─" * 60)
    print("Step 6: Online Learning Demo (new shot)...")
    print("─" * 60)
    
    new_shot_data = engine.generate()[0][:500]  # Simulate new 500-sample shot
    hybrid.online_update(new_shot_data, verbose=verbose)
    
    print("\n✅ Pipeline complete — CausalShield-RL is ready for deployment!")
    print("   First-ever causal RL for tokamak plasma control. 🔥")
    
    return hybrid, history, eval_metrics


if __name__ == "__main__":
    hybrid, history, metrics = main()
