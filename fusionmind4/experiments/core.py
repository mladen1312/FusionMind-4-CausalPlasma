#!/usr/bin/env python3
"""
AEDE PoC — Active Experiment Design Engine
============================================
Patent Family PF5: Designs optimal experiments to resolve
causal ambiguities discovered by CPDE.

Instead of random exploration, AEDE uses information-theoretic
criteria to select interventions that maximize causal knowledge gain.

Author: Dr. Mladen Mester
Date: March 2026
License: BSL-1.1 (converts to Apache-2.0 on 2030-03-05)
"""

import numpy as np
from typing import Dict, List, Tuple
from itertools import combinations


# ============================================================================
# BOOTSTRAP UNCERTAINTY ESTIMATOR
# ============================================================================

class BootstrapCausalUncertainty:
    """
    Estimate uncertainty in discovered causal edges using bootstrap resampling.
    High uncertainty edges are candidates for experimental validation.
    """

    def __init__(self, n_bootstrap: int = 50):
        self.n_bootstrap = n_bootstrap

    def estimate(self, data: np.ndarray, cpde_class, seed: int = 42) -> Dict:
        """
        Bootstrap CPDE to get edge confidence intervals.

        Args:
            data: (n, d) time series
            cpde_class: CPDE class to instantiate for each bootstrap
            seed: random seed

        Returns:
            edge_uncertainty: dict with confidence intervals per edge
        """
        rng = np.random.RandomState(seed)
        n = data.shape[0]
        d = data.shape[1]

        # Collect adjacency matrices from bootstrap samples
        adj_samples = []

        for b in range(self.n_bootstrap):
            # Block bootstrap (preserve temporal structure)
            block_size = min(100, n // 10)
            n_blocks = n // block_size + 1
            indices = []
            for _ in range(n_blocks):
                start = rng.randint(0, n - block_size)
                indices.extend(range(start, start + block_size))
            indices = indices[:n]

            boot_data = data[indices]

            # Run CPDE on bootstrap sample
            try:
                cpde = cpde_class()
                result = cpde.discover(boot_data)
                adj_samples.append(result['adjacency'])
            except:
                continue

        if len(adj_samples) == 0:
            return {'error': 'All bootstrap samples failed'}

        adj_stack = np.array(adj_samples)

        # Compute statistics per edge
        edge_stats = {}
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                values = adj_stack[:, i, j]
                nonzero_frac = np.mean(values != 0)

                edge_stats[(i, j)] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'presence_rate': nonzero_frac,
                    'ci_low': np.percentile(values, 2.5),
                    'ci_high': np.percentile(values, 97.5),
                    'uncertain': 0.3 < nonzero_frac < 0.7,  # Neither certain present nor absent
                }

        return edge_stats


# ============================================================================
# EXPERIMENT DESIGNER
# ============================================================================

class ExperimentDesigner:
    """
    Design optimal experiments to resolve causal uncertainties.

    Uses information-theoretic criteria:
    - Expected Information Gain (EIG) per intervention
    - Cost-effectiveness ranking
    - Safety-aware experiment scheduling
    """

    ACTUATOR_INDICES = [0, 1, 2, 3]

    # Experiment cost model [arbitrary units]
    EXPERIMENT_COSTS = {
        0: 5.0,   # NBI perturbation - moderate cost
        1: 3.0,   # ECRH perturbation - low cost (fast response)
        2: 2.0,   # Gas puff - cheap
        3: 8.0,   # Ip change - expensive (slow, risky)
    }

    # Safety risk per actuator perturbation
    SAFETY_RISK = {
        0: 0.1,  # NBI - low risk
        1: 0.1,  # ECRH - low risk
        2: 0.2,  # Gas - moderate (density limit)
        3: 0.5,  # Ip - high risk (disruption)
    }

    def __init__(self, edge_uncertainty: Dict):
        self.edge_uncertainty = edge_uncertainty

    def rank_experiments(self, n_top: int = 10) -> List[Dict]:
        """
        Rank possible experiments by expected information gain / cost.

        An "experiment" = perturbing one actuator and observing downstream
        effects to validate/refute uncertain causal edges.
        """
        experiments = []

        for act_idx in self.ACTUATOR_INDICES:
            # Which edges from this actuator are uncertain?
            uncertain_targets = []
            for (i, j), stats in self.edge_uncertainty.items():
                if i == act_idx and stats.get('uncertain', False):
                    uncertain_targets.append((j, stats))

            if not uncertain_targets:
                # Check if actuator resolves uncertainty in downstream edges
                for (i, j), stats in self.edge_uncertainty.items():
                    if stats.get('uncertain', False):
                        # Does perturbing actuator affect variable i?
                        act_to_i = self.edge_uncertainty.get((act_idx, i), {})
                        if act_to_i.get('presence_rate', 0) > 0.5:
                            uncertain_targets.append((j, stats))

            # Expected information gain
            eig = sum(
                self._entropy_reduction(stats)
                for _, stats in uncertain_targets
            )

            cost = self.EXPERIMENT_COSTS[act_idx]
            risk = self.SAFETY_RISK[act_idx]

            if eig > 0:
                experiments.append({
                    'actuator': act_idx,
                    'actuator_name': ['P_NBI', 'P_ECRH', 'gas_puff', 'Ip'][act_idx],
                    'eig': eig,
                    'cost': cost,
                    'risk': risk,
                    'value': eig / (cost * (1 + risk)),  # Cost-risk adjusted value
                    'resolves_edges': len(uncertain_targets),
                    'experiment_type': self._suggest_experiment_type(act_idx),
                })

        # Sort by value
        experiments.sort(key=lambda x: x['value'], reverse=True)
        return experiments[:n_top]

    def _entropy_reduction(self, stats: Dict) -> float:
        """Expected entropy reduction from resolving this edge"""
        p = stats.get('presence_rate', 0.5)
        if p <= 0 or p >= 1:
            return 0
        # Binary entropy
        h = -p * np.log2(p + 1e-10) - (1-p) * np.log2(1-p + 1e-10)
        return h

    def _suggest_experiment_type(self, act_idx: int) -> str:
        """Suggest specific experiment protocol"""
        protocols = {
            0: "NBI step perturbation: 5MW → 15MW → 5MW (2s each)",
            1: "ECRH modulation: sinusoidal ±3MW at 10Hz for 1s",
            2: "Gas puff pulse: 3x baseline for 500ms",
            3: "Ip ramp: +10% over 2s (with safety monitor)",
        }
        return protocols[act_idx]

    def generate_experiment_plan(self, budget: float = 20.0) -> Dict:
        """
        Generate optimal experiment plan within budget.

        Uses greedy selection: pick highest-value experiment that fits budget.
        """
        ranked = self.rank_experiments(n_top=20)
        plan = []
        remaining_budget = budget
        total_eig = 0

        for exp in ranked:
            if exp['cost'] <= remaining_budget:
                plan.append(exp)
                remaining_budget -= exp['cost']
                total_eig += exp['eig']

        return {
            'experiments': plan,
            'total_cost': budget - remaining_budget,
            'total_eig': total_eig,
            'remaining_budget': remaining_budget,
            'n_experiments': len(plan),
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("AEDE PoC — Active Experiment Design Engine")
    print("Patent Family PF5 · FusionMind 4.0")
    print("=" * 60)

    from fm3_lite_physics import FM3LiteSimulator, FM3LiteConfig, N_VARS

    # Generate data
    print("\n[1] Generating plasma data...")
    sim = FM3LiteSimulator(FM3LiteConfig(n_timesteps=2000, seed=42))
    data, _, _ = sim.simulate()

    # Simulate edge uncertainty (from bootstrap — simplified for PoC)
    print("[2] Simulating bootstrap uncertainty...")
    rng = np.random.RandomState(42)
    edge_uncertainty = {}
    for i in range(N_VARS):
        for j in range(N_VARS):
            if i == j:
                continue
            # Simulate: strong edges have high presence, weak ones are uncertain
            from fm3_lite_physics import get_ground_truth_adjacency
            gt = get_ground_truth_adjacency()
            if gt[i, j] != 0:
                # True edge: mostly present but some uncertainty
                presence = 0.6 + 0.3 * rng.random()
                mean_w = gt[i, j] + 0.1 * rng.randn()
            else:
                # No true edge: mostly absent
                presence = 0.1 * rng.random()
                mean_w = 0.05 * rng.randn()

            edge_uncertainty[(i, j)] = {
                'mean': mean_w,
                'std': abs(mean_w) * 0.3 + 0.05,
                'presence_rate': presence,
                'ci_low': mean_w - 0.2,
                'ci_high': mean_w + 0.2,
                'uncertain': 0.3 < presence < 0.7,
            }

    # Design experiments
    print("[3] Designing optimal experiments...")
    designer = ExperimentDesigner(edge_uncertainty)

    ranked = designer.rank_experiments(n_top=10)
    print(f"\n  Top experiments by value:")
    for i, exp in enumerate(ranked):
        print(f"    {i+1}. {exp['actuator_name']:10s} | "
              f"EIG={exp['eig']:.2f} | cost={exp['cost']:.1f} | "
              f"risk={exp['risk']:.1f} | value={exp['value']:.3f}")
        print(f"       Protocol: {exp['experiment_type']}")

    # Generate experiment plan
    plan = designer.generate_experiment_plan(budget=15.0)
    print(f"\n  Experiment Plan (budget=15.0):")
    print(f"    Total experiments: {plan['n_experiments']}")
    print(f"    Total EIG: {plan['total_eig']:.2f}")
    print(f"    Total cost: {plan['total_cost']:.1f}")

    n_uncertain = sum(1 for v in edge_uncertainty.values() if v.get('uncertain'))
    print(f"\n  Uncertain edges to resolve: {n_uncertain}")

    return designer, plan


if __name__ == "__main__":
    designer, plan = main()
