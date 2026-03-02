"""Interventional Scoring — do-calculus validation for causal edges.

Uses interventional data (do(actuator=value)) to validate whether
manipulating an actuator has the expected downstream effect.
"""
import numpy as np
from typing import Dict, Optional
from ..utils.plasma_vars import N_VARS, ACTUATOR_IDS


class InterventionalScorer:
    """Score causal edges using interventional (do-calculus) data.

    For each actuator, compares the distribution of downstream variables
    under low vs high actuator settings. Large shifts indicate causal effect.

    Args:
        effect_threshold: Minimum Cohen's d to consider causal
    """

    def __init__(self, effect_threshold: float = 0.3):
        self.effect_threshold = effect_threshold

    def score(self, interventions: Dict) -> np.ndarray:
        """Score all actuator→downstream edges using interventional data.

        Args:
            interventions: {actuator_id: (data_low, data_high)} from FM3Lite

        Returns:
            scores: (N_VARS, N_VARS) matrix of interventional effect scores [0,1]
        """
        scores = np.zeros((N_VARS, N_VARS))

        for act_id, (data_low, data_high) in interventions.items():
            if act_id not in ACTUATOR_IDS:
                continue

            for j in range(N_VARS):
                if j == act_id:
                    continue

                # Cohen's d effect size
                d = self._cohens_d(data_low[:, j], data_high[:, j])

                if abs(d) > self.effect_threshold:
                    # Normalize to [0, 1] with sigmoid-like scaling
                    scores[act_id, j] = min(1.0, abs(d) / 2.0)

        return scores

    @staticmethod
    def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std < 1e-10:
            return 0.0

        return (np.mean(group2) - np.mean(group1)) / pooled_std
