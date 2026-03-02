"""
Counterfactual Plasma Controller (CPC) v2.0 — UNIFIED

Merged from both implementations:
  - Named variable API with dataclass decisions (modular)
  - Intervention + Counterfactual engines (modular)
  - explain_causal_path: trace causal chains through DAG (modular)
  - retrospective_analysis: post-hoc "would have been better?" (modular)
  - disruption_avoidance: emergency safety intervention (flat)
  - Safety limits with confidence scoring (modular)

Control loop:
  1. Observe current plasma state via diagnostics
  2. Generate candidate actions (grid + perturbations)
  3. For each: compute do-calculus prediction P(Y|do(X))
  4. Check safety constraints via counterfactual prediction
  5. Score by target achievement + safety margin
  6. Select best action with causal explanation
  7. Explain WHY via causal path tracing

Part of: FusionMind 4.0 / Patent Family PF2
Author: Dr. Mladen Mester, March 2026
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .scm import PlasmaSCM
from .interventions import InterventionEngine, CounterfactualEngine


@dataclass
class ControlDecision:
    """A control decision with causal explanation."""
    action: Dict[str, float]
    predicted_state: Dict[str, float]
    causal_explanation: List[str]
    counterfactual_comparison: Dict
    confidence: float
    safety_ok: bool


class CounterfactualPlasmaController:
    """
    Main controller: causal reasoning instead of correlational RL.

    Unlike RL controllers (DeepMind, KSTAR) which learn π(a|s),
    this controller reasons:
      P(s' | do(a), s) — "what state results if I DO action a"
      P(s'_a | a=a₀, s=s₀) — "what WOULD have happened?"
    """

    def __init__(self, scm: PlasmaSCM,
                 actuator_names: List[str],
                 target_names: List[str],
                 safety_limits: Dict[str, Tuple[float, float]] = None):
        self.scm = scm
        self.actuators = actuator_names
        self.targets = target_names
        self.intervention_engine = InterventionEngine(scm)
        self.counterfactual_engine = CounterfactualEngine(scm)

        self.safety_limits = safety_limits or {
            'beta_N': (0.0, 0.8),
            'MHD_amp': (0.0, 0.5),
            'n_impurity': (0.0, 0.3),
            'q': (0.3, 1.0),
        }

        self.history: List[ControlDecision] = []

    def compute_action(self, current_state: Dict[str, float],
                       targets: Dict[str, float],
                       n_candidates: int = 20) -> ControlDecision:
        """
        Choose best actuator settings to achieve targets via causal reasoning.

        Algorithm:
        1. Generate candidate actions (current + perturbations + random)
        2. For each: compute do-calculus prediction
        3. Check safety constraints
        4. Score by target achievement + safety margin
        5. Select best with causal explanation
        """
        candidates = self._generate_candidates(current_state, n_candidates)

        scored = []
        for action in candidates:
            result = self.intervention_engine.do(action, current_state)

            target_score = 0
            for tgt, desired in targets.items():
                predicted = result.outcomes.get(tgt, 0)
                target_score -= (predicted - desired) ** 2

            safe = self._check_safety(result.outcomes)
            if not safe:
                target_score -= 100

            scored.append((action, result, target_score, safe))

        scored.sort(key=lambda x: x[2], reverse=True)
        best_action, best_result, best_score, best_safe = scored[0]

        explanation = self._explain_decision(
            current_state, best_action, best_result, targets
        )
        comparison = self._compare_alternatives(
            current_state, scored[:3], targets
        )

        decision = ControlDecision(
            action=best_action,
            predicted_state=best_result.outcomes,
            causal_explanation=explanation,
            counterfactual_comparison=comparison,
            confidence=self._compute_confidence(scored),
            safety_ok=best_safe
        )

        self.history.append(decision)
        return decision

    def explain_causal_path(self, cause: str, effect: str,
                            state: Dict[str, float]) -> List[Dict]:
        """
        Trace ALL causal paths from cause to effect through the DAG.

        Returns sorted by total effect strength.
        Example output:
          "P_ECRH →(+0.52)→ Te →(+0.35)→ beta_N →(+0.61)→ MHD_amp"
        """
        G = nx.DiGraph()
        for i in range(self.scm.n_vars):
            for j in range(self.scm.n_vars):
                if abs(self.scm.dag[i, j]) > 0.01:
                    G.add_edge(self.scm.names[i], self.scm.names[j],
                               weight=self.scm.dag[i, j])

        paths = []
        try:
            for path in nx.all_simple_paths(G, cause, effect, cutoff=4):
                path_str = []
                total_effect = 1.0
                for k in range(len(path) - 1):
                    u, v = path[k], path[k + 1]
                    eq = self.scm.equations.get(v)
                    coeff = eq.coefficients.get(u, 0) if eq else 0
                    total_effect *= coeff
                    path_str.append(f"{u} →({coeff:+.3f})→ {v}")

                paths.append({
                    'path': ' | '.join(path_str),
                    'nodes': path,
                    'total_effect': total_effect
                })
        except nx.NetworkXNoPath:
            paths.append({'path': f"No causal path from {cause} to {effect}",
                          'nodes': [], 'total_effect': 0})

        return sorted(paths, key=lambda x: abs(x['total_effect']), reverse=True)

    def retrospective_analysis(self, factual_state: Dict[str, float],
                               alternative_action: Dict[str, float]) -> Dict:
        """
        Post-hoc analysis: "Would the outcome have been better if we had done X?"
        Uses Level 3 (counterfactual) reasoning.
        """
        cf_result = self.counterfactual_engine.counterfactual(
            factual_state, alternative_action
        )

        comparison = {}
        for var in self.scm.names:
            factual = factual_state.get(var, 0)
            counterfactual = cf_result.counterfactual_outcomes.get(var, 0)
            comparison[var] = {
                'factual': factual,
                'counterfactual': counterfactual,
                'difference': counterfactual - factual,
            }

        return {
            'alternative_action': alternative_action,
            'comparison': comparison,
            'counterfactual_state': cf_result.counterfactual_outcomes,
            'noise_terms': cf_result.abducted_noise
        }

    def disruption_avoidance(self, current_state: Dict[str, float]) -> Dict:
        """
        Emergency intervention: "What actuator changes prevent imminent disruption?"
        Targets: reduce βN to safe level, increase q, minimize MHD activity.
        """
        # Resolve variable names (handle βN vs beta_N etc.)
        beta_name = None
        for name in ['beta_N', 'βN', 'betaN']:
            if name in self.scm.idx:
                beta_name = name
                break

        safety_targets = {}
        target_map = {'q': 3.5, 'MHD_amp': 0.1}
        if beta_name:
            target_map[beta_name] = 1.5

        for var, val in target_map.items():
            if var in self.scm.idx:
                safety_targets[var] = val

        if not safety_targets:
            return {'error': 'Safety target variables not found in SCM'}

        # Use first target for optimization
        primary_target = list(safety_targets.keys())[0]
        return self.intervention_engine.find_optimal_intervention(
            target_var=primary_target,
            target_value=safety_targets[primary_target],
            actuators=self.actuators,
            current_state=current_state,
        )

    def _generate_candidates(self, state: Dict[str, float],
                             n: int) -> List[Dict[str, float]]:
        """Generate candidate actions: current + perturbations + random."""
        candidates = []
        current = {a: state.get(a, 0.5) for a in self.actuators}
        candidates.append(current)

        # Single-actuator perturbations
        for act in self.actuators:
            for delta in [-0.2, -0.1, 0.1, 0.2]:
                c = dict(current)
                c[act] = np.clip(c[act] + delta, 0.1, 0.95)
                candidates.append(c)

        # Random combinations
        while len(candidates) < n:
            c = {act: np.clip(current[act] + np.random.uniform(-0.15, 0.15),
                              0.1, 0.95) for act in self.actuators}
            candidates.append(c)

        return candidates[:n]

    def _check_safety(self, predicted_state: Dict[str, float]) -> bool:
        for var, (lo, hi) in self.safety_limits.items():
            val = predicted_state.get(var, 0)
            if val < lo or val > hi:
                return False
        return True

    def _explain_decision(self, state, action, result, targets) -> List[str]:
        explanations = []
        for act_name, act_val in action.items():
            old_val = state.get(act_name, 0)
            if abs(act_val - old_val) > 0.01:
                direction = "↑" if act_val > old_val else "↓"
                explanations.append(
                    f"  {act_name}: {old_val:.2f} → {act_val:.2f} ({direction})"
                )
                for tgt in self.targets:
                    paths = self.explain_causal_path(act_name, tgt, state)
                    if paths and paths[0]['total_effect'] != 0:
                        eff = result.causal_effects.get(tgt, 0)
                        if abs(eff) > 0.01:
                            explanations.append(
                                f"    → {tgt}: {eff:+.3f} via {paths[0]['path']}"
                            )
        return explanations

    def _compare_alternatives(self, state, top_actions, targets) -> Dict:
        comparison = {}
        for i, (action, result, score, safe) in enumerate(top_actions):
            comparison[f"option_{i + 1}"] = {
                'action': action,
                'score': score,
                'safe': safe,
                'outcomes': {t: result.outcomes.get(t, 0) for t in self.targets}
            }
        return comparison

    def _compute_confidence(self, scored) -> float:
        if len(scored) < 2:
            return 0.5
        scores = [s[2] for s in scored[:5]]
        gap = scores[0] - scores[1] if len(scores) > 1 else 0
        return min(0.95, 0.5 + gap * 10)
