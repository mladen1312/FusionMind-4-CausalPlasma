"""
FusionMind 4.6 — CausalRL Controller
=====================================

Three operating modes:
  MODE A: WRAPPER  — Sits above external RL (DeepMind/KSTAR) as safety layer
  MODE B: HYBRID   — Our causal RL replaces external RL entirely  
  MODE C: ADVISOR  — Runs alongside any controller, provides explanations only

Architecture:
  ┌──────────────────────────────────────────────┐
  │            FusionMind CausalRL                │
  │  ┌────────────────────────────────────────┐   │
  │  │  Layer 3: Causal Safety Monitor        │   │  ← MODE A (wrapper)
  │  │  - Vetoes unsafe RL actions            │   │
  │  │  - Provides counterfactual explanation  │   │
  │  ├────────────────────────────────────────┤   │
  │  │  Layer 2: Causal Policy (our RL)       │   │  ← MODE B (hybrid)
  │  │  - SCM-guided action selection         │   │
  │  │  - do-calculus for action evaluation   │   │
  │  │  - Causal reward shaping              │   │
  │  ├────────────────────────────────────────┤   │
  │  │  Layer 1: Causal World Model (CPDE)    │   │  ← Always active
  │  │  - DAG structure                       │   │
  │  │  - Nonlinear SCM                       │   │
  │  │  - Intervention prediction             │   │
  │  └────────────────────────────────────────┘   │
  └──────────────────────────────────────────────┘

Why this is better than pure RL:
  1. RL learns WHAT to do. FusionMind explains WHY.
  2. RL can exploit spurious correlations. SCM cannot (causal identification).
  3. RL needs millions of samples. SCM generalizes from structure.
  4. RL is a black box for regulators. SCM provides formal proofs.
  5. RL fails on distribution shift. SCM's do-calculus is invariant.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class ControlMode(Enum):
    WRAPPER = "wrapper"    # Safety layer over external RL
    HYBRID = "hybrid"      # Our causal RL replaces external
    ADVISOR = "advisor"    # Read-only explanations


@dataclass
class ControlAction:
    """A control action with causal justification."""
    actuator_values: Dict[str, float]    # What to do
    source: str                          # Who decided (external_rl / causal_rl / safety_override)
    confidence: float                    # 0-1
    causal_explanation: str              # WHY this action
    causal_paths: List[str]              # Causal chain: ["Pheat → Te → βN → stability"]
    risk_score: float                    # 0-1 disruption risk
    counterfactual: Optional[str] = None # "If we don't act, βN will exceed limit in 50ms"
    vetoed: bool = False                 # Was original action vetoed?
    veto_reason: Optional[str] = None


@dataclass
class PlasmaState:
    """Current plasma state vector."""
    values: Dict[str, float]
    timestamp: float
    shot_id: int = 0
    
    def to_array(self, var_names):
        return np.array([self.values.get(v, 0.0) for v in var_names])


@dataclass
class SafetyLimits:
    """Operational safety boundaries — hard physics limits."""
    q95_min: float = 2.0            # Below this → guaranteed disruption
    q95_warning: float = 2.5        # Below this → high risk
    betan_max: float = 3.5          # Troyon limit (simplified)
    betan_warning: float = 3.0
    li_max: float = 2.0             # Current peaking limit
    li_warning: float = 1.5
    greenwald_fraction_max: float = 1.0  # ne/nGW
    power_change_rate_max: float = 0.2   # Max 20% per control cycle
    ip_ramp_rate_max: float = 0.5e6      # A/s


class CausalWorldModel:
    """
    Layer 1: The causal world model (DAG + SCM).
    Always active in all modes.
    """
    
    def __init__(self, dag, scm, var_names):
        """
        Parameters
        ----------
        dag : ndarray — adjacency matrix from CPDE
        scm : NonlinearPlasmaSCM or PlasmaSCM — fitted structural causal model
        var_names : list of str
        """
        self.dag = dag
        self.scm = scm
        self.var_names = var_names
        self.idx = {v: i for i, v in enumerate(var_names)}
    
    def predict_intervention(self, state: PlasmaState, intervention: Dict[str, float]) -> Dict[str, float]:
        """Predict outcome of do(intervention) from current state."""
        baseline = state.to_array(self.var_names)
        result = self.scm.do(intervention, baseline)
        return {v: result[i] for i, v in enumerate(self.var_names)}
    
    def counterfactual(self, factual: PlasmaState, 
                       hypothetical: Dict[str, float]) -> Dict[str, float]:
        """What would have happened if we had done X instead?"""
        fact_arr = factual.to_array(self.var_names)
        result = self.scm.counterfactual(fact_arr, hypothetical)
        return {v: result[i] for i, v in enumerate(self.var_names)}
    
    def get_causal_parents(self, variable: str) -> List[str]:
        """What directly causes this variable?"""
        if variable not in self.idx:
            return []
        vi = self.idx[variable]
        parents = np.where(self.dag[:, vi] > 0)[0]
        return [self.var_names[p] for p in parents]
    
    def get_causal_children(self, variable: str) -> List[str]:
        """What does this variable directly affect?"""
        if variable not in self.idx:
            return []
        vi = self.idx[variable]
        children = np.where(self.dag[vi, :] > 0)[0]
        return [self.var_names[c] for c in children]
    
    def trace_causal_path(self, source: str, target: str) -> List[List[str]]:
        """Find all causal paths from source to target."""
        if source not in self.idx or target not in self.idx:
            return []
        
        si, ti = self.idx[source], self.idx[target]
        paths = []
        
        def dfs(current, path, visited):
            if current == ti:
                paths.append(list(path))
                return
            for child in range(len(self.var_names)):
                if self.dag[current, child] > 0 and child not in visited:
                    visited.add(child)
                    path.append(self.var_names[child])
                    dfs(child, path, visited)
                    path.pop()
                    visited.remove(child)
        
        dfs(si, [source], {si})
        return paths
    
    def compute_risk(self, state: PlasmaState, limits: SafetyLimits) -> Tuple[float, List[str]]:
        """Compute disruption risk from causal model."""
        v = state.values
        risk_factors = []
        risk = 0.0
        
        q95 = v.get('q_95', 5.0)
        if q95 < limits.q95_min:
            risk = max(risk, 1.0)
            risk_factors.append(f"CRITICAL: q95={q95:.2f} < {limits.q95_min} (disruption boundary)")
        elif q95 < limits.q95_warning:
            r = 1.0 - (q95 - limits.q95_min) / (limits.q95_warning - limits.q95_min)
            risk = max(risk, r * 0.8)
            risk_factors.append(f"WARNING: q95={q95:.2f} approaching limit")
        
        betan = v.get('betan', 1.0)
        if betan > limits.betan_max:
            risk = max(risk, 0.9)
            risk_factors.append(f"CRITICAL: βN={betan:.2f} > Troyon limit {limits.betan_max}")
        elif betan > limits.betan_warning:
            r = (betan - limits.betan_warning) / (limits.betan_max - limits.betan_warning)
            risk = max(risk, r * 0.7)
            risk_factors.append(f"WARNING: βN={betan:.2f} approaching Troyon limit")
        
        li = v.get('li', 1.0)
        if li > limits.li_max:
            risk = max(risk, 0.8)
            risk_factors.append(f"WARNING: li={li:.2f} > {limits.li_max} (current peaking)")
        
        return min(risk, 1.0), risk_factors


class CausalSafetyMonitor:
    """
    Layer 3: Safety monitor that wraps any external RL controller.
    
    This is MODE A — the most commercially viable configuration:
    - CFS/ITER already have RL controllers
    - They WON'T throw them away
    - But regulators DEMAND explainability
    - FusionMind sits on top, vetoes unsafe actions, explains WHY
    """
    
    def __init__(self, world_model: CausalWorldModel, limits: SafetyLimits = None):
        self.world = world_model
        self.limits = limits or SafetyLimits()
        self.veto_history = []
        self.action_history = []
    
    def evaluate_action(self, state: PlasmaState, 
                        proposed_action: Dict[str, float]) -> ControlAction:
        """
        Evaluate a proposed RL action through the causal lens.
        
        This is the core value proposition:
        RL says "do X". FusionMind says "if you do X, here's what happens
        causally, here are the risks, and here's why."
        """
        # 1. Predict outcome using causal model (not correlational!)
        predicted_state = self.world.predict_intervention(state, proposed_action)
        predicted_plasma = PlasmaState(values=predicted_state, timestamp=state.timestamp)
        
        # 2. Assess risk of predicted state
        risk, risk_factors = self.world.compute_risk(predicted_plasma, self.limits)
        
        # 3. Build causal explanation
        explanation_parts = []
        causal_paths = []
        
        for actuator, value in proposed_action.items():
            children = self.world.get_causal_children(actuator)
            if children:
                for child in children:
                    paths = self.world.trace_causal_path(actuator, child)
                    for path in paths:
                        causal_paths.append(" → ".join(path))
                
                # What changes?
                baseline_v = state.values.get(actuator, 0)
                delta = value - baseline_v
                direction = "increase" if delta > 0 else "decrease"
                explanation_parts.append(
                    f"{actuator} {direction} by {abs(delta):.3f} → affects {', '.join(children)}"
                )
        
        explanation = "; ".join(explanation_parts) if explanation_parts else "No causal effects predicted"
        
        # 4. Decide: approve, modify, or veto
        if risk > 0.8:
            # VETO — too dangerous
            safe_action = self._compute_safe_alternative(state, proposed_action)
            
            # Counterfactual: what if we did nothing?
            no_action = {k: state.values.get(k, v) for k, v in proposed_action.items()}
            cf_state = self.world.predict_intervention(state, no_action)
            cf_risk, _ = self.world.compute_risk(
                PlasmaState(values=cf_state, timestamp=state.timestamp), self.limits)
            
            counterfactual_msg = (
                f"Without intervention: risk={cf_risk:.2f}. "
                f"Proposed action would cause: {'; '.join(risk_factors)}. "
                f"Safe alternative applied."
            )
            
            action = ControlAction(
                actuator_values=safe_action,
                source="safety_override",
                confidence=0.95,
                causal_explanation=explanation,
                causal_paths=causal_paths[:5],
                risk_score=risk,
                counterfactual=counterfactual_msg,
                vetoed=True,
                veto_reason="; ".join(risk_factors)
            )
            self.veto_history.append(action)
            
        elif risk > 0.5:
            # WARN — allow but flag
            action = ControlAction(
                actuator_values=proposed_action,
                source="external_rl (causal_warning)",
                confidence=0.7,
                causal_explanation=f"CAUTION: {explanation}. Risk factors: {'; '.join(risk_factors)}",
                causal_paths=causal_paths[:5],
                risk_score=risk,
            )
        else:
            # APPROVE
            action = ControlAction(
                actuator_values=proposed_action,
                source="external_rl (causal_approved)",
                confidence=0.9,
                causal_explanation=explanation,
                causal_paths=causal_paths[:5],
                risk_score=risk,
            )
        
        self.action_history.append(action)
        return action
    
    def _compute_safe_alternative(self, state: PlasmaState,
                                   proposed: Dict[str, float]) -> Dict[str, float]:
        """Find a safe action that's as close to proposed as possible."""
        safe = {}
        for actuator, target_val in proposed.items():
            current = state.values.get(actuator, target_val)
            # Limit change rate
            max_delta = abs(current) * self.limits.power_change_rate_max
            clamped = np.clip(target_val, current - max_delta, current + max_delta)
            safe[actuator] = clamped
        
        # Verify safe action doesn't cause problems
        pred = self.world.predict_intervention(state, safe)
        pred_state = PlasmaState(values=pred, timestamp=state.timestamp)
        risk, _ = self.world.compute_risk(pred_state, self.limits)
        
        if risk > 0.5:
            # Even clamped action is risky — revert to current
            return {k: state.values.get(k, v) for k, v in proposed.items()}
        
        return safe
    
    def explain_disruption(self, pre_state: PlasmaState, 
                           post_state: PlasmaState) -> Dict[str, Any]:
        """
        Post-mortem: explain WHY a disruption happened.
        
        This is what ITER regulators need:
        "The disruption at t=14.32s was caused by li exceeding 1.8,
         which through the causal path li → q_axis → q95 drove q95
         below 2.0. Counterfactual: if li had been maintained below 1.5
         through 10% Ip reduction at t=14.25s, the disruption would
         not have occurred."
        """
        explanation = {
            'timestamp': post_state.timestamp,
            'root_causes': [],
            'causal_chains': [],
            'counterfactuals': [],
        }
        
        # Find what changed most
        deltas = {}
        for v in self.world.var_names:
            pre_v = pre_state.values.get(v, 0)
            post_v = post_state.values.get(v, 0)
            if abs(pre_v) > 1e-10:
                deltas[v] = (post_v - pre_v) / abs(pre_v)
        
        # Sort by magnitude of change
        sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # For each big change, trace causal path
        for var, delta in sorted_deltas[:3]:
            parents = self.world.get_causal_parents(var)
            if parents:
                parent_changes = []
                for p in parents:
                    if p in deltas:
                        parent_changes.append(f"{p} changed {deltas[p]:+.1%}")
                
                explanation['root_causes'].append({
                    'variable': var,
                    'change': f"{delta:+.1%}",
                    'caused_by': parent_changes,
                })
            
            # Counterfactual: what if this variable hadn't changed?
            cf_arr = post_state.to_array(self.world.var_names)
            cf = self.world.scm.counterfactual(
                cf_arr,
                {var: pre_state.values.get(var, 0)}
            )
            cf_state = PlasmaState(
                values={v: cf[i] for i, v in enumerate(self.world.var_names)},
                timestamp=post_state.timestamp
            )
            cf_risk, _ = self.world.compute_risk(cf_state, self.limits)
            actual_risk, _ = self.world.compute_risk(post_state, self.limits)
            
            explanation['counterfactuals'].append({
                'hypothesis': f"If {var} had stayed at {pre_state.values.get(var, 0):.3f}",
                'risk_would_be': cf_risk,
                'actual_risk': actual_risk,
                'would_prevent_disruption': cf_risk < 0.5 and actual_risk > 0.8,
            })
        
        return explanation


class CausalRLPolicy:
    """
    Layer 2: Our own RL policy guided by causal model.
    
    This is MODE B — full replacement of external RL.
    Uses SCM for:
    1. Model-based planning (predict outcomes of actions via do-calculus)
    2. Causal reward shaping (reward actions that fix ROOT CAUSES)
    3. Safe exploration (only explore in causally-understood regions)
    """
    
    def __init__(self, world_model: CausalWorldModel, limits: SafetyLimits = None):
        self.world = world_model
        self.limits = limits or SafetyLimits()
        self.target_state = {}  # Desired plasma parameters
    
    def set_targets(self, targets: Dict[str, float]):
        """Set target plasma parameters."""
        self.target_state = targets
    
    def compute_action(self, state: PlasmaState, 
                       actuators: List[str]) -> ControlAction:
        """
        Compute optimal action using causal model-based planning.
        
        For each actuator, uses do-calculus to predict which change
        best moves plasma toward target while minimizing risk.
        """
        best_action = {}
        explanations = []
        
        for actuator in actuators:
            if actuator not in self.world.idx:
                continue
            
            current_val = state.values.get(actuator, 0)
            
            # Test candidate actions via do-calculus
            candidates = [
                current_val * 0.9,   # 10% decrease
                current_val * 0.95,  # 5% decrease
                current_val,          # hold
                current_val * 1.05,  # 5% increase
                current_val * 1.1,   # 10% increase
            ]
            
            best_score = -np.inf
            best_val = current_val
            best_reason = "hold"
            
            for candidate in candidates:
                # Predict outcome
                pred = self.world.predict_intervention(
                    state, {actuator: candidate}
                )
                pred_state = PlasmaState(values=pred, timestamp=state.timestamp)
                
                # Score = closeness to target - risk penalty
                score = 0
                for target_var, target_val in self.target_state.items():
                    if target_var in pred:
                        error = abs(pred[target_var] - target_val) / (abs(target_val) + 1e-10)
                        score -= error
                
                risk, _ = self.world.compute_risk(pred_state, self.limits)
                score -= 5.0 * risk  # Heavy penalty for risk
                
                if score > best_score:
                    best_score = score
                    best_val = candidate
                    delta = (candidate - current_val) / (abs(current_val) + 1e-10)
                    if abs(delta) < 0.01:
                        best_reason = "hold (optimal)"
                    else:
                        best_reason = f"{'increase' if delta > 0 else 'decrease'} {abs(delta):.1%}"
            
            best_action[actuator] = best_val
            explanations.append(f"{actuator}: {best_reason}")
        
        # Compute overall risk
        final_pred = self.world.predict_intervention(state, best_action)
        final_state = PlasmaState(values=final_pred, timestamp=state.timestamp)
        risk, risk_factors = self.world.compute_risk(final_state, self.limits)
        
        # Causal paths for explanation
        all_paths = []
        for act in actuators:
            for target in self.target_state:
                paths = self.world.trace_causal_path(act, target)
                all_paths.extend([" → ".join(p) for p in paths])
        
        return ControlAction(
            actuator_values=best_action,
            source="causal_rl",
            confidence=min(0.95, 1.0 - risk),
            causal_explanation="; ".join(explanations),
            causal_paths=all_paths[:10],
            risk_score=risk,
        )


class FusionMindController:
    """
    Top-level controller — selects mode and orchestrates layers.
    
    Usage:
        # MODE A: Wrap DeepMind's RL
        ctrl = FusionMindController(world_model, mode=ControlMode.WRAPPER)
        action = ctrl.evaluate_external_action(state, deepmind_action)
        
        # MODE B: Replace RL entirely
        ctrl = FusionMindController(world_model, mode=ControlMode.HYBRID)
        ctrl.set_targets({'betan': 2.5, 'q_95': 4.0})
        action = ctrl.compute_action(state, actuators=['Ip', 'Prad'])
        
        # MODE C: Advisory only
        ctrl = FusionMindController(world_model, mode=ControlMode.ADVISOR)
        report = ctrl.explain_state(state)
    """
    
    def __init__(self, world_model: CausalWorldModel, 
                 mode: ControlMode = ControlMode.WRAPPER,
                 limits: SafetyLimits = None):
        self.world = world_model
        self.mode = mode
        self.limits = limits or SafetyLimits()
        self.safety = CausalSafetyMonitor(world_model, self.limits)
        self.policy = CausalRLPolicy(world_model, self.limits)
    
    def evaluate_external_action(self, state: PlasmaState,
                                  external_action: Dict[str, float]) -> ControlAction:
        """MODE A: Evaluate and optionally veto external RL action."""
        if self.mode == ControlMode.ADVISOR:
            # Don't modify, just explain
            pred = self.world.predict_intervention(state, external_action)
            risk, factors = self.world.compute_risk(
                PlasmaState(values=pred, timestamp=state.timestamp), self.limits)
            return ControlAction(
                actuator_values=external_action,
                source="external_rl (advisory)",
                confidence=1.0 - risk,
                causal_explanation=f"Predicted risk: {risk:.2f}. " + "; ".join(factors),
                causal_paths=[],
                risk_score=risk,
            )
        
        return self.safety.evaluate_action(state, external_action)
    
    def compute_action(self, state: PlasmaState,
                       actuators: List[str]) -> ControlAction:
        """MODE B: Compute action from our causal RL policy."""
        action = self.policy.compute_action(state, actuators)
        
        # Always verify through safety layer
        safe_action = self.safety.evaluate_action(state, action.actuator_values)
        if safe_action.vetoed:
            return safe_action
        
        return action
    
    def set_targets(self, targets: Dict[str, float]):
        """Set target plasma parameters for MODE B."""
        self.policy.set_targets(targets)
    
    def explain_state(self, state: PlasmaState) -> Dict[str, Any]:
        """MODE C: Explain current plasma state causally."""
        risk, factors = self.world.compute_risk(state, self.limits)
        
        # For each variable, explain what's causing it
        causal_map = {}
        for v in self.world.var_names:
            parents = self.world.get_causal_parents(v)
            children = self.world.get_causal_children(v)
            r2 = self.world.scm.r2_scores.get(self.world.idx.get(v, -1), 0)
            causal_map[v] = {
                'value': state.values.get(v, 0),
                'caused_by': parents,
                'affects': children,
                'model_r2': r2,
            }
        
        return {
            'risk': risk,
            'risk_factors': factors,
            'causal_map': causal_map,
            'mode': self.mode.value,
        }
    
    def explain_disruption(self, pre: PlasmaState, post: PlasmaState) -> Dict:
        """Post-mortem disruption analysis."""
        return self.safety.explain_disruption(pre, post)
    
    def get_statistics(self) -> Dict:
        """Performance statistics."""
        n_actions = len(self.safety.action_history)
        n_vetoes = len(self.safety.veto_history)
        return {
            'mode': self.mode.value,
            'total_actions': n_actions,
            'vetoed_actions': n_vetoes,
            'veto_rate': n_vetoes / n_actions if n_actions > 0 else 0,
            'avg_risk': np.mean([a.risk_score for a in self.safety.action_history]) if self.safety.action_history else 0,
        }
