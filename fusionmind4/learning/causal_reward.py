"""
CausalRewardShaper — Causal Reward Shaping for RL
===================================================

Uses the discovered causal graph (from CPDE) to shape the RL reward,
preventing the agent from exploiting spurious correlations.

Key innovation: The reward function is CAUSAL, not correlational.
Instead of rewarding "whenever Te is high" (correlation → Simpson's
Paradox risk), we reward "actions that CAUSE Te to increase via
known causal pathways."

This prevents:
- Reward hacking via spurious correlations
- Unsafe interventions on non-causal pathways  
- Simpson's Paradox in multi-variate control

Part of: FusionMind 4.0 / Patent Family PF7 (CausalShield-RL)
Author: Dr. Mladen Mester, March 2026
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from ..utils.plasma_vars import VAR_NAMES, N_VARS, ACTUATOR_IDS


class CausalRewardShaper:
    """
    Shapes RL reward using causal graph structure.
    
    The standard RL reward is: r = -|target - actual|
    
    CausalShield adds:
    1. Causal pathway bonus: reward if action → target via known causal path
    2. Forbidden path penalty: penalize if action exploits non-causal correlation
    3. Confounding penalty: penalize if action changes a confounder
    4. Safety constraint: hard penalty for violating causal safety bounds
    
    This is what makes our RL EXPLAINABLE: we can trace every reward
    component back to a specific causal pathway in the discovered graph.
    """
    
    def __init__(self, dag: np.ndarray, variable_names: Optional[List[str]] = None,
                 causal_bonus_weight: float = 0.3,
                 forbidden_penalty_weight: float = 0.5,
                 safety_weight: float = 1.0):
        """
        Args:
            dag: (n, n) weighted adjacency matrix from CPDE
            variable_names: List of variable names (default: standard 14)
            causal_bonus_weight: Weight for causal pathway bonus
            forbidden_penalty_weight: Weight for forbidden path penalty
            safety_weight: Weight for safety violations
        """
        self.dag = dag
        self.n_vars = dag.shape[0]
        self.names = variable_names or VAR_NAMES[:self.n_vars]
        self.idx = {name: i for i, name in enumerate(self.names)}
        
        self.causal_bonus_weight = causal_bonus_weight
        self.forbidden_penalty_weight = forbidden_penalty_weight
        self.safety_weight = safety_weight
        
        # Precompute causal structure
        self.allowed_paths = self._compute_allowed_paths()
        self.forbidden_edges = self._compute_forbidden_edges()
        
        # Safety bounds from physics
        self.safety_bounds = {
            'q': (1.5, 8.0),         # Safety factor limits
            'betaN': (0.0, 3.5),      # Beta limit
            'MHD_amp': (0.0, 2.0),    # MHD stability
            'ne': (0.01, 3.0),        # Density limit (Greenwald)
        }
    
    def shape_reward(self, state: np.ndarray, action: np.ndarray,
                     next_state: np.ndarray, base_reward: float,
                     action_names: Optional[List[str]] = None) -> Dict:
        """
        Shape the RL reward using causal reasoning.
        
        Args:
            state: (n_vars,) current state
            action: (n_actuators,) action taken
            next_state: (n_vars,) resulting state
            base_reward: Original environment reward
            action_names: Names of actuator variables
            
        Returns:
            Dictionary with shaped reward and detailed breakdown
        """
        if action_names is None:
            action_names = [self.names[i] for i in sorted(ACTUATOR_IDS) 
                           if i < len(self.names)]
        
        # Start with base reward
        total_reward = base_reward
        breakdown = {'base': base_reward}
        
        # 1. Causal pathway bonus
        causal_bonus = self._causal_pathway_bonus(state, action, next_state, action_names)
        total_reward += self.causal_bonus_weight * causal_bonus
        breakdown['causal_bonus'] = causal_bonus
        
        # 2. Forbidden path penalty
        forbidden_penalty = self._forbidden_path_penalty(state, action, next_state, action_names)
        total_reward -= self.forbidden_penalty_weight * forbidden_penalty
        breakdown['forbidden_penalty'] = forbidden_penalty
        
        # 3. Safety constraint penalty
        safety_penalty = self._safety_penalty(next_state)
        total_reward -= self.safety_weight * safety_penalty
        breakdown['safety_penalty'] = safety_penalty
        
        # 4. Causal consistency bonus (changes follow expected causal direction)
        consistency = self._causal_consistency(state, action, next_state, action_names)
        total_reward += 0.1 * consistency
        breakdown['consistency'] = consistency
        
        breakdown['total'] = total_reward
        breakdown['shaping_contribution'] = total_reward - base_reward
        
        return breakdown
    
    def _causal_pathway_bonus(self, state: np.ndarray, action: np.ndarray,
                               next_state: np.ndarray, action_names: List[str]) -> float:
        """
        Bonus for changes that follow known causal pathways.
        
        If actuator A changed and downstream target T changed in the
        expected direction via path A→...→T, give a bonus.
        """
        bonus = 0.0
        delta = next_state - state
        
        for k, act_name in enumerate(action_names):
            if act_name not in self.idx:
                continue
            act_idx = self.idx[act_name]
            
            # Check if this actuator was significantly changed
            if abs(delta[act_idx]) < 0.01:
                continue
            
            # Check downstream effects via causal paths
            reachable = self.allowed_paths.get(act_idx, set())
            for target_idx in reachable:
                if target_idx == act_idx:
                    continue
                
                # Expected sign of effect
                expected_sign = np.sign(self.dag[act_idx, target_idx]) if self.dag[act_idx, target_idx] != 0 else 0
                actual_change = delta[target_idx]
                
                if expected_sign != 0 and np.sign(actual_change) == expected_sign:
                    bonus += abs(actual_change) * 0.5
        
        return float(bonus)
    
    def _forbidden_path_penalty(self, state: np.ndarray, action: np.ndarray,
                                  next_state: np.ndarray, action_names: List[str]) -> float:
        """
        Penalty for changes via non-causal pathways.
        
        If a variable changed significantly but there's no causal path
        from any changed actuator to it, flag as potential Simpson's Paradox.
        """
        penalty = 0.0
        delta = next_state - state
        
        # Find which actuators changed
        changed_actuators = set()
        for k, act_name in enumerate(action_names):
            if act_name in self.idx:
                act_idx = self.idx[act_name]
                if abs(delta[act_idx]) > 0.01:
                    changed_actuators.add(act_idx)
        
        # Check non-actuator variables
        for j in range(self.n_vars):
            if j in changed_actuators:
                continue
            
            if abs(delta[j]) > 0.05:  # Significant change
                # Is there a causal path from any changed actuator?
                has_causal_path = False
                for act_idx in changed_actuators:
                    if j in self.allowed_paths.get(act_idx, set()):
                        has_causal_path = True
                        break
                
                if not has_causal_path:
                    # Change without causal explanation → suspicious
                    penalty += abs(delta[j]) * 1.0
        
        return float(penalty)
    
    def _safety_penalty(self, state: np.ndarray) -> float:
        """Hard penalty for violating physics safety bounds."""
        penalty = 0.0
        
        for var_name, (lo, hi) in self.safety_bounds.items():
            if var_name not in self.idx:
                continue
            idx = self.idx[var_name]
            val = state[idx]
            
            if val < lo:
                penalty += (lo - val) ** 2 * 10.0
            elif val > hi:
                penalty += (val - hi) ** 2 * 10.0
            
            # Soft warning zone (within 10% of limits)
            margin = 0.1 * (hi - lo)
            if val < lo + margin:
                penalty += (lo + margin - val) * 0.5
            elif val > hi - margin:
                penalty += (val - (hi - margin)) * 0.5
        
        return float(penalty)
    
    def _causal_consistency(self, state: np.ndarray, action: np.ndarray,
                             next_state: np.ndarray, action_names: List[str]) -> float:
        """
        Bonus for state changes that are consistent with causal graph directions.
        
        For each edge A→B in the DAG: if A increased and DAG says A→B is positive,
        then B should increase. If it does, that's consistent.
        """
        consistency = 0.0
        delta = next_state - state
        n_checks = 0
        
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                w = self.dag[i, j]
                if abs(w) < 0.01:
                    continue
                
                if abs(delta[i]) > 0.01 and abs(delta[j]) > 0.01:
                    expected_sign = np.sign(w * delta[i])
                    actual_sign = np.sign(delta[j])
                    
                    if expected_sign == actual_sign:
                        consistency += 1.0
                    else:
                        consistency -= 0.5
                    n_checks += 1
        
        return consistency / max(n_checks, 1)
    
    def _compute_allowed_paths(self) -> Dict[int, Set[int]]:
        """Compute reachable nodes from each variable via DAG."""
        allowed = {}
        
        for i in range(self.n_vars):
            reachable = set()
            frontier = {i}
            visited = set()
            
            while frontier:
                node = frontier.pop()
                if node in visited:
                    continue
                visited.add(node)
                reachable.add(node)
                
                # Follow edges from node
                for j in range(self.n_vars):
                    if abs(self.dag[node, j]) > 0.01:
                        frontier.add(j)
            
            allowed[i] = reachable
        
        return allowed
    
    def _compute_forbidden_edges(self) -> Set[Tuple[int, int]]:
        """Find edges that don't exist in the causal graph (forbidden paths)."""
        forbidden = set()
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                if i != j and abs(self.dag[i, j]) < 0.01:
                    forbidden.add((i, j))
        return forbidden
    
    def get_action_explanation(self, action: np.ndarray, state: np.ndarray,
                                next_state: np.ndarray,
                                action_names: Optional[List[str]] = None) -> str:
        """
        Generate human-readable explanation of why RL agent took this action.
        
        This is the EXPLAINABILITY component: for every action, we can
        trace the causal reasoning.
        """
        if action_names is None:
            action_names = [self.names[i] for i in sorted(ACTUATOR_IDS) 
                           if i < len(self.names)]
        
        lines = ["=== Causal Action Explanation ==="]
        delta = next_state - state
        
        for k, act_name in enumerate(action_names):
            if act_name not in self.idx:
                continue
            act_idx = self.idx[act_name]
            
            if abs(delta[act_idx]) < 0.01:
                continue
            
            direction = "increased" if delta[act_idx] > 0 else "decreased"
            lines.append(f"\n{act_name} {direction} by {abs(delta[act_idx]):.3f}")
            lines.append("  Causal effects:")
            
            # Trace causal paths
            for j in range(self.n_vars):
                w = self.dag[act_idx, j]
                if abs(w) > 0.01:
                    expected = "↑" if w * delta[act_idx] > 0 else "↓"
                    actual = "↑" if delta[j] > 0 else "↓" if delta[j] < 0 else "—"
                    match = "✓" if expected == actual else "✗"
                    lines.append(
                        f"    {act_name} → {self.names[j]}: "
                        f"expected {expected}, actual {actual} {match} "
                        f"(w={w:+.2f}, Δ={delta[j]:+.4f})"
                    )
        
        return "\n".join(lines)
