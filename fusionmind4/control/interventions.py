"""
do-Calculus Intervention Engine & Counterfactual Reasoning

Implements Pearl's three levels of causal reasoning for plasma control:

Level 2 — INTERVENTIONS: P(Y | do(X=x))
  "What happens to Te if we SET P_ECRH = 10MW?"
  Implementation: Cut all incoming edges to X, set X=x, propagate.

Level 3 — COUNTERFACTUALS: P(Y_x | X=x', Y=y')
  "Given that Te was 8keV when P_ECRH was 5MW, what WOULD Te have been 
   if P_ECRH had been 10MW instead?"
  Implementation: Abduction (infer noise) → Action (intervene) → Prediction

Part of: FusionMind 4.0 / Patent Family PF2
Author: Dr. Mladen Mešter, dr.med., March 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .scm import PlasmaSCM


@dataclass
class InterventionResult:
    """Result of a do-calculus intervention."""
    intervention: Dict[str, float]    # do(X=x)
    outcomes: Dict[str, float]        # Resulting values
    causal_effects: Dict[str, float]  # Change from baseline per variable
    baseline: Dict[str, float]        # Pre-intervention state


@dataclass
class CounterfactualResult:
    """Result of a counterfactual query."""
    factual_state: Dict[str, float]        # What actually happened
    counterfactual_intervention: Dict[str, float]  # What we wish had happened
    counterfactual_outcomes: Dict[str, float]       # What would have happened
    effects: Dict[str, float]                       # Difference
    abducted_noise: Dict[str, float]                # Inferred noise terms


class InterventionEngine:
    """
    do-Calculus engine for plasma control.
    
    Answers: "If we SET actuator X to value x, what happens to plasma state Y?"
    
    This is fundamentally different from correlation:
    - Correlation: P(Te | P_ECRH=10) — "When P_ECRH is 10, Te tends to be..."
    - Intervention: P(Te | do(P_ECRH=10)) — "If we FORCE P_ECRH to 10, Te WILL be..."
    
    The key difference: do(X=x) removes all incoming edges to X,
    eliminating confounding. This is what makes causal control possible.
    """

    def __init__(self, scm: PlasmaSCM):
        self.scm = scm

    def do(self, interventions: Dict[str, float], 
           current_state: Optional[Dict[str, float]] = None) -> InterventionResult:
        """
        Perform do-calculus intervention.
        
        do(X=x) means:
        1. In the SCM, replace equation for X with X := x (constant)
        2. Cut all edges INTO X (remove confounding)
        3. Propagate through remaining equations
        
        Args:
            interventions: {variable_name: forced_value}
            current_state: Current plasma state (for baseline comparison)
        """
        if not self.scm._fitted:
            raise RuntimeError("SCM must be fitted before interventions")

        # Compute baseline (no intervention)
        if current_state is None:
            current_state = {name: eq.intercept 
                            for name, eq in self.scm.equations.items()}
        baseline = self.scm.predict(current_state)

        # Create mutilated SCM (cut incoming edges to intervened variables)
        # Then propagate with intervened values
        intervened_state = dict(current_state)
        intervened_state.update(interventions)

        # Forward propagation through DAG with interventions fixed
        order = self.scm._topological_order()
        result = {}
        
        for var in order:
            if var in interventions:
                result[var] = interventions[var]  # Fixed by do()
            elif var in current_state and self.scm.equations[var].parents == []:
                result[var] = current_state[var]  # Exogenous, unchanged
            else:
                eq = self.scm.equations[var]
                val = eq.intercept
                for parent, coeff in eq.coefficients.items():
                    parent_val = result.get(parent, current_state.get(parent, 0))
                    val += coeff * parent_val
                result[var] = val

        # Compute causal effects
        effects = {var: result.get(var, 0) - baseline.get(var, 0) 
                   for var in self.scm.names}

        return InterventionResult(
            intervention=interventions,
            outcomes=result,
            causal_effects=effects,
            baseline=baseline
        )

    def average_causal_effect(self, cause: str, effect: str,
                               values: np.ndarray,
                               current_state: Dict[str, float]) -> np.ndarray:
        """
        Compute ACE: E[Y | do(X=x)] for a range of x values.
        This gives the causal dose-response curve.
        """
        effects = []
        for x in values:
            result = self.do({cause: x}, current_state)
            effects.append(result.outcomes.get(effect, 0))
        return np.array(effects)

    def find_optimal_intervention(self, target_var: str, target_value: float,
                                    actuators: List[str],
                                    current_state: Dict[str, float],
                                    bounds: Dict[str, Tuple[float, float]] = None) -> Dict:
        """
        Find actuator settings that achieve desired target via causal reasoning.
        
        This is the CONTROL application: "What do I need to DO to get Te = 10keV?"
        Uses the causal graph to find the most efficient intervention path.
        """
        from scipy.optimize import minimize

        if bounds is None:
            bounds = {a: (0.1, 0.95) for a in actuators}

        def objective(x):
            intervention = {act: x[i] for i, act in enumerate(actuators)}
            result = self.do(intervention, current_state)
            predicted = result.outcomes.get(target_var, 0)
            return (predicted - target_value) ** 2

        x0 = np.array([current_state.get(a, 0.5) for a in actuators])
        bds = [bounds[a] for a in actuators]

        opt = minimize(objective, x0, bounds=bds, method='L-BFGS-B')

        optimal_intervention = {act: opt.x[i] for i, act in enumerate(actuators)}
        result = self.do(optimal_intervention, current_state)

        return {
            'optimal_actuators': optimal_intervention,
            'predicted_outcome': result.outcomes,
            'target_achieved': result.outcomes.get(target_var, 0),
            'target_error': abs(result.outcomes.get(target_var, 0) - target_value),
            'causal_effects': result.causal_effects,
            'optimization_success': opt.success
        }


class CounterfactualEngine:
    """
    Counterfactual reasoning engine for plasma control.
    
    Answers questions like:
    "Given that we observed Te=8keV when P_ECRH=5MW and P_NBI=10MW,
     what WOULD Te have been if P_ECRH had been 8MW instead?"
    
    Pearl's 3-step algorithm:
    1. ABDUCTION: Infer noise terms U from observed state
       U_j = X_j_observed - f_j(PA_j_observed)
    2. ACTION: Apply the counterfactual intervention do(X=x')
       Modify structural equations, keep same noise U
    3. PREDICTION: Propagate through modified SCM with original noise
    """

    def __init__(self, scm: PlasmaSCM):
        self.scm = scm

    def counterfactual(self, factual_state: Dict[str, float],
                        intervention: Dict[str, float]) -> CounterfactualResult:
        """
        Full counterfactual query.
        
        Args:
            factual_state: What actually happened {var: observed_value}
            intervention: What we wish had happened {var: counterfactual_value}
        """
        # Step 1: ABDUCTION — infer noise terms from observed state
        noise = self._abduction(factual_state)

        # Step 2 & 3: ACTION + PREDICTION — intervene and propagate with same noise
        cf_outcomes = self._predict_counterfactual(factual_state, intervention, noise)

        # Compute effects
        effects = {var: cf_outcomes.get(var, 0) - factual_state.get(var, 0)
                   for var in self.scm.names}

        return CounterfactualResult(
            factual_state=factual_state,
            counterfactual_intervention=intervention,
            counterfactual_outcomes=cf_outcomes,
            effects=effects,
            abducted_noise=noise
        )

    def _abduction(self, observed: Dict[str, float]) -> Dict[str, float]:
        """
        Step 1: Infer exogenous noise from observed data.
        
        For each variable j:  U_j = X_j - f_j(PA_j)
        This "explains" why the observed values are what they are.
        """
        noise = {}
        order = self.scm._topological_order()
        
        for var in order:
            eq = self.scm.equations[var]
            
            if not eq.parents:
                # Exogenous: noise = deviation from mean
                noise[var] = observed.get(var, eq.intercept) - eq.intercept
            else:
                # Compute predicted value from parents
                predicted = eq.intercept
                for parent, coeff in eq.coefficients.items():
                    predicted += coeff * observed.get(parent, 0)
                
                # Noise = residual
                noise[var] = observed.get(var, predicted) - predicted
        
        return noise

    def _predict_counterfactual(self, factual: Dict[str, float],
                                  intervention: Dict[str, float],
                                  noise: Dict[str, float]) -> Dict[str, float]:
        """
        Steps 2+3: Apply intervention and propagate with abducted noise.
        
        Key insight: We use the SAME noise terms as the factual world.
        This is what makes it a counterfactual (same exogenous conditions,
        different intervention) rather than just an intervention.
        """
        result = {}
        order = self.scm._topological_order()
        
        for var in order:
            if var in intervention:
                result[var] = intervention[var]  # Overridden by do()
            else:
                eq = self.scm.equations[var]
                
                if not eq.parents:
                    # Exogenous: use factual value (same world)
                    result[var] = factual.get(var, eq.intercept)
                else:
                    # Compute from parents + SAME noise
                    val = eq.intercept
                    for parent, coeff in eq.coefficients.items():
                        parent_val = result.get(parent, factual.get(parent, 0))
                        val += coeff * parent_val
                    val += noise.get(var, 0)  # Add back the factual noise!
                    result[var] = val
        
        return result

    def what_if_analysis(self, factual_state: Dict[str, float],
                          cause: str, effect: str,
                          cf_values: np.ndarray) -> Dict:
        """
        "What if" sweep: compute counterfactual outcomes across a range.
        
        Example: "What would Te have been if P_ECRH were 3, 5, 7, 9, 11 MW?"
        """
        results = []
        for val in cf_values:
            cf = self.counterfactual(factual_state, {cause: val})
            results.append({
                'cause_value': val,
                'effect_value': cf.counterfactual_outcomes.get(effect, 0),
                'effect_change': cf.effects.get(effect, 0)
            })
        
        return {
            'cause': cause,
            'effect': effect,
            'factual_cause': factual_state.get(cause, 0),
            'factual_effect': factual_state.get(effect, 0),
            'counterfactual_sweep': results
        }
