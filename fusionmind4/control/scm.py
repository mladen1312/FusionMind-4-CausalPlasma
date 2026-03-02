"""
Structural Causal Model (SCM) for Tokamak Plasma

Implements Pearl's SCM framework adapted for plasma physics:
- Each variable has a structural equation: X_j = f_j(PA_j, U_j)
  where PA_j are causal parents and U_j is exogenous noise
- Supports interventions via do-calculus: P(Y | do(X=x))
- Supports counterfactuals: P(Y_x | X=x', Y=y')

This is the mathematical backbone of the Counterfactual Controller.

Part of: FusionMind 4.0 / Patent Family PF2
Author: Dr. Mladen Mester, March 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class StructuralEquation:
    """One structural equation: X_j = f_j(parents) + noise"""
    variable: str
    parents: List[str]
    coefficients: Dict[str, float]    # Linear coefficients per parent
    nonlinear_fn: Optional[Callable] = None  # Optional nonlinear transform
    noise_std: float = 0.02
    intercept: float = 0.0


class PlasmaSCM:
    """
    Structural Causal Model for tokamak plasma.
    
    Given a discovered causal DAG (from CPDE), this class:
    1. Fits structural equations to observed data
    2. Answers interventional queries: P(Y | do(X=x))
    3. Answers counterfactual queries: P(Y_x | evidence)
    
    Pearl's Causal Hierarchy:
    Level 1 (Association):    P(Y | X=x)        — "what if I observe X=x?"
    Level 2 (Intervention):   P(Y | do(X=x))    — "what if I SET X=x?"
    Level 3 (Counterfactual): P(Y_x | X=x',Y=y) — "what WOULD HAVE happened?"
    """

    def __init__(self, variable_names: List[str], dag: np.ndarray):
        """
        Args:
            variable_names: list of variable names
            dag: (n, n) weighted adjacency matrix from CPDE. dag[i,j] = i→j
        """
        self.names = variable_names
        self.n_vars = len(variable_names)
        self.dag = dag
        self.idx = {name: i for i, name in enumerate(variable_names)}
        
        self.equations: Dict[str, StructuralEquation] = {}
        self.noise_distributions: Dict[str, np.ndarray] = {}
        self._fitted = False

    def fit(self, data: np.ndarray, verbose: bool = True):
        """
        Fit structural equations from data given the causal graph.
        
        For each variable j, find its parents (from DAG) and fit:
        X_j = Σ_i β_i * X_i + U_j    (for parents i of j)
        
        Store the noise terms U_j = X_j - Σ β_i X_i for counterfactuals.
        """
        if verbose:
            print("Fitting Structural Causal Model...")
        
        for j in range(self.n_vars):
            var_name = self.names[j]
            
            # Find parents from DAG
            parent_indices = np.where(np.abs(self.dag[:, j]) > 0.01)[0]
            parent_names = [self.names[i] for i in parent_indices]
            
            if len(parent_indices) == 0:
                # Exogenous variable (no parents) — e.g., actuators
                self.equations[var_name] = StructuralEquation(
                    variable=var_name,
                    parents=[],
                    coefficients={},
                    noise_std=float(np.std(data[:, j])),
                    intercept=float(np.mean(data[:, j]))
                )
                self.noise_distributions[var_name] = data[:, j] - np.mean(data[:, j])
            else:
                # Fit linear structural equation via OLS
                X_parents = data[:, parent_indices]
                y = data[:, j]
                
                # OLS: β = (X'X)^{-1} X'y
                X_aug = np.column_stack([X_parents, np.ones(len(y))])
                try:
                    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
                except:
                    beta = np.zeros(len(parent_indices) + 1)
                
                coefficients = {parent_names[k]: float(beta[k]) 
                                for k in range(len(parent_indices))}
                intercept = float(beta[-1])
                
                # Compute residuals (noise terms)
                y_pred = X_aug @ beta
                residuals = y - y_pred
                
                self.equations[var_name] = StructuralEquation(
                    variable=var_name,
                    parents=parent_names,
                    coefficients=coefficients,
                    noise_std=float(np.std(residuals)),
                    intercept=intercept
                )
                self.noise_distributions[var_name] = residuals
                
                if verbose:
                    eq_str = f"  {var_name} = "
                    terms = [f"{v:+.3f}·{k}" for k, v in coefficients.items()]
                    eq_str += " ".join(terms)
                    eq_str += f" + {intercept:+.3f} + U (σ={np.std(residuals):.4f})"
                    print(eq_str)
        
        self._fitted = True
        if verbose:
            print(f"\n  Fitted {len(self.equations)} structural equations")
            n_exo = sum(1 for eq in self.equations.values() if not eq.parents)
            print(f"  Exogenous variables: {n_exo}")
            print(f"  Endogenous variables: {len(self.equations) - n_exo}")

    def predict(self, values: Dict[str, float]) -> Dict[str, float]:
        """
        Forward pass through SCM: compute all variables given parent values.
        Uses topological ordering of the DAG.
        """
        result = dict(values)
        order = self._topological_order()
        
        for var in order:
            if var in result:
                continue  # Already set (intervention or exogenous)
            
            eq = self.equations[var]
            val = eq.intercept
            for parent, coeff in eq.coefficients.items():
                if parent in result:
                    val += coeff * result[parent]
            result[var] = val
        
        return result

    def _topological_order(self) -> List[str]:
        """Return variables in causal (topological) order."""
        import networkx as nx
        G = nx.DiGraph()
        for i in range(self.n_vars):
            G.add_node(self.names[i])
            for j in range(self.n_vars):
                if abs(self.dag[i, j]) > 0.01:
                    G.add_edge(self.names[i], self.names[j])
        
        try:
            return list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # If cycles exist, use approximate ordering
            return self.names

    def get_equation_string(self, var: str) -> str:
        """Human-readable structural equation."""
        eq = self.equations[var]
        if not eq.parents:
            return f"{var} = {eq.intercept:.3f} + U_{var}"
        terms = [f"{v:+.3f}·{k}" for k, v in eq.coefficients.items()]
        return f"{var} = {' '.join(terms)} {eq.intercept:+.3f} + U_{var}"
