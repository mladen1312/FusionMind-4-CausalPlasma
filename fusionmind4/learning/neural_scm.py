"""
NeuralSCM — Differentiable Structural Causal Model for Plasma
===============================================================

Upgrades the linear PlasmaSCM (PF2) to a learnable, differentiable model:
- Each structural equation X_j = f_θ(PA_j) + U_j is a small neural network
- Supports gradient descent for online adaptation
- Preserves causal graph structure (edges from CPDE)
- Enables model-based RL: agent can backprop through the world model

Key innovation: Neural networks are CONSTRAINED to the DAG topology.
Only parent → child connections exist, so the model stays causal.

Part of: FusionMind 4.0 / Patent Family PF7 (CausalShield-RL)
Author: Dr. Mladen Mester, March 2026
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════
# Lightweight neural network (NumPy — no PyTorch/JAX dependency needed)
# ═══════════════════════════════════════════════════════════════════════

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


@dataclass
class MLPParams:
    """Parameters of a 2-layer MLP: input → hidden → output."""
    W1: np.ndarray      # (hidden, input)
    b1: np.ndarray      # (hidden,)
    W2: np.ndarray      # (1, hidden)
    b2: np.ndarray      # (1,)
    
    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


class NeuralEquation:
    """
    One neural structural equation: X_j = MLP_θ(parents_j) + U_j
    
    The MLP maps parent values to the child's predicted value.
    This is a drop-in replacement for the linear equation in PlasmaSCM.
    """
    
    def __init__(self, variable: str, parents: List[str], 
                 hidden_dim: int = 16, lr: float = 1e-3):
        self.variable = variable
        self.parents = parents
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        input_dim = max(len(parents), 1)
        
        # Xavier initialization
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        
        self.params = MLPParams(
            W1=np.random.randn(hidden_dim, input_dim) * scale1,
            b1=np.zeros(hidden_dim),
            W2=np.random.randn(1, hidden_dim) * scale2,
            b2=np.zeros(1)
        )
        
        # Noise distribution (learned from data)
        self.noise_std = 0.05
        self.noise_samples: Optional[np.ndarray] = None
        
        # Adam optimizer state
        self._m = {k: np.zeros_like(getattr(self.params, k)) for k in ['W1', 'b1', 'W2', 'b2']}
        self._v = {k: np.zeros_like(getattr(self.params, k)) for k in ['W1', 'b1', 'W2', 'b2']}
        self._t = 0
    
    def forward(self, parent_values: np.ndarray) -> float:
        """Forward pass: parent_values → predicted child value."""
        x = parent_values.reshape(-1)
        if len(x) == 0:
            x = np.zeros(1)
        
        # Layer 1
        h = self.params.W1 @ x + self.params.b1
        h = _relu(h)
        
        # Layer 2
        out = self.params.W2 @ h + self.params.b2
        return float(out[0])
    
    def forward_batch(self, parent_data: np.ndarray) -> np.ndarray:
        """Batch forward pass: (n_samples, n_parents) → (n_samples,)."""
        if parent_data.ndim == 1:
            parent_data = parent_data.reshape(-1, 1)
        if parent_data.shape[1] == 0:
            parent_data = np.zeros((parent_data.shape[0], 1))
        
        # Layer 1: (n, hidden)
        H = parent_data @ self.params.W1.T + self.params.b1
        H = _relu(H)
        
        # Layer 2: (n, 1)
        out = H @ self.params.W2.T + self.params.b2
        return out.ravel()
    
    def fit(self, parent_data: np.ndarray, target: np.ndarray, 
            n_epochs: int = 200, batch_size: int = 256) -> float:
        """
        Fit neural equation to data using mini-batch Adam.
        
        Args:
            parent_data: (n_samples, n_parents) parent variable data
            target: (n_samples,) child variable data
            n_epochs: Training epochs
            batch_size: Mini-batch size
            
        Returns:
            Final MSE loss
        """
        n = len(target)
        if parent_data.ndim == 1:
            parent_data = parent_data.reshape(-1, 1)
        if parent_data.shape[1] == 0:
            parent_data = np.zeros((n, 1))
        
        best_loss = float('inf')
        
        for epoch in range(n_epochs):
            # Mini-batch sampling
            idx = np.random.choice(n, min(batch_size, n), replace=False)
            X_batch = parent_data[idx]
            y_batch = target[idx]
            
            # Forward
            H_pre = X_batch @ self.params.W1.T + self.params.b1  # (B, H)
            H = _relu(H_pre)
            y_pred = (H @ self.params.W2.T + self.params.b2).ravel()  # (B,)
            
            # Loss
            residuals = y_pred - y_batch
            loss = float(np.mean(residuals ** 2))
            
            # Backward (analytical gradients)
            B = len(idx)
            d_out = 2 * residuals / B  # (B,)
            
            # Layer 2 gradients
            dW2 = d_out.reshape(1, -1) @ H         # (1, H)
            db2 = np.array([d_out.sum()])            # (1,)
            
            # Through ReLU
            d_h = np.outer(d_out, self.params.W2[0]) * _relu_grad(H_pre)  # (B, H)
            
            # Layer 1 gradients
            dW1 = d_h.T @ X_batch                    # (H, input)
            db1 = d_h.sum(axis=0)                    # (H,)
            
            # Adam update
            grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
            self._adam_step(grads)
            
            if loss < best_loss:
                best_loss = loss
        
        # Store noise distribution
        y_full_pred = self.forward_batch(parent_data)
        self.noise_samples = target - y_full_pred
        self.noise_std = float(np.std(self.noise_samples))
        
        return best_loss
    
    def _adam_step(self, grads: Dict[str, np.ndarray],
                   beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """Adam optimizer step."""
        self._t += 1
        for key in ['W1', 'b1', 'W2', 'b2']:
            g = grads[key]
            self._m[key] = beta1 * self._m[key] + (1 - beta1) * g
            self._v[key] = beta2 * self._v[key] + (1 - beta2) * g ** 2
            m_hat = self._m[key] / (1 - beta1 ** self._t)
            v_hat = self._v[key] / (1 - beta2 ** self._t)
            
            current = getattr(self.params, key)
            setattr(self.params, key, current - self.lr * m_hat / (np.sqrt(v_hat) + eps))


# ═══════════════════════════════════════════════════════════════════════
# NeuralSCM — Full differentiable SCM
# ═══════════════════════════════════════════════════════════════════════

class NeuralSCM:
    """
    Differentiable Structural Causal Model for tokamak plasma.
    
    Upgrades PlasmaSCM from linear to neural structural equations while
    preserving the causal DAG topology discovered by CPDE.
    
    Features:
    - Neural structural equations: X_j = MLP_θ(PA_j) + U_j
    - Online adaptation: refit from new shot data
    - Counterfactual reasoning: same 3-step procedure as linear SCM
    - do-calculus interventions: graph surgery + neural forward pass
    - Jacobian computation: ∂Y/∂X for sensitivity analysis
    
    This serves as the differentiable world model for CausalShield-RL.
    
    Part of: Patent Family PF7 (CausalShield-RL)
    Author: Dr. Mladen Mester, March 2026
    """
    
    def __init__(self, variable_names: List[str], dag: np.ndarray,
                 hidden_dim: int = 16, lr: float = 1e-3):
        """
        Args:
            variable_names: List of plasma variable names
            dag: (n, n) weighted adjacency matrix from CPDE (dag[i,j] = i→j)
            hidden_dim: Hidden layer size for neural equations
            lr: Learning rate for Adam optimizer
        """
        self.names = variable_names
        self.n_vars = len(variable_names)
        self.dag = dag
        self.idx = {name: i for i, name in enumerate(variable_names)}
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        self.equations: Dict[str, NeuralEquation] = {}
        self._fitted = False
        self._fit_losses: Dict[str, float] = {}
    
    def fit(self, data: np.ndarray, n_epochs: int = 300, verbose: bool = True) -> Dict:
        """
        Fit all neural structural equations from data.
        
        Args:
            data: (n_samples, n_vars) observed data
            n_epochs: Training epochs per equation
            verbose: Print progress
            
        Returns:
            Dictionary with per-variable fit losses
        """
        if verbose:
            print("Fitting Neural SCM...")
            print(f"  Variables: {self.n_vars}, Samples: {data.shape[0]}")
        
        total_params = 0
        
        for j in range(self.n_vars):
            var_name = self.names[j]
            
            # Find parents from DAG
            parent_indices = np.where(np.abs(self.dag[:, j]) > 0.01)[0]
            parent_names = [self.names[i] for i in parent_indices]
            
            # Create neural equation
            eq = NeuralEquation(
                variable=var_name,
                parents=parent_names,
                hidden_dim=self.hidden_dim,
                lr=self.lr
            )
            
            if len(parent_indices) == 0:
                # Exogenous variable — store empirical distribution
                eq.noise_samples = data[:, j] - np.mean(data[:, j])
                eq.noise_std = float(np.std(data[:, j]))
                eq.params.b2 = np.array([np.mean(data[:, j])])
                loss = float(np.var(data[:, j]))
            else:
                # Fit neural equation
                parent_data = data[:, parent_indices]
                target = data[:, j]
                loss = eq.fit(parent_data, target, n_epochs=n_epochs)
            
            self.equations[var_name] = eq
            self._fit_losses[var_name] = loss
            total_params += eq.params.n_params()
            
            if verbose:
                parents_str = ", ".join(parent_names) if parent_names else "(exogenous)"
                print(f"  {var_name:12s} ← {parents_str:40s} | MSE={loss:.6f} | σ_noise={eq.noise_std:.4f}")
        
        self._fitted = True
        
        if verbose:
            avg_loss = np.mean(list(self._fit_losses.values()))
            print(f"\n  Total learnable parameters: {total_params:,}")
            print(f"  Average MSE: {avg_loss:.6f}")
        
        return self._fit_losses
    
    def predict(self, values: Dict[str, float]) -> Dict[str, float]:
        """
        Forward pass through Neural SCM.
        
        Computes all variables in topological order using neural equations.
        """
        result = dict(values)
        order = self._topological_order()
        
        for var in order:
            if var in result:
                continue
            
            eq = self.equations[var]
            if not eq.parents:
                result[var] = float(eq.params.b2[0])
                continue
            
            parent_vals = np.array([result.get(p, 0.0) for p in eq.parents])
            result[var] = eq.forward(parent_vals)
        
        return result
    
    def predict_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Batch forward pass: predict all variables from exogenous inputs.
        
        Args:
            data: (n_samples, n_vars) — only exogenous columns need to be filled
            
        Returns:
            (n_samples, n_vars) with all variables predicted
        """
        result = data.copy()
        order = self._topological_order()
        
        for var in order:
            j = self.idx[var]
            eq = self.equations[var]
            
            if not eq.parents:
                continue  # Keep exogenous values as-is
            
            parent_indices = [self.idx[p] for p in eq.parents]
            parent_data = result[:, parent_indices]
            result[:, j] = eq.forward_batch(parent_data)
        
        return result
    
    def do_intervention(self, state: Dict[str, float], 
                        interventions: Dict[str, float]) -> Dict[str, float]:
        """
        do-Calculus intervention: P(Y | do(X=x)).
        
        Graph surgery: cut all incoming edges to intervened variables,
        set them to specified values, propagate through neural equations.
        """
        result = {}
        order = self._topological_order()
        
        for var in order:
            if var in interventions:
                result[var] = interventions[var]  # Fixed by do()
            elif var in state and not self.equations[var].parents:
                result[var] = state[var]  # Exogenous, unchanged
            else:
                eq = self.equations[var]
                if not eq.parents:
                    result[var] = float(eq.params.b2[0])
                else:
                    parent_vals = np.array([
                        result.get(p, state.get(p, 0.0)) for p in eq.parents
                    ])
                    result[var] = eq.forward(parent_vals)
        
        return result
    
    def counterfactual(self, factual: Dict[str, float],
                       intervention: Dict[str, float]) -> Dict[str, float]:
        """
        Counterfactual query via Pearl's 3-step procedure.
        
        Step 1 (Abduction): Infer noise U from factual observations
        Step 2 (Action): Apply counterfactual do(X=x')
        Step 3 (Prediction): Propagate with original noise
        """
        # Step 1: Abduction — compute noise terms
        noise = self._abduction(factual)
        
        # Steps 2+3: Action + Prediction
        result = {}
        order = self._topological_order()
        
        for var in order:
            if var in intervention:
                result[var] = intervention[var]
            elif not self.equations[var].parents:
                result[var] = factual.get(var, 0.0)
            else:
                eq = self.equations[var]
                parent_vals = np.array([
                    result.get(p, factual.get(p, 0.0)) for p in eq.parents
                ])
                result[var] = eq.forward(parent_vals) + noise.get(var, 0.0)
        
        return result
    
    def _abduction(self, observed: Dict[str, float]) -> Dict[str, float]:
        """Infer exogenous noise terms from observations."""
        noise = {}
        order = self._topological_order()
        
        for var in order:
            eq = self.equations[var]
            if not eq.parents:
                noise[var] = observed.get(var, 0.0) - float(eq.params.b2[0])
            else:
                parent_vals = np.array([observed.get(p, 0.0) for p in eq.parents])
                predicted = eq.forward(parent_vals)
                noise[var] = observed.get(var, predicted) - predicted
        
        return noise
    
    def jacobian(self, state: Dict[str, float]) -> np.ndarray:
        """
        Compute Jacobian ∂Y/∂X at given state using finite differences.
        
        Returns (n_vars, n_vars) matrix where J[i,j] = ∂x_j / ∂x_i.
        Useful for sensitivity analysis and reward shaping.
        """
        eps = 1e-5
        J = np.zeros((self.n_vars, self.n_vars))
        
        baseline = self.do_intervention(state, {})
        
        for i in range(self.n_vars):
            var_i = self.names[i]
            perturbed = dict(state)
            perturbed[var_i] = state.get(var_i, 0.0) + eps
            
            result = self.do_intervention(perturbed, {var_i: perturbed[var_i]})
            
            for j in range(self.n_vars):
                var_j = self.names[j]
                J[i, j] = (result.get(var_j, 0.0) - baseline.get(var_j, 0.0)) / eps
        
        return J
    
    def online_update(self, new_data: np.ndarray, n_epochs: int = 50):
        """
        Online adaptation: update neural equations with new shot data.
        
        This enables continuous learning after each plasma shot.
        AEDE (PF5) selects informative shots → CPDE updates graph →
        NeuralSCM refines equations.
        """
        for j in range(self.n_vars):
            var_name = self.names[j]
            eq = self.equations[var_name]
            
            if not eq.parents:
                continue
            
            parent_indices = [self.idx[p] for p in eq.parents]
            parent_data = new_data[:, parent_indices]
            target = new_data[:, j]
            
            # Fine-tune with lower learning rate
            old_lr = eq.lr
            eq.lr = old_lr * 0.1  # Conservative update
            eq.fit(parent_data, target, n_epochs=n_epochs)
            eq.lr = old_lr
    
    def get_causal_effect_matrix(self, state: Dict[str, float]) -> np.ndarray:
        """
        Compute total causal effects between all variable pairs.
        
        Uses do-calculus: effect[i,j] = E[x_j | do(x_i = x_i + δ)] - E[x_j]
        
        Returns (n_vars, n_vars) matrix of causal effects.
        """
        delta = 0.1  # Perturbation size
        effects = np.zeros((self.n_vars, self.n_vars))
        
        baseline = self.do_intervention(state, {})
        
        for i in range(self.n_vars):
            var_i = self.names[i]
            new_val = state.get(var_i, 0.0) + delta
            
            result = self.do_intervention(state, {var_i: new_val})
            
            for j in range(self.n_vars):
                var_j = self.names[j]
                effects[i, j] = (result.get(var_j, 0.0) - baseline.get(var_j, 0.0)) / delta
        
        return effects
    
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
            return list(self.names)
    
    def summary(self) -> str:
        """Human-readable summary of the Neural SCM."""
        lines = ["Neural SCM Summary", "=" * 50]
        total_params = 0
        
        for var in self._topological_order():
            eq = self.equations[var]
            n_p = eq.params.n_params()
            total_params += n_p
            parents_str = ", ".join(eq.parents) if eq.parents else "(exogenous)"
            loss = self._fit_losses.get(var, 0)
            lines.append(
                f"  {var:12s} ← {parents_str:35s} | "
                f"params={n_p:4d} | MSE={loss:.6f}"
            )
        
        lines.append(f"\nTotal parameters: {total_params:,}")
        lines.append(f"Variables: {self.n_vars}")
        lines.append(f"Edges: {int(np.sum(np.abs(self.dag) > 0.01))}")
        
        return "\n".join(lines)
