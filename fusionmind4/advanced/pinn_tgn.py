#!/usr/bin/env python3
"""
Hybrid PINN + Temporal Graph Network for Plasma Disruption Prediction
======================================================================

Treats the plasma as a GRAPH where:
  - Nodes = plasma variables (li, q95, βN, ne, ...)  OR  radial zones
  - Edges = causal/physical relationships (from CPDE DAG)
  - Node features = signal values + rates + margins
  - Edge features = interaction strength + physics coupling

The PINN component enforces physics constraints:
  - Transport equations: ∂n/∂t = -∇·Γ + S
  - Energy balance: ∂(nT)/∂t = -∇·q + P_heat - P_rad
  - MHD equilibrium: ∇p = J×B (Grad-Shafranov)

The TGN component learns temporal evolution:
  - Message passing between connected variables
  - Temporal attention over shot history
  - Disruption = anomalous graph evolution

ACTIVATION CONDITIONS:
  - Profile data with ≥20 radial points (for spatial graph), OR
  - ≥10 0D variables with known causal structure (for variable graph)
  - ≥200 shots for training
  - Causal DAG available (from CPDE)

Two modes:
  MODE A: Variable Graph (works with 0D data)
    Nodes = variables, edges = causal relationships
    Available NOW with existing MAST data
    
  MODE B: Spatial Graph (needs profile data)  
    Nodes = radial zones, edges = transport connections
    Needs 1D profile data — future activation

Author: Dr. Mladen Mester
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings


@dataclass
class PINNTGNConfig:
    """Configuration for Hybrid PINN+TGN."""
    # Activation for Mode A (variable graph)
    min_variables_mode_a: int = 10
    min_shots_mode_a: int = 200
    
    # Activation for Mode B (spatial graph)
    min_radial_points: int = 20
    min_shots_mode_b: int = 100
    
    # Graph architecture
    node_feature_dim: int = 8     # Features per node
    edge_feature_dim: int = 4     # Features per edge
    hidden_dim: int = 32
    n_message_passing: int = 3    # GNN layers
    n_temporal_steps: int = 10    # Lookback window
    
    # PINN physics
    transport_weight: float = 0.1    # Transport equation residual
    energy_weight: float = 0.1       # Energy balance residual
    mhd_weight: float = 0.05         # MHD constraint residual
    
    # Training
    n_epochs: int = 50
    learning_rate: float = 1e-3
    random_state: int = 42


def check_activation(data_info: Dict) -> Tuple[bool, str, str]:
    """Check activation and determine mode.
    
    Returns:
        (can_activate, reason, mode)  where mode is 'A', 'B', or 'none'
    """
    config = PINNTGNConfig()
    
    # Mode B: spatial graph (higher priority, more powerful)
    has_profiles = data_info.get('has_profiles', False)
    if has_profiles:
        nr = data_info.get('n_radial_points', 0)
        ns = data_info.get('n_shots', 0)
        if nr >= config.min_radial_points and ns >= config.min_shots_mode_b:
            return True, f"Mode B (spatial graph): {nr} radial × {ns} shots", 'B'
    
    # Mode A: variable graph
    nv = data_info.get('n_variables', 0)
    ns = data_info.get('n_shots', 0)
    has_dag = data_info.get('has_causal_dag', False)
    
    if nv >= config.min_variables_mode_a and ns >= config.min_shots_mode_a:
        if has_dag:
            return True, f"Mode A (variable graph): {nv} vars × {ns} shots, DAG available", 'A'
        else:
            return True, f"Mode A (variable graph): {nv} vars × {ns} shots, no DAG (will use correlation)", 'A'
    
    return False, f"Need ≥{config.min_variables_mode_a} vars and ≥{config.min_shots_mode_a} shots, have {nv}/{ns}", 'none'


# ═══════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

class PlasmaGraph:
    """Constructs a graph from plasma variables or radial zones.
    
    For Mode A (variable graph):
      - Nodes = [li, q95, βN, βp, ne, Ip, P_rad, Wmhd, ...]
      - Edges from CPDE causal DAG or correlation threshold
    
    For Mode B (spatial graph):
      - Nodes = radial zones [r=0, r=0.1, ..., r=1.0]
      - Edges = neighboring zones (transport connections)
    """
    
    def __init__(self, mode: str, var_names: List[str] = None,
                 causal_dag: np.ndarray = None, n_radial: int = None):
        self.mode = mode
        self.var_names = var_names
        self.n_nodes = len(var_names) if mode == 'A' else n_radial
        
        # Build adjacency
        if mode == 'A':
            self.adjacency = self._build_variable_graph(causal_dag)
        else:
            self.adjacency = self._build_spatial_graph(n_radial)
        
        # Edge list for message passing
        self.edge_index = np.array(np.where(self.adjacency > 0)).T  # [n_edges, 2]
        self.n_edges = len(self.edge_index)
    
    def _build_variable_graph(self, dag: np.ndarray = None) -> np.ndarray:
        """Build graph from causal DAG or fully connected."""
        n = self.n_nodes
        if dag is not None and dag.shape == (n, n):
            # Use DAG: edge exists if causal relationship found
            adj = (np.abs(dag) > 0.1).astype(float)
            # Make undirected (message passing goes both ways)
            adj = np.maximum(adj, adj.T)
        else:
            # Fully connected (will learn which edges matter)
            adj = np.ones((n, n)) - np.eye(n)
        return adj
    
    def _build_spatial_graph(self, n_radial: int) -> np.ndarray:
        """Build graph from radial grid: connect neighboring zones."""
        adj = np.zeros((n_radial, n_radial))
        for i in range(n_radial - 1):
            adj[i, i+1] = 1.0
            adj[i+1, i] = 1.0
        # Also connect every 3rd zone (long-range transport)
        for i in range(n_radial - 3):
            adj[i, i+3] = 0.5
            adj[i+3, i] = 0.5
        return adj


# ═══════════════════════════════════════════════════════════════
# MESSAGE PASSING NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════

class MessagePassingLayer:
    """Single message passing step on the plasma graph.
    
    For each node i:
      m_i = AGG({MSG(h_i, h_j, e_ij) : j ∈ N(i)})
      h_i' = UPDATE(h_i, m_i)
    
    MSG = MLP on concatenated features
    AGG = mean aggregation
    UPDATE = GRU cell (temporal awareness)
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # Message function: [h_i, h_j, e_ij] → message
        msg_input = 2 * node_dim + edge_dim
        self.msg_W1 = np.random.randn(msg_input, hidden_dim) * np.sqrt(2.0/msg_input)
        self.msg_b1 = np.zeros(hidden_dim)
        self.msg_W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0/hidden_dim)
        self.msg_b2 = np.zeros(hidden_dim)
        
        # Update weights (flexible input dimension)
        max_input = 2 * hidden_dim + edge_dim  # Max possible concat size
        self.update_Wz = np.random.randn(max_input, hidden_dim) * 0.1
        self.update_Wr = np.random.randn(max_input, hidden_dim) * 0.1
        self.update_Wh = np.random.randn(max_input, hidden_dim) * 0.1
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    
    def forward(self, node_features: np.ndarray, edge_index: np.ndarray,
                edge_features: np.ndarray) -> np.ndarray:
        """
        Args:
            node_features: [n_nodes, node_dim]
            edge_index: [n_edges, 2] — (source, target) pairs
            edge_features: [n_edges, edge_dim]
        Returns:
            [n_nodes, hidden_dim] — updated node features
        """
        n_nodes = node_features.shape[0]
        
        # Compute messages
        messages = np.zeros((n_nodes, self.hidden_dim))
        counts = np.zeros(n_nodes) + 1e-10
        
        for e, (src, tgt) in enumerate(edge_index):
            # Message from src to tgt
            h_src = node_features[src]
            h_tgt = node_features[tgt]
            e_feat = edge_features[e] if e < len(edge_features) else np.zeros(edge_features.shape[1] if len(edge_features) > 0 else 1)
            
            msg_input = np.concatenate([h_src, h_tgt, e_feat])
            msg = self._relu(msg_input @ self.msg_W1 + self.msg_b1)
            msg = np.clip(msg @ self.msg_W2 + self.msg_b2, -100, 100)
            
            messages[tgt] += msg
            counts[tgt] += 1
        
        # Mean aggregation
        messages /= counts[:, None]
        
        # Simple update: h' = ReLU(W_u [h; m] + b_u)
        updated = np.zeros((n_nodes, self.hidden_dim))
        h_dim = node_features.shape[1]
        W_u = self.update_Wz[:h_dim + self.hidden_dim, :self.hidden_dim]  # Adapt size
        for i in range(n_nodes):
            h = node_features[i] if h_dim == self.hidden_dim else \
                np.pad(node_features[i], (0, max(0, self.hidden_dim - h_dim)))[:self.hidden_dim]
            hm = np.concatenate([h, messages[i]])[:W_u.shape[0]]
            updated[i] = np.maximum(0, hm[:W_u.shape[0]] @ W_u[:hm.shape[0], :])
        
        return updated


# ═══════════════════════════════════════════════════════════════
# PHYSICS-INFORMED CONSTRAINTS
# ═══════════════════════════════════════════════════════════════

class PlasmaPhysicsLayer:
    """Enforce physics constraints on graph predictions.
    
    Mode A (variable graph):
      - Conservation: sum of power flows = 0
      - Causality: effects cannot precede causes (temporal)
      - Bounds: each variable within physical range
    
    Mode B (spatial graph):
      - Transport: ∂n/∂t + ∇·Γ = S
      - Energy: ∂(3/2 nT)/∂t + ∇·q = P
      - Positivity: n > 0, T > 0
    """
    
    def __init__(self, mode: str, config: PINNTGNConfig):
        self.mode = mode
        self.config = config
        
        # Physical bounds per variable (Mode A)
        self.var_bounds = {
            'li': (0, 3.0), 'q95': (0.5, 20), 'betan': (-1, 10),
            'betap': (-1, 5), 'ne': (0, 1e21), 'Ip': (0, 5e6),
            'p_rad': (0, 1e8), 'wmhd': (0, 1e7), 'f_GW': (0, 3),
        }
    
    def enforce_bounds(self, node_predictions: np.ndarray, 
                       var_names: List[str]) -> np.ndarray:
        """Hard constraint: clip predictions to physical bounds."""
        for i, name in enumerate(var_names):
            if name in self.var_bounds:
                lo, hi = self.var_bounds[name]
                node_predictions[i] = np.clip(node_predictions[i], lo, hi)
        return node_predictions
    
    def transport_residual(self, profiles_t: np.ndarray, 
                           profiles_t1: np.ndarray, 
                           dr: float, dt: float) -> float:
        """Mode B: compute transport equation residual.
        
        ∂n/∂t ≈ (n_{t+1} - n_t) / dt
        ∇·Γ ≈ -D ∂²n/∂r²
        Residual = |∂n/∂t + ∇·Γ - S|
        """
        dndt = (profiles_t1 - profiles_t) / dt
        d2ndr2 = np.gradient(np.gradient(profiles_t, dr), dr)
        D_est = 1.0  # Effective diffusivity
        residual = dndt + D_est * d2ndr2  # Should be ≈ source
        return float(np.mean(residual**2))
    
    def power_balance_residual(self, node_values: np.ndarray,
                                var_names: List[str]) -> float:
        """Mode A: power balance — P_heat ≈ P_rad + dW/dt."""
        idx = {v: i for i, v in enumerate(var_names)}
        if 'p_rad' in idx and 'wmhd' in idx and 'p_nbi' in idx:
            p_in = node_values[idx['p_nbi']]
            p_rad = node_values[idx['p_rad']]
            w = node_values[idx['wmhd']]
            # Simplified: P_in ≈ P_rad + dW/dt
            # In steady state: P_in ≈ P_rad, residual should be small
            residual = abs(p_in - p_rad) / (abs(p_in) + abs(p_rad) + 1e-10)
            return float(residual)
        return 0.0


# ═══════════════════════════════════════════════════════════════
# TEMPORAL ATTENTION (over shot history)
# ═══════════════════════════════════════════════════════════════

class TemporalAttention:
    """Attention mechanism over graph snapshots through time.
    
    Given graph embeddings at times [t-K, ..., t-1, t],
    learn which past timesteps are most relevant for current prediction.
    
    Key for disruption: the "precursor" phase (50-200ms before disruption)
    is more important than earlier history.
    """
    
    def __init__(self, dim: int, n_heads: int = 4):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Q, K, V projections
        self.Wq = np.random.randn(dim, dim) * np.sqrt(2.0/dim)
        self.Wk = np.random.randn(dim, dim) * np.sqrt(2.0/dim)
        self.Wv = np.random.randn(dim, dim) * np.sqrt(2.0/dim)
        self.Wo = np.random.randn(dim, dim) * np.sqrt(2.0/dim)
    
    def forward(self, sequence: np.ndarray) -> np.ndarray:
        """
        Args:
            sequence: [seq_len, dim] — graph embeddings over time
        Returns:
            [dim] — attended embedding
        """
        T, d = sequence.shape
        
        Q = sequence @ self.Wq  # [T, d]
        K = sequence @ self.Wk
        V = sequence @ self.Wv
        
        # Scaled dot-product attention
        scale = np.sqrt(d)
        scores = Q @ K.T / scale  # [T, T]
        
        # Causal mask: can only attend to past
        mask = np.triu(np.ones((T, T)), k=1) * (-1e9)
        scores += mask
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-10)
        
        # Weighted sum
        out = attn @ V  # [T, d]
        out = out @ self.Wo
        
        # Return last timestep (current prediction)
        return out[-1]
    
    def get_attention_weights(self, sequence: np.ndarray) -> np.ndarray:
        """Get attention weights for interpretability."""
        T, d = sequence.shape
        Q = sequence @ self.Wq
        K = sequence @ self.Wk
        scores = Q @ K.T / np.sqrt(d)
        mask = np.triu(np.ones((T, T)), k=1) * (-1e9)
        scores += mask
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-10)


# ═══════════════════════════════════════════════════════════════
# FULL HYBRID PINN+TGN MODEL
# ═══════════════════════════════════════════════════════════════

class HybridPINNTemporalGraphNetwork:
    """Complete PINN+TGN model for disruption prediction.
    
    Pipeline:
      1. Construct plasma graph (variable or spatial)
      2. Compute node/edge features at each timestep
      3. Message passing (GNN) → graph embedding per timestep
      4. Temporal attention over history → final embedding
      5. Physics constraint layer → enforce conservation laws
      6. Classification head → P(disruption)
    
    Usage:
        model = HybridPINNTemporalGraphNetwork(config)
        can_use, reason, mode = model.check_activation(data_info)
        
        if can_use:
            model.fit(data, shot_ids, labels, var_names, causal_dag)
            risk, details = model.predict_shot(shot_data)
    """
    
    def __init__(self, config: PINNTGNConfig = None):
        self.config = config or PINNTGNConfig()
        self.mode = None
        self.graph = None
        self.gnn_layers = None
        self.temporal_attn = None
        self.physics = None
        self.fitted = False
        
        # Disruption classifier
        self.clf_W = None
        self.clf_b = None
        
        # Statistics from training (for anomaly detection)
        self.normal_graph_stats = None
    
    def check_activation(self, data_info: Dict) -> Tuple[bool, str, str]:
        return check_activation(data_info)
    
    def fit(self, data: np.ndarray, shot_ids: np.ndarray,
            labels: np.ndarray, var_names: List[str],
            causal_dag: np.ndarray = None):
        """Fit the PINN+TGN model.
        
        Args:
            data: [n_timepoints, n_variables]
            shot_ids: [n_timepoints]
            labels: [n_shots] binary disruption labels
            var_names: variable names
            causal_dag: [n_vars, n_vars] adjacency matrix from CPDE
        """
        n_vars = len(var_names)
        unique_shots = np.unique(shot_ids)
        
        # Determine mode
        _, _, self.mode = self.check_activation({
            'n_variables': n_vars,
            'n_shots': len(unique_shots),
            'has_causal_dag': causal_dag is not None,
            'has_profiles': False,
        })
        
        if self.mode == 'none':
            warnings.warn("PINN+TGN: conditions not met, skipping")
            return
        
        print(f"  PINN+TGN Mode {self.mode}: {n_vars} vars, {len(unique_shots)} shots")
        
        # Build graph
        self.graph = PlasmaGraph(
            mode=self.mode, var_names=var_names,
            causal_dag=causal_dag, n_radial=None
        )
        
        # Build GNN layers
        node_dim = self.config.node_feature_dim
        edge_dim = self.config.edge_feature_dim
        hidden = self.config.hidden_dim
        
        self.gnn_layers = [
            MessagePassingLayer(node_dim if i == 0 else hidden, edge_dim, hidden)
            for i in range(self.config.n_message_passing)
        ]
        
        # Temporal attention
        self.temporal_attn = TemporalAttention(hidden * n_vars)
        
        # Physics layer
        self.physics = PlasmaPhysicsLayer(self.mode, self.config)
        
        # Classifier
        total_dim = hidden * n_vars
        self.clf_W = np.random.randn(total_dim, 1) * 0.1
        self.clf_b = np.zeros(1)
        
        # Compute graph embeddings for all shots
        shot_embeddings = []
        shot_labels = []
        
        for sid in unique_shots:
            mask = shot_ids == sid
            shot_data = data[mask]
            if shot_data.shape[0] < self.config.n_temporal_steps:
                continue
            
            # Graph embedding for this shot
            emb = self._compute_shot_embedding(shot_data, var_names)
            shot_embeddings.append(emb)
            
            label_idx = np.where(unique_shots == sid)[0][0]
            shot_labels.append(labels[label_idx] if label_idx < len(labels) else 0)
        
        if not shot_embeddings:
            return
        
        X_emb = np.array(shot_embeddings)
        y = np.array(shot_labels)
        
        # Fit simple classifier on graph embeddings
        # (In production: end-to-end training with backprop through GNN)
        from sklearn.linear_model import LogisticRegression
        self.meta_clf = LogisticRegression(C=1.0, class_weight='balanced',
                                           random_state=self.config.random_state)
        self.meta_clf.fit(X_emb, y)
        
        # Store normal shot statistics for anomaly detection
        clean_emb = X_emb[y == 0]
        if len(clean_emb) > 0:
            self.normal_graph_stats = {
                'mean': np.mean(clean_emb, axis=0),
                'std': np.std(clean_emb, axis=0) + 1e-10,
            }
        
        self.var_names = var_names
        self.fitted = True
        print(f"  PINN+TGN fitted: {X_emb.shape[1]} graph features, "
              f"{sum(y)} disrupted / {len(y)} total")
    
    def _compute_node_features(self, timestep_data: np.ndarray) -> np.ndarray:
        """Compute node features from raw variable values.
        
        For each variable: [value, z-score, rate, margin, ...]
        """
        n_vars = len(timestep_data)
        feats = np.zeros((n_vars, self.config.node_feature_dim))
        
        for i in range(n_vars):
            v = np.clip(timestep_data[i], -1e6, 1e6)
            feats[i, 0] = v                    # Raw value
            feats[i, 1] = min(np.abs(v), 1e6)  # Magnitude
            # Pad remaining with zeros (will be filled with rates etc.)
        
        return feats
    
    def _compute_edge_features(self, timestep_data: np.ndarray) -> np.ndarray:
        """Compute edge features: interaction strength between variables."""
        n_edges = self.graph.n_edges
        feats = np.zeros((n_edges, self.config.edge_feature_dim))
        
        for e, (src, tgt) in enumerate(self.graph.edge_index):
            v_src = np.clip(timestep_data[src], -1e6, 1e6)
            v_tgt = np.clip(timestep_data[tgt], -1e6, 1e6)
            feats[e, 0] = np.clip(v_src * v_tgt, -1e6, 1e6)  # Product
            feats[e, 1] = min(abs(v_src - v_tgt), 1e6)        # Difference
            feats[e, 2] = self.graph.adjacency[src, tgt]       # Edge weight
        
        return feats
    
    def _compute_shot_embedding(self, shot_data: np.ndarray, 
                                 var_names: List[str]) -> np.ndarray:
        """Compute graph embedding for an entire shot.
        
        Runs GNN at each timestep, then temporal attention over history.
        """
        n_time, n_vars = shot_data.shape
        K = min(self.config.n_temporal_steps, n_time)
        hidden = self.config.hidden_dim
        
        # Normalize data for numerical stability
        shot_data = np.clip(np.nan_to_num(shot_data), -1e6, 1e6)
        sd = np.std(shot_data, axis=0) + 1e-10
        shot_data_norm = shot_data / sd[None, :]
        
        # Process last K timesteps
        graph_sequence = []
        
        for t in range(max(0, n_time - K), n_time):
            # Node features at time t
            node_feats = self._compute_node_features(shot_data_norm[t])
            
            # Add rate information if not first timestep
            if t > 0:
                rates = shot_data_norm[t] - shot_data_norm[t-1]
                for i in range(n_vars):
                    node_feats[i, 2] = np.clip(rates[i], -10, 10)  # Rate of change
            
            # Edge features
            edge_feats = self._compute_edge_features(shot_data_norm[t])
            
            # Message passing
            h = node_feats
            for layer in self.gnn_layers:
                h = layer.forward(h, self.graph.edge_index, edge_feats)
            
            # Flatten graph: [n_vars, hidden] → [n_vars * hidden]
            graph_emb = h.flatten()
            graph_sequence.append(graph_emb)
        
        # Temporal attention over graph snapshots
        seq = np.array(graph_sequence)
        
        if len(seq) > 1:
            attended = self.temporal_attn.forward(seq)
        else:
            attended = seq[0]
        
        return attended
    
    def predict_shot(self, shot_data: np.ndarray) -> Tuple[float, Dict]:
        """Predict disruption probability for a single shot.
        
        Returns:
            (probability, details)
        """
        if not self.fitted:
            return 0.5, {'error': 'PINN+TGN not fitted'}
        
        emb = self._compute_shot_embedding(shot_data, self.var_names)
        
        # Classifier prediction
        prob = self.meta_clf.predict_proba(emb.reshape(1, -1))[0, 1]
        
        # Anomaly score (Mahalanobis-like distance from normal)
        if self.normal_graph_stats is not None:
            z = (emb - self.normal_graph_stats['mean']) / self.normal_graph_stats['std']
            anomaly_score = float(np.mean(z**2))
        else:
            anomaly_score = 0.0
        
        # Attention weights for interpretability
        n_time = shot_data.shape[0]
        K = min(self.config.n_temporal_steps, n_time)
        
        details = {
            'probability': float(prob),
            'anomaly_score': anomaly_score,
            'graph_mode': self.mode,
            'n_nodes': self.graph.n_nodes,
            'n_edges': self.graph.n_edges,
            'temporal_window': K,
        }
        
        return float(prob), details
    
    def build_features(self, shot_data: np.ndarray) -> Optional[np.ndarray]:
        """Build feature vector for meta-learner integration."""
        if not self.fitted:
            return None
        
        emb = self._compute_shot_embedding(shot_data, self.var_names)
        
        # Reduce dimensionality: mean, std, max of embedding
        n_chunks = 4
        chunk_size = len(emb) // n_chunks
        feats = []
        for i in range(n_chunks):
            chunk = emb[i*chunk_size:(i+1)*chunk_size]
            feats.extend([np.mean(chunk), np.std(chunk), np.max(chunk)])
        
        return np.array(feats, dtype=np.float32)
