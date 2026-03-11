#!/usr/bin/env python3
"""
Self-Supervised Pretraining for Plasma Time Series
====================================================

Pre-train a representation model on UNLABELED plasma data using:
1. Masked Signal Prediction (MSP): mask 15% of signals, predict them
2. Contrastive Temporal Learning (CTL): same shot = positive, different = negative
3. Next-State Prediction (NSP): predict next timestep from current window

Then fine-tune the learned representations on the small labeled disruption set.

ACTIVATION CONDITIONS:
  - ≥ 1,000,000 unlabeled timepoints (for meaningful pretraining)
  - ≥ 8 signal channels
  - ≥ 500 unique shots

When conditions are NOT met, returns None.

Why this helps: With only 83 disrupted shots, supervised learning hits a ceiling.
Self-supervised pretraining learns GENERAL plasma dynamics from millions of
unlabeled timepoints, then transfers that knowledge to disruption detection.

Analogy: BERT learns language from unlabeled text, then fine-tunes on 
labeled sentiment. We learn plasma physics from unlabeled shots, then
fine-tune on labeled disruptions.

Estimated gain: +5-15% AUC when ≥10M timepoints available.
Current data (268K tp): marginal gain, not worth the complexity.

Author: Dr. Mladen Mester
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class SSPTConfig:
    """Self-Supervised Pre-Training configuration."""
    # Activation thresholds
    min_timepoints: int = 1_000_000
    min_channels: int = 8
    min_shots: int = 500
    
    # Architecture
    embedding_dim: int = 64
    n_encoder_layers: int = 3
    window_size: int = 32         # Input window length
    mask_ratio: float = 0.15      # Fraction of signals to mask (MSP)
    temperature: float = 0.07     # Contrastive loss temperature
    
    # Pretraining
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-3
    pretrain_batch_size: int = 256
    
    # Fine-tuning
    finetune_epochs: int = 20
    finetune_lr: float = 1e-4
    
    random_state: int = 42


def check_activation(data_info: Dict) -> Tuple[bool, str]:
    """Check if self-supervised pretraining should activate.
    
    Args:
        data_info: {
            'n_timepoints': int,
            'n_channels': int,
            'n_shots': int,
        }
    """
    config = SSPTConfig()
    
    ntp = data_info.get('n_timepoints', 0)
    if ntp < config.min_timepoints:
        return False, (f"Need ≥{config.min_timepoints:,} timepoints for pretraining, "
                      f"have {ntp:,}. Gain would be marginal.")
    
    nch = data_info.get('n_channels', 0)
    if nch < config.min_channels:
        return False, f"Need ≥{config.min_channels} channels, have {nch}"
    
    ns = data_info.get('n_shots', 0)
    if ns < config.min_shots:
        return False, f"Need ≥{config.min_shots} shots, have {ns}"
    
    return True, f"SSPT ready: {ntp:,} tp × {nch} ch × {ns} shots"


# ═══════════════════════════════════════════════════════════════
# ENCODER ARCHITECTURE (NumPy — portable, no framework dependency)
# ═══════════════════════════════════════════════════════════════

class TemporalEncoder:
    """Lightweight temporal encoder for plasma time series.
    
    Architecture: Conv1D stack → Global Average Pool → Embedding
    
    In production, replace with:
    - Transformer encoder (for longer sequences)
    - Temporal Convolutional Network (for real-time)
    - Mamba/S4 (for very long sequences)
    """
    
    def __init__(self, n_channels: int, embedding_dim: int, n_layers: int = 3):
        self.n_channels = n_channels
        self.emb_dim = embedding_dim
        
        # Conv1D layers: channels → embedding_dim
        dims = [n_channels] + [embedding_dim] * n_layers
        self.conv_weights = []
        self.conv_biases = []
        
        for i in range(n_layers):
            # Kernel size 3 conv
            W = np.random.randn(dims[i+1], dims[i], 3) * np.sqrt(2.0 / (dims[i] * 3))
            b = np.zeros(dims[i+1])
            self.conv_weights.append(W)
            self.conv_biases.append(b)
        
        # Final projection
        self.proj_W = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2.0/embedding_dim)
        self.proj_b = np.zeros(embedding_dim)
    
    def _conv1d(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Simple 1D convolution: x [batch, ch_in, time], W [ch_out, ch_in, kernel]."""
        batch, ch_in, T = x.shape
        ch_out, _, K = W.shape
        pad = K // 2
        x_pad = np.pad(x, ((0,0), (0,0), (pad, pad)), mode='constant')
        out = np.zeros((batch, ch_out, T))
        for k in range(K):
            out += np.einsum('oi,bit->bot', W[:, :, k], x_pad[:, :, k:k+T])
        return out + b[None, :, None]
    
    def _gelu(self, x):
        return x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))) / 2
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode time series window to embedding.
        
        Args:
            x: [batch, n_channels, window_size]
        Returns:
            [batch, embedding_dim]
        """
        h = x
        for W, b in zip(self.conv_weights, self.conv_biases):
            h = self._gelu(self._conv1d(h, W, b))
        
        # Global average pooling over time
        h = np.mean(h, axis=-1)  # [batch, embedding_dim]
        
        # Final projection
        h = h @ self.proj_W + self.proj_b
        
        return h
    
    def get_params(self) -> List[np.ndarray]:
        params = []
        for W, b in zip(self.conv_weights, self.conv_biases):
            params.extend([W, b])
        params.extend([self.proj_W, self.proj_b])
        return params


# ═══════════════════════════════════════════════════════════════
# PRETEXT TASKS
# ═══════════════════════════════════════════════════════════════

class MaskedSignalPrediction:
    """Task 1: Mask random signals, predict their values.
    
    Like BERT's masked language modeling, but for plasma signals.
    Forces the encoder to learn cross-signal relationships.
    """
    
    def __init__(self, n_channels: int, embedding_dim: int, mask_ratio: float = 0.15):
        self.mask_ratio = mask_ratio
        self.predictor_W = np.random.randn(embedding_dim, n_channels) * 0.1
        self.predictor_b = np.zeros(n_channels)
    
    def create_masked_input(self, x: np.ndarray, rng: np.random.RandomState):
        """Mask random channels and return masked input + target.
        
        Args:
            x: [batch, n_channels, window_size]
        Returns:
            x_masked, mask, targets
        """
        batch, ch, T = x.shape
        mask = rng.random((batch, ch, 1)) < self.mask_ratio
        mask = np.broadcast_to(mask, x.shape)
        
        x_masked = x.copy()
        x_masked[mask] = 0  # Zero out masked channels
        
        return x_masked, mask, x
    
    def compute_loss(self, embeddings: np.ndarray, targets: np.ndarray, 
                     mask: np.ndarray) -> float:
        """Predict masked values from embeddings."""
        # Predict all channels from embedding
        pred = embeddings @ self.predictor_W + self.predictor_b  # [batch, n_channels]
        
        # Average target over time for each channel
        target_avg = np.mean(targets, axis=-1)  # [batch, n_channels]
        
        # Loss only on masked channels
        channel_mask = mask[:, :, 0]  # [batch, n_channels]
        if channel_mask.sum() == 0:
            return 0.0
        
        errors = (pred - target_avg)**2 * channel_mask
        return float(np.sum(errors) / (channel_mask.sum() + 1e-10))


class ContrastiveTemporalLearning:
    """Task 2: Learn that windows from same shot are similar.
    
    Positive pairs: two windows from the same shot
    Negative pairs: windows from different shots
    
    InfoNCE loss: encourages same-shot embeddings to be close.
    """
    
    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature
    
    def compute_loss(self, embeddings: np.ndarray, shot_ids: np.ndarray) -> float:
        """Contrastive loss on batch of embeddings.
        
        Args:
            embeddings: [batch, emb_dim] L2-normalized
            shot_ids: [batch] which shot each window came from
        """
        batch = len(embeddings)
        if batch < 4:
            return 0.0
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        emb_norm = embeddings / norms
        
        # Similarity matrix
        sim = emb_norm @ emb_norm.T / self.temperature  # [batch, batch]
        
        # Positive mask: same shot
        pos_mask = shot_ids[:, None] == shot_ids[None, :]
        np.fill_diagonal(pos_mask, False)  # Don't count self
        
        if pos_mask.sum() == 0:
            return 0.0
        
        # InfoNCE: for each anchor, maximize similarity to positives vs negatives
        # Softmax over columns for each row
        exp_sim = np.exp(sim - np.max(sim, axis=1, keepdims=True))
        log_softmax = np.log(exp_sim / (exp_sim.sum(axis=1, keepdims=True) + 1e-10))
        
        # Average log-prob of positive pairs
        loss = -np.sum(log_softmax * pos_mask) / (pos_mask.sum() + 1e-10)
        return float(loss)


class NextStatePrediction:
    """Task 3: Predict the next timestep from current window.
    
    This teaches the model about plasma DYNAMICS —
    crucial for understanding how disruptions develop.
    """
    
    def __init__(self, embedding_dim: int, n_channels: int):
        self.pred_W = np.random.randn(embedding_dim, n_channels) * 0.1
        self.pred_b = np.zeros(n_channels)
    
    def compute_loss(self, embeddings: np.ndarray, 
                     next_values: np.ndarray) -> float:
        """Predict next timestep values from embedding.
        
        Args:
            embeddings: [batch, emb_dim]
            next_values: [batch, n_channels] — values at t+1
        """
        pred = embeddings @ self.pred_W + self.pred_b
        return float(np.mean((pred - next_values)**2))


# ═══════════════════════════════════════════════════════════════
# FULL SELF-SUPERVISED PRETRAINER
# ═══════════════════════════════════════════════════════════════

class SelfSupervisedPretrainer:
    """Self-supervised pretraining for plasma time series.
    
    Usage:
        sspt = SelfSupervisedPretrainer(config)
        
        can_use, reason = sspt.check_activation(data_info)
        if not can_use:
            print(f"SSPT skipped: {reason}")
            return
        
        # Pretrain on all unlabeled data
        sspt.pretrain(all_data, all_shot_ids)
        
        # Extract features for disruption prediction
        features = sspt.extract_features(shot_data)
    """
    
    def __init__(self, config: SSPTConfig = None):
        self.config = config or SSPTConfig()
        self.encoder = None
        self.pretrained = False
        self.finetuned = False
    
    def check_activation(self, data_info: Dict) -> Tuple[bool, str]:
        return check_activation(data_info)
    
    def pretrain(self, data: np.ndarray, shot_ids: np.ndarray):
        """Pretrain encoder on unlabeled time series.
        
        Args:
            data: [n_timepoints, n_channels]
            shot_ids: [n_timepoints] — which shot each timepoint belongs to
        """
        n_tp, n_ch = data.shape
        config = self.config
        ws = config.window_size
        
        print(f"  SSPT pretraining: {n_tp:,} timepoints, {n_ch} channels")
        
        # Build encoder
        self.encoder = TemporalEncoder(n_ch, config.embedding_dim, config.n_encoder_layers)
        
        # Build pretext task heads
        self.msp = MaskedSignalPrediction(n_ch, config.embedding_dim, config.mask_ratio)
        self.ctl = ContrastiveTemporalLearning(config.temperature)
        self.nsp = NextStatePrediction(config.embedding_dim, n_ch)
        
        rng = np.random.RandomState(config.random_state)
        unique_shots = np.unique(shot_ids)
        
        # Pretraining loop
        for epoch in range(min(config.pretrain_epochs, 10)):  # Cap for PoC
            # Sample random windows
            batch_windows = []
            batch_shot_ids = []
            batch_next_values = []
            
            for _ in range(config.pretrain_batch_size):
                # Random shot, random start position
                sid = rng.choice(unique_shots)
                mask = shot_ids == sid
                indices = np.where(mask)[0]
                if len(indices) < ws + 1:
                    continue
                start = rng.randint(0, len(indices) - ws - 1)
                
                window = data[indices[start:start+ws]].T  # [n_ch, ws]
                batch_windows.append(window)
                batch_shot_ids.append(sid)
                batch_next_values.append(data[indices[start+ws]])
            
            if len(batch_windows) < 4:
                continue
            
            X = np.array(batch_windows)        # [batch, n_ch, ws]
            sids = np.array(batch_shot_ids)
            next_vals = np.array(batch_next_values)  # [batch, n_ch]
            
            # Task 1: Masked Signal Prediction
            X_masked, mask_arr, targets = self.msp.create_masked_input(X, rng)
            embeddings = self.encoder.encode(X_masked)
            loss_msp = self.msp.compute_loss(embeddings, targets, mask_arr)
            
            # Task 2: Contrastive (on unmasked)
            embeddings_clean = self.encoder.encode(X)
            loss_ctl = self.ctl.compute_loss(embeddings_clean, sids)
            
            # Task 3: Next State Prediction
            loss_nsp = self.nsp.compute_loss(embeddings_clean, next_vals)
            
            total_loss = loss_msp + 0.5 * loss_ctl + loss_nsp
            
            if epoch % 2 == 0:
                print(f"    Epoch {epoch}: MSP={loss_msp:.4f} CTL={loss_ctl:.4f} "
                      f"NSP={loss_nsp:.4f} Total={total_loss:.4f}")
            
            # NOTE: In production, compute gradients and update weights here.
            # This PoC demonstrates the architecture and loss computation.
            # For actual training, wrap in PyTorch/JAX with autograd.
        
        self.pretrained = True
        print(f"  SSPT pretraining complete")
    
    def extract_features(self, shot_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract pretrained features for one shot.
        
        Args:
            shot_data: [n_timepoints, n_channels] — one shot
        Returns:
            [feature_dim] aggregated features, or None if not pretrained
        """
        if not self.pretrained or self.encoder is None:
            return None
        
        ws = self.config.window_size
        if shot_data.shape[0] < ws:
            return None
        
        # Extract embeddings for overlapping windows
        embeddings = []
        for start in range(0, shot_data.shape[0] - ws, ws // 2):
            window = shot_data[start:start+ws].T[None, :, :]  # [1, ch, ws]
            emb = self.encoder.encode(window)[0]  # [emb_dim]
            embeddings.append(emb)
        
        if not embeddings:
            return None
        
        emb_array = np.array(embeddings)  # [n_windows, emb_dim]
        n30 = max(int(0.3 * len(emb_array)), 1)
        
        # Aggregate: mean, std, late mean, trend
        feats = np.concatenate([
            np.mean(emb_array, axis=0),                           # Global mean
            np.std(emb_array, axis=0),                            # Global std
            np.mean(emb_array[-n30:], axis=0),                    # Late mean
            np.mean(emb_array[-n30:], axis=0) - np.mean(emb_array[:n30], axis=0),  # Trend
        ])
        
        return feats.astype(np.float32)
    
    def build_features_batch(self, data: np.ndarray, shot_ids: np.ndarray,
                              unique_shots: np.ndarray) -> Optional[np.ndarray]:
        """Extract features for all shots. Returns [n_shots, feature_dim] or None."""
        if not self.pretrained:
            return None
        
        all_feats = []
        for sid in unique_shots:
            mask = shot_ids == sid
            shot_data = data[mask]
            feats = self.extract_features(shot_data)
            if feats is None:
                feats = np.zeros(self.config.embedding_dim * 4)
            all_feats.append(feats)
        
        return np.array(all_feats, dtype=np.float32)
