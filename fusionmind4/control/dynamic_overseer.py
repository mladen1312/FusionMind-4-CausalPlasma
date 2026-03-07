"""
Dynamic Overseer — Real-Time Track Selection and Correction
============================================================

Inspired by radar data fusion (Mimosa pattern):
- 4 parallel tracks run every timestep
- Overseer selects best track OR blends based on disagreement
- Stateful memory smooths transitions (last N timesteps)
- Physics tracks (C, D) get priority when disagreement is high

Tracks:
  A: Correlational ML (GBT on all features)
  B: Causal SCM (only DAG parents)  
  C: Physics thresholds (Troyon, q-margin, li-margin)
  D: Fast diagnostics anomaly (MHD n=2, Dα, li_rate, βp_rate)

The Overseer does NOT predict disruptions itself.
It decides WHICH track to trust at each moment.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class TrackOutput:
    """Output from a single track at one timestep."""
    name: str
    prob: float         # Disruption probability [0, 1]
    confidence: float   # Track self-confidence [0, 1]
    explanation: str = ""  # Why this track thinks what it thinks


@dataclass
class OverseerDecision:
    """Overseer output at one timestep."""
    final_prob: float
    best_track: str
    disagreement: float
    correction_applied: bool
    explanation: str
    track_weights: Dict[str, float]
    warning_level: int  # 0=safe, 1=watch, 2=warning, 3=alarm


class DynamicOverseer:
    """
    Mimosa-style multi-track arbitrator for plasma disruption prediction.
    
    Key design: physics tracks (C, D) are trusted MORE when tracks disagree,
    because correlational models (A) exploit spurious patterns while physics
    tracks have guaranteed safety bounds.
    """
    
    def __init__(self, 
                 history_len: int = 10,
                 disagreement_threshold: float = 0.20,
                 alarm_threshold: float = 0.5,
                 warning_threshold: float = 0.3,
                 physics_priority: float = 1.5):
        """
        Args:
            history_len: Number of past decisions to remember
            disagreement_threshold: When std(probs) exceeds this, switch to physics
            alarm_threshold: Above this → ALARM
            warning_threshold: Above this → WARNING
            physics_priority: Confidence multiplier for physics tracks (C, D)
        """
        self.history = deque(maxlen=history_len)
        self.disagreement_threshold = disagreement_threshold
        self.alarm_threshold = alarm_threshold
        self.warning_threshold = warning_threshold
        self.physics_priority = physics_priority
        self._track_performance = {'A': 0.5, 'B': 0.5, 'C': 0.5, 'D': 0.5}
        
    def decide(self, tracks: Dict[str, TrackOutput]) -> OverseerDecision:
        """
        Main decision function — called every timestep (~10ms).
        
        Returns OverseerDecision with final probability and explanation.
        """
        probs = {name: t.prob for name, t in tracks.items()}
        confs = {name: t.confidence for name, t in tracks.items()}
        
        # 1. Compute disagreement
        prob_values = np.array(list(probs.values()))
        disagreement = float(np.std(prob_values))
        
        # 2. Compute weighted scores (confidence × physics priority)
        scores = {}
        for name in tracks:
            base = confs[name]
            if name in ('C', 'D'):
                base *= self.physics_priority
            # Bonus for consistency with recent history
            if self.history:
                recent_track = self.history[-1]['best_track']
                if name == recent_track:
                    base *= 1.1  # Stability bonus
            scores[name] = base
        
        # 3. Decision logic
        if disagreement > self.disagreement_threshold:
            # HIGH DISAGREEMENT: tracks don't agree → trust physics
            # Weighted blend of C and D (physics + fast diagnostics)
            w_c = confs.get('C', 0.5)
            w_d = confs.get('D', 0.5)
            total_w = w_c + w_d + 1e-10
            
            corrected = (w_c * probs.get('C', 0) + w_d * probs.get('D', 0)) / total_w
            best_track = 'D' if probs.get('D', 0) > probs.get('C', 0) else 'C'
            correction_applied = True
            explanation = (f"High disagreement ({disagreement:.2f}>{self.disagreement_threshold}). "
                          f"Trusting physics tracks: C={probs.get('C',0):.2f}, D={probs.get('D',0):.2f}")
        else:
            # LOW DISAGREEMENT: tracks agree → use best track with smoothing
            best_track = max(scores, key=scores.get)
            
            # Smooth with history (Mimosa-style exponential moving average)
            if self.history:
                prev = self.history[-1]['prob']
                alpha = 0.7  # Current weight
                corrected = alpha * probs[best_track] + (1 - alpha) * prev
            else:
                corrected = probs[best_track]
            
            correction_applied = False
            explanation = f"Low disagreement ({disagreement:.2f}). Best track: {best_track} (conf={confs[best_track]:.2f})"
        
        # 4. Safety override: if ANY track says ALARM, take it seriously
        max_prob = max(prob_values)
        if max_prob > self.alarm_threshold and corrected < self.alarm_threshold * 0.8:
            # Some track sees danger that the blend is washing out
            alarm_track = max(probs, key=probs.get)
            corrected = max(corrected, max_prob * 0.7)
            explanation += f" | Safety override: {alarm_track} sees {max_prob:.2f}"
        
        # 5. Warning level
        if corrected > self.alarm_threshold:
            warning_level = 3  # ALARM
        elif corrected > self.warning_threshold:
            warning_level = 2  # WARNING
        elif corrected > 0.15:
            warning_level = 1  # WATCH
        else:
            warning_level = 0  # SAFE
        
        # 6. Compute track weights for transparency
        total_score = sum(scores.values()) + 1e-10
        track_weights = {name: round(s / total_score, 3) for name, s in scores.items()}
        
        # 7. Update history
        self.history.append({
            'prob': corrected,
            'best_track': best_track,
            'disagreement': disagreement,
            'warning_level': warning_level,
        })
        
        # 8. Update track performance (for adaptive weighting)
        # If we later learn the outcome, we can call update_performance()
        
        return OverseerDecision(
            final_prob=float(np.clip(corrected, 0, 1)),
            best_track=best_track,
            disagreement=disagreement,
            correction_applied=correction_applied,
            explanation=explanation,
            track_weights=track_weights,
            warning_level=warning_level,
        )
    
    def update_performance(self, track_name: str, was_correct: bool):
        """Update track reliability based on outcome (for online learning)."""
        alpha = 0.1
        current = self._track_performance.get(track_name, 0.5)
        self._track_performance[track_name] = current + alpha * ((1.0 if was_correct else 0.0) - current)
    
    def reset(self):
        """Reset state for new shot."""
        self.history.clear()
    
    def get_state(self) -> dict:
        """Return current overseer state for logging/debugging."""
        return {
            'history_len': len(self.history),
            'last_decision': dict(self.history[-1]) if self.history else None,
            'track_performance': dict(self._track_performance),
        }
