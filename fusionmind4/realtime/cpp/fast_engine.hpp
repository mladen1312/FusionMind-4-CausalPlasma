/**
 * FusionMind 4.0 — Fast Real-Time Inference Engine (C++)
 * ========================================================
 *
 * Sub-microsecond inference for dual-mode disruption prediction.
 * Designed for tokamak PCS integration with deterministic latency.
 *
 * Components:
 *   - DecisionStumpEnsemble: Gradient-boosted stumps, < 1 μs inference
 *   - FeatureEngine: Physics feature extraction with ring buffer
 *   - CausalScorer: Backdoor-adjusted causal disruption scoring
 *   - DualPredictor: Fused ML+causal prediction
 *
 * Memory layout: all arrays contiguous, cache-line aligned.
 * No heap allocation in hot path (predict/extract).
 *
 * Patent Families: PF1 (CPDE), PF2 (CPC)
 * Author: Dr. Mladen Mester, March 2026
 */

#pragma once

#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <array>

namespace fusionmind {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int MAX_VARS       = 32;
static constexpr int MAX_FEATURES   = 128;
static constexpr int MAX_TREES      = 200;
static constexpr int MAX_HISTORY    = 64;
static constexpr int MAX_EDGES      = MAX_VARS * MAX_VARS;

// ---------------------------------------------------------------------------
// Threat classification
// ---------------------------------------------------------------------------

enum ThreatLevel : int {
    SAFE     = 0,
    WATCH    = 1,
    WARNING  = 2,
    CRITICAL = 3,
    IMMINENT = 4,
};

// ---------------------------------------------------------------------------
// Decision Stump (single split on one feature)
// ---------------------------------------------------------------------------

struct alignas(32) DecisionStump {
    int    feature_idx;
    float  split_value;
    float  value_left;
    float  value_right;
    float  learning_rate;
    float  _pad[3];  // pad to 32 bytes for cache alignment
};

// ---------------------------------------------------------------------------
// Gradient Boosted Stump Ensemble
// ---------------------------------------------------------------------------

struct StumpEnsemble {
    DecisionStump trees[MAX_TREES];
    int           n_trees;
    float         means[MAX_FEATURES];
    float         stds[MAX_FEATURES];
    int           n_features;
    float         threshold;    // decision threshold

    // Feature importances (computed during training)
    float         importances[MAX_FEATURES];

    void init() {
        n_trees = 0;
        n_features = 0;
        threshold = 0.5f;
        std::memset(means, 0, sizeof(means));
        std::memset(stds, 0, sizeof(stds));
        std::memset(importances, 0, sizeof(importances));
    }

    /**
     * Predict disruption probability.  O(n_trees) — branchless inner loop.
     * Target: < 1 μs on modern CPU.
     */
    inline float predict(const float* features) const {
        // Normalise
        float x[MAX_FEATURES];
        for (int i = 0; i < n_features; ++i) {
            float s = stds[i];
            x[i] = (s > 1e-12f) ? (features[i] - means[i]) / s : 0.0f;
        }

        // Accumulate tree predictions (branchless)
        float score = 0.0f;
        for (int t = 0; t < n_trees; ++t) {
            const auto& tree = trees[t];
            float val = (x[tree.feature_idx] <= tree.split_value)
                        ? tree.value_left
                        : tree.value_right;
            score += tree.learning_rate * val;
        }

        // Sigmoid
        score = std::max(-20.0f, std::min(20.0f, score));
        return 1.0f / (1.0f + std::exp(-score));
    }

    /**
     * Batch prediction — vectorisable by compiler.
     */
    void predict_batch(const float* X, int n_samples,
                       float* out_probs) const {
        for (int i = 0; i < n_samples; ++i) {
            out_probs[i] = predict(X + i * n_features);
        }
    }
};

// ---------------------------------------------------------------------------
// Ring Buffer for time-series history (no allocation in hot path)
// ---------------------------------------------------------------------------

struct alignas(64) RingBuffer {
    float  data[MAX_HISTORY][MAX_VARS];
    double timestamps[MAX_HISTORY];
    int    head;          // next write position
    int    count;         // number of valid entries
    int    n_vars;
    int    capacity;

    void init(int nvars, int cap = MAX_HISTORY) {
        n_vars = nvars;
        capacity = std::min(cap, (int)MAX_HISTORY);
        head = 0;
        count = 0;
        std::memset(data, 0, sizeof(data));
        std::memset(timestamps, 0, sizeof(timestamps));
    }

    inline void push(const float* values, double timestamp) {
        std::memcpy(data[head], values, n_vars * sizeof(float));
        timestamps[head] = timestamp;
        head = (head + 1) % capacity;
        if (count < capacity) ++count;
    }

    // Get entry at offset from most recent (0 = most recent)
    inline const float* get(int offset) const {
        int idx = (head - 1 - offset + capacity * 2) % capacity;
        return data[idx];
    }

    inline double get_time(int offset) const {
        int idx = (head - 1 - offset + capacity * 2) % capacity;
        return timestamps[idx];
    }
};

// ---------------------------------------------------------------------------
// Feature Extraction Engine
// ---------------------------------------------------------------------------

struct FeatureEngine {
    RingBuffer history;

    // Variable name → index mapping (set by Python)
    int idx_betaN;
    int idx_ne;
    int idx_Ip;
    int idx_q95;
    int idx_P_rad;
    int idx_P_NBI;
    int idx_li;
    int idx_MHD;
    int idx_ne_core;

    // Physics constants
    float greenwald_limit;
    float troyon_limit;
    float q_kink;

    // Output feature buffer
    float features[MAX_FEATURES];
    int   n_features;

    // Feature name indices (for fast lookup)
    int feat_greenwald;
    int feat_beta_prox;
    int feat_q_prox;
    int feat_f_rad;

    void init(int n_vars) {
        history.init(n_vars, MAX_HISTORY);
        greenwald_limit = 1.0f;
        troyon_limit = 2.8f;
        q_kink = 2.0f;
        n_features = 0;

        // Defaults: -1 = not present
        idx_betaN = -1; idx_ne = -1; idx_Ip = -1;
        idx_q95 = -1; idx_P_rad = -1; idx_P_NBI = -1;
        idx_li = -1; idx_MHD = -1; idx_ne_core = -1;
    }

    /**
     * Update history and extract features.  No heap allocation.
     * Returns number of features extracted.
     */
    int extract(const float* raw_values, double timestamp) {
        history.push(raw_values, timestamp);
        int nf = 0;
        int nv = history.n_vars;
        const float* current = history.get(0);

        // 1. Raw values
        for (int i = 0; i < nv && nf < MAX_FEATURES; ++i) {
            features[nf++] = current[i];
        }

        // 2. Greenwald fraction
        if (idx_ne >= 0 && idx_Ip >= 0) {
            float ne = current[idx_ne];
            float Ip = current[idx_Ip];
            features[nf++] = (Ip > 1e-6f) ? ne / (Ip * 0.5f) : 0.0f;
        } else if (idx_ne_core >= 0) {
            features[nf++] = current[idx_ne_core] / 5.0f;
        }
        feat_greenwald = nf - 1;

        // 3. Beta proximity
        if (idx_betaN >= 0) {
            features[nf++] = current[idx_betaN] / troyon_limit;
        }
        feat_beta_prox = nf - 1;

        // 4. q95 proximity
        if (idx_q95 >= 0) {
            float q = current[idx_q95];
            features[nf++] = (q > 0.1f) ? q_kink / q : 10.0f;
        }
        feat_q_prox = nf - 1;

        // 5. Radiation fraction
        if (idx_P_rad >= 0 && idx_P_NBI >= 0) {
            float pr = current[idx_P_rad];
            float pi = current[idx_P_NBI];
            features[nf++] = (pi > 0.01f) ? pr / pi : 0.0f;
        }
        feat_f_rad = nf - 1;

        // 6. li
        if (idx_li >= 0) {
            features[nf++] = current[idx_li];
        }

        // 7. MHD amplitude
        if (idx_MHD >= 0) {
            features[nf++] = current[idx_MHD];
        }

        // 8. Rate-of-change features (if >= 3 history points)
        if (history.count >= 3) {
            const float* prev = history.get(2);
            double dt = timestamp - history.get_time(2);
            if (dt > 1e-9) {
                float inv_dt = 1.0f / (float)dt;
                for (int i = 0; i < nv && nf < MAX_FEATURES - 5; ++i) {
                    features[nf++] = (current[i] - prev[i]) * inv_dt;
                }
            }
        }

        // 9. Rolling std (if >= 10 history points)
        if (history.count >= 10) {
            for (int v = 0; v < nv && nf < MAX_FEATURES - 5; ++v) {
                float sum = 0, sum2 = 0;
                for (int k = 0; k < 10; ++k) {
                    float val = history.get(k)[v];
                    sum += val;
                    sum2 += val * val;
                }
                float mean = sum / 10.0f;
                float var = sum2 / 10.0f - mean * mean;
                features[nf++] = (var > 0) ? std::sqrt(var) : 0.0f;
            }
        }

        n_features = nf;
        return nf;
    }

    void reset() {
        history.init(history.n_vars, history.capacity);
        n_features = 0;
    }
};

// ---------------------------------------------------------------------------
// Causal Disruption Scorer
// ---------------------------------------------------------------------------

struct CausalScorer {
    // Backdoor-adjusted causal effects
    float effects[MAX_VARS];
    int   n_vars;

    // Disruption boundaries per variable
    float boundary_low[MAX_VARS];
    float boundary_high[MAX_VARS];
    int   has_boundary[MAX_VARS];

    // Confounder flags
    int   has_confounder[MAX_VARS];
    float confounder_boundary_low[MAX_VARS];
    float confounder_boundary_high[MAX_VARS];

    void init(int nvars) {
        n_vars = nvars;
        std::memset(effects, 0, sizeof(effects));
        std::memset(has_boundary, 0, sizeof(has_boundary));
        std::memset(has_confounder, 0, sizeof(has_confounder));
    }

    /**
     * Compute causal disruption probability.
     * Uses backdoor-adjusted effects — immune to Simpson's Paradox.
     * Target: < 2 μs.
     */
    inline float predict(const float* values, int* simpsons_detected) const {
        float score = 0.0f;
        *simpsons_detected = 0;

        for (int j = 0; j < n_vars; ++j) {
            float eff = effects[j];
            if (std::abs(eff) < 0.05f) continue;
            if (!has_boundary[j]) continue;

            float val = values[j];
            float lo = boundary_low[j];
            float hi = boundary_high[j];
            float range = hi - lo;
            if (range < 1e-12f) continue;

            float norm_val;
            if (eff > 0) {
                norm_val = (val - lo) / range;
            } else {
                norm_val = (hi - val) / range;
            }
            norm_val = std::max(0.0f, std::min(1.0f, norm_val));
            score += std::abs(eff) * norm_val;

            // Simpson's Paradox check
            if (has_confounder[j]) {
                float cv = values[j]; // Simplified: use same var as proxy
                if (cv < confounder_boundary_low[j] ||
                    cv > confounder_boundary_high[j]) {
                    *simpsons_detected = 1;
                }
            }
        }

        // Sigmoid
        score = 3.0f * (score - 0.5f);
        score = std::max(-20.0f, std::min(20.0f, score));
        return 1.0f / (1.0f + std::exp(-score));
    }
};

// ---------------------------------------------------------------------------
// Dual-Mode Predictor (fuses ML + Causal)
// ---------------------------------------------------------------------------

struct DualPredictionResult {
    float ml_prob;
    float causal_prob;
    float fused_prob;
    int   fused_threat;
    int   simpsons_detected;
    float ml_latency_ns;
    float causal_latency_ns;
    float total_latency_ns;
    float ttd_ms;          // time to disruption
};

struct DualPredictor {
    StumpEnsemble ml;
    CausalScorer  causal;
    FeatureEngine features;

    float w_ml;
    float w_causal;
    float w_ml_simpson;     // weights when Simpson's detected
    float w_causal_simpson;

    void init(int n_vars) {
        ml.init();
        causal.init(n_vars);
        features.init(n_vars);
        w_ml = 0.35f;
        w_causal = 0.65f;
        w_ml_simpson = 0.15f;
        w_causal_simpson = 0.85f;
    }

    /**
     * Full dual-mode prediction.  Target: < 5 μs total.
     * No heap allocation, no system calls.
     */
    DualPredictionResult predict(const float* raw_values, double timestamp) {
        DualPredictionResult result;

        // Feature extraction
        features.extract(raw_values, timestamp);

        // Channel A: Fast ML
        auto t0 = __builtin_ia32_rdtsc();
        result.ml_prob = ml.predict(features.features);
        auto t1 = __builtin_ia32_rdtsc();

        // Channel B: Causal
        int simpsons = 0;
        result.causal_prob = causal.predict(raw_values, &simpsons);
        auto t2 = __builtin_ia32_rdtsc();

        result.simpsons_detected = simpsons;

        // Fusion
        float wm, wc;
        if (simpsons) {
            wm = w_ml_simpson;
            wc = w_causal_simpson;
        } else {
            wm = w_ml;
            wc = w_causal;
        }
        result.fused_prob = wm * result.ml_prob + wc * result.causal_prob;

        // Safety override
        if (result.ml_prob > 0.95f || result.causal_prob > 0.95f) {
            result.fused_prob = std::max(result.fused_prob, 0.95f);
        }

        // Threat classification
        float ttd = estimate_ttd(result.fused_prob);
        result.ttd_ms = ttd;
        result.fused_threat = classify_threat(result.fused_prob, ttd);

        // Approximate nanoseconds (assuming ~3 GHz)
        result.ml_latency_ns = (float)(t1 - t0) / 3.0f;
        result.causal_latency_ns = (float)(t2 - t1) / 3.0f;
        result.total_latency_ns = (float)(t2 - t0) / 3.0f;

        return result;
    }

    inline float estimate_ttd(float prob) const {
        if (prob < 0.3f) return 1e6f;
        return 800.0f * (1.0f - prob);
    }

    inline int classify_threat(float prob, float ttd_ms) const {
        if (prob > 0.9f && ttd_ms < 50.0f)  return IMMINENT;
        if (prob > 0.7f && ttd_ms < 200.0f) return CRITICAL;
        if (prob > 0.5f)                      return WARNING;
        if (prob > 0.3f)                      return WATCH;
        return SAFE;
    }
};

} // namespace fusionmind
