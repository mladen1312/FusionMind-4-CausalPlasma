// dynamic_overseer.hpp — Real-time multi-track arbitrator
// FusionMind 4.0 | Dr. Mladen Mester | March 2026
// Latency target: <500ns per decision (8 float ops + history lookup)
#pragma once
#include <array>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace fusionmind {

struct TrackResult {
    float prob;        // [0, 1]
    float confidence;  // [0, 1]
};

struct OverseerConfig {
    float disagreement_threshold = 0.20f;
    float alarm_threshold = 0.35f;
    float warning_threshold = 0.25f;
    float physics_priority = 1.5f;
    float smoothing_alpha = 0.7f;
    int history_len = 10;
};

struct OverseerDecision {
    float final_prob;
    int best_track;       // 0=A, 1=B, 2=C, 3=D
    float disagreement;
    int warning_level;    // 0=safe, 1=watch, 2=warning, 3=alarm
    bool correction_applied;
};

class DynamicOverseer {
public:
    static constexpr int N_TRACKS = 4;
    static constexpr int MAX_HISTORY = 32;
    
    DynamicOverseer() { reset(); }
    
    void configure(const OverseerConfig& cfg) { cfg_ = cfg; }
    
    void reset() {
        hist_idx_ = 0;
        hist_count_ = 0;
        std::memset(history_, 0, sizeof(history_));
    }
    
    OverseerDecision decide(const TrackResult tracks[N_TRACKS]) {
        OverseerDecision d;
        
        // 1. Compute disagreement (std of probabilities)
        float mean = 0;
        for (int i = 0; i < N_TRACKS; i++) mean += tracks[i].prob;
        mean /= N_TRACKS;
        
        float var = 0;
        for (int i = 0; i < N_TRACKS; i++) {
            float diff = tracks[i].prob - mean;
            var += diff * diff;
        }
        d.disagreement = std::sqrt(var / N_TRACKS);
        
        // 2. Compute weighted scores (physics tracks get priority)
        float scores[N_TRACKS];
        for (int i = 0; i < N_TRACKS; i++) {
            scores[i] = tracks[i].confidence;
            if (i >= 2) scores[i] *= cfg_.physics_priority; // C, D = physics
        }
        
        // Find best track
        d.best_track = 0;
        float best_score = scores[0];
        for (int i = 1; i < N_TRACKS; i++) {
            if (scores[i] > best_score) {
                best_score = scores[i];
                d.best_track = i;
            }
        }
        
        // 3. Decision logic
        if (d.disagreement > cfg_.disagreement_threshold) {
            // HIGH DISAGREEMENT → trust physics (tracks 2, 3)
            float w_c = tracks[2].confidence;
            float w_d = tracks[3].confidence;
            float total = w_c + w_d + 1e-10f;
            d.final_prob = (w_c * tracks[2].prob + w_d * tracks[3].prob) / total;
            d.correction_applied = true;
            d.best_track = (tracks[3].prob > tracks[2].prob) ? 3 : 2;
        } else {
            // LOW DISAGREEMENT → best track + smoothing from history
            float current = tracks[d.best_track].prob;
            if (hist_count_ > 0) {
                int prev_idx = (hist_idx_ - 1 + MAX_HISTORY) % MAX_HISTORY;
                float prev = history_[prev_idx];
                d.final_prob = cfg_.smoothing_alpha * current + (1.0f - cfg_.smoothing_alpha) * prev;
            } else {
                d.final_prob = current;
            }
            d.correction_applied = false;
        }
        
        // 4. Safety override
        float max_prob = 0;
        for (int i = 0; i < N_TRACKS; i++) {
            if (tracks[i].prob > max_prob) max_prob = tracks[i].prob;
        }
        if (max_prob > cfg_.alarm_threshold && d.final_prob < cfg_.alarm_threshold * 0.8f) {
            d.final_prob = std::max(d.final_prob, max_prob * 0.7f);
        }
        
        // Clamp
        d.final_prob = std::max(0.0f, std::min(1.0f, d.final_prob));
        
        // 5. Warning level
        if (d.final_prob > cfg_.alarm_threshold) d.warning_level = 3;
        else if (d.final_prob > cfg_.warning_threshold) d.warning_level = 2;
        else if (d.final_prob > 0.15f) d.warning_level = 1;
        else d.warning_level = 0;
        
        // 6. Update history
        history_[hist_idx_] = d.final_prob;
        hist_idx_ = (hist_idx_ + 1) % MAX_HISTORY;
        if (hist_count_ < MAX_HISTORY) hist_count_++;
        
        return d;
    }
    
private:
    OverseerConfig cfg_;
    float history_[MAX_HISTORY];
    int hist_idx_;
    int hist_count_;
};

} // namespace fusionmind
