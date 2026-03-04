/**
 * FusionMind 4.0 — C-API for Python ctypes binding
 *
 * Exposes the fast C++ inference engine as a shared library.
 * All functions use C linkage, POD arguments, no exceptions in hot path.
 *
 * Build:
 *   g++ -O3 -march=native -ffast-math -shared -fPIC \
 *       -o libfusionmind_rt.so fast_engine_api.cpp
 *
 * Author: Dr. Mladen Mester, March 2026
 */

#include "fast_engine.hpp"
#include <cstring>
#include <chrono>

using namespace fusionmind;

// Global instances (single-threaded PCS context)
static DualPredictor g_predictor;
static bool g_initialized = false;

extern "C" {

// ─── Lifecycle ───────────────────────────────────────────────────────────

/**
 * Initialize the real-time engine.
 * @param n_vars  Number of plasma variables
 * @return 0 on success
 */
int fm_init(int n_vars) {
    g_predictor.init(n_vars);
    g_initialized = true;
    return 0;
}

/**
 * Set variable index mappings for feature extraction.
 */
void fm_set_var_indices(int betaN, int ne, int Ip, int q95,
                        int P_rad, int P_NBI, int li, int MHD,
                        int ne_core) {
    auto& f = g_predictor.features;
    f.idx_betaN  = betaN;
    f.idx_ne     = ne;
    f.idx_Ip     = Ip;
    f.idx_q95    = q95;
    f.idx_P_rad  = P_rad;
    f.idx_P_NBI  = P_NBI;
    f.idx_li     = li;
    f.idx_MHD    = MHD;
    f.idx_ne_core = ne_core;
}

// ─── ML Model Loading ───────────────────────────────────────────────────

/**
 * Load a trained decision stump ensemble.
 * @param n_trees      Number of trees
 * @param n_features   Number of input features
 * @param features     Flat array [n_trees * 1] of feature indices
 * @param splits       Flat array [n_trees * 1] of split values
 * @param left_vals    Flat array [n_trees * 1] of left predictions
 * @param right_vals   Flat array [n_trees * 1] of right predictions
 * @param lrs          Flat array [n_trees * 1] of learning rates
 * @param means        Array [n_features] of feature means
 * @param stds         Array [n_features] of feature stds
 * @param threshold    Decision threshold
 * @return 0 on success
 */
int fm_load_ml_model(int n_trees, int n_features,
                     const int* features,
                     const float* splits,
                     const float* left_vals,
                     const float* right_vals,
                     const float* lrs,
                     const float* means,
                     const float* stds,
                     float threshold) {
    if (n_trees > MAX_TREES || n_features > MAX_FEATURES) return -1;

    auto& ml = g_predictor.ml;
    ml.n_trees = n_trees;
    ml.n_features = n_features;
    ml.threshold = threshold;

    for (int i = 0; i < n_trees; ++i) {
        ml.trees[i].feature_idx = features[i];
        ml.trees[i].split_value = splits[i];
        ml.trees[i].value_left  = left_vals[i];
        ml.trees[i].value_right = right_vals[i];
        ml.trees[i].learning_rate = lrs[i];
    }

    std::memcpy(ml.means, means, n_features * sizeof(float));
    std::memcpy(ml.stds, stds, n_features * sizeof(float));

    return 0;
}

/**
 * Load causal model parameters.
 * @param n_vars     Number of variables
 * @param effects    Array [n_vars] of causal effects
 * @param has_bound  Array [n_vars] 0/1 whether boundary exists
 * @param bound_low  Array [n_vars] of lower boundaries
 * @param bound_high Array [n_vars] of upper boundaries
 */
int fm_load_causal_model(int n_vars,
                         const float* effects,
                         const int* has_bound,
                         const float* bound_low,
                         const float* bound_high) {
    if (n_vars > MAX_VARS) return -1;

    auto& c = g_predictor.causal;
    c.n_vars = n_vars;
    std::memcpy(c.effects, effects, n_vars * sizeof(float));
    std::memcpy(c.has_boundary, has_bound, n_vars * sizeof(int));
    std::memcpy(c.boundary_low, bound_low, n_vars * sizeof(float));
    std::memcpy(c.boundary_high, bound_high, n_vars * sizeof(float));

    return 0;
}

/**
 * Set fusion weights.
 */
void fm_set_weights(float w_ml, float w_causal,
                    float w_ml_simpson, float w_causal_simpson) {
    g_predictor.w_ml = w_ml;
    g_predictor.w_causal = w_causal;
    g_predictor.w_ml_simpson = w_ml_simpson;
    g_predictor.w_causal_simpson = w_causal_simpson;
}

// ─── Inference (HOT PATH — no allocation, no syscalls) ──────────────────

/**
 * Run dual-mode prediction on a single plasma snapshot.
 * This is the main hot-path function — target < 5 μs.
 *
 * @param raw_values   Array [n_vars] of current plasma values
 * @param timestamp_s  Seconds since start-of-discharge
 * @param out          Pointer to result struct (pre-allocated by caller)
 * @return 0 on success
 */
int fm_predict(const float* raw_values, double timestamp_s,
               DualPredictionResult* out) {
    if (!g_initialized) return -1;

    *out = g_predictor.predict(raw_values, timestamp_s);
    return 0;
}

/**
 * Run ML-only prediction (fastest path, < 1 μs).
 * @param features  Pre-extracted features [n_features]
 * @return disruption probability [0, 1]
 */
float fm_predict_ml_only(const float* features) {
    return g_predictor.ml.predict(features);
}

/**
 * Run causal-only prediction.
 * @param raw_values    Raw plasma values [n_vars]
 * @param simpsons_out  Output: 1 if Simpson's Paradox detected
 * @return disruption probability [0, 1]
 */
float fm_predict_causal_only(const float* raw_values,
                             int* simpsons_out) {
    return g_predictor.causal.predict(raw_values, simpsons_out);
}

/**
 * Batch ML prediction.
 * @param X          Feature matrix [n_samples * n_features], row-major
 * @param n_samples  Number of samples
 * @param out_probs  Output array [n_samples]
 */
void fm_predict_batch(const float* X, int n_samples, float* out_probs) {
    g_predictor.ml.predict_batch(X, n_samples, out_probs);
}

/**
 * Extract features from raw values (updates internal ring buffer).
 * @param raw_values  Array [n_vars]
 * @param timestamp_s Seconds since SOD
 * @param out_features Output feature array [MAX_FEATURES]
 * @return number of features extracted
 */
int fm_extract_features(const float* raw_values, double timestamp_s,
                        float* out_features) {
    int nf = g_predictor.features.extract(raw_values, timestamp_s);
    std::memcpy(out_features, g_predictor.features.features,
                nf * sizeof(float));
    return nf;
}

/**
 * Reset the feature history buffer.
 */
void fm_reset() {
    g_predictor.features.reset();
}

// ─── Benchmarking ────────────────────────────────────────────────────────

/**
 * Run latency benchmark.
 * @param raw_values   Sample input [n_vars]
 * @param n_iterations Number of iterations
 * @param out_mean_ns  Output: mean latency in nanoseconds
 * @param out_p99_ns   Output: p99 latency in nanoseconds
 * @param out_max_ns   Output: max latency in nanoseconds
 */
void fm_benchmark(const float* raw_values, int n_iterations,
                  double* out_mean_ns, double* out_p99_ns,
                  double* out_max_ns) {
    // Warmup
    DualPredictionResult dummy;
    for (int i = 0; i < 100; ++i) {
        fm_predict(raw_values, (double)i * 0.001, &dummy);
    }
    g_predictor.features.reset();

    // Benchmark
    double* latencies = new double[n_iterations];

    for (int i = 0; i < n_iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fm_predict(raw_values, (double)i * 0.001, &dummy);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }

    // Sort for percentiles
    std::sort(latencies, latencies + n_iterations);

    double sum = 0;
    for (int i = 0; i < n_iterations; ++i) sum += latencies[i];
    *out_mean_ns = sum / n_iterations;
    *out_p99_ns = latencies[(int)(n_iterations * 0.99)];
    *out_max_ns = latencies[n_iterations - 1];

    delete[] latencies;
}

/**
 * Get version info.
 */
const char* fm_version() {
    return "FusionMind-RT 4.5.0 (C++ fast engine)";
}

} // extern "C"
