/**
 * FusionMind 4.0 — C API for Stack Engine (ctypes interface)
 * 
 * Compile: g++ -O3 -march=native -shared -fPIC -o libfusionmind_stack.so stack_api.cpp
 */

#include "stack_engine.hpp"

extern "C" {

using namespace fusionmind;

// ── Create / Destroy ──

FusionMindStack_CPP* fm_stack_create(int n_vars, int phase) {
    auto* s = new FusionMindStack_CPP();
    s->init(n_vars, phase);
    return s;
}

void fm_stack_destroy(FusionMindStack_CPP* s) {
    delete s;
}

// ── Configuration ──

void fm_stack_set_phase(FusionMindStack_CPP* s, int phase) {
    s->phase = phase;
}

void fm_stack_set_setpoints(FusionMindStack_CPP* s, const float* sp) {
    s->set_setpoints(sp);
}

void fm_stack_load_dag(FusionMindStack_CPP* s, const int* adj_flat) {
    s->L2.load_dag(adj_flat);
}

void fm_stack_load_equation(FusionMindStack_CPP* s, int var_idx,
                            const int* parents, const float* coefs,
                            float intercept, int n_parents, float r2) {
    s->L2.load_equation(var_idx, parents, coefs, intercept, n_parents, r2);
}

void fm_stack_set_safety_limit(FusionMindStack_CPP* s, int var_idx,
                               float min_crit, float min_warn,
                               float max_warn, float max_crit,
                               int has_min, int has_max) {
    s->L3.set_limit(var_idx, min_crit, min_warn, max_warn, max_crit,
                    has_min != 0, has_max != 0);
}

void fm_stack_load_policy_weights(FusionMindStack_CPP* s,
                                  const float* w1, const float* b1,
                                  const float* w2, const float* b2,
                                  const float* w3, const float* b3) {
    s->L1.load_weights(w1, b1, w2, b2, w3, b3);
}

// ── Main Step ──

void fm_stack_step(FusionMindStack_CPP* s,
                   const float* raw_values, float timestamp,
                   const float* ext_action,  // NULL if no external RL
                   int n_act, const int* act_var_map,
                   StackResult* result) {
    s->step(raw_values, timestamp, ext_action, n_act, act_var_map, *result);
}

// ── Queries ──

void fm_stack_do_intervention(FusionMindStack_CPP* s,
                              const float* baseline,
                              const int* interv_mask,
                              const float* interv_vals,
                              float* result) {
    s->L2.do_intervention(baseline, result, interv_mask, interv_vals);
}

void fm_stack_counterfactual(FusionMindStack_CPP* s,
                             const float* factual,
                             const int* interv_mask,
                             const float* interv_vals,
                             float* result) {
    s->L2.counterfactual(factual, result, interv_mask, interv_vals);
}

float fm_stack_fast_risk(FusionMindStack_CPP* s, const float* values) {
    float old_vals[MAX_VARS];
    std::memcpy(old_vals, s->L0.values, sizeof(float) * s->n_vars);
    std::memcpy(s->L0.values, values, sizeof(float) * s->n_vars);
    float risk = s->L0.fast_risk(s->L3.limits);
    std::memcpy(s->L0.values, old_vals, sizeof(float) * s->n_vars);
    return risk;
}

// ── Statistics ──

int fm_stack_get_cycle_count(FusionMindStack_CPP* s) { return s->cycle_count; }
int fm_stack_get_n_vetoed(FusionMindStack_CPP* s)    { return s->L3.n_vetoed; }
int fm_stack_get_n_approved(FusionMindStack_CPP* s)  { return s->L3.n_approved; }
int fm_stack_get_n_warned(FusionMindStack_CPP* s)    { return s->L3.n_warned; }
int fm_stack_get_phase(FusionMindStack_CPP* s)       { return s->phase; }

} // extern "C"
