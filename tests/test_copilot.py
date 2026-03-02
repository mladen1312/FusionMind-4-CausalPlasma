"""Tests for FusionMind Causal Copilot (PF8)."""
import pytest
import numpy as np
from fusionmind4.copilot import CausalContext, QueryEngine, QueryClassifier, create_example_queries


# ── Fixtures ──────────────────────────────────────────────

def make_sample_context():
    """Create a sample CausalContext with realistic fusion data."""
    var_names = ['I_p', 'P_NBI', 'P_ECRH', 'gas_puff',
                 'n_e', 'T_e', 'T_i', 'beta_N', 'q95', 'W_MHD', 'P_rad']
    ctx = CausalContext(variable_names=var_names)
    ctx.set_dag([
        ('I_p', 'q95', -0.85),
        ('I_p', 'W_MHD', 0.45),
        ('P_NBI', 'T_e', 0.72),
        ('P_NBI', 'T_i', 0.68),
        ('P_NBI', 'W_MHD', 0.81),
        ('P_NBI', 'beta_N', 0.65),
        ('P_ECRH', 'T_e', 0.55),
        ('gas_puff', 'n_e', 0.90),
        ('n_e', 'P_rad', 0.78),
        ('n_e', 'T_e', -0.35),
        ('T_e', 'beta_N', 0.42),
    ])
    ctx.set_scm_equations({
        'T_e': 'T_e = 0.72*P_NBI + 0.55*P_ECRH - 0.35*n_e + ε',
        'q95': 'q95 = -0.85*I_p + ε',
        'W_MHD': 'W_MHD = 0.45*I_p + 0.81*P_NBI + ε',
        'beta_N': 'beta_N = 0.65*P_NBI + 0.42*T_e + ε',
        'P_rad': 'P_rad = 0.78*n_e + ε',
        'n_e': 'n_e = 0.90*gas_puff + ε',
    })
    ctx.set_state({
        'I_p': 1.0, 'P_NBI': 5.0, 'P_ECRH': 2.0, 'gas_puff': 3.0,
        'n_e': 4.5e19, 'T_e': 5.2, 'T_i': 4.8, 'beta_N': 2.1,
        'q95': 3.5, 'W_MHD': 1.2, 'P_rad': 0.8,
    })
    return ctx


# ── CausalContext Tests ───────────────────────────────────

class TestCausalContext:

    def test_create_empty(self):
        ctx = CausalContext()
        assert ctx.dag is None
        assert ctx.variable_names == []

    def test_set_dag(self):
        ctx = make_sample_context()
        assert 'P_NBI' in ctx.dag
        assert len(ctx.dag['P_NBI']) == 4  # T_e, T_i, W_MHD, beta_N

    def test_find_paths_direct(self):
        ctx = make_sample_context()
        paths = ctx.find_all_paths('P_NBI', 'T_e')
        assert len(paths) >= 1
        assert ['P_NBI', 'T_e'] in paths

    def test_find_paths_indirect(self):
        ctx = make_sample_context()
        # P_NBI → T_e → beta_N (indirect path)
        paths = ctx.find_all_paths('P_NBI', 'beta_N')
        assert len(paths) >= 2  # direct + via T_e

    def test_find_paths_no_path(self):
        ctx = make_sample_context()
        paths = ctx.find_all_paths('P_rad', 'I_p')
        assert paths == []

    def test_get_parents(self):
        ctx = make_sample_context()
        parents = ctx.get_parents('T_e')
        assert 'P_NBI' in parents
        assert 'P_ECRH' in parents
        assert 'n_e' in parents

    def test_get_children(self):
        ctx = make_sample_context()
        children = ctx.get_children('P_NBI')
        assert 'T_e' in children
        assert 'W_MHD' in children

    def test_get_confounders(self):
        ctx = make_sample_context()
        # T_e and W_MHD both have P_NBI as ancestor
        conf = ctx.get_confounders('T_e', 'W_MHD')
        assert 'P_NBI' in conf

    def test_check_safety_ok(self):
        ctx = make_sample_context()
        safety = ctx.check_safety()
        assert safety['safe'] is True

    def test_check_safety_critical(self):
        ctx = make_sample_context()
        ctx.set_state({'beta_N': 5.0, 'q95': 1.2})
        safety = ctx.check_safety()
        assert safety['safe'] is False
        assert len(safety['alerts']) >= 2

    def test_build_system_prompt(self):
        ctx = make_sample_context()
        prompt = ctx.build_system_prompt()
        assert 'FusionMind Causal Copilot' in prompt
        assert 'CAUSAL GRAPH' in prompt
        assert 'SCM' in prompt or 'STRUCTURAL' in prompt
        assert 'CURRENT PLASMA STATE' in prompt
        assert 'REASONING INSTRUCTIONS' in prompt

    def test_build_system_prompt_length(self):
        ctx = make_sample_context()
        prompt = ctx.build_system_prompt()
        # Should be substantial but not enormous
        assert 500 < len(prompt) < 10000

    def test_build_query_context(self):
        ctx = make_sample_context()
        qc = ctx.build_query_context("What happens if we increase P_NBI?")
        assert 'Causal path' in qc or qc == ""

    def test_add_intervention_history(self):
        ctx = make_sample_context()
        ctx.add_intervention('P_NBI', 8.0, {'T_e': 7.1, 'W_MHD': 1.8})
        assert len(ctx.intervention_history) == 1
        prompt = ctx.build_system_prompt()
        assert 'INTERVENTION HISTORY' in prompt

    def test_to_dict(self):
        ctx = make_sample_context()
        d = ctx.to_dict()
        assert 'dag' in d
        assert 'system_prompt' in d
        assert isinstance(d['system_prompt'], str)

    def test_from_fusionmind(self):
        cpde_results = {
            'variable_names': ['A', 'B', 'C'],
            'edges': [('A', 'B', 0.5), ('B', 'C', 0.3)],
        }
        ctx = CausalContext.from_fusionmind(cpde_results)
        assert ctx.variable_names == ['A', 'B', 'C']
        assert 'A' in ctx.dag

    def test_extract_variables_from_query(self):
        ctx = make_sample_context()
        vars_found = ctx._extract_variables_from_query(
            "What happens to electron temperature if we increase NBI?"
        )
        assert 'T_e' in vars_found or 'P_NBI' in vars_found


# ── QueryClassifier Tests ─────────────────────────────────

class TestQueryClassifier:

    def test_observation_query(self):
        result = QueryClassifier.classify("What is the current density?")
        assert result['level'] == 1

    def test_intervention_query(self):
        result = QueryClassifier.classify("What happens if we increase P_NBI to 8 MW?")
        assert result['level'] == 2

    def test_counterfactual_query(self):
        result = QueryClassifier.classify(
            "What would Te have been if we had used ECRH instead of NBI?"
        )
        assert result['level'] == 3

    def test_hypothesis_query(self):
        result = QueryClassifier.classify(
            "What new causal relationships should we test?"
        )
        assert result['level_name'] == 'hypothesis'

    def test_explanation_query(self):
        result = QueryClassifier.classify("Why did the stored energy drop?")
        assert result['level_name'] == 'explanation'

    def test_safety_query(self):
        result = QueryClassifier.classify("Is there a disruption risk?")
        assert result['level_name'] == 'safety'

    def test_croatian_intervention(self):
        result = QueryClassifier.classify("Što se dogodi ako povećamo snagu?")
        assert result['level'] == 2

    def test_confidence_increases_with_matches(self):
        r1 = QueryClassifier.classify("increase power")
        r2 = QueryClassifier.classify("What happens if we increase P_NBI and change gas?")
        assert r2['confidence'] >= r1['confidence']

    def test_counterfactual_beats_intervention(self):
        # "would have" + "if we" → counterfactual wins
        result = QueryClassifier.classify(
            "What would have happened if we had increased power?"
        )
        assert result['level'] == 3


# ── QueryEngine Tests ─────────────────────────────────────

class TestQueryEngine:

    def test_process_observation(self):
        ctx = make_sample_context()
        engine = QueryEngine(ctx)
        result = engine.process_query("What is the current Te?")
        assert 'system_prompt' in result
        assert 'user_message' in result
        assert 'classification' in result

    def test_process_intervention(self):
        ctx = make_sample_context()
        engine = QueryEngine(ctx)
        result = engine.process_query("What happens if we set P_NBI to 10 MW?")
        assert 'do-calculus' in result['user_message'].lower() or \
               'intervention' in result['user_message'].lower()
        assert result['classification']['level'] == 2

    def test_process_counterfactual(self):
        ctx = make_sample_context()
        engine = QueryEngine(ctx)
        result = engine.process_query(
            "Would Te have been higher if we had used ECRH instead of NBI?"
        )
        assert result['classification']['level'] == 3
        assert 'abduction' in result['user_message'].lower() or \
               'counterfactual' in result['user_message'].lower()

    def test_process_hypothesis(self):
        ctx = make_sample_context()
        engine = QueryEngine(ctx)
        result = engine.process_query("Suggest new experiments to test")
        assert 'hypothes' in result['user_message'].lower() or \
               'experiment' in result['user_message'].lower()

    def test_format_response(self):
        ctx = make_sample_context()
        engine = QueryEngine(ctx)
        classification = {'level': 2, 'level_name': 'intervention',
                          'confidence': 0.8, 'suggested_operation': 'do_intervention'}
        result = engine.format_response_for_display("Test answer", classification)
        assert result['pearl_level'] == 2
        assert result['answer'] == "Test answer"


# ── Example Queries ───────────────────────────────────────

class TestExampleQueries:

    def test_example_queries_exist(self):
        examples = create_example_queries()
        assert len(examples) >= 10

    def test_all_levels_covered(self):
        examples = create_example_queries()
        levels = set(e['level'] for e in examples)
        assert 'observation' in levels
        assert 'intervention' in levels
        assert 'counterfactual' in levels
        assert 'hypothesis' in levels

    def test_examples_classify_correctly(self):
        examples = create_example_queries()
        level_map = {
            'observation': [1],
            'intervention': [2],
            'counterfactual': [3],
            'explanation': ['explanation'],
            'hypothesis': ['hypothesis'],
            'safety': ['safety'],
        }
        correct = 0
        for ex in examples:
            result = QueryClassifier.classify(ex['query'])
            expected = level_map.get(ex['level'], [])
            if result['level'] in expected or result['level_name'] in [ex['level']]:
                correct += 1
        # At least 70% should classify correctly
        assert correct / len(examples) >= 0.7, \
            f"Only {correct}/{len(examples)} classified correctly"
