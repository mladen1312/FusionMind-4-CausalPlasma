"""
FusionMind Causal Copilot (PF8) — LLM-Augmented Causal Plasma Reasoning.

Combines Pearl's causal inference framework with large language models
to enable natural language interaction with tokamak plasma causal structure.

Components:
  CausalContext    — Encodes DAG + SCM + state for LLM consumption
  QueryEngine      — Classifies queries and builds structured prompts
  QueryClassifier  — Maps NL queries to Pearl's Ladder levels
"""
__all__ = ['CausalContext', 'QueryEngine', 'QueryClassifier', 'create_example_queries']

from .causal_context import CausalContext
from .query_engine import QueryEngine, QueryClassifier, create_example_queries
