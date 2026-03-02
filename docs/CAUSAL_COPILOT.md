# PF8: LLM-Augmented Causal Plasma Reasoning

## Overview

The Causal Copilot combines Pearl's causal inference framework with a large language model (LLM) to create a natural language interface for tokamak plasma control. Unlike standard AI assistants, the Copilot has access to the **discovered causal structure** of the plasma — its responses are grounded in the DAG, SCM equations, and validated physics.

This is the first system in fusion that integrates formal causal reasoning with an LLM.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Causal Copilot (PF8)                    │
│                                                      │
│  ┌────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │   Query     │    │   Causal     │    │   LLM    │ │
│  │ Classifier  │───▶│  Context     │───▶│ (Claude) │ │
│  │ (Pearl's    │    │  Builder     │    │          │ │
│  │  Ladder)    │    │ (DAG+SCM+   │    │ Grounded │ │
│  │             │    │  State)      │    │ Response │ │
│  └────────────┘    └──────────────┘    └──────────┘ │
│                                                      │
│  Data sources:                                       │
│  ├── CPDE v3.2 causal graph (15 edges, F1=91.9%)   │
│  ├── SCM equations (fitted to MAST data)            │
│  ├── Current plasma state (real-time)               │
│  ├── Safety limits (βN, q95, Greenwald)             │
│  └── Intervention history                            │
└─────────────────────────────────────────────────────┘
```

## Key Innovation

Standard fusion AI chatbots would simply pattern-match on plasma terminology. The Causal Copilot is fundamentally different:

1. **Causal Graph as Context**: The LLM receives the full DAG discovered by CPDE, including edge weights, signs, and physical mechanisms. It can trace causal paths, identify confounders, and detect Simpson's Paradox risks.

2. **Pearl's Ladder Classification**: Every query is classified into:
   - Level 1 (Observation): "What is the current Te?"
   - Level 2 (Intervention): "What happens if we SET P_NBI = 8 MW?" → do-calculus
   - Level 3 (Counterfactual): "What WOULD have happened if we had used ECRH?" → abduction

3. **Structured Reasoning Protocol**: The LLM follows a specific protocol:
   - Identify query type → Trace causal paths → Check confounders → Use SCM equations → Verify safety → Provide confidence

4. **Hypothesis Generation**: The LLM can propose new causal relationships to test, grounded in the existing graph structure and physics constraints.

## Components

### CausalContext (`fusionmind4/copilot/causal_context.py`)

Builds the structured LLM prompt from FusionMind pipeline outputs:
- DAG edges with weights and mechanisms
- SCM equations (human-readable)
- Current plasma state
- Safety limits and alerts
- Intervention history
- Causal path analysis (find_all_paths, get_confounders)

### QueryClassifier (`fusionmind4/copilot/query_engine.py`)

Regex-based classifier that maps natural language to Pearl's Ladder levels. Supports English and Croatian. Priority ordering ensures counterfactuals beat interventions when both patterns match.

### QueryEngine (`fusionmind4/copilot/query_engine.py`)

Orchestrates the full pipeline: classify → extract relevant subgraph → build enriched prompt → format response.

### React Dashboard (`dashboards/FM4_CausalCopilot.jsx`)

Interactive UI with:
- Chat interface calling Claude API with causal system prompt
- Causal graph SVG visualization with hover highlighting
- Example queries for each Pearl's Ladder level
- Pearl's Ladder level badges on each message

## Usage

### Python

```python
from fusionmind4.copilot import CausalContext, QueryEngine

# Build context from CPDE results
ctx = CausalContext.from_fusionmind(cpde_results, scm=scm, state=current_state)

# Process a query
engine = QueryEngine(ctx)
result = engine.process_query("What happens to Te if we increase NBI to 8 MW?")

# result['system_prompt'] → send to LLM API
# result['classification'] → Pearl's Level 2 (intervention)
```

### React Dashboard

The dashboard calls the Anthropic API directly with the causal context as the system prompt. See `dashboards/FM4_CausalCopilot.jsx`.

## Novelty Assessment

- **Novelty**: 9/10 — No prior art combining formal causal inference + LLM for fusion
- **Prior art search**: No fusion AI system uses LLM + causal graph for plasma reasoning
- **Closest work**: General causal LLM papers (2024-2025) but none applied to plasma physics
- **Patent strategy**: File as PF8 under FusionMind 4.0 portfolio

## Use Cases

| Use Case | User | Value |
|----------|------|-------|
| Control room assistant | Tokamak operators | NL interface for causal plasma queries |
| Explainability demo | Investors | Show that FusionMind "understands" physics |
| Hypothesis generation | Researchers | AI-assisted causal discovery |
| Training tool | Students | Learn plasma physics through causal reasoning |
| Post-shot analysis | Scientists | "Why did this shot disrupt?" with causal explanation |
