"""
Causal Query Engine (PF8).

Classifies natural language queries into Pearl's Ladder levels and
generates structured causal operations. The LLM then uses these
operations to produce grounded answers.

Query Types:
  Level 1 (Observation):  "What is the current Te?"
  Level 2 (Intervention): "What happens if we increase P_NBI to 8 MW?"
  Level 3 (Counterfactual): "Would Te have been higher if we had used ECRH instead?"
  Meta (Hypothesis):       "What new causal relationships should we test?"
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from .causal_context import CausalContext


class QueryClassifier:
    """Classify natural language queries into Pearl's Ladder levels."""

    # Pattern → level mapping
    INTERVENTION_PATTERNS = [
        r'\bwhat\s+happens?\s+if\b',
        r'\bwhat\s+would\s+happen\b',
        r'\bif\s+we\s+(set|increase|decrease|change|adjust|raise|lower|turn)',
        r'\beffect\s+of\s+(increasing|decreasing|changing|setting)',
        r'\bdo\s*\(',           # do-calculus notation
        r'\bintervene\b',
        r'\bwhat\s+if\s+we\b',
        r'\bšto\s+(se\s+)?dogodi\s+ako\b',   # Croatian
        r'\bšto\s+ako\b',
        r'\bkad\s+bi\s+postavili\b',
    ]

    COUNTERFACTUAL_PATTERNS = [
        r'\bwould\s+have\b',
        r'\bcould\s+have\b',
        r'\bshould\s+have\b',
        r'\bwhat\s+would\s+.+\s+have\s+been\b',
        r'\bhad\s+we\s+(used|chosen|set|increased)\b',
        r'\binstead\s+of\b',
        r'\bif\s+.+\s+had\s+been\b',
        r'\bretrospecti',
        r'\bcounterfactual\b',
        r'\bda\s+smo\b',         # Croatian
        r'\bšto\s+bi\s+bilo\b',
    ]

    HYPOTHESIS_PATTERNS = [
        r'\bwhat\s+(new\s+)?causal\b',
        r'\bhypothes[ie]s\b',
        r'\bsuggest\s+(new\s+)?(experiment|test|relationship)',
        r'\bwhat\s+should\s+we\s+test\b',
        r'\bunknown\s+(causal\s+)?relationship',
        r'\bmissing\s+edge',
        r'\bwhat\s+(else\s+)?could\s+cause\b',
        r'\bpredloži\b',          # Croatian
    ]

    EXPLANATION_PATTERNS = [
        r'\bwhy\s+(did|does|is|was|has)\b',
        r'\bexplain\b',
        r'\bcause\s+of\b',
        r'\breason\s+for\b',
        r'\bzašto\b',            # Croatian
        r'\bobjasni\b',
    ]

    SAFETY_PATTERNS = [
        r'\bsafe\b',
        r'\bdisruption\b',
        r'\brisk\b',
        r'\blimit\b',
        r'\bgreenwald\b',
        r'\bbeta\s+limit\b',
        r'\bsigurno\b',
    ]

    @classmethod
    def classify(cls, query: str) -> Dict[str, Any]:
        """Classify query into Pearl's Ladder level with confidence.
        
        Returns:
            {
                'level': 1|2|3|'hypothesis'|'explanation'|'safety',
                'confidence': float,
                'patterns_matched': list,
                'suggested_operation': str,
            }
        """
        q = query.lower().strip()
        results = []

        # Check each level
        for pattern in cls.COUNTERFACTUAL_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                results.append(('counterfactual', 3, pattern))

        for pattern in cls.INTERVENTION_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                results.append(('intervention', 2, pattern))

        for pattern in cls.HYPOTHESIS_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                results.append(('hypothesis', 'hypothesis', pattern))

        for pattern in cls.EXPLANATION_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                results.append(('explanation', 'explanation', pattern))

        for pattern in cls.SAFETY_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                results.append(('safety', 'safety', pattern))

        if not results:
            return {
                'level': 1,
                'level_name': 'observation',
                'confidence': 0.5,
                'patterns_matched': [],
                'suggested_operation': 'observe',
            }

        # Priority: counterfactual > intervention > hypothesis > explanation > safety
        priority = {'counterfactual': 5, 'intervention': 4, 'hypothesis': 3,
                     'explanation': 2, 'safety': 1}
        results.sort(key=lambda x: priority.get(x[0], 0), reverse=True)
        best = results[0]

        level_names = {1: 'observation', 2: 'intervention', 3: 'counterfactual'}
        level = best[1]
        level_name = best[0] if isinstance(level, str) else level_names.get(level, str(level))

        operations = {
            'counterfactual': 'counterfactual_query',
            'intervention': 'do_intervention',
            'hypothesis': 'generate_hypotheses',
            'explanation': 'trace_causal_paths',
            'safety': 'check_safety',
        }

        return {
            'level': level,
            'level_name': level_name,
            'confidence': min(1.0, 0.5 + 0.15 * len(results)),
            'patterns_matched': [r[2] for r in results],
            'suggested_operation': operations.get(best[0], 'observe'),
        }


class QueryEngine:
    """Process natural language queries using causal context.
    
    This engine prepares the full prompt for the LLM by:
    1. Classifying the query (Pearl's Ladder level)
    2. Extracting relevant causal subgraph
    3. Building structured context
    4. Generating the prompt with reasoning instructions
    """

    def __init__(self, context: CausalContext):
        self.context = context
        self.classifier = QueryClassifier()

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query and return the full LLM prompt + metadata.
        
        Returns:
            {
                'system_prompt': str,     # Full system prompt with causal context
                'user_message': str,      # Enriched user query
                'classification': dict,   # Query classification
                'relevant_paths': list,   # Causal paths relevant to query
            }
        """
        # 1. Classify
        classification = self.classifier.classify(query)

        # 2. Build system prompt
        system_prompt = self.context.build_system_prompt()

        # 3. Build query-specific context
        query_context = self.context.build_query_context(query)

        # 4. Build enriched user message
        level_label = f"[Pearl Level {classification['level']}]" \
            if isinstance(classification['level'], int) \
            else f"[{classification['level_name'].upper()}]"

        user_parts = [
            f"{level_label} {query}",
        ]
        if query_context:
            user_parts.append(f"\n{query_context}")

        # Add operation hint
        op = classification['suggested_operation']
        if op == 'do_intervention':
            user_parts.append(
                "\nUse do-calculus to answer. Trace the causal paths from "
                "the intervention variable to the target. Quantify using SCM equations."
            )
        elif op == 'counterfactual_query':
            user_parts.append(
                "\nUse the abduction→action→prediction protocol for counterfactual reasoning. "
                "First infer exogenous noise from the factual observation, then apply "
                "the counterfactual intervention, then predict."
            )
        elif op == 'trace_causal_paths':
            user_parts.append(
                "\nTrace all causal paths relevant to this question. "
                "Check for confounders and Simpson's Paradox risks."
            )
        elif op == 'generate_hypotheses':
            user_parts.append(
                "\nPropose 3-5 testable causal hypotheses. For each, specify: "
                "expected edge (X→Y), expected sign (+/−), proposed experiment "
                "to test it, and expected outcome."
            )

        return {
            'system_prompt': system_prompt,
            'user_message': "\n".join(user_parts),
            'classification': classification,
        }

    def format_response_for_display(self, raw_response: str,
                                     classification: Dict) -> Dict[str, Any]:
        """Post-process LLM response for structured display."""
        level = classification.get('level', 1)
        level_name = classification.get('level_name', 'observation')

        return {
            'answer': raw_response,
            'pearl_level': level,
            'pearl_level_name': level_name,
            'confidence': classification.get('confidence', 0.5),
            'operation': classification.get('suggested_operation', 'observe'),
        }


def create_example_queries() -> List[Dict[str, str]]:
    """Generate example queries for each Pearl's Ladder level."""
    return [
        # Level 1: Observation
        {'query': 'What is the current electron temperature?',
         'level': 'observation', 'category': 'Status'},
        {'query': 'What are the direct causes of stored energy?',
         'level': 'observation', 'category': 'Graph'},
        # Level 2: Intervention
        {'query': 'What happens to Te if we increase P_NBI to 8 MW?',
         'level': 'intervention', 'category': 'Heating'},
        {'query': 'What is the effect of decreasing gas puff on beta_N?',
         'level': 'intervention', 'category': 'Fueling'},
        {'query': 'If we set P_ECRH to 5 MW, what happens to q95?',
         'level': 'intervention', 'category': 'Profile control'},
        # Level 3: Counterfactual
        {'query': 'Would the disruption have been avoided if we had reduced density 100ms earlier?',
         'level': 'counterfactual', 'category': 'Safety'},
        {'query': 'What would Te have been if we had used ECRH instead of NBI?',
         'level': 'counterfactual', 'category': 'Scenario'},
        # Explanation
        {'query': 'Why did the stored energy drop suddenly?',
         'level': 'explanation', 'category': 'Diagnosis'},
        {'query': 'Explain the causal relationship between density and radiation',
         'level': 'explanation', 'category': 'Physics'},
        # Hypothesis
        {'query': 'What new causal relationships should we test in the next experiment?',
         'level': 'hypothesis', 'category': 'Experiment design'},
        {'query': 'Suggest experiments to distinguish NBI vs ECRH effects on confinement',
         'level': 'hypothesis', 'category': 'Experiment design'},
        # Safety
        {'query': 'Is the current plasma state safe? What are the disruption risks?',
         'level': 'safety', 'category': 'Safety'},
    ]
