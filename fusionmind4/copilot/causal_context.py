"""
Causal Context Builder for LLM Integration (PF8).

Converts FusionMind's causal graph, SCM equations, and plasma state
into structured context that an LLM can reason over. This is NOT just
"send data to chatbot" — it encodes Pearl's causal hierarchy so the LLM
can perform interventional and counterfactual reasoning grounded in the
discovered causal structure.

Patent Family: PF8 — LLM-Augmented Causal Plasma Reasoning
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class CausalContext:
    """Structured causal context for LLM consumption.
    
    Encodes:
    - Causal DAG (directed edges with weights and physics semantics)
    - SCM equations (structural equations for each variable)
    - Current plasma state (with uncertainty)
    - Causal paths (all directed paths between variables)
    - Intervention history (past do-calculus operations)
    - Physics constraints (hard limits, safety boundaries)
    """

    def __init__(self, dag=None, scm=None, variable_names=None):
        self.dag = dag
        self.scm = scm
        self.variable_names = variable_names or []
        self.current_state = {}
        self.intervention_history = []
        self.safety_limits = {
            'beta_N': {'max': 4.0, 'warning': 3.5, 'unit': ''},
            'q95': {'min': 1.5, 'warning': 2.0, 'unit': ''},
            'n_e': {'max': 1.5e20, 'warning': 1.2e20, 'unit': 'm⁻³'},
            'P_rad_frac': {'max': 0.8, 'warning': 0.6, 'unit': ''},
        }

    def set_dag(self, dag_edges: List[Tuple[str, str, float]]):
        """Set causal DAG from list of (cause, effect, weight) tuples."""
        self.dag = {}
        for cause, effect, weight in dag_edges:
            if cause not in self.dag:
                self.dag[cause] = []
            self.dag[cause].append({'effect': effect, 'weight': weight})

    def set_scm_equations(self, equations: Dict[str, str]):
        """Set SCM equations as human-readable strings."""
        self.scm = equations

    def set_state(self, state: Dict[str, float]):
        """Set current plasma state."""
        self.current_state = state

    def add_intervention(self, variable: str, value: float, result: Dict[str, float]):
        """Record an intervention for context."""
        self.intervention_history.append({
            'variable': variable,
            'value': value,
            'result': result,
        })

    # ----------------------------------------------------------------
    # Causal path analysis
    # ----------------------------------------------------------------

    def find_all_paths(self, source: str, target: str, max_depth: int = 5) -> List[List[str]]:
        """Find all directed causal paths from source to target."""
        if not self.dag:
            return []
        paths = []
        self._dfs_paths(source, target, [source], set(), paths, max_depth)
        return paths

    def _dfs_paths(self, current, target, path, visited, paths, max_depth):
        if len(path) > max_depth:
            return
        if current == target and len(path) > 1:
            paths.append(list(path))
            return
        visited.add(current)
        for edge in self.dag.get(current, []):
            next_node = edge['effect']
            if next_node not in visited:
                path.append(next_node)
                self._dfs_paths(next_node, target, path, visited, paths, max_depth)
                path.pop()
        visited.discard(current)

    def get_parents(self, variable: str) -> List[str]:
        """Get direct causal parents of a variable."""
        parents = []
        if not self.dag:
            return parents
        for cause, edges in self.dag.items():
            for e in edges:
                if e['effect'] == variable:
                    parents.append(cause)
        return parents

    def get_children(self, variable: str) -> List[str]:
        """Get direct causal children of a variable."""
        if not self.dag:
            return []
        return [e['effect'] for e in self.dag.get(variable, [])]

    def get_confounders(self, var_a: str, var_b: str) -> List[str]:
        """Find common causal ancestors (potential confounders)."""
        ancestors_a = self._get_ancestors(var_a)
        ancestors_b = self._get_ancestors(var_b)
        return list(ancestors_a & ancestors_b)

    def _get_ancestors(self, variable: str, visited=None) -> set:
        if visited is None:
            visited = set()
        parents = self.get_parents(variable)
        for p in parents:
            if p not in visited:
                visited.add(p)
                self._get_ancestors(p, visited)
        return visited

    # ----------------------------------------------------------------
    # Safety analysis
    # ----------------------------------------------------------------

    def check_safety(self, state: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Check current state against safety limits."""
        s = state or self.current_state
        alerts = []
        for var, limits in self.safety_limits.items():
            val = s.get(var)
            if val is None:
                continue
            if 'max' in limits and val > limits['max']:
                alerts.append({'variable': var, 'level': 'CRITICAL',
                               'message': f"{var} = {val:.3f} exceeds limit {limits['max']}"})
            elif 'max' in limits and val > limits['warning']:
                alerts.append({'variable': var, 'level': 'WARNING',
                               'message': f"{var} = {val:.3f} approaching limit {limits['max']}"})
            if 'min' in limits and val < limits['min']:
                alerts.append({'variable': var, 'level': 'CRITICAL',
                               'message': f"{var} = {val:.3f} below limit {limits['min']}"})
            elif 'min' in limits and val < limits['warning']:
                alerts.append({'variable': var, 'level': 'WARNING',
                               'message': f"{var} = {val:.3f} approaching limit {limits['min']}"})
        return {'safe': len([a for a in alerts if a['level'] == 'CRITICAL']) == 0,
                'alerts': alerts}

    # ----------------------------------------------------------------
    # LLM context generation (the key innovation)
    # ----------------------------------------------------------------

    def build_system_prompt(self) -> str:
        """Build the structured system prompt encoding all causal knowledge.
        
        This is the core PF8 innovation: we don't just describe the plasma —
        we encode the CAUSAL STRUCTURE so the LLM can reason at Pearl's
        Ladder Levels 2–3.
        """
        sections = []

        # Identity
        sections.append(
            "You are FusionMind Causal Copilot — an AI assistant for tokamak "
            "plasma control that reasons using Pearl's causal inference framework. "
            "You have access to the discovered causal graph (DAG), structural "
            "causal model (SCM) equations, and current plasma state. You can "
            "answer interventional questions (what happens if we DO X?) and "
            "counterfactual questions (what WOULD have happened if X had been Y?)."
        )

        # Causal graph
        if self.dag:
            sections.append(self._format_dag_section())

        # SCM equations
        if self.scm:
            sections.append(self._format_scm_section())

        # Current state
        if self.current_state:
            sections.append(self._format_state_section())

        # Safety
        safety = self.check_safety()
        if safety['alerts']:
            sections.append(self._format_safety_section(safety))

        # Intervention history
        if self.intervention_history:
            sections.append(self._format_history_section())

        # Reasoning instructions
        sections.append(self._format_reasoning_instructions())

        return "\n\n".join(sections)

    def build_query_context(self, query: str) -> str:
        """Build query-specific context by analyzing which variables are relevant."""
        relevant_vars = self._extract_variables_from_query(query)
        context_parts = []

        if relevant_vars:
            # Add relevant causal paths
            for i, v1 in enumerate(relevant_vars):
                for v2 in relevant_vars[i + 1:]:
                    paths_fwd = self.find_all_paths(v1, v2)
                    paths_rev = self.find_all_paths(v2, v1)
                    if paths_fwd:
                        for p in paths_fwd:
                            context_parts.append(f"Causal path: {' → '.join(p)}")
                    if paths_rev:
                        for p in paths_rev:
                            context_parts.append(f"Causal path: {' → '.join(p)}")

            # Add confounders
            for i, v1 in enumerate(relevant_vars):
                for v2 in relevant_vars[i + 1:]:
                    confounders = self.get_confounders(v1, v2)
                    if confounders:
                        context_parts.append(
                            f"Confounders between {v1} and {v2}: {', '.join(confounders)}"
                        )

        if context_parts:
            return "Relevant causal context:\n" + "\n".join(context_parts)
        return ""

    def _extract_variables_from_query(self, query: str) -> List[str]:
        """Extract plasma variable references from natural language query."""
        aliases = {
            'temperature': ['T_e', 'T_i'],
            'electron temperature': ['T_e'],
            'ion temperature': ['T_i'],
            'density': ['n_e'],
            'current': ['I_p'],
            'plasma current': ['I_p'],
            'beta': ['beta_N'],
            'safety factor': ['q95'],
            'stored energy': ['W_MHD'],
            'radiation': ['P_rad'],
            'nbi': ['P_NBI'],
            'ecrh': ['P_ECRH'],
            'icrh': ['P_ICRH'],
            'gas': ['gas_puff'],
            'heating': ['P_NBI', 'P_ECRH', 'P_ICRH'],
            'disruption': ['beta_N', 'q95', 'n_e', 'MHD_activity'],
            'confinement': ['W_MHD', 'tau_E', 'H98'],
            'power': ['P_NBI', 'P_ECRH', 'P_ICRH', 'P_rad'],
            'elongation': ['kappa'],
            'impurity': ['Z_eff', 'P_rad'],
            'te': ['T_e'],
            'ti': ['T_i'],
            'ne': ['n_e'],
            'pnbi': ['P_NBI'],
            'q95': ['q95'],
            'wmhd': ['W_MHD'],
            'betan': ['beta_N'],
        }
        query_lower = query.lower()
        found = set()
        # Direct variable name match
        for var in self.variable_names:
            if var.lower() in query_lower:
                found.add(var)
        # Alias match
        for alias, vars_list in aliases.items():
            if alias in query_lower:
                for v in vars_list:
                    if v in self.variable_names:
                        found.add(v)
        return list(found)

    # ----------------------------------------------------------------
    # Formatting helpers
    # ----------------------------------------------------------------

    def _format_dag_section(self) -> str:
        lines = ["CAUSAL GRAPH (Discovered by CPDE v3.2):"]
        for cause, edges in sorted(self.dag.items()):
            for e in edges:
                sign = "+" if e['weight'] > 0 else "−"
                lines.append(f"  {cause} →({sign}{abs(e['weight']):.2f}) {e['effect']}")
        lines.append(f"Total edges: {sum(len(v) for v in self.dag.values())}")
        return "\n".join(lines)

    def _format_scm_section(self) -> str:
        lines = ["STRUCTURAL CAUSAL MODEL (SCM equations):"]
        for var, eq in sorted(self.scm.items()):
            lines.append(f"  {var} = {eq}")
        return "\n".join(lines)

    def _format_state_section(self) -> str:
        lines = ["CURRENT PLASMA STATE:"]
        for var, val in sorted(self.current_state.items()):
            lines.append(f"  {var} = {val:.4g}")
        return "\n".join(lines)

    def _format_safety_section(self, safety: Dict) -> str:
        lines = ["SAFETY STATUS:"]
        for alert in safety['alerts']:
            lines.append(f"  [{alert['level']}] {alert['message']}")
        return "\n".join(lines)

    def _format_history_section(self) -> str:
        lines = [f"INTERVENTION HISTORY (last {min(5, len(self.intervention_history))}):"]
        for entry in self.intervention_history[-5:]:
            lines.append(
                f"  do({entry['variable']} = {entry['value']:.3g}) → "
                + ", ".join(f"{k}={v:.3g}" for k, v in list(entry['result'].items())[:4])
            )
        return "\n".join(lines)

    def _format_reasoning_instructions(self) -> str:
        return (
            "REASONING INSTRUCTIONS:\n"
            "When answering questions, follow this protocol:\n"
            "1. IDENTIFY the query type: Observation (Level 1), Intervention (Level 2), "
            "or Counterfactual (Level 3).\n"
            "2. TRACE causal paths in the DAG relevant to the question.\n"
            "3. CHECK for confounders — warn if correlation ≠ causation.\n"
            "4. USE SCM equations for quantitative estimates.\n"
            "5. VERIFY against physics constraints and safety limits.\n"
            "6. If a Simpson's Paradox risk exists, explicitly flag it.\n"
            "7. Provide a CONFIDENCE LEVEL (high/medium/low) with reasoning.\n"
            "\n"
            "For interventional queries: Use do-calculus. State P(Y|do(X=x)).\n"
            "For counterfactual queries: Use abduction→action→prediction.\n"
            "For hypothesis generation: Propose testable causal hypotheses with "
            "expected edges, signs, and suggested experiments."
        )

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialize context to dict (for JSON transport to frontend)."""
        return {
            'dag': self.dag,
            'scm': self.scm,
            'variable_names': self.variable_names,
            'current_state': self.current_state,
            'intervention_history': self.intervention_history,
            'safety': self.check_safety(),
            'system_prompt': self.build_system_prompt(),
        }

    @classmethod
    def from_fusionmind(cls, cpde_results: Dict, scm_obj=None,
                        state: Optional[Dict] = None):
        """Build CausalContext from FusionMind pipeline outputs.
        
        Args:
            cpde_results: Output from EnsembleCPDE.discover()
            scm_obj: Fitted PlasmaSCM instance
            state: Current plasma state dict
        """
        ctx = cls()
        ctx.variable_names = cpde_results.get('variable_names', [])

        # Extract edges from results
        edges = cpde_results.get('edges', [])
        if edges:
            ctx.set_dag(edges)

        # Extract SCM equations
        if scm_obj is not None and hasattr(scm_obj, 'get_equations'):
            ctx.set_scm_equations(scm_obj.get_equations())

        if state:
            ctx.set_state(state)

        return ctx
