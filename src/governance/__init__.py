"""
governance — Evidence critic and governance validator for Agentic Ops.

Public API
----------
validate_rca_and_remediation(agent_run, remediation_plan, graph_context)
    → governance review dict
"""

from .evidence_critic import validate_rca_and_remediation

__all__ = ["validate_rca_and_remediation"]
