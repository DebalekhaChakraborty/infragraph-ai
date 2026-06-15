"""
runbook_retrieval — SOP/runbook retrieval, reranking, and policy filtering.

Public API
----------
retrieve_candidate_runbooks(...)  → list[dict]
rerank_runbooks(...)              → list[dict]
apply_runbook_policy(...)         → list[dict]
RUNBOOK_CATALOG                   → list[dict]
get_runbook_by_id(runbook_id)     → dict | None
"""

from .retriever     import retrieve_candidate_runbooks
from .reranker      import rerank_runbooks
from .policy_filter import apply_runbook_policy
from .kb_schema     import RUNBOOK_CATALOG, get_runbook_by_id

__all__ = [
    "retrieve_candidate_runbooks",
    "rerank_runbooks",
    "apply_runbook_policy",
    "RUNBOOK_CATALOG",
    "get_runbook_by_id",
]
