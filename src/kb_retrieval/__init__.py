"""
kb_retrieval — SOP/KB/Known-Resolution retrieval for remediation grounding.

Public API
----------
build_kb_index(...)           build or update the ChromaDB vector index
retrieve_kb_evidence(...)     query the index and return evidence dicts
build_query_from_rca_context  build a search query string from a context dict
"""
from .indexer import build_kb_index
from .retriever import build_query_from_rca_context, retrieve_kb_evidence

__all__ = [
    "build_kb_index",
    "build_query_from_rca_context",
    "retrieve_kb_evidence",
]
