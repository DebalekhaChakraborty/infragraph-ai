"""
reranker.py — Rerank candidate runbooks using RCA context.
"""
from __future__ import annotations

_SEV_RANK: dict[str, int] = {
    "critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0,
}


def rerank_runbooks(candidates: list[dict], rca_context: dict) -> list[dict]:
    """
    Rerank candidate runbooks using RCA context signals.

    Parameters
    ----------
    candidates  : list[dict] — runbook dicts from retriever
    rca_context : dict — may include: root_cause, rca_source, confidence,
                  impacted_diagrams, severity, node_type

    Returns
    -------
    list[dict] — reranked runbooks (highest relevance first)
    """
    if not candidates:
        return []

    src        = (rca_context.get("rca_source") or "").lower()
    is_gnn     = "gnn" in src
    severity   = rca_context.get("severity", "medium")
    sev_rank   = _SEV_RANK.get(severity, 2)
    confidence = float(rca_context.get("confidence") or 0.0)
    node_type  = (rca_context.get("node_type") or "").lower()
    n_imp      = len(rca_context.get("impacted_diagrams") or [])

    scored: list[dict] = []
    for rb in candidates:
        s = 0.0

        # GNN-backed high-confidence → prefer automation-eligible runbooks
        if is_gnn and confidence >= 0.75:
            s += 0.25

        # Exact node type match
        rb_types = [t.lower() for t in rb.get("applicable_node_types", [])]
        if node_type and node_type in rb_types:
            s += 0.40

        # Severity alignment
        rb_risk  = rb.get("risk_level", "medium").lower()
        rb_srank = _SEV_RANK.get(rb_risk, 2)
        if abs(sev_rank - rb_srank) <= 1:
            s += 0.15

        # Cross-diagram preference
        if n_imp > 1 and rb.get("domain") == "enterprise":
            s += 0.20

        # Automation bonus
        if confidence >= 0.80 and rb.get("automation_eligible"):
            s += 0.10

        scored.append({"runbook": rb, "rerank_score": round(s, 4)})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return [item["runbook"] for item in scored]
