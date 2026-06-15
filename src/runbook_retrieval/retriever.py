"""
retriever.py — Candidate runbook retrieval from the in-code catalog.

retrieve_candidate_runbooks(...) -> list[dict]
    Scores and returns candidate runbooks based on node type, alert types,
    impacted diagram count, and root-cause string matching.
"""
from __future__ import annotations

from .kb_schema import RUNBOOK_CATALOG

_DOMAIN_FOR_NODE_TYPE: dict[str, str] = {
    "firewall":      "firewall",
    "router":        "router",
    "wan":           "wan",
    "switch":        "router",
    "load_balancer": "load_balancer",
    "database":      "database",
    "server":        "server",
    "compute":       "server",
    "storage":       "storage",
    "gateway":       "shared_services",
    "service":       "shared_services",
    "dns":           "shared_services",
}


def retrieve_candidate_runbooks(
    root_cause: str = "",
    root_cause_diagram: str = "",
    node_type: str = "",
    alert_timeline: list[dict] | None = None,
    impacted_diagrams: list[str] | None = None,
    evidence_summary: list[str] | None = None,
) -> list[dict]:
    """
    Score every runbook in the catalog and return the top 5.

    Scoring:
      +3  — node_type is in applicable_node_types
      +2  — each matching alert_type in the timeline
      +2  — cross-diagram incident AND runbook domain == "enterprise"
      +1  — root_cause string contains the runbook domain
      +1  — root_cause_diagram contains the runbook domain
    """
    timeline   = list(alert_timeline or [])
    imp_diags  = list(impacted_diagrams or [])
    is_cross   = len(imp_diags) > 1

    # Collect alert types from timeline
    al_types: set[str] = set()
    for ev in timeline:
        at = ev.get("alert_type", "") or ev.get("type", "")
        if at:
            al_types.add(at.lower())

    # Normalise node type (infer from root_cause if missing)
    nt = (node_type or "").lower()
    if not nt and root_cause:
        rc_lower = root_cause.lower()
        for kw in _DOMAIN_FOR_NODE_TYPE:
            if kw in rc_lower:
                nt = kw
                break

    scored: list[dict] = []
    for rb in RUNBOOK_CATALOG:
        score = 0

        if nt and nt in [t.lower() for t in rb.get("applicable_node_types", [])]:
            score += 3

        rb_types = {a.lower() for a in rb.get("applicable_alert_types", [])}
        score += 2 * len(al_types & rb_types)

        if is_cross and rb.get("domain") == "enterprise":
            score += 2

        rc_lower = root_cause.lower()
        dom      = rb.get("domain", "").lower()
        if dom and dom in rc_lower:
            score += 1
        if root_cause_diagram and dom in root_cause_diagram.lower():
            score += 1

        # Always keep the cross-diagram enterprise runbook when relevant
        if score > 0 or (is_cross and rb.get("domain") == "enterprise"):
            scored.append({"runbook": rb, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate
    seen: set[str] = set()
    result: list[dict] = []
    for item in scored[:7]:
        rb = item["runbook"]
        if rb["runbook_id"] not in seen:
            seen.add(rb["runbook_id"])
            result.append(rb)
        if len(result) >= 5:
            break
    return result
