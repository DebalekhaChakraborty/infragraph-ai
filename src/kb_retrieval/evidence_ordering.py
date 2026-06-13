"""
evidence_ordering.py — Domain-first KB evidence ordering.

Ensures root-cause-domain SOP/KR evidence appears before neighboring or
downstream SOP evidence in both the user message content (retrieved_kb_evidence)
and the assistant target output (evidence_ids_used, evidence_from_graph).

Public API
----------
infer_domain(root_cause: str) -> str
sort_kb_evidence_domain_first(kb_evidence, root_cause) -> list[dict]
"""
from __future__ import annotations

# Lowercase substring patterns matched against root_cause.lower()
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "firewall": [
        "dc-fw", "fw-", "-fw", "firewall", "packet_drop", "packet drop", "acl",
    ],
    "load_balancer": [
        "app-lb", "-lb-", "-lb", "lb-", "load_balancer", "load balancer",
        "backend_pool", "backend pool", "health_probe", "health probe", "unhealthy",
    ],
    "database": [
        "db-master", "db_master", "db-", "-db", "database", "connection_pool",
        "connection pool", "replication", "replica",
    ],
    "wan": [
        "wan-pe", "wan_pe", "pe-0", "pe-1", "pe-", "-pe", "wan",
        "bgp", "ospf", "mpls", "circuit", "carrier", "sfp", "link_down", "link down",
    ],
}

# KB-ID prefixes that are primary for each domain.
# Checked against metadata.kb_id (e.g., "SOP-APP-LB-001") or the kb_id
# portion of evidence_id after stripping the "KB-" prefix.
DOMAIN_PRIMARY_PREFIXES: dict[str, tuple[str, ...]] = {
    "firewall":      ("SOP-DC-FW", "KR-DC-FW", "DC-FW"),
    "load_balancer": ("SOP-APP-LB", "KR-APP-LB", "APP-LB"),
    "database":      ("SOP-DB", "KR-DB"),
    "wan":           ("SOP-WAN", "KR-WAN"),
}

# Expected first-KB-ID prefixes for the validation check (includes "KB-").
# Used in training-data scripts to assert domain-first ordering.
DOMAIN_EXPECTED_FIRST_KB: dict[str, tuple[str, ...]] = {
    domain: tuple(f"KB-{p}" for p in prefixes)
    for domain, prefixes in DOMAIN_PRIMARY_PREFIXES.items()
}


def infer_domain(root_cause: str) -> str:
    """Return the infra domain inferred from a root-cause node ID. Empty string if unknown."""
    rc = root_cause.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in rc:
                return domain
    return ""


def _kb_id_of(item: dict) -> str:
    """
    Extract the kb_id portion from a KB evidence item for prefix matching.

    Handles both raw retriever format (metadata.kb_id = "SOP-APP-LB-001")
    and the evidence_id fallback ("KB-SOP-APP-LB-001-000" → "SOP-APP-LB-001-000").
    """
    meta = item.get("metadata") or {}
    kb_id = meta.get("kb_id", "")
    if kb_id:
        return kb_id
    eid = str(item.get("evidence_id", "") or meta.get("evidence_id", ""))
    if eid.upper().startswith("KB-"):
        return eid[3:]  # strip "KB-" prefix
    return eid


def _domain_priority(item: dict, primary_prefixes: tuple[str, ...]) -> int:
    """0 = domain-primary evidence, 1 = everything else."""
    kb_id = _kb_id_of(item)
    for prefix in primary_prefixes:
        if kb_id.startswith(prefix):
            return 0
    return 1


def sort_kb_evidence_domain_first(
    kb_evidence: list[dict],
    root_cause: str,
) -> list[dict]:
    """
    Return a new list with domain-primary KB evidence sorted to the front.

    Uses a stable sort so that within each priority tier the original
    relative order (e.g., reranker score order) is preserved.

    Does not mutate the input list.
    If the domain cannot be inferred, the original order is returned unchanged.
    """
    if not kb_evidence:
        return list(kb_evidence)
    domain = infer_domain(root_cause)
    if not domain:
        return list(kb_evidence)
    primary_prefixes = DOMAIN_PRIMARY_PREFIXES.get(domain, ())
    if not primary_prefixes:
        return list(kb_evidence)
    return sorted(
        kb_evidence,
        key=lambda item: _domain_priority(item, primary_prefixes),
    )


def apply_domain_first_ordering(context: dict, root_cause: str) -> None:
    """
    Sort retrieved_kb_evidence and retrieved_graph_memory_evidence in-place
    so domain-primary KB evidence appears before neighboring/downstream evidence.

    Mutates context in place. Call this BEFORE generate_template_remediation().
    """
    sorted_kb = sort_kb_evidence_domain_first(
        context.get("retrieved_kb_evidence", []) or [], root_cause
    )
    context["retrieved_kb_evidence"] = sorted_kb

    # Preserve non-KB items (e.g., graph memory summaries) at their original position;
    # replace only the KB items with the sorted set.
    rgme = context.get("retrieved_graph_memory_evidence", []) or []
    non_kb = [
        item for item in rgme
        if not (isinstance(item, dict) and str(item.get("evidence_id", "")).startswith("KB-"))
    ]
    context["retrieved_graph_memory_evidence"] = non_kb + sorted_kb
