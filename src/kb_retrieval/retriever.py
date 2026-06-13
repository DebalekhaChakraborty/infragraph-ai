"""
KB evidence retriever.

build_query_from_rca_context : build a search query string from a context dict
retrieve_kb_evidence          : query the ChromaDB index and return evidence dicts
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import DEFAULT_COLLECTION, DEFAULT_INDEX_DIR


def build_query_from_rca_context(context: dict) -> str:
    """
    Build a free-text search query from a remediation context dict.

    Uses root cause, diagram, alert types, node types, correlation reasons,
    and causal evidence summaries to produce a rich query string.
    """
    parts: list[str] = []

    root_cause = str(context.get("root_cause", "")).strip()
    if root_cause:
        parts.append(f"root cause node: {root_cause}")

    rc_diagram = str(context.get("root_cause_diagram", "") or context.get("selected_diagram_id", "")).strip()
    if rc_diagram:
        parts.append(f"topology diagram: {rc_diagram}")

    imp_diagrams = context.get("impacted_diagrams", []) or []
    if imp_diagrams:
        parts.append(f"impacted diagrams: {', '.join(str(d) for d in imp_diagrams[:4])}")

    # Alert types from timeline
    alert_types: list[str] = []
    for ev in (context.get("alert_timeline", []) or []):
        if isinstance(ev, dict):
            atype = str(ev.get("alert_type", ev.get("alert_message", ""))).strip()
            if atype and atype not in alert_types:
                alert_types.append(atype)
    if alert_types:
        parts.append(f"alert types: {', '.join(alert_types[:5])}")

    # Severities (highest first)
    sev_order = {"critical": 4, "high": 3, "warning": 2, "medium": 1, "low": 0}
    severities = sorted(
        {str(ev.get("severity", "")).lower() for ev in (context.get("alert_timeline", []) or []) if isinstance(ev, dict) and ev.get("severity")},
        key=lambda s: -sev_order.get(s, -1),
    )
    if severities:
        parts.append(f"severity: {', '.join(severities[:2])}")

    # Node types from candidate ranking
    node_types: list[str] = []
    for c in (context.get("candidate_ranking", []) or []):
        if isinstance(c, dict):
            nt = str(c.get("node_type", "")).strip()
            if nt and nt not in node_types:
                node_types.append(nt)
    if node_types:
        parts.append(f"node types: {', '.join(node_types[:3])}")

    # Correlation reasons
    for reason in (context.get("correlation_reasons", []) or [])[:3]:
        parts.append(str(reason)[:200])

    # Causal evidence summaries
    for item in (context.get("causal_evidence", []) or [])[:2]:
        if isinstance(item, dict):
            summary = str(item.get("summary", "")).strip()
            if summary:
                parts.append(summary[:200])

    return "; ".join(parts)


def retrieve_kb_evidence(
    *,
    context: dict,
    index_dir: Path | str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
) -> list[dict]:
    """
    Query the KB index and return a ranked list of evidence dicts.

    Each returned dict matches the shape expected by prompt_builder
    ``_fmt_retrieved_evidence`` and template_mode KB helpers:

      evidence_id : str  e.g. "KB-SOP-DC-FW-001-000"
      text        : str  (chunk text for prompt injection)
      score       : float (relevance score, higher = more relevant)
      metadata    : dict  (kb_id, title, doc_type, section, applies_to_*, ...)

    Returns an empty list (not an exception) if chromadb/sentence-transformers
    are not installed or the index does not exist.
    """
    try:
        import chromadb
    except ImportError:
        return []

    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        return []

    from .schema import DEFAULT_EMBED_MODEL

    _idx_dir = Path(index_dir or DEFAULT_INDEX_DIR)
    if not _idx_dir.exists():
        return []

    try:
        client = chromadb.PersistentClient(path=str(_idx_dir))
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=DEFAULT_EMBED_MODEL)
        collection = client.get_collection(name=collection_name, embedding_function=embed_fn)
    except Exception:
        return []

    query_text = build_query_from_rca_context(context)
    if not query_text.strip():
        return []

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    evidence: list[dict] = []
    docs      = (results.get("documents") or [[]])[0]
    metas     = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for doc_text, meta, dist in zip(docs, metas, distances):
        # Convert L2 distance to a 0–1 relevance score
        score = round(1.0 / (1.0 + float(dist)), 4)

        # Deserialize any JSON-serialized list fields
        deserialized_meta = _deserialize_metadata(meta or {})
        evidence_id = deserialized_meta.get("evidence_id", "KB-unknown")

        evidence.append({
            "evidence_id": evidence_id,
            "text":        doc_text or "",
            "score":       score,
            "metadata":    deserialized_meta,
        })

    # Rerank with domain-aware strong boosts before returning top_k
    evidence = _rerank_by_overlap(evidence, context)

    return evidence[:top_k]


# ── Domain inference ──────────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "firewall": [
        "dc-fw", "fw-", "-fw", "firewall", "packet_drop", "packet drop", "acl",
    ],
    "load_balancer": [
        "app-lb", "-lb-", "-lb", "lb-", "load_balancer", "load balancer",
        "backend_pool", "backend pool", "unhealthy", "health_probe", "health probe",
    ],
    "database": [
        "db-master", "db_master", "db-", "-db", "database", "connection_pool",
        "connection pool", "replica", "replication",
    ],
    "wan": [
        "wan-pe", "wan_pe", "pe-0", "pe-1", "pe-", "-pe", "wan",
        "bgp", "ospf", "mpls", "circuit", "link_down", "link down", "sfp", "carrier",
    ],
}

_DOMAIN_KB_PREFIXES: dict[str, list[str]] = {
    "firewall":      ["SOP-DC-FW", "DC-FW", "KR-DC-FW", "RB-DC-FW"],
    "load_balancer": ["SOP-APP-LB", "KR-APP-LB", "APP-LB", "RB-APP-LB"],
    "database":      ["SOP-DB", "KR-DB", "RB-DB"],
    "wan":           ["SOP-WAN", "KR-WAN", "RB-WAN"],
}

# Runbook domain → runbook_id prefix for guaranteed inclusion
_DOMAIN_RUNBOOK_IDS: dict[str, str] = {
    "firewall":      "DC-FW",
    "load_balancer": "APP-LB",
    "database":      "DB",
    "wan":           "WAN",
}


def _infer_domain(root_cause: str) -> str:
    """Infer infrastructure domain from a root-cause node ID or description."""
    rc = root_cause.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in rc:
                return domain
    return ""


def _rerank_by_overlap(evidence: list[dict], context: dict) -> list[dict]:
    """
    Boost evidence using domain inference and multi-signal overlap.

    Boost weights (additive on top of embedding score):
      root_cause text in chunk body   +0.30
      root_cause in kb_id / title     +0.25
      domain prefix match             +0.20
      root_cause in evidence_tags     +0.15
      node_type overlap (per match)   +0.10
      diagram overlap (per match)     +0.08
      alert type overlap (per match)  +0.06
      doc_type sop                    +0.05
      doc_type known_resolution       +0.02
      cross-diagram runbook (>=3 diag) -0.10
    """
    root_cause = str(context.get("root_cause", "")).strip().lower()
    domain = _infer_domain(root_cause)
    domain_prefixes = _DOMAIN_KB_PREFIXES.get(domain, [])

    context_diagrams: set[str] = set(
        str(d).lower() for d in (context.get("impacted_diagrams", []) or [])
    )
    rc_diagram = str(context.get("root_cause_diagram", "")).lower()
    if rc_diagram:
        context_diagrams.add(rc_diagram)

    alert_types_raw: list[str] = []
    for ev in (context.get("alert_timeline", []) or []):
        if isinstance(ev, dict):
            a = str(ev.get("alert_type", ev.get("alert_message", ""))).strip().lower()
            if a:
                alert_types_raw.append(a)
    context_alert_types = set(alert_types_raw)

    context_node_types: set[str] = set()
    for c in (context.get("candidate_ranking", []) or []):
        if isinstance(c, dict):
            nt = str(c.get("node_type", "")).strip().lower()
            if nt:
                context_node_types.add(nt)

    for item in evidence:
        meta        = item.get("metadata", {})
        text        = item.get("text", "").lower()
        boost       = 0.0

        kb_id_orig  = str(meta.get("kb_id", ""))
        kb_id       = kb_id_orig.lower()
        title       = str(meta.get("title", "")).lower()
        doc_type    = str(meta.get("doc_type", "")).lower()
        ev_tags     = [str(t).lower() for t in _ensure_list(meta.get("evidence_tags"))]
        kb_diagrams = set(str(d).lower() for d in _ensure_list(meta.get("applies_to_diagrams")))
        kb_alerts   = set(str(a).lower() for a in _ensure_list(meta.get("applies_to_alert_types")))
        kb_ntypes   = set(str(n).lower() for n in _ensure_list(meta.get("applies_to_node_types")))

        # Strong signal: root_cause text appears in chunk body
        if root_cause and root_cause in text:
            boost += 0.30

        # Strong signal: root_cause appears in kb_id or title
        if root_cause and (root_cause in kb_id or root_cause in title):
            boost += 0.25

        # Domain prefix: kb_id starts with one of the inferred domain prefixes
        if domain_prefixes and any(
            kb_id_orig.upper().startswith(p.upper()) for p in domain_prefixes
        ):
            boost += 0.20

        # Root_cause appears in evidence_tags
        if root_cause and any(root_cause in tag or tag in root_cause for tag in ev_tags):
            boost += 0.15

        # Per-match structural overlaps
        for nt in context_node_types:
            if nt in kb_ntypes:
                boost += 0.10

        boost += len(context_diagrams & kb_diagrams) * 0.08
        boost += len(context_alert_types & kb_alerts) * 0.06

        # Doc-type minor bonuses
        if doc_type == "sop":
            boost += 0.05
        elif doc_type == "known_resolution":
            boost += 0.02

        # Domain runbook: strong boost when runbook domain matches inferred domain
        if doc_type == "runbook":
            rb_domain = str(meta.get("domain", "")).lower()
            if rb_domain and domain and rb_domain == domain:
                boost += 0.35
            elif doc_type == "runbook" and len(kb_diagrams) >= 3:
                # Cross-diagram runbook without domain match: mild penalty vs. domain SOPs
                boost -= 0.10

        item["score"] = round(min(1.0, item["score"] + boost), 4)

    return sorted(evidence, key=lambda x: x["score"], reverse=True)


def build_runbook_chain(evidence: list[dict], context: dict) -> list[dict]:
    """
    Extract runbook-type evidence chunks into a structured runbook_chain list.

    Always includes the cross-diagram validation runbook (ENT-XDIAG-001) when
    the context has multiple impacted diagrams, in addition to the domain runbook.

    Each chain item:
      runbook_id        : str
      title             : str
      domain            : str
      source            : str
      approval_required : bool
      automation_eligible: bool
      execution_mode    : str
      tool_name         : str
      connector         : str
      dry_run_supported : bool
      evidence_ids      : list[str]  (RB-* chunk IDs from this runbook)
      sections_retrieved: list[str]  (section headings found in evidence)
    """
    impacted_diagrams = context.get("impacted_diagrams") or []
    is_cross_diagram  = len(impacted_diagrams) >= 2

    # Group runbook chunks by runbook_id
    rb_groups: dict[str, list[dict]] = {}
    for item in evidence:
        meta     = item.get("metadata", {}) if isinstance(item, dict) else {}
        doc_type = str(meta.get("doc_type", "")).lower()
        if doc_type != "runbook":
            continue
        rb_id = str(meta.get("runbook_id", "")).strip()
        if not rb_id:
            continue
        rb_groups.setdefault(rb_id, []).append(item)

    chain: list[dict] = []
    seen_rb_ids: set[str] = set()

    def _add_rb(rb_id: str, chunks: list[dict]) -> None:
        if rb_id in seen_rb_ids or not chunks:
            return
        seen_rb_ids.add(rb_id)
        first_meta = chunks[0].get("metadata", {})
        chain.append({
            "runbook_id":          rb_id,
            "title":               str(first_meta.get("title", "")),
            "domain":              str(first_meta.get("domain", "")),
            "source":              str(first_meta.get("source", "approved_kb_repo")),
            "approval_required":   bool(first_meta.get("approval_required", True)),
            "automation_eligible": bool(first_meta.get("automation_eligible", False)),
            "execution_mode":      str(first_meta.get("execution_mode", "manual")),
            "tool_name":           str(first_meta.get("tool_name", "")),
            "connector":           str(first_meta.get("connector", "")),
            "dry_run_supported":   bool(first_meta.get("dry_run_supported", False)),
            "evidence_ids":        [c.get("evidence_id", "") for c in chunks if c.get("evidence_id")],
            "sections_retrieved":  [
                str(c.get("metadata", {}).get("section", ""))
                for c in chunks if c.get("metadata", {}).get("section")
            ],
        })

    # Add domain-specific runbook first (highest priority)
    root_cause = str(context.get("root_cause", "")).strip()
    domain     = _infer_domain(root_cause)
    domain_rb_prefix = _DOMAIN_RUNBOOK_IDS.get(domain, "")
    for rb_id, chunks in rb_groups.items():
        if domain_rb_prefix and rb_id.startswith(domain_rb_prefix):
            _add_rb(rb_id, chunks)

    # Add remaining retrieved runbooks (excluding cross-diagram, handled separately)
    for rb_id, chunks in rb_groups.items():
        if rb_id != "ENT-XDIAG-001":
            _add_rb(rb_id, chunks)

    # Always append cross-diagram runbook when incident spans >= 2 diagrams
    if is_cross_diagram and "ENT-XDIAG-001" in rb_groups:
        _add_rb("ENT-XDIAG-001", rb_groups["ENT-XDIAG-001"])

    return chain


def _ensure_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [value] if value else []
    return []


def _deserialize_metadata(meta: dict) -> dict:
    """Restore list fields that were JSON-serialized for ChromaDB storage."""
    _LIST_FIELDS = {
        "applies_to_node_types",
        "applies_to_diagrams",
        "applies_to_alert_types",
        "rca_patterns",
        "evidence_tags",
    }
    out = dict(meta)
    for field in _LIST_FIELDS:
        if field in out and isinstance(out[field], str):
            try:
                parsed = json.loads(out[field])
                if isinstance(parsed, list):
                    out[field] = parsed
            except Exception:
                pass
    return out
