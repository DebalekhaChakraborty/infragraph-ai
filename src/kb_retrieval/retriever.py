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

    # Optional reranking: boost items whose applies_to_diagrams / applies_to_alert_types
    # overlap with the context
    evidence = _rerank_by_overlap(evidence, context)

    return evidence[:top_k]


def _rerank_by_overlap(evidence: list[dict], context: dict) -> list[dict]:
    """
    Boost evidence items whose KB metadata overlaps with context fields.

    Overlap boost is additive, applied on top of the embedding score,
    then the list is re-sorted descending.
    """
    context_diagrams = set(str(d).lower() for d in (context.get("impacted_diagrams", []) or []))
    context_diagrams.add(str(context.get("root_cause_diagram", "")).lower())

    alert_types_raw: list[str] = []
    for ev in (context.get("alert_timeline", []) or []):
        if isinstance(ev, dict):
            a = str(ev.get("alert_type", ev.get("alert_message", ""))).strip().lower()
            if a:
                alert_types_raw.append(a)
    context_alert_types = set(alert_types_raw)

    for item in evidence:
        meta    = item.get("metadata", {})
        boost   = 0.0

        kb_diagrams    = set(str(d).lower() for d in _ensure_list(meta.get("applies_to_diagrams")))
        kb_alert_types = set(str(a).lower() for a in _ensure_list(meta.get("applies_to_alert_types")))

        diag_overlap  = len(context_diagrams  & kb_diagrams)
        alert_overlap = len(context_alert_types & kb_alert_types)

        boost += diag_overlap  * 0.05
        boost += alert_overlap * 0.05

        item["score"] = round(min(1.0, item["score"] + boost), 4)

    return sorted(evidence, key=lambda x: x["score"], reverse=True)


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
