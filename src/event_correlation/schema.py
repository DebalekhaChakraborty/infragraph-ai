"""
schema.py — Cluster output schema builders and integrity enforcement.

No remediation, root-cause, or evaluation fields are ever produced here.
"""
from __future__ import annotations

import hashlib

# Keys that must NEVER appear in any cluster output file
FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "root_cause",
    "root_cause_node",
    "root_cause_diagram",
    "expected_root_cause",
    "correct",
    "remediation",
    "remediation_steps",
    "validation_steps",
    "rollback",
    "rollback_steps",
    "resolution",
    "resolution_steps",
    "itsm_ticket_summary",
    "recommended_actions",
    "evaluation",
    "ground_truth_node",
    "correct_top1",
    "correct_top_k",
    "reciprocal_rank",
})

_CORRELATION_ROLES: frozenset[str] = frozenset({
    "cluster_seed",
    "propagation_signal",
    "peer_signal",
    "noise_candidate",
})

_EVIDENCE_STAGES: frozenset[str] = frozenset({
    "temporal_correlation",
    "alert_sequence",
    "topology_proximity",
    "cross_diagram_correlation",
    "propagation_hypothesis",
})


def make_event_in_cluster(event: dict, correlation_role: str) -> dict:
    """Produce a cluster-event dict containing only observable fields."""
    if correlation_role not in _CORRELATION_ROLES:
        raise ValueError(f"Unknown correlation role: {correlation_role!r}")
    return {
        "event_id":         event.get("event_id", ""),
        "node":             event.get("node", ""),
        "alert_type":       event.get("alert_type", ""),
        "severity":         event.get("severity", ""),
        "time_offset_min":  event.get("time_offset_min", 0),
        "diagram_id":       event.get("diagram_id", ""),
        "correlation_role": correlation_role,
    }


def make_causal_evidence_item(
    evidence_id: str,
    stage: str,
    summary: str,
    supporting_events: list[str],
    supporting_nodes: list[str],
    confidence: float,
) -> dict:
    """Produce one causal evidence item.  Stage must be one of _EVIDENCE_STAGES."""
    if stage not in _EVIDENCE_STAGES:
        raise ValueError(f"Unknown evidence stage: {stage!r}")
    return {
        "evidence_id":       evidence_id,
        "stage":             stage,
        "summary":           summary,
        "supporting_events": list(supporting_events),
        "supporting_nodes":  list(supporting_nodes),
        "confidence":        round(float(confidence), 4),
    }


def _cluster_fingerprint(events: list[dict]) -> str:
    atypes = sorted({e.get("alert_type", "") for e in events if e.get("alert_type")})
    nodes  = sorted({e.get("node", "")       for e in events if e.get("node")})[:3]
    raw    = "|".join(atypes) + "@" + "|".join(nodes)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _time_window(events: list[dict]) -> dict[str, int]:
    times = [e.get("time_offset_min", 0) for e in events]
    if not times:
        return {"start_offset_min": 0, "end_offset_min": 0}
    return {"start_offset_min": int(min(times)), "end_offset_min": int(max(times))}


def make_cluster(
    cluster_id: str,
    case_id: str,
    scenario_id: str,
    mode: str,
    diagram_scope: list[str],
    cluster_score: float,
    correlation_dimensions: dict[str, float],
    correlation_reasons: list[str],
    cluster_events: list[dict],
    causal_evidence: list[dict],
) -> dict:
    """Assemble and return a validated cluster dict."""
    obj: dict = {
        "cluster_id":             cluster_id,
        "case_id":                case_id,
        "scenario_id":            scenario_id,
        "mode":                   mode,
        "diagram_scope":          list(diagram_scope),
        "event_count":            len(cluster_events),
        "time_window":            _time_window(cluster_events),
        "cluster_score":          round(float(cluster_score), 4),
        "correlation_dimensions": {k: round(float(v), 4) for k, v in correlation_dimensions.items()},
        "correlation_reasons":    list(correlation_reasons),
        "events":                 cluster_events,
        "cluster_fingerprint":    _cluster_fingerprint(cluster_events),
        "causal_evidence":        causal_evidence,
    }
    # Integrity guard: fail fast on any forbidden key
    for key in FORBIDDEN_KEYS:
        if key in obj:
            raise ValueError(f"Cluster schema violation — forbidden key: {key!r}")
    return obj


def make_cluster_output(
    case_id: str,
    scenario_id: str,
    mode: str,
    clusters: list[dict],
) -> dict:
    """Top-level output wrapper for a case's cluster file."""
    return {
        "case_id":       case_id,
        "scenario_id":   scenario_id,
        "mode":          mode,
        "cluster_count": len(clusters),
        "clusters":      clusters,
    }
