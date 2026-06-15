"""
features.py — Alert feature engineering for AI/graph-aware correlation.

build_alert_features(alerts, enterprise_graph_by_scenario, gnn_result_by_scenario)
    → list[dict]  — one feature dict per alert, index-aligned

compute_pairwise_similarity(feat_a, feat_b) → float  — score in [0, 1]

No root-cause labels, remediation steps, or evaluation fields produced here.
"""
from __future__ import annotations

import hashlib
from typing import Any

_SEVERITY_SCORE: dict[str, float] = {
    "critical": 1.0, "high": 0.75, "medium": 0.50, "low": 0.25, "info": 0.10,
}

_NODE_TYPE_IDX: dict[str, int] = {
    "firewall": 0, "router": 1, "switch": 2, "server": 3, "compute": 4,
    "database": 5, "load_balancer": 6, "gateway": 7, "storage": 8, "wan": 9,
    "cloud": 10, "service": 11, "dns": 12, "network": 13, "unknown": 14,
}

# Pairs of node types that are infrastructure-adjacent (higher type compat)
_ADJACENT_TYPES: frozenset[frozenset] = frozenset({
    frozenset({"firewall", "router"}),
    frozenset({"router", "wan"}),
    frozenset({"router", "switch"}),
    frozenset({"switch", "server"}),
    frozenset({"server", "load_balancer"}),
    frozenset({"load_balancer", "database"}),
    frozenset({"server", "storage"}),
    frozenset({"gateway", "service"}),
})


def _hash_str(s: str) -> float:
    """Deterministic float in [0, 1] from a string (categorical encoding)."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _node_degree(node_id: str, enterprise_graph: dict) -> int:
    degree = 0
    all_edges = enterprise_graph.get("edges", []) + enterprise_graph.get("cross_diagram_edges", [])
    for e in all_edges:
        if e.get("source") == node_id or e.get("target") == node_id:
            degree += 1
    return degree


def _cross_diagram_degree(node_id: str, enterprise_graph: dict) -> int:
    degree = 0
    for e in enterprise_graph.get("cross_diagram_edges", []):
        if e.get("source") == node_id or e.get("target") == node_id:
            degree += 1
    return degree


def _gnn_proximity(node_id: str, gnn_result: dict | None) -> float:
    """1.0 if node IS the GNN root cause, else decays by candidate rank."""
    if not gnn_result or not node_id:
        return 0.0
    rc = gnn_result.get("predicted_root_cause") or gnn_result.get("root_cause", "")
    if rc and rc == node_id:
        return 1.0
    candidates = gnn_result.get("top_candidates") or gnn_result.get("ranking") or []
    for i, c in enumerate(candidates[:5]):
        if c.get("node_id") == node_id:
            return max(0.0, 1.0 - i * 0.18)
    return 0.0


def _max_timestamp(alerts: list[dict]) -> float:
    t = max((float(a.get("timestamp_offset", 0)) for a in alerts), default=1.0)
    return t if t > 0 else 1.0


def build_alert_features(
    alerts: list[dict],
    enterprise_graph_by_scenario: dict[str, dict] | None = None,
    gnn_result_by_scenario: dict[str, dict] | None = None,
) -> list[dict]:
    """
    Build a feature vector dict for each alert.
    Returns a list, index-aligned with `alerts`.
    """
    egbs  = enterprise_graph_by_scenario or {}
    grbs  = gnn_result_by_scenario or {}
    t_max = _max_timestamp(alerts)

    feature_list: list[dict] = []
    for al in alerts:
        sid   = al.get("scenario_id", "")
        nid   = al.get("node_id", "")
        eg    = egbs.get(sid, {})
        gnn   = grbs.get(sid)

        t_off = float(al.get("timestamp_offset", 0))
        sev   = al.get("severity", "medium")
        ntype = al.get("node_type", "unknown")
        diag  = al.get("diagram", al.get("diagram_id", ""))
        svc   = al.get("service", "")

        feat: dict[str, Any] = {
            "alert_id":               al.get("alert_id", ""),
            "node_id":                nid,
            "scenario_id":            sid,
            # numeric features
            "severity_score":         _SEVERITY_SCORE.get(sev, 0.25),
            "timestamp_offset_norm":  t_off / t_max,
            "node_type_idx":          _NODE_TYPE_IDX.get(ntype, 14) / 14.0,
            # hash-based categorical encoding
            "node_type_hash":         _hash_str(ntype),
            "diagram_hash":           _hash_str(diag),
            "service_hash":           _hash_str(svc),
            "scenario_hash":          _hash_str(sid),
            # graph topology features
            "node_degree":            min(_node_degree(nid, eg) / 10.0, 1.0),
            "cross_diagram_degree":   min(_cross_diagram_degree(nid, eg) / 5.0, 1.0),
            # GNN proximity feature
            "gnn_proximity":          _gnn_proximity(nid, gnn),
            # raw values for explain (prefixed _)
            "_node_type":             ntype,
            "_diagram":               diag,
            "_service":               svc,
            "_severity":              sev,
            "_timestamp_offset":      t_off,
        }
        feature_list.append(feat)
    return feature_list


def compute_pairwise_similarity(feat_a: dict, feat_b: dict) -> float:
    """
    Score how likely two alerts share a common causal root, in [0, 1].

    Weighted dimensions:
      temporal_close    (0.25): close in time
      same_scenario     (0.22): same scenario (strong signal in demo topology data)
      same_domain       (0.20): same diagram or service layer
      severity_compat   (0.15): compatible severity levels
      gnn_proximity_sum (0.10): both nodes near GNN predicted root cause
      type_compat       (0.08): related infrastructure node types
    """
    # Temporal closeness
    t_diff = abs(feat_a["timestamp_offset_norm"] - feat_b["timestamp_offset_norm"])
    temporal_close = max(0.0, 1.0 - t_diff * 3.0)

    # Same scenario (correlated topology scope)
    same_scenario = 1.0 if feat_a["scenario_hash"] == feat_b["scenario_hash"] else 0.0

    # Same diagram or same service layer
    same_diag    = feat_a["diagram_hash"] == feat_b["diagram_hash"]
    same_service = feat_a["service_hash"]  == feat_b["service_hash"]
    same_domain  = 1.0 if same_diag else (0.5 if same_service else 0.0)

    # Severity compatibility
    sev_diff        = abs(feat_a["severity_score"] - feat_b["severity_score"])
    severity_compat = max(0.0, 1.0 - sev_diff * 2.0)

    # Both near GNN root cause
    gnn_sum = (feat_a["gnn_proximity"] + feat_b["gnn_proximity"]) / 2.0

    # Infrastructure type adjacency
    nt_a = feat_a.get("_node_type", "")
    nt_b = feat_b.get("_node_type", "")
    if nt_a == nt_b:
        type_compat = 1.0
    elif frozenset({nt_a, nt_b}) in _ADJACENT_TYPES:
        type_compat = 0.75
    else:
        type_diff   = abs(feat_a["node_type_idx"] - feat_b["node_type_idx"])
        type_compat = max(0.0, 1.0 - type_diff * 2.0)

    score = (
        0.25 * temporal_close
        + 0.22 * same_scenario
        + 0.20 * same_domain
        + 0.15 * severity_compat
        + 0.10 * gnn_sum
        + 0.08 * type_compat
    )
    return round(min(1.0, max(0.0, score)), 4)
