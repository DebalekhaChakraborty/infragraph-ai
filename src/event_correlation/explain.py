"""
explain.py — Plain-English explanations for alert pairs and clusters.

explain_pair(alert_a, alert_b, feat_a, feat_b, score) -> list[str]
explain_cluster(cluster_id, alerts_in_cluster, features_by_id)  -> list[str]
"""
from __future__ import annotations

_SEV_RANK: dict[str, int] = {
    "critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0,
}


def explain_pair(
    alert_a: dict,
    alert_b: dict,
    feat_a: dict,
    feat_b: dict,
    score: float,
) -> list[str]:
    """Return human-readable reasons why two alerts are correlated."""
    reasons: list[str] = []

    t_a   = feat_a.get("_timestamp_offset", 0)
    t_b   = feat_b.get("_timestamp_offset", 0)
    t_diff = abs(t_a - t_b)
    if t_diff <= 2:
        reasons.append(f"Co-occurred simultaneously (t-offset delta {int(t_diff)} min)")
    elif t_diff <= 10:
        reasons.append(f"Close temporal proximity — {int(t_diff)} min apart")
    elif t_diff <= 20:
        reasons.append(f"Near-simultaneous events — {int(t_diff)} min apart")

    if feat_a.get("diagram_hash") == feat_b.get("diagram_hash"):
        diag = feat_a.get("_diagram") or "same diagram"
        reasons.append(f"Same topology domain: {diag}")
    elif feat_a.get("service_hash") == feat_b.get("service_hash"):
        svc = feat_a.get("_service") or "same service layer"
        if svc:
            reasons.append(f"Same service layer: {svc}")

    if feat_a.get("scenario_hash") == feat_b.get("scenario_hash"):
        reasons.append("Same scenario — correlated topology scope")

    gnn_a = feat_a.get("gnn_proximity", 0)
    gnn_b = feat_b.get("gnn_proximity", 0)
    if gnn_a >= 0.8 or gnn_b >= 0.8:
        reasons.append("Alert node is the GNN predicted root-cause candidate")
    elif gnn_a > 0 or gnn_b > 0:
        reasons.append("One or both nodes are near GNN predicted root cause")

    sev_a = feat_a.get("_severity", "")
    sev_b = feat_b.get("_severity", "")
    if sev_a == sev_b and sev_a in ("critical", "high"):
        reasons.append(f"Both alerts are {sev_a} severity")
    elif _SEV_RANK.get(sev_a, 0) >= 3 and _SEV_RANK.get(sev_b, 0) >= 3:
        reasons.append("Both alerts are high or critical severity")

    nt_a = feat_a.get("_node_type", "")
    nt_b = feat_b.get("_node_type", "")
    if nt_a and nt_b and nt_a == nt_b:
        reasons.append(f"Same node type: {nt_a}")

    if not reasons:
        reasons.append(f"Graph-aware similarity score: {score:.2f}")
    return reasons


def explain_cluster(
    cluster_id: str,
    alerts_in_cluster: list[dict],
    features_by_id: dict[str, dict],
) -> list[str]:
    """Return cluster-level correlation explanation bullets."""
    if not alerts_in_cluster:
        return ["No alerts in cluster"]

    reasons: list[str] = []

    diagrams = list(dict.fromkeys(
        a.get("diagram", a.get("diagram_id", ""))
        for a in alerts_in_cluster
        if a.get("diagram", a.get("diagram_id"))
    ))
    if len(diagrams) > 1:
        reasons.append(f"Cross-diagram scope: {', '.join(diagrams[:3])}")
    elif diagrams:
        reasons.append(f"Single-diagram cluster: {diagrams[0]}")

    sevs    = [a.get("severity", "low") for a in alerts_in_cluster]
    max_sev = max(sevs, key=lambda s: _SEV_RANK.get(s, 0))
    reasons.append(f"Highest severity in cluster: {max_sev}")

    scenarios = list(dict.fromkeys(
        a.get("scenario_id", "") for a in alerts_in_cluster if a.get("scenario_id")
    ))
    if len(scenarios) > 1:
        reasons.append(f"Spans {len(scenarios)} scenarios")

    gnn_alerts = [
        a for a in alerts_in_cluster
        if features_by_id.get(a.get("alert_id", ""), {}).get("gnn_proximity", 0) > 0
    ]
    if gnn_alerts:
        reasons.append(f"{len(gnn_alerts)} alert(s) near GNN predicted root-cause node")

    ntypes = list(dict.fromkeys(
        a.get("node_type", "unknown") for a in alerts_in_cluster
    ))
    if ntypes:
        reasons.append(f"Node types involved: {', '.join(ntypes[:4])}")

    # Timing window
    offsets = [float(a.get("timestamp_offset", 0)) for a in alerts_in_cluster]
    span_min = int(max(offsets) - min(offsets)) if len(offsets) > 1 else 0
    if span_min == 0:
        reasons.append(f"{len(alerts_in_cluster)} alert(s) co-occurred at same timestamp")
    else:
        reasons.append(f"{len(alerts_in_cluster)} alert(s) span {span_min} min window")

    return reasons
