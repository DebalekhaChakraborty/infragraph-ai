"""
inference.py — Public API for AI/graph-aware alert correlation.

correlate_alerts(alerts, enterprise_graph_by_scenario, gnn_result_by_scenario, threshold)
    Returns enriched correlated_alerts + clusters dict.

No root-cause labels, remediation steps, or evaluation fields produced.
"""
from __future__ import annotations

from .features import build_alert_features, compute_pairwise_similarity
from .clustering import cluster_alerts
from .explain import explain_pair, explain_cluster

_GROUP_COLORS = ["#8b5cf6", "#0ea5e9", "#10b981", "#f59e0b", "#f43f5e"]

_NODE_TYPE_TITLES: dict[str, str] = {
    "firewall":      "Firewall Policy Violation",
    "router":        "BGP Route Instability",
    "switch":        "STP Topology Disruption",
    "compute":       "Compute Resource Failure",
    "load_balancer": "Load Balancer Degradation",
    "database":      "Database Replication Alert",
    "server":        "Server I/O Saturation",
    "gateway":       "Gateway Connectivity Failure",
    "storage":       "Storage Mount Failure",
    "wan":           "WAN Link Degradation",
}

_NODE_TYPE_SERVICES: dict[str, str] = {
    "firewall":      "Security / Firewall",
    "router":        "Network / Routing",
    "switch":        "Network / Switching",
    "compute":       "Infrastructure / Compute",
    "load_balancer": "Application / LB",
    "database":      "Data / Database",
    "server":        "Infrastructure / Server",
    "gateway":       "Network / Gateway",
    "storage":       "Infrastructure / Storage",
    "wan":           "Network / WAN",
}

_SEV_RANK: dict[str, int] = {
    "critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0,
}


def correlate_alerts(
    alerts: list[dict],
    enterprise_graph_by_scenario: dict[str, dict] | None = None,
    gnn_result_by_scenario: dict[str, dict] | None = None,
    threshold: float = 0.62,
) -> dict:
    """
    Run AI/graph-aware event correlation on a list of alert dicts.

    Parameters
    ----------
    alerts : list[dict]
        Alert objects from the Agentic Ops alert stream.
    enterprise_graph_by_scenario : dict[scenario_id -> enterprise_graph_dict]
    gnn_result_by_scenario : dict[scenario_id -> gnn_result_dict]
    threshold : float
        Pairwise similarity threshold for clustering (default 0.62).

    Returns
    -------
    dict with keys:
      correlated_alerts   : list[dict]  — alerts enriched with correlation_group,
                            color, correlation_score, correlation_explanation
      clusters            : list[dict]  — cluster dicts with metadata + scores
      method              : "graph_aware_event_correlation"
      clustering_method   : str  — actual algo used
      n_clusters          : int
      correlation_score_avg : float
      top_reasons         : list[str]
      feature_summary     : dict
      warnings            : list[str]
    """
    warnings_out: list[str] = []

    if not alerts:
        return {
            "correlated_alerts":     [],
            "clusters":              [],
            "method":                "graph_aware_event_correlation",
            "clustering_method":     "empty",
            "n_clusters":            0,
            "correlation_score_avg": 0.0,
            "top_reasons":           [],
            "feature_summary":       {},
            "warnings":              ["No alerts to correlate"],
        }

    # ── Feature engineering ───────────────────────────────────────────────────
    try:
        features = build_alert_features(
            alerts, enterprise_graph_by_scenario, gnn_result_by_scenario
        )
    except Exception as exc:
        warnings_out.append(f"Feature extraction failed ({exc}) — using minimal features")
        features = [
            {
                "alert_id":              a.get("alert_id", ""),
                "node_id":               a.get("node_id", ""),
                "scenario_id":           a.get("scenario_id", ""),
                "severity_score":        0.5,
                "timestamp_offset_norm": float(a.get("timestamp_offset", 0)) / 60.0,
                "node_type_idx":         0.5,
                "node_type_hash":        0.5,
                "diagram_hash":          0.5,
                "service_hash":          0.5,
                "scenario_hash":         hash(a.get("scenario_id", "")) % 1000 / 1000.0,
                "node_degree":           0.0,
                "cross_diagram_degree":  0.0,
                "gnn_proximity":         0.0,
                "_node_type":            a.get("node_type", ""),
                "_diagram":              a.get("diagram", ""),
                "_service":              a.get("service", ""),
                "_severity":             a.get("severity", ""),
                "_timestamp_offset":     float(a.get("timestamp_offset", 0)),
            }
            for a in alerts
        ]

    feat_by_id = {f["alert_id"]: f for f in features if f.get("alert_id")}

    # ── Clustering ────────────────────────────────────────────────────────────
    try:
        cl_result = cluster_alerts(alerts, features, threshold=threshold)
        labels    = cl_result["labels"]
        cl_method = cl_result.get("method", "graph_connected_components")
    except Exception as exc:
        warnings_out.append(
            f"Clustering failed ({exc}) — falling back to scenario grouping"
        )
        # scenario-grouping fallback
        sid_list: list[str] = []
        for a in alerts:
            sid = a.get("scenario_id", "default")
            if sid not in sid_list:
                sid_list.append(sid)
        labels    = [sid_list.index(a.get("scenario_id", "default")) for a in alerts]
        cl_method = "scenario_grouping_fallback"
        warnings_out.append(
            "Graph-aware correlation unavailable — using scenario grouping fallback."
        )

    # ── Group alerts by cluster label ─────────────────────────────────────────
    cluster_groups: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        cluster_groups.setdefault(lbl, []).append(i)

    n_clusters = len(cluster_groups)

    # Assign IDs and colors in sorted label order
    sorted_labels = sorted(cluster_groups.keys())
    cid_map:   dict[int, str] = {lbl: f"CORR-{ci+1:03d}" for ci, lbl in enumerate(sorted_labels)}
    color_map: dict[int, str] = {
        lbl: _GROUP_COLORS[ci % len(_GROUP_COLORS)] for ci, lbl in enumerate(sorted_labels)
    }

    # ── Compute per-cluster scores and explanations ───────────────────────────
    cluster_scores:   dict[int, float]       = {}
    cluster_expls:    dict[int, list[str]]   = {}
    cluster_signals:  dict[int, list[str]]   = {}
    cluster_max_sev:  dict[int, str]         = {}

    for lbl, idxs in cluster_groups.items():
        cl_alerts_raw = [alerts[i] for i in idxs]
        cl_feats_raw  = [features[i] for i in idxs]

        # Average pairwise similarity within cluster
        if len(idxs) > 1:
            sims: list[float] = []
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    sims.append(
                        compute_pairwise_similarity(cl_feats_raw[i], cl_feats_raw[j])
                    )
            cluster_scores[lbl] = round(sum(sims) / len(sims), 4) if sims else 0.75
        else:
            cluster_scores[lbl] = 0.75  # singleton — nominal score

        cluster_expls[lbl] = explain_cluster(cid_map[lbl], cl_alerts_raw, feat_by_id)

        # Root signal nodes: highest GNN proximity, then highest severity
        signals = sorted(
            cl_alerts_raw,
            key=lambda a: (
                feat_by_id.get(a.get("alert_id", ""), {}).get("gnn_proximity", 0),
                _SEV_RANK.get(a.get("severity", "low"), 0),
            ),
            reverse=True,
        )
        cluster_signals[lbl] = [
            a["node_id"] for a in signals[:3] if a.get("node_id")
        ]

        sevs = [a.get("severity", "low") for a in cl_alerts_raw]
        cluster_max_sev[lbl] = max(sevs, key=lambda s: _SEV_RANK.get(s, 0))

    # ── Enrich each alert ─────────────────────────────────────────────────────
    enriched_alerts: list[dict] = []
    for i, al in enumerate(alerts):
        lbl  = labels[i]
        feat = features[i]
        cid  = cid_map[lbl]
        col  = color_map[lbl]

        # Per-alert explanation: compare with best peer in same cluster
        peers = [j for j in cluster_groups[lbl] if j != i]
        if peers:
            best_j = max(
                peers,
                key=lambda j: compute_pairwise_similarity(feat, features[j])
            )
            pair_score = compute_pairwise_similarity(feat, features[best_j])
            explanation = explain_pair(
                al, alerts[best_j], feat, features[best_j], pair_score
            )
        else:
            explanation = ["Singleton alert — no peers in correlation window"]

        enriched = dict(al)
        enriched["correlation_group"]       = cid
        enriched["color"]                   = col
        enriched["correlation_score"]       = cluster_scores[lbl]
        enriched["correlation_explanation"] = explanation
        enriched_alerts.append(enriched)

    # ── Build cluster output dicts ────────────────────────────────────────────
    clusters_out: list[dict] = []
    for lbl in sorted_labels:
        idxs     = cluster_groups[lbl]
        cl_items = [enriched_alerts[i] for i in idxs]
        cid      = cid_map[lbl]
        col      = color_map[lbl]
        scenario_ids = list(dict.fromkeys(
            a.get("scenario_id", "") for a in cl_items
        ))
        ntypes  = [a.get("node_type", "network") for a in cl_items]
        dom     = max(set(ntypes), key=ntypes.count) if ntypes else "network"

        clusters_out.append({
            "cluster_id":             cid,
            "color":                  col,
            "alerts":                 [a["alert_id"] for a in cl_items],
            "scenario_ids":           scenario_ids,
            "scenario_id":            scenario_ids[0] if scenario_ids else "",
            "primary_service":        _NODE_TYPE_SERVICES.get(dom, "Network / Other"),
            "severity":               cluster_max_sev[lbl],
            "correlation_score":      cluster_scores[lbl],
            "root_signal_nodes":      cluster_signals[lbl],
            "explanations":           cluster_expls[lbl],
            "title":                  _NODE_TYPE_TITLES.get(dom, f"{dom.title()} Incident"),
            "service":                _NODE_TYPE_SERVICES.get(dom, "Network / Other"),
        })

    # ── Summary stats ─────────────────────────────────────────────────────────
    avg_score = (
        sum(cluster_scores.values()) / len(cluster_scores) if cluster_scores else 0.0
    )
    avg_gnn = sum(f.get("gnn_proximity", 0) for f in features) / max(len(features), 1)

    all_reasons: list[str] = []
    for exps in cluster_expls.values():
        all_reasons.extend(exps)
    top_reasons = list(dict.fromkeys(all_reasons))[:5]

    return {
        "correlated_alerts":     enriched_alerts,
        "clusters":              clusters_out,
        "method":                "graph_aware_event_correlation",
        "clustering_method":     cl_method,
        "n_clusters":            n_clusters,
        "correlation_score_avg": round(avg_score, 4),
        "top_reasons":           top_reasons,
        "feature_summary": {
            "alert_count":       len(alerts),
            "n_clusters":        n_clusters,
            "avg_gnn_proximity": round(avg_gnn, 4),
            "threshold":         threshold,
        },
        "warnings": warnings_out,
    }
