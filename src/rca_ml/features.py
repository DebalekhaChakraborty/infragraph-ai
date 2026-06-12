"""
features.py — Feature engineering for topology RCA ML.

One feature row is computed per node per case.  No remediation keys appear.
"""
from __future__ import annotations

import math
from pathlib import Path

import networkx as nx

# ── Scoring tables ─────────────────────────────────────────────────────────────

SEVERITY_SCORE: dict[str, float] = {
    "critical": 1.0,
    "high":     0.8,
    "warning":  0.6,
    "medium":   0.5,
    "low":      0.3,
    "info":     0.1,
    "unknown":  0.0,
}

NODE_TYPE_PRIORITY: dict[str, float] = {
    "firewall":     0.95,
    "router":       0.90,
    "cloud":        0.85,
    "wan":          0.85,
    "load_balancer":0.80,
    "switch":       0.70,
    "database":     0.65,
    "server":       0.55,
    "service":      0.50,
    "unknown":      0.10,
}

_LARGE = 999   # sentinel for "not applicable / not reachable"


# ── Path normalisation ─────────────────────────────────────────────────────────

def normalize_repo_path(repo_root: Path, raw_path: str) -> Path:
    """
    Return an absolute Path, handling Windows backslashes and relative paths.

    Supports:
      "datasets\\infragraph_v3\\..."   (Windows backslash)
      "datasets/infragraph_v3/..."
      "C:\\absolute\\path"
    """
    normalized = raw_path.replace("\\", "/")
    p = Path(normalized)
    if p.is_absolute():
        return p
    return repo_root / p


# ── Graph construction ─────────────────────────────────────────────────────────

def build_digraph(local_graph: dict) -> nx.DiGraph:
    """Build a directed graph from a local_graph dict."""
    G: nx.DiGraph = nx.DiGraph()
    for node in local_graph.get("nodes", []):
        nid = node.get("id", "")
        if nid:
            G.add_node(nid, **{k: v for k, v in node.items() if k != "id"})
    for edge in local_graph.get("edges", []):
        src, tgt = edge.get("source", ""), edge.get("target", "")
        if src and tgt:
            G.add_edge(src, tgt)
    return G


def _safe_centralities(G: nx.DiGraph) -> tuple[dict, dict, dict]:
    """Compute pagerank, betweenness, closeness with safe fallbacks."""
    n = len(G)
    if n == 0:
        return {}, {}, {}
    uniform = {v: 1.0 / n for v in G.nodes()}
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        pr = uniform.copy()
    try:
        bc = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        bc = {v: 0.0 for v in G.nodes()}
    G_und = G.to_undirected()
    try:
        cl = nx.closeness_centrality(G_und)
    except Exception:
        cl = {v: 0.0 for v in G.nodes()}
    return pr, bc, cl


# ── Per-node priority resolution ──────────────────────────────────────────────

def _type_priority(node_type: str) -> float:
    t = node_type.lower()
    if t in NODE_TYPE_PRIORITY:
        return NODE_TYPE_PRIORITY[t]
    for key, score in NODE_TYPE_PRIORITY.items():
        if key in t:
            return score
    return 0.10


# ── Main feature builder ───────────────────────────────────────────────────────

def compute_case_features(
    case_id: str,
    split: str,
    scenario_id: str,
    diagram_id: str,
    events: list[dict],
    local_graph: dict,
    root_cause_node: str | None = None,
) -> list[dict]:
    """
    Return one feature row per node in local_graph.

    root_cause_node: if provided, sets label_is_root=1 for that node.
    No remediation content is produced.
    """
    G = build_digraph(local_graph)
    G_und = G.to_undirected()
    pr, bc, cl = _safe_centralities(G)

    # Index events by node
    alert_map: dict[str, list[dict]] = {}
    for ev in events:
        node = ev.get("node", "")
        if node:
            alert_map.setdefault(node, []).append(ev)

    alerted_set: set[str] = set(alert_map.keys())

    # Time-rank table (rank 1 = first alerted)
    first_times = sorted(
        [(n, min(e.get("time_offset_min", 0) for e in evts))
         for n, evts in alert_map.items()],
        key=lambda x: x[1],
    )
    time_rank: dict[str, int] = {n: i + 1 for i, (n, _) in enumerate(first_times)}

    rows: list[dict] = []
    for node_data in local_graph.get("nodes", []):
        nid = node_data.get("id", "")
        if not nid:
            continue

        evts = alert_map.get(nid, [])
        is_alerted = len(evts) > 0
        alert_count = len(evts)

        severities = [SEVERITY_SCORE.get(e.get("severity", "").lower(), 0.0) for e in evts]
        times = [e.get("time_offset_min", 0) for e in evts]

        max_sev = max(severities) if severities else 0.0
        first_t = min(times) if times else _LARGE
        mean_t  = sum(times) / len(times) if times else _LARGE
        min_rank = time_rank.get(nid, len(alerted_set) + 1)

        in_deg    = G.in_degree(nid)  if nid in G else 0
        out_deg   = G.out_degree(nid) if nid in G else 0
        total_deg = in_deg + out_deg
        is_src  = int(nid in G and in_deg  == 0 and total_deg > 0)
        is_sink = int(nid in G and out_deg == 0 and total_deg > 0)

        # Undirected distances to alerted nodes
        dists: list[int] = [0] if is_alerted else []
        others = alerted_set - {nid}
        if nid in G_und:
            for an in others:
                if an in G_und:
                    try:
                        dists.append(nx.shortest_path_length(G_und, nid, an))
                    except nx.NetworkXNoPath:
                        pass
        min_dist  = min(dists) if dists else _LARGE
        mean_dist = sum(dists) / len(dists) if dists else _LARGE

        # Directed reachability
        reach_to = reach_from = 0
        if nid in G:
            non_self = alerted_set - {nid}
            reach_to   = len(nx.descendants(G, nid) & non_self)
            reach_from = len(nx.ancestors(G, nid)   & non_self)

        node_type = node_data.get("type", "unknown").lower()
        type_prio = _type_priority(node_type)

        # Severity-weighted score: heavier weight for high severity at earlier time
        sev_wt = sum(
            s / math.log(t + math.e)
            for s, t in zip(severities, times)
        ) if severities else 0.0

        rows.append({
            "case_id":               case_id,
            "split":                 split,
            "scenario_id":           scenario_id,
            "diagram_id":            diagram_id,
            "node_id":               nid,
            "node_type":             node_type,
            "zone":                  node_data.get("zone", "unknown"),
            "is_shared_entity":      int(node_data.get("is_shared_entity", False)),
            "is_alerted":            int(is_alerted),
            "alert_count":           alert_count,
            "max_severity_score":    max_sev,
            "first_alert_time":      first_t,
            "mean_alert_time":       mean_t,
            "min_time_rank":         min_rank,
            "in_degree":             in_deg,
            "out_degree":            out_deg,
            "total_degree":          total_deg,
            "pagerank":              pr.get(nid, 0.0),
            "betweenness_centrality":bc.get(nid, 0.0),
            "closeness_centrality":  cl.get(nid, 0.0),
            "is_source_node":        is_src,
            "is_sink_node":          is_sink,
            "min_undirected_distance_to_alert":    min_dist,
            "mean_undirected_distance_to_alert":   mean_dist,
            "directed_reachability_to_alert_count":  reach_to,
            "reverse_reachability_from_alert_count": reach_from,
            "node_type_priority_score":   type_prio,
            "severity_weighted_alert_score": sev_wt,
            "label_is_root": int(nid == root_cause_node) if root_cause_node else 0,
        })

    return rows
