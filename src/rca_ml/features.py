"""
features.py — Feature engineering for topology RCA ML.

One feature row is computed per node per case.  No remediation keys appear.
"""
from __future__ import annotations

import math
from collections import defaultdict
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
    "firewall":      0.95,
    "router":        0.90,
    "cloud":         0.85,
    "wan":           0.85,
    "load_balancer": 0.80,
    "switch":        0.70,
    "database":      0.65,
    "server":        0.55,
    "service":       0.50,
    "unknown":       0.10,
}

_LARGE = 999   # sentinel for "not applicable / not reachable"


# ── Alert type classification ──────────────────────────────────────────────────

ALERT_TYPE_BUCKETS: dict[str, list[str]] = {
    "cpu":                    ["cpu_spike", "cpu_high", "high_cpu"],
    "latency":                ["latency", "api_latency", "high_latency", "slow_response", "response_time"],
    "packet_drop":            ["packet_drop", "packet_loss"],
    "link_errors":            ["link_errors", "link_down", "interface_errors", "port_down",
                               "interface_flap", "port_flap"],
    "connection_timeout":     ["connection_timeout", "connection_refused", "connection_drop",
                               "connection_reset"],
    "auth_errors":            ["auth_errors", "auth_failure", "authentication_failed",
                               "auth_denied"],
    "backend_pool_unhealthy": ["backend_pool_unhealthy", "unhealthy_backend", "pool_error",
                               "backend_failure"],
    "user_timeout":           ["user_timeout", "session_timeout", "request_timeout"],
}

# node_type → compatible alert buckets
NODE_ALERT_COMPAT_MAP: dict[str, frozenset[str]] = {
    "firewall":      frozenset({"packet_drop", "connection_timeout", "link_errors", "auth_errors"}),
    "router":        frozenset({"packet_drop", "link_errors", "latency"}),
    "cloud":         frozenset({"packet_drop", "link_errors", "latency"}),
    "wan":           frozenset({"packet_drop", "link_errors", "latency"}),
    "load_balancer": frozenset({"backend_pool_unhealthy", "connection_timeout", "latency"}),
    "server":        frozenset({"cpu", "latency", "connection_timeout", "user_timeout"}),
    "database":      frozenset({"latency", "connection_timeout", "cpu"}),
    "service":       frozenset({"auth_errors", "latency", "connection_timeout"}),
    "switch":        frozenset({"link_errors", "packet_drop"}),
}


def _classify_alert_type(alert_type: str) -> str:
    at = alert_type.lower().replace("-", "_").replace(" ", "_")
    for bucket, patterns in ALERT_TYPE_BUCKETS.items():
        if any(p in at for p in patterns):
            return bucket
    return "other"


def _alert_compat_score(node_type: str, alert_types: list[str]) -> float:
    if not alert_types:
        return 0.0
    nt = _type_key(node_type)
    compat = NODE_ALERT_COMPAT_MAP.get(nt, frozenset())
    if not compat:
        return 0.0
    classified = [_classify_alert_type(at) for at in alert_types]
    return sum(1 for c in classified if c in compat) / len(classified)


# ── Path normalisation ─────────────────────────────────────────────────────────

def normalize_repo_path(repo_root: Path, raw_path: str) -> Path:
    """Return an absolute Path, handling Windows backslashes and relative paths."""
    normalized = raw_path.replace("\\", "/")
    p = Path(normalized)
    if p.is_absolute():
        return p
    return repo_root / p


# ── Graph construction ─────────────────────────────────────────────────────────

def build_digraph(local_graph: dict) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()
    for node in local_graph.get("nodes", []):
        nid = node.get("id", "")
        if nid:
            G.add_node(nid, **{k: v for k, v in node.items() if k != "id"})
    for edge in local_graph.get("edges", []):
        src, tgt = edge.get("source", ""), edge.get("target", "")
        if src and tgt and src != tgt:
            G.add_edge(src, tgt)
    return G


def _safe_centralities(G: nx.DiGraph) -> tuple[dict, dict, dict]:
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


# ── Node type helpers ──────────────────────────────────────────────────────────

def _type_key(node_type: str) -> str:
    t = node_type.lower()
    if t in NODE_TYPE_PRIORITY:
        return t
    for key in NODE_TYPE_PRIORITY:
        if key in t:
            return key
    return "unknown"


def _type_priority(node_type: str) -> float:
    return NODE_TYPE_PRIORITY.get(_type_key(node_type), 0.10)


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

    Includes base graph/alert features plus alert-type, temporal, and
    propagation context features.  No remediation content is produced.
    """
    G     = build_digraph(local_graph)
    G_und = G.to_undirected()
    pr, bc, cl = _safe_centralities(G)

    # ── Alert index ────────────────────────────────────────────────────────────
    alert_map: dict[str, list[dict]] = {}
    for ev in events:
        node = ev.get("node", "")
        if node:
            alert_map.setdefault(node, []).append(ev)

    alerted_set: set[str] = set(alert_map.keys())

    # Time rank (rank 1 = first alerted)
    first_times = sorted(
        [(n, min(e.get("time_offset_min", 0) for e in evts))
         for n, evts in alert_map.items()],
        key=lambda x: x[1],
    )
    time_rank: dict[str, int]     = {n: i + 1 for i, (n, _) in enumerate(first_times)}
    first_time_map: dict[str, float] = {n: t for n, t in first_times}
    num_alerted = len(alerted_set)

    global_first_alerted = first_times[0][0]  if first_times else None
    global_last_alerted  = first_times[-1][0] if first_times else None

    # Max severity per alerted node
    sev_map: dict[str, float] = {
        nid: max(SEVERITY_SCORE.get(e.get("severity", "").lower(), 0.0) for e in evts)
        for nid, evts in alert_map.items()
    }

    # Alert type bucket counts per node
    at_map: dict[str, dict[str, int]] = {}
    for nid, evts in alert_map.items():
        bc_counts: dict[str, int] = defaultdict(int)
        for ev in evts:
            bc_counts[_classify_alert_type(ev.get("alert_type", ""))] += 1
        at_map[nid] = dict(bc_counts)

    # ── Pre-compute ancestors / descendants ────────────────────────────────────
    ancestor_map:    dict[str, set[str]] = {}
    descendant_map:  dict[str, set[str]] = {}
    for nid in G.nodes():
        try:
            ancestor_map[nid]   = nx.ancestors(G, nid)
        except Exception:
            ancestor_map[nid]   = set()
        try:
            descendant_map[nid] = nx.descendants(G, nid)
        except Exception:
            descendant_map[nid] = set()

    # ── Per-node features ──────────────────────────────────────────────────────
    rows: list[dict] = []

    for node_data in local_graph.get("nodes", []):
        nid = node_data.get("id", "")
        if not nid:
            continue

        evts        = alert_map.get(nid, [])
        is_alerted  = len(evts) > 0
        alert_count = len(evts)
        severities  = [SEVERITY_SCORE.get(e.get("severity", "").lower(), 0.0) for e in evts]
        times       = [e.get("time_offset_min", 0) for e in evts]

        max_sev  = max(severities) if severities else 0.0
        first_t  = min(times) if times else _LARGE
        mean_t   = sum(times) / len(times) if times else _LARGE
        min_rank = time_rank.get(nid, num_alerted + 1)

        # Degree
        in_deg    = G.in_degree(nid)  if nid in G else 0
        out_deg   = G.out_degree(nid) if nid in G else 0
        total_deg = in_deg + out_deg
        is_src    = int(nid in G and in_deg  == 0 and total_deg > 0)
        is_sink   = int(nid in G and out_deg == 0 and total_deg > 0)

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

        # Directed reachability (existing)
        anc  = ancestor_map.get(nid, set())
        desc = descendant_map.get(nid, set())
        reach_to   = len(desc & (alerted_set - {nid}))
        reach_from = len(anc  & (alerted_set - {nid}))

        # Node type + priority
        node_type = node_data.get("type", "unknown").lower()
        type_prio = _type_priority(node_type)
        sev_wt    = sum(
            s / math.log(t + math.e)
            for s, t in zip(severities, times)
        ) if severities else 0.0

        # ── Alert type counts ──────────────────────────────────────────────────
        at_counts = at_map.get(nid, {})
        alert_type_count_cpu    = at_counts.get("cpu", 0)
        alert_type_count_lat    = at_counts.get("latency", 0)
        alert_type_count_pkd    = at_counts.get("packet_drop", 0)
        alert_type_count_lke    = at_counts.get("link_errors", 0)
        alert_type_count_cto    = at_counts.get("connection_timeout", 0)
        alert_type_count_auth   = at_counts.get("auth_errors", 0)
        alert_type_count_bpu    = at_counts.get("backend_pool_unhealthy", 0)
        alert_type_count_ut     = at_counts.get("user_timeout", 0)
        alert_type_count_other  = at_counts.get("other", 0)

        # Node-alert compatibility
        alert_types_list = [ev.get("alert_type", "") for ev in evts]
        compat_score = _alert_compat_score(node_type, alert_types_list)

        # ── Temporal features ──────────────────────────────────────────────────
        is_first_alerted = int(nid == global_first_alerted)
        is_last_alerted  = int(nid == global_last_alerted)
        time_span   = (max(times) - min(times)) if len(times) >= 2 else 0
        burst_score = alert_count / (time_span + 1)

        if is_alerted and num_alerted > 1:
            seq_pos_norm = (min_rank - 1) / (num_alerted - 1)
        elif is_alerted:
            seq_pos_norm = 0.0  # only alerted node
        else:
            seq_pos_norm = 1.0  # not alerted → last position sentinel

        # ── Propagation features ───────────────────────────────────────────────
        cand_t = first_time_map.get(nid, None)

        upstream_alerted   = anc  & alerted_set
        downstream_alerted = desc & alerted_set
        upstream_alert_count   = len(upstream_alerted)
        downstream_alert_count = len(downstream_alerted)

        upstream_critical = sum(1 for an in upstream_alerted if sev_map.get(an, 0.0) >= 1.0)
        downstream_warn   = sum(1 for dn in downstream_alerted if sev_map.get(dn, 0.0) <= 0.6)

        if cand_t is not None:
            downstream_after = sum(
                1 for dn in downstream_alerted
                if first_time_map.get(dn, cand_t + 1) > cand_t
            )
            upstream_before = sum(
                1 for an in upstream_alerted
                if first_time_map.get(an, cand_t) < cand_t
            )
        else:
            downstream_after = len(downstream_alerted)
            upstream_before  = 0  # no reference time for unalerted node

        if num_alerted > 0:
            early_score = (1.0 - (min_rank - 1) / max(1, num_alerted)) if is_alerted else 0.4
            down_frac   = downstream_after / max(1, num_alerted)
            up_clean    = 1.0 - (upstream_alert_count / max(1, num_alerted))
            prop_cons   = min(1.0, max(0.0,
                early_score * 0.35 + down_frac * 0.45 + up_clean * 0.20
            ))
        else:
            prop_cons = 0.0

        rows.append({
            # ── row identifiers (not features) ─────────────────────────────────
            "case_id":     case_id,
            "split":       split,
            "scenario_id": scenario_id,
            "diagram_id":  diagram_id,
            "node_id":     nid,
            # ── categorical features ────────────────────────────────────────────
            "node_type":        node_type,
            "zone":             node_data.get("zone", "unknown"),
            # ── numeric: base ──────────────────────────────────────────────────
            "is_shared_entity":               int(node_data.get("is_shared_entity", False)),
            "is_alerted":                     int(is_alerted),
            "alert_count":                    alert_count,
            "max_severity_score":             max_sev,
            "first_alert_time":               first_t,
            "mean_alert_time":                mean_t,
            "min_time_rank":                  min_rank,
            "in_degree":                      in_deg,
            "out_degree":                     out_deg,
            "total_degree":                   total_deg,
            "pagerank":                       pr.get(nid, 0.0),
            "betweenness_centrality":         bc.get(nid, 0.0),
            "closeness_centrality":           cl.get(nid, 0.0),
            "is_source_node":                 is_src,
            "is_sink_node":                   is_sink,
            "min_undirected_distance_to_alert":  min_dist,
            "mean_undirected_distance_to_alert": mean_dist,
            "directed_reachability_to_alert_count":  reach_to,
            "reverse_reachability_from_alert_count": reach_from,
            "node_type_priority_score":       type_prio,
            "severity_weighted_alert_score":  sev_wt,
            # ── numeric: alert type counts ─────────────────────────────────────
            "alert_type_count_cpu":                    alert_type_count_cpu,
            "alert_type_count_latency":                alert_type_count_lat,
            "alert_type_count_packet_drop":            alert_type_count_pkd,
            "alert_type_count_link_errors":            alert_type_count_lke,
            "alert_type_count_connection_timeout":     alert_type_count_cto,
            "alert_type_count_auth_errors":            alert_type_count_auth,
            "alert_type_count_backend_pool_unhealthy": alert_type_count_bpu,
            "alert_type_count_user_timeout":           alert_type_count_ut,
            "alert_type_count_other":                  alert_type_count_other,
            # ── numeric: alert compatibility ───────────────────────────────────
            "node_alert_compatibility_score": compat_score,
            # ── numeric: temporal ──────────────────────────────────────────────
            "is_first_alerted_node":       is_first_alerted,
            "is_last_alerted_node":        is_last_alerted,
            "alert_time_span":             time_span,
            "alert_burst_score":           burst_score,
            "alert_sequence_position_norm": seq_pos_norm,
            # ── numeric: propagation ───────────────────────────────────────────
            "upstream_alert_count":                        upstream_alert_count,
            "downstream_alert_count":                      downstream_alert_count,
            "upstream_critical_alert_count":               upstream_critical,
            "downstream_warning_alert_count":              downstream_warn,
            "downstream_after_candidate_count":            downstream_after,
            "alerts_reachable_downstream_after_candidate": downstream_after,
            "alerts_reachable_upstream_before_candidate":  upstream_before,
            "propagation_consistency_score":               prop_cons,
            # ── label ──────────────────────────────────────────────────────────
            "label_is_root": int(nid == root_cause_node) if root_cause_node else 0,
        })

    return rows
