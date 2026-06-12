"""
enterprise_gnn_dataset.py — Build torch graph data dicts for Enterprise RCA GNN.

Requires PyTorch (for torch.save).  Call check_torch_requirement() in scripts.
Does NOT require torch_geometric at build time.

No remediation content is produced here.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

from .features import (
    ALERT_TYPE_BUCKETS,
    NODE_ALERT_COMPAT_MAP,
    SEVERITY_SCORE,
    _LARGE,
    normalize_repo_path,
)

# ── Dependency check ────────────────────────────────────────────────────────────

def check_torch_requirement() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[ERROR] PyTorch is required for enterprise GNN RCA (needed for torch.save).")
        print("        Install from: https://pytorch.org/get-started/locally/")
        sys.exit(1)


# ── Vocabularies ────────────────────────────────────────────────────────────────

NODE_TYPES: list[str] = [
    "router", "switch", "firewall", "server", "database",
    "load_balancer", "cloud", "wan", "service", "unknown",
]
DIAGRAM_TYPES: list[str] = [
    "branch_topology", "wan_topology", "datacenter_topology",
    "app_db_topology", "shared_services_topology", "unknown",
]
_NODE_TYPE_IDX: dict[str, int]  = {t: i for i, t in enumerate(NODE_TYPES)}
_DIAG_TYPE_IDX: dict[str, int]  = {t: i for i, t in enumerate(DIAGRAM_TYPES)}

NUM_NODE_TYPES    = len(NODE_TYPES)     # 10
NUM_DIAGRAM_TYPES = len(DIAGRAM_TYPES)  # 6
NUMERIC_DIM       = 18
ALERT_TYPE_DIM    = 11   # multi-hot: 8 shared buckets + route_flap + dependency_error + other
TEMPORAL_PROP_DIM = 9    # temporal + propagation + compat
IN_DIM = NUM_NODE_TYPES + NUM_DIAGRAM_TYPES + NUMERIC_DIM + ALERT_TYPE_DIM + TEMPORAL_PROP_DIM  # 54

# Enterprise-specific alert type order (must match _ent_alert_type_vec)
ENT_ALERT_TYPES: list[str] = [
    "cpu", "latency", "packet_drop", "link_errors", "connection_timeout",
    "auth_errors", "backend_pool_unhealthy", "user_timeout",
    "route_flap", "dependency_error", "other",
]
_ENT_AT_IDX: dict[str, int] = {t: i for i, t in enumerate(ENT_ALERT_TYPES)}

FEATURE_NAMES: list[str] = (
    [f"nt_{t}" for t in NODE_TYPES]
    + [f"dt_{t}" for t in DIAGRAM_TYPES]
    + [
        # original 18 numeric
        "is_shared_entity",
        "is_alerted",
        "alert_count_norm",
        "max_severity_score",
        "first_alert_time_norm",
        "mean_alert_time_norm",
        "min_time_rank_norm",
        "degree_norm",
        "in_degree_norm",
        "out_degree_norm",
        "cross_diagram_degree_norm",
        "pagerank",
        "betweenness_centrality",
        "closeness_centrality",
        "distance_to_alert_norm",
        "reverse_reachability_norm",
        "source_like_score",
        "sink_like_score",
    ]
    + [f"at_{t}" for t in ENT_ALERT_TYPES]   # 11 alert-type multi-hot
    + [
        # 9 temporal + propagation + compat
        "is_first_alerted_node",
        "is_last_alerted_node",
        "alert_sequence_position_norm",
        "upstream_alert_count_norm",
        "downstream_alert_count_norm",
        "upstream_critical_count_norm",
        "downstream_warning_count_norm",
        "propagation_consistency_score",
        "node_alert_compatibility_score",
    ]
)
assert len(FEATURE_NAMES) == IN_DIM, f"{len(FEATURE_NAMES)} != {IN_DIM}"


# ── Enterprise-specific alert type classifier ──────────────────────────────────

_ENT_EXTRA_BUCKETS: dict[str, list[str]] = {
    "route_flap":       ["route_flap", "bgp_flap", "ospf_flap", "routing_instability",
                         "route_instability"],
    "dependency_error": ["dependency_error", "downstream_failure", "service_dependency",
                         "dependency_fail", "upstream_unavailable"],
}


def _classify_ent_alert_type(alert_type: str) -> str:
    """Map alert_type string to one of ENT_ALERT_TYPES bucket names."""
    at = alert_type.lower().replace("-", "_").replace(" ", "_")
    for bucket, patterns in _ENT_EXTRA_BUCKETS.items():
        if any(p in at for p in patterns):
            return bucket
    for bucket, patterns in ALERT_TYPE_BUCKETS.items():
        if any(p in at for p in patterns):
            return bucket
    return "other"


def _ent_alert_type_vec(at_counts: dict[str, int]) -> list[float]:
    """Return 11-element multi-hot vector of alert type presence (0 or 1)."""
    return [float(at_counts.get(t, 0) > 0) for t in ENT_ALERT_TYPES]


def _ent_compat_score(node_type: str, at_counts: dict[str, int]) -> float:
    """Fraction of present alert types that are compatible with node_type."""
    present = {t for t, c in at_counts.items() if c > 0}
    if not present:
        return 0.0
    nt = node_type.lower()
    if nt not in NODE_ALERT_COMPAT_MAP:
        for key in NODE_ALERT_COMPAT_MAP:
            if key in nt:
                nt = key
                break
    compat = NODE_ALERT_COMPAT_MAP.get(nt, frozenset())
    if not compat:
        return 0.0
    # Map alert type bucket names to compat keys (same naming)
    return sum(1 for t in present if t in compat) / len(present)


# ── Graph construction helpers ─────────────────────────────────────────────────

def _build_enterprise_nx_graph(enterprise_graph: dict) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()
    for node in enterprise_graph.get("nodes", []):
        nid = node.get("id", "")
        if nid:
            G.add_node(nid, **{k: v for k, v in node.items() if k != "id"})

    seen_edges: set[tuple[str, str]] = set()

    # Primary edges
    for edge in enterprise_graph.get("edges", []):
        s, t = edge.get("source", ""), edge.get("target", "")
        if not (s and t and s != t):
            continue
        scope = edge.get("edge_scope", "") or edge.get("edge_type", "")
        etype = "cross_diagram" if "cross_diagram" in scope else "local"
        key = (s, t)
        if key not in seen_edges:
            seen_edges.add(key)
            G.add_edge(s, t, edge_type=etype)

    # Cross-diagram edges list (may duplicate entries already in edges[])
    for edge in enterprise_graph.get("cross_diagram_edges", []):
        s, t = edge.get("source", ""), edge.get("target", "")
        if not (s and t and s != t):
            continue
        key = (s, t)
        if key not in seen_edges:
            seen_edges.add(key)
            G.add_edge(s, t, edge_type="cross_diagram")

    return G


def _cross_diagram_degrees(enterprise_graph: dict) -> dict[str, int]:
    """
    Count cross-diagram edges per node (undirected degree).

    Reads both:
      - enterprise_graph["cross_diagram_edges"]
      - enterprise_graph["edges"] where edge_scope or edge_type == "cross_diagram"

    Deduplicates by (source, target) pair and filters self-loops.
    """
    seen: set[tuple[str, str]] = set()

    def _add(s: str, t: str) -> None:
        if s and t and s != t:
            seen.add((min(s, t), max(s, t)))  # canonical order for dedup

    for edge in enterprise_graph.get("cross_diagram_edges", []):
        _add(edge.get("source", ""), edge.get("target", ""))

    for edge in enterprise_graph.get("edges", []):
        scope = edge.get("edge_scope", "") or edge.get("edge_type", "")
        if "cross_diagram" in scope:
            _add(edge.get("source", ""), edge.get("target", ""))

    deg: dict[str, int] = defaultdict(int)
    for s, t in seen:
        deg[s] += 1
        deg[t] += 1
    return dict(deg)


def _safe_ent_centralities(G: nx.DiGraph) -> tuple[dict, dict, dict]:
    n = len(G)
    if n == 0:
        return {}, {}, {}
    unif = {v: 1.0 / n for v in G.nodes()}
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=300)
    except Exception:
        pr = unif.copy()
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


# ── Per-node feature vector ─────────────────────────────────────────────────────

def _normalise_node_type(raw: str) -> str:
    raw = raw.lower()
    if raw in _NODE_TYPE_IDX:
        return raw
    for key in NODE_TYPES:
        if key in raw:
            return key
    return "unknown"


def _normalise_diagram_type(raw: str) -> str:
    raw = raw.lower()
    if raw in _DIAG_TYPE_IDX:
        return raw
    for key in DIAGRAM_TYPES:
        if key in raw:
            return key
    return "unknown"


def _node_feature_vec(node_data: dict, ctx: dict) -> list[float]:
    """
    Build a 54-dim feature vector for one node.

    ctx must contain: alert_map, G, G_und, pr, bc, cl, alerted_set,
    cross_diag_deg, total_nodes, time_rank, num_alerted, max_alert_count,
    first_time_map, sev_map, global_first_alerted, global_last_alerted,
    ancestor_map, descendant_map, alert_type_map.
    """
    nid = node_data.get("id", "")

    alert_map          = ctx["alert_map"]
    G                  = ctx["G"]
    G_und              = ctx["G_und"]
    pr, bc, cl         = ctx["pr"], ctx["bc"], ctx["cl"]
    alerted_set        = ctx["alerted_set"]
    cross_diag_deg     = ctx["cross_diag_deg"]
    total_nodes        = ctx["total_nodes"]
    time_rank          = ctx["time_rank"]
    num_alerted        = ctx["num_alerted"]
    max_alert_count    = ctx["max_alert_count"]
    first_time_map     = ctx["first_time_map"]
    sev_map            = ctx["sev_map"]
    global_first       = ctx["global_first_alerted"]
    global_last        = ctx["global_last_alerted"]
    ancestor_map       = ctx["ancestor_map"]
    descendant_map     = ctx["descendant_map"]
    alert_type_map     = ctx["alert_type_map"]   # {nid: {bucket: count}}

    # One-hot: node type
    nt_key = _normalise_node_type(node_data.get("type", "unknown"))
    nt_vec = [0.0] * NUM_NODE_TYPES
    nt_vec[_NODE_TYPE_IDX[nt_key]] = 1.0

    # One-hot: diagram type
    dt_raw = node_data.get("diagram_type", node_data.get("diagram_id", "unknown"))
    dt_key = _normalise_diagram_type(dt_raw)
    dt_vec = [0.0] * NUM_DIAGRAM_TYPES
    dt_vec[_DIAG_TYPE_IDX[dt_key]] = 1.0

    # Alert features
    evts        = alert_map.get(nid, [])
    is_alerted  = len(evts) > 0
    alert_count = len(evts)
    severities  = [SEVERITY_SCORE.get(e.get("severity", "").lower(), 0.0) for e in evts]
    times       = [e.get("time_offset_min", 0) for e in evts]
    max_sev     = max(severities) if severities else 0.0
    first_t     = min(times) if times else 0
    mean_t      = sum(times) / len(times) if times else 0
    rank        = time_rank.get(nid, num_alerted + 1)

    # Structural
    n_scale   = max(1, total_nodes)
    in_deg    = G.in_degree(nid)  if nid in G else 0
    out_deg   = G.out_degree(nid) if nid in G else 0
    total_deg = in_deg + out_deg
    xd        = cross_diag_deg.get(nid, 0)
    is_src    = float(nid in G and in_deg  == 0 and total_deg > 0)
    is_sink   = float(nid in G and out_deg == 0 and total_deg > 0)

    # Undirected distances to alerted nodes
    others = alerted_set - {nid}
    dists: list[int] = [0] if is_alerted else []
    if nid in G_und:
        for an in others:
            if an in G_und:
                try:
                    dists.append(nx.shortest_path_length(G_und, nid, an))
                except nx.NetworkXNoPath:
                    pass
    min_dist = min(dists) if dists else _LARGE

    # Reverse reachability
    anc        = ancestor_map.get(nid, set())
    desc       = descendant_map.get(nid, set())
    rr_count   = len(anc & (alerted_set - {nid}))

    # Normalise base features
    denom_time = 1440.0
    dist_norm  = min(1.0, min_dist / n_scale) if min_dist < _LARGE else 1.0
    rr_norm    = rr_count / max(1, len(alerted_set))
    rank_norm  = rank / max(1, num_alerted + 1)

    base_numeric: list[float] = [
        float(node_data.get("is_shared_entity", False)),
        float(is_alerted),
        alert_count / max(1, max_alert_count),
        max_sev,
        first_t / denom_time,
        mean_t  / denom_time,
        rank_norm,
        total_deg / n_scale,
        in_deg    / n_scale,
        out_deg   / n_scale,
        xd        / n_scale,
        pr.get(nid, 0.0),
        bc.get(nid, 0.0),
        cl.get(nid, 0.0),
        dist_norm,
        rr_norm,
        is_src,
        is_sink,
    ]
    assert len(base_numeric) == NUMERIC_DIM

    # Alert-type multi-hot (11)
    at_counts  = alert_type_map.get(nid, {})
    at_vec     = _ent_alert_type_vec(at_counts)

    # Temporal features
    is_first = float(nid == global_first) if global_first else 0.0
    is_last  = float(nid == global_last)  if global_last  else 0.0
    if is_alerted and num_alerted > 1:
        seq_pos = (rank - 1) / (num_alerted - 1)
    elif is_alerted:
        seq_pos = 0.0
    else:
        seq_pos = 1.0  # unalerted nodes treated as last

    # Propagation features
    upstream_alerted   = anc  & alerted_set
    downstream_alerted = desc & alerted_set
    up_count  = len(upstream_alerted)
    dn_count  = len(downstream_alerted)

    up_crit = sum(1 for a in upstream_alerted   if sev_map.get(a, 0.0) >= 1.0)
    dn_warn = sum(1 for a in downstream_alerted if sev_map.get(a, 0.0) <= 0.6)

    cand_t = first_time_map.get(nid, None)
    if cand_t is not None:
        dn_after = sum(
            1 for dn in downstream_alerted
            if first_time_map.get(dn, cand_t + 1) > cand_t
        )
    else:
        dn_after = dn_count

    if num_alerted > 0:
        early_sc = (1.0 - (rank - 1) / max(1, num_alerted)) if is_alerted else 0.4
        dn_frac  = dn_after / max(1, num_alerted)
        up_clean = 1.0 - (up_count / max(1, num_alerted))
        prop_cons = min(1.0, max(0.0,
            early_sc * 0.35 + dn_frac * 0.45 + up_clean * 0.20
        ))
    else:
        prop_cons = 0.0

    compat = _ent_compat_score(node_data.get("type", "unknown"), at_counts)

    prop_vec: list[float] = [
        is_first,
        is_last,
        seq_pos,
        up_count / max(1, num_alerted),
        dn_count / max(1, num_alerted),
        up_crit  / max(1, num_alerted),
        dn_warn  / max(1, num_alerted),
        prop_cons,
        compat,
    ]
    assert len(prop_vec) == TEMPORAL_PROP_DIM

    return nt_vec + dt_vec + base_numeric + at_vec + prop_vec


# ── Graph dict builder ─────────────────────────────────────────────────────────

def build_graph_dict(
    case_id: str,
    scenario_id: str,
    split: str,
    enterprise_graph: dict,
    events: list[dict],
    root_cause_node: str | None = None,
) -> dict | None:
    """
    Build a graph dict with torch tensors for torch.save.

    Returns None if root_cause_node is provided but not present in the graph nodes.
    Does not include remediation content.

    Adds alert_event_diagram_map: node_id -> time-sorted list of diagram_ids
    from events (used by inference to pick root_cause_diagram for shared nodes).
    """
    import torch

    raw_nodes = enterprise_graph.get("nodes", [])
    if not raw_nodes:
        return None

    node_ids   = [n["id"] for n in raw_nodes if n.get("id")]
    nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    num_nodes  = len(node_ids)

    if root_cause_node and root_cause_node not in nid_to_idx:
        return None

    # Graph and centralities
    G         = _build_enterprise_nx_graph(enterprise_graph)
    G_und     = G.to_undirected()
    pr, bc, cl = _safe_ent_centralities(G)
    cross_deg  = _cross_diagram_degrees(enterprise_graph)

    # Alert index
    alert_map: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        node = ev.get("node", "")
        if node:
            alert_map[node].append(ev)
    alerted_set     = set(alert_map.keys())
    max_alert_count = max((len(v) for v in alert_map.values()), default=1)

    # Time rank (rank 1 = first alerted)
    first_times = sorted(
        [(n, min(e.get("time_offset_min", 0) for e in ev_list))
         for n, ev_list in alert_map.items()],
        key=lambda x: x[1],
    )
    time_rank      = {n: i + 1 for i, (n, _) in enumerate(first_times)}
    first_time_map = {n: t for n, t in first_times}
    num_alerted    = len(alerted_set)

    global_first_alerted = first_times[0][0]  if first_times else None
    global_last_alerted  = first_times[-1][0] if first_times else None

    sev_map: dict[str, float] = {
        nid: max(SEVERITY_SCORE.get(e.get("severity", "").lower(), 0.0) for e in evts)
        for nid, evts in alert_map.items()
    }

    # Alert type map per node
    alert_type_map: dict[str, dict[str, int]] = {}
    for nid, evts in alert_map.items():
        counts: dict[str, int] = defaultdict(int)
        for ev in evts:
            counts[_classify_ent_alert_type(ev.get("alert_type", ""))] += 1
        alert_type_map[nid] = dict(counts)

    # Ancestors / descendants (precompute for propagation features)
    ancestor_map:   dict[str, set[str]] = {}
    descendant_map: dict[str, set[str]] = {}
    for nid in G.nodes():
        try:
            ancestor_map[nid]   = nx.ancestors(G, nid)
        except Exception:
            ancestor_map[nid]   = set()
        try:
            descendant_map[nid] = nx.descendants(G, nid)
        except Exception:
            descendant_map[nid] = set()

    # Alert event diagram map (Part B): node_id → sorted list of diagram_ids from events
    node_data_map = {n["id"]: n for n in raw_nodes if n.get("id")}
    _aedm: dict[str, list[tuple[float, str]]] = defaultdict(list)  # (time, diagram_id)

    for ev in events:
        nid = ev.get("node", "")
        if not nid:
            continue
        # Prefer explicit diagram_id on event, fall back to node's canonical diagram_id
        diag = ev.get("diagram_id", "") or node_data_map.get(nid, {}).get("diagram_id", "")
        if diag:
            t = ev.get("time_offset_min", 0)
            _aedm[nid].append((t, diag))

    # Also include non-event nodes with a canonical diagram_id (not via events)
    alert_event_diagram_map: dict[str, list[str]] = {}
    for nid in node_ids:
        if nid in _aedm:
            seen_diags: list[str] = []
            for _, d in sorted(_aedm[nid]):  # sort by time
                if d not in seen_diags:
                    seen_diags.append(d)
            alert_event_diagram_map[nid] = seen_diags
        else:
            canonical = node_data_map.get(nid, {}).get("diagram_id", "")
            alert_event_diagram_map[nid] = [canonical] if canonical else []

    # Build feature context
    ctx: dict = {
        "alert_map":           alert_map,
        "G":                   G,
        "G_und":               G_und,
        "pr":                  pr,
        "bc":                  bc,
        "cl":                  cl,
        "alerted_set":         alerted_set,
        "cross_diag_deg":      cross_deg,
        "total_nodes":         num_nodes,
        "time_rank":           time_rank,
        "num_alerted":         num_alerted,
        "max_alert_count":     max_alert_count,
        "first_time_map":      first_time_map,
        "sev_map":             sev_map,
        "global_first_alerted": global_first_alerted,
        "global_last_alerted":  global_last_alerted,
        "ancestor_map":        ancestor_map,
        "descendant_map":      descendant_map,
        "alert_type_map":      alert_type_map,
    }

    # Feature matrix
    x_rows = []
    for nid in node_ids:
        nd   = node_data_map[nid]
        fvec = _node_feature_vec(nd, ctx)
        x_rows.append(fvec)

    x = torch.FloatTensor(x_rows)  # [N, IN_DIM]

    # Edge index — bidirectional
    fwd_edges: list[tuple[int, int]] = []
    for src, tgt in G.edges():
        if src in nid_to_idx and tgt in nid_to_idx:
            fwd_edges.append((nid_to_idx[src], nid_to_idx[tgt]))
    all_edges = list(set(fwd_edges + [(t, s) for s, t in fwd_edges]))
    if all_edges:
        edge_index = torch.LongTensor(all_edges).t().contiguous()  # [2, E]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    y = (torch.tensor(nid_to_idx[root_cause_node], dtype=torch.long)
         if root_cause_node else torch.tensor(-1, dtype=torch.long))

    return {
        "x":                      x,
        "edge_index":             edge_index,
        "y":                      y,
        "num_nodes":              num_nodes,
        "case_id":                case_id,
        "scenario_id":            scenario_id,
        "split":                  split,
        "event_count":            len(events),
        "node_ids":               node_ids,
        "node_type_list":         [node_data_map[nid].get("type", "unknown") for nid in node_ids],
        "diagram_id_list":        [node_data_map[nid].get("diagram_id", "")   for nid in node_ids],
        "root_cause_node":        root_cause_node or "",
        "alert_event_diagram_map": alert_event_diagram_map,
    }


# ── Case loader ────────────────────────────────────────────────────────────────

def load_enterprise_case(
    lib_root: Path,
    manifest_row: dict,
    repo_root: Path,
) -> tuple[list[dict], dict, dict, dict, dict]:
    """
    Returns (events, labels, graph_ref, enterprise_graph, stitch_map).
    stitch_map is {} if file is absent.
    """
    def _r(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8"))

    events_doc = _r(lib_root / manifest_row["events_path"])
    labels     = _r(lib_root / manifest_row["labels_path"])
    graph_ref  = _r(lib_root / manifest_row["graph_ref_path"])

    eg_path          = normalize_repo_path(repo_root, graph_ref["enterprise_graph_path"])
    enterprise_graph = _r(eg_path)

    stitch_map: dict = {}
    sm_raw = graph_ref.get("stitch_map_path", "")
    if sm_raw:
        sm_path = normalize_repo_path(repo_root, sm_raw)
        if sm_path.exists():
            stitch_map = _r(sm_path)

    return events_doc.get("events", []), labels, graph_ref, enterprise_graph, stitch_map


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_graph_dataset(
    manifest_rows: list[dict],
    lib_root: Path,
    repo_root: Path,
) -> tuple[list[dict], list[dict]]:
    """
    Build list of graph_dicts and case_index.

    Skips cases with root_cause_in_scope=False (not labelled for training).
    """
    graphs: list[dict] = []
    case_index: list[dict] = []
    skipped = 0

    for row in manifest_rows:
        if not row.get("root_cause_in_scope", False):
            continue
        case_id     = row["case_id"]
        scenario_id = row.get("scenario_id", "")
        split       = row.get("split", "train")

        try:
            events, labels, graph_ref, enterprise_graph, stitch_map = (
                load_enterprise_case(lib_root, row, repo_root)
            )
        except Exception as exc:
            print(f"  [skip] {case_id}: load error: {exc}")
            skipped += 1
            continue

        root_cause_node    = labels.get("root_cause_node", "")
        root_cause_diagram = labels.get("root_cause_diagram", "")

        g = build_graph_dict(
            case_id=case_id,
            scenario_id=scenario_id,
            split=split,
            enterprise_graph=enterprise_graph,
            events=events,
            root_cause_node=root_cause_node,
        )
        if g is None:
            print(f"  [skip] {case_id}: root_cause_node '{root_cause_node}' not in graph nodes")
            skipped += 1
            continue

        graphs.append(g)
        case_index.append({
            "case_id":             case_id,
            "scenario_id":         scenario_id,
            "split":               split,
            "root_cause_node":     root_cause_node,
            "root_cause_diagram":  root_cause_diagram,
            "root_cause_pattern":  labels.get("root_cause_pattern", ""),
            "root_index":          int(g["y"].item()),
            "node_count":          g["num_nodes"],
            "edge_count":          int(g["edge_index"].shape[1]),
            "event_count":         len(events),
            "node_id_to_index":    {nid: i for i, nid in enumerate(g["node_ids"])},
        })

    if skipped:
        print(f"  [warning] skipped {skipped} case(s).")
    return graphs, case_index
