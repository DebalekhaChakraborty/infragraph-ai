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

from .features import SEVERITY_SCORE, _LARGE, normalize_repo_path

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
IN_DIM            = NUM_NODE_TYPES + NUM_DIAGRAM_TYPES + NUMERIC_DIM  # 34

FEATURE_NAMES: list[str] = (
    [f"nt_{t}" for t in NODE_TYPES]
    + [f"dt_{t}" for t in DIAGRAM_TYPES]
    + [
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
)
assert len(FEATURE_NAMES) == IN_DIM, f"{len(FEATURE_NAMES)} != {IN_DIM}"


# ── Graph construction helpers ─────────────────────────────────────────────────

def _build_enterprise_nx_graph(enterprise_graph: dict) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()
    for node in enterprise_graph.get("nodes", []):
        nid = node.get("id", "")
        if nid:
            G.add_node(nid, **{k: v for k, v in node.items() if k != "id"})
    for edge in enterprise_graph.get("edges", []):
        s, t = edge.get("source", ""), edge.get("target", "")
        if s and t and s != t:
            G.add_edge(s, t, edge_type="local")
    for edge in enterprise_graph.get("cross_diagram_edges", []):
        s, t = edge.get("source", ""), edge.get("target", "")
        if s and t and s != t:
            G.add_edge(s, t, edge_type="cross_diagram")
    return G


def _cross_diagram_degrees(enterprise_graph: dict) -> dict[str, int]:
    deg: dict[str, int] = defaultdict(int)
    for edge in enterprise_graph.get("cross_diagram_edges", []):
        s, t = edge.get("source", ""), edge.get("target", "")
        if s:
            deg[s] += 1
        if t:
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


def _node_feature_vec(
    node_data: dict,
    alert_map: dict,
    time_rank: dict,
    num_alerted: int,
    max_alert_count: int,
    G: nx.DiGraph,
    G_und: nx.Graph,
    pr: dict, bc: dict, cl: dict,
    alerted_set: set,
    cross_diag_deg: dict,
    total_nodes: int,
) -> list[float]:
    nid = node_data.get("id", "")

    # One-hot: node type
    nt_key = _normalise_node_type(node_data.get("type", "unknown"))
    nt_vec = [0.0] * NUM_NODE_TYPES
    nt_vec[_NODE_TYPE_IDX[nt_key]] = 1.0

    # One-hot: diagram type (prefer diagram_type field, fall back to diagram_id)
    dt_raw  = node_data.get("diagram_type", node_data.get("diagram_id", "unknown"))
    dt_key  = _normalise_diagram_type(dt_raw)
    dt_vec  = [0.0] * NUM_DIAGRAM_TYPES
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

    # Structural features
    n_scale = max(1, total_nodes)
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

    # Reverse reachability: alerted ancestors
    reverse_reach = 0
    if nid in G:
        reverse_reach = len(nx.ancestors(G, nid) & (alerted_set - {nid}))

    # Normalize
    denom_time = 1440.0  # 1 day in minutes
    dist_norm  = min(1.0, min_dist / n_scale) if min_dist < _LARGE else 1.0
    rr_norm    = reverse_reach / max(1, len(alerted_set))
    rank_norm  = rank / max(1, num_alerted + 1)

    numeric: list[float] = [
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
    assert len(numeric) == NUMERIC_DIM
    return nt_vec + dt_vec + numeric


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
    time_rank   = {n: i + 1 for i, (n, _) in enumerate(first_times)}
    num_alerted = len(alerted_set)

    node_data_map = {n["id"]: n for n in raw_nodes if n.get("id")}

    # Feature matrix
    x_rows = []
    for nid in node_ids:
        nd   = node_data_map[nid]
        fvec = _node_feature_vec(
            nd, alert_map, time_rank, num_alerted, max_alert_count,
            G, G_und, pr, bc, cl, alerted_set, cross_deg, num_nodes,
        )
        x_rows.append(fvec)

    x = torch.FloatTensor(x_rows)  # [N, IN_DIM]

    # Edge index — bidirectional
    fwd_edges: list[tuple[int, int]] = []
    for src, tgt in G.edges():
        if src in nid_to_idx and tgt in nid_to_idx:
            fwd_edges.append((nid_to_idx[src], nid_to_idx[tgt]))
    all_edges = list(set(fwd_edges + [(t, s) for s, t in fwd_edges]))
    if all_edges:
        import torch as _t
        edge_index = _t.LongTensor(all_edges).t().contiguous()  # [2, E]
    else:
        import torch as _t
        edge_index = _t.zeros((2, 0), dtype=_t.long)

    y = (torch.tensor(nid_to_idx[root_cause_node], dtype=torch.long)
         if root_cause_node else torch.tensor(-1, dtype=torch.long))

    return {
        "x":               x,
        "edge_index":      edge_index,
        "y":               y,
        "num_nodes":       num_nodes,
        "case_id":         case_id,
        "scenario_id":     scenario_id,
        "split":           split,
        "event_count":     len(events),
        "node_ids":        node_ids,
        "node_type_list":  [node_data_map[nid].get("type", "unknown") for nid in node_ids],
        "diagram_id_list": [node_data_map[nid].get("diagram_id", "")   for nid in node_ids],
        "root_cause_node": root_cause_node or "",
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
