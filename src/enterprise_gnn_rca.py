"""
enterprise_gnn_rca.py

Shared GNN utilities for InfraGraph AI Enterprise RCA.

Used by:
  scripts/train_enterprise_gnn_rca.py  -- training
  scripts/run_enterprise_gnn_inference.py  -- per-scenario inference

Exports:
  Constants:   NODE_TYPES, DIAGRAM_TYPES, FEATURE_NAMES, IN_FEAT, SEV_SCORE
  Model:       EnterpriseGCN
  Features:    _extract_features, _build_adj_norm
  Scenarios:   _load_scenario, find_scenario_dir
  Inference:   _score_scenario, run_inference_for_scenario
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

# ── PyTorch (required) ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print(
        "[ERROR] PyTorch is not installed.\n"
        "        Install with:  pip install torch"
    )
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# VOCABULARY & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NODE_TYPES = [
    "router", "switch", "firewall", "server",
    "database", "load_balancer", "cloud_or_wan", "service", "unknown",
]
DIAGRAM_TYPES = [
    "branch_topology", "wan_topology", "datacenter_topology",
    "app_db_topology", "shared_services_topology", "unknown",
]

_NODE_T_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
_DIAG_T_IDX = {d: i for i, d in enumerate(DIAGRAM_TYPES)}

SEV_SCORE = {
    "critical": 1.0, "high": 0.8, "warning": 0.6,
    "medium": 0.5, "low": 0.3, "info": 0.1,
}

# Feature vector (34-dimensional) — must match _extract_features() exactly
FEATURE_NAMES: list[str] = (
    [f"type_{t}" for t in NODE_TYPES]           # 9  one-hot node type
    + [f"diag_{d}" for d in DIAGRAM_TYPES]      # 6  one-hot diagram type
    + ["has_alert",                              # 1
       "max_severity_score",                     # 1
       "earliest_alert_score",                   # 1
       "alert_count_norm"]                       # 1
    + ["is_impacted_node",                       # 1
       "is_shared_entity",                       # 1
       "is_cross_diagram_bridge"]                # 1
    + ["in_degree_norm",                         # 1
       "out_degree_norm",                        # 1
       "total_degree_norm",                      # 1
       "downstream_reach_norm",                  # 1
       "upstream_reach_norm",                    # 1
       "cross_diag_degree_norm",                 # 1
       "cluster_size_norm"]                      # 1
    + ["rc_prior_firewall",                      # 1
       "rc_prior_router",                        # 1
       "rc_prior_lb",                            # 1
       "rc_prior_database",                      # 1
       "rc_prior_service"]                       # 1
)
IN_FEAT = len(FEATURE_NAMES)   # 34


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _bfs(start: int, adj: dict[int, list[int]]) -> int:
    visited: set[int] = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nb in adj.get(cur, []):
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return len(visited) - 1


def _extract_features(
    graph: dict,
    alerts: dict,
) -> tuple[np.ndarray, dict[str, int], list[dict]]:
    """Build the (N, 34) feature matrix for one enterprise scenario."""
    nodes    = graph["nodes"]
    edges    = graph["edges"]
    n        = len(nodes)
    node_idx = {nd["id"]: i for i, nd in enumerate(nodes)}

    alert_map: dict[str, list[dict]] = defaultdict(list)
    for a in alerts.get("alerts", []):
        alert_map[a["node"]].append(a)
    impacted_set = set(alerts.get("impacted_nodes", []))
    total_alerts = max(len(alerts.get("alerts", [])), 1)

    in_deg  = [0] * n
    out_deg = [0] * n
    cd_deg  = [0] * n
    fwd_adj: dict[int, list[int]] = defaultdict(list)
    rev_adj: dict[int, list[int]] = defaultdict(list)

    for edge in edges:
        si = node_idx.get(edge["source"])
        ti = node_idx.get(edge["target"])
        if si is None or ti is None:
            continue
        out_deg[si] += 1
        in_deg[ti]  += 1
        fwd_adj[si].append(ti)
        rev_adj[ti].append(si)
        if edge.get("edge_scope") == "cross_diagram":
            cd_deg[si] += 1
            cd_deg[ti] += 1

    down_reach = [_bfs(i, fwd_adj) for i in range(n)]
    up_reach   = [_bfs(i, rev_adj) for i in range(n)]

    cl_size: dict[str, int] = {}
    clusters = graph.get("diagram_clusters", [])
    if isinstance(clusters, list):
        for cl in clusters:
            sz = len(cl.get("node_ids", []))
            for nid in cl.get("node_ids", []):
                cl_size[nid] = sz
    elif isinstance(clusters, dict):
        for _, cl in clusters.items():
            if isinstance(cl, dict):
                sz = len(cl.get("node_ids", []))
                for nid in cl.get("node_ids", []):
                    cl_size[nid] = sz

    max_n = max(n - 1, 1)
    X: list[list[float]] = []

    for i, nd in enumerate(nodes):
        nid   = nd["id"]
        ntype = nd.get("type", "unknown")
        dtype = nd.get("diagram_type", "unknown")

        t_oh = [1.0 if ntype == t else 0.0 for t in NODE_TYPES]
        d_oh = [1.0 if dtype == d else 0.0 for d in DIAGRAM_TYPES]

        nalerts     = alert_map.get(nid, [])
        has_alert   = 1.0 if nalerts else 0.0
        max_sev     = max((SEV_SCORE.get(a.get("severity", "info"), 0.1) for a in nalerts), default=0.0)
        early_t     = min((a.get("time_offset_min", 999) for a in nalerts), default=0)
        early_score = 1.0 / (1.0 + early_t) if nalerts else 0.0
        ac_norm     = len(nalerts) / total_alerts

        is_impacted = 1.0 if nid in impacted_set else 0.0
        is_shared   = 1.0 if nd.get("is_shared_entity", False) else 0.0
        is_bridge   = 1.0 if nd.get("is_cross_diagram_bridge", False) else 0.0

        in_d  = in_deg[i]  / max_n
        out_d = out_deg[i] / max_n
        tot_d = (in_deg[i] + out_deg[i]) / (2 * max_n)
        dn_r  = down_reach[i] / max_n
        up_r  = up_reach[i]   / max_n
        cd_d  = cd_deg[i]     / max_n
        cl_sz = cl_size.get(nid, 1) / max_n

        is_fw  = 1.0 if ntype == "firewall"      else 0.0
        is_rtr = 1.0 if ntype == "router"         else 0.0
        is_lb  = 1.0 if ntype == "load_balancer"  else 0.0
        is_db  = 1.0 if ntype == "database"       else 0.0
        is_svc = 1.0 if ntype == "service"        else 0.0

        row = (t_oh + d_oh
               + [has_alert, max_sev, early_score, ac_norm]
               + [is_impacted, is_shared, is_bridge]
               + [in_d, out_d, tot_d, dn_r, up_r, cd_d, cl_sz]
               + [is_fw, is_rtr, is_lb, is_db, is_svc])
        X.append(row)

    return np.array(X, dtype=np.float32), node_idx, nodes


def _build_adj_norm(n: int, node_idx: dict[str, int], edges: list[dict]) -> np.ndarray:
    """Symmetric D^{-1/2} A D^{-1/2} normalisation with self-loops."""
    A = np.zeros((n, n), dtype=np.float32)
    for edge in edges:
        si = node_idx.get(edge["source"])
        ti = node_idx.get(edge["target"])
        if si is None or ti is None:
            continue
        A[si, ti] = 1.0
        A[ti, si] = 1.0
    A += np.eye(n, dtype=np.float32)
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    D = np.diag(d_inv_sqrt)
    return (D @ A @ D).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class _GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return torch.mm(adj_norm, torch.mm(x, self.weight))


class EnterpriseGCN(nn.Module):
    """3-layer GCN: input_dim -> hidden1 -> hidden2 -> 1 (node score)."""

    def __init__(
        self,
        in_dim:  int,
        hidden1: int = 96,
        hidden2: int = 48,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gc1  = _GCNLayer(in_dim,  hidden1)
        self.gc2  = _GCNLayer(hidden1, hidden2)
        self.gc3  = _GCNLayer(hidden2, 1)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.gc1(x, adj_norm))
        h1 = self.drop(h1)
        h2 = F.relu(self.gc2(h1, adj_norm))
        h2 = self.drop(h2)
        return self.gc3(h2, adj_norm).squeeze(-1)   # (N,)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: Path, device: "torch.device | None" = None) -> EnterpriseGCN:
    """Load an EnterpriseGCN from a checkpoint saved by save_model()."""
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(str(model_path), map_location=device)
    arch = ckpt.get("architecture", {})
    model = EnterpriseGCN(
        in_dim  = arch.get("in_dim",  IN_FEAT),
        hidden1 = arch.get("hidden1", 96),
        hidden2 = arch.get("hidden2", 48),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_scenario(scenario_dir: Path) -> "dict | None":
    """Load one scenario directory into a feature dict ready for inference."""
    g_path = scenario_dir / "enterprise_graph.json"
    a_path = scenario_dir / "alerts.json"
    if not g_path.exists() or not a_path.exists():
        return None
    try:
        graph  = json.loads(g_path.read_text(encoding="utf-8"))
        alerts = json.loads(a_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [warn] Could not load {scenario_dir.name}: {exc}")
        return None

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes:
        return None

    node_idx = {nd["id"]: i for i, nd in enumerate(nodes)}
    root_cause = alerts.get("root_cause", "")

    X, node_idx, nodes = _extract_features(graph, alerts)
    A = _build_adj_norm(len(nodes), node_idx, edges)

    rc_i = node_idx.get(root_cause)
    labels = np.zeros(len(nodes), dtype=np.float32)
    if rc_i is not None:
        labels[rc_i] = 1.0

    return {
        "scenario_id":  graph.get("scenario_id", scenario_dir.name),
        "X":            X,
        "A":            A,
        "labels":       labels,
        "rc_idx":       rc_i,
        "node_ids":     [nd["id"] for nd in nodes],
        "node_data":    nodes,
        "graph_data":   graph,
        "alerts_data":  alerts,
        "graph_path":   str(g_path),
        "alerts_path":  str(a_path),
    }


SPLITS = ("train", "val", "test")


def find_scenario_dir(
    dataset_root: Path,
    scenario_id: str,
    split: "str | None" = None,
) -> "tuple[Path, str] | None":
    """
    Locate the directory for a scenario_id.

    If split is given, look only there.
    Otherwise search train → val → test.

    Returns (scenario_dir, split_name) or None.
    """
    search = [split] if split else list(SPLITS)
    for sp in search:
        d = dataset_root / "scenarios" / sp / scenario_id
        if d.is_dir():
            return d, sp
    return None


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _score_scenario(
    model: EnterpriseGCN,
    sc: dict,
    device: "torch.device",
) -> "tuple[np.ndarray, list[tuple[int, float]]]":
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(sc["X"]).to(device)
        A = torch.from_numpy(sc["A"]).to(device)
        scores = torch.sigmoid(model(X, A)).cpu().numpy()
    ranked = sorted(enumerate(scores.tolist()), key=lambda x: -x[1])
    return scores, ranked


def run_inference_for_scenario(
    model: EnterpriseGCN,
    sc: dict,
    out_dir: Path,
    device: "torch.device",
    model_path: "Path | None" = None,
) -> dict:
    """
    Run GNN inference for one loaded scenario, save result JSON, return result dict.

    The result file is named:  <out_dir>/<scenario_id>_enterprise_gnn_rca_result.json
    """
    scores, ranked = _score_scenario(model, sc, device)

    alerts_data = sc["alerts_data"]
    graph_data  = sc["graph_data"]
    node_data   = sc["node_data"]
    node_ids    = sc["node_ids"]
    rc_node     = alerts_data.get("root_cause", "")

    pred_node = node_ids[ranked[0][0]]
    rc_rank   = next(
        (i + 1 for i, (idx, _) in enumerate(ranked) if node_ids[idx] == rc_node),
        -1,
    )

    top_candidates: list[dict] = []
    impacted_set = set(alerts_data.get("impacted_nodes", []))
    for rank, (idx, score) in enumerate(ranked[:10], start=1):
        nd  = node_data[idx]
        nid = node_ids[idx]
        node_alerts = [a for a in alerts_data.get("alerts", []) if a["node"] == nid]
        top_candidates.append({
            "rank":                    rank,
            "node_id":                 nid,
            "score":                   round(float(score), 6),
            "type":                    nd.get("type", "unknown"),
            "diagram_id":              nd.get("diagram_id", ""),
            "diagram_type":            nd.get("diagram_type", ""),
            "has_alert":               bool(node_alerts),
            "is_impacted_node":        nid in impacted_set,
            "is_shared_entity":        bool(nd.get("is_shared_entity", False)),
            "is_cross_diagram_bridge": bool(nd.get("is_cross_diagram_bridge", False)),
        })

    cross_edge_count = sum(
        1 for e in graph_data.get("edges", []) if e.get("edge_scope") == "cross_diagram"
    )

    result: dict[str, Any] = {
        "scenario_id":             sc["scenario_id"],
        "model_type":              "Enterprise GCN RCA",
        "backend":                 "torch",
        "inference_source":        "trained_enterprise_gnn",
        "predicted_root_cause":    pred_node,
        "ground_truth_root_cause": rc_node,
        "is_correct":              pred_node == rc_node,
        "ground_truth_rank":       rc_rank,
        "root_cause_diagram":      alerts_data.get("root_cause_diagram", ""),
        "impacted_diagrams":       alerts_data.get("impacted_diagrams", []),
        "alert_count":             len(alerts_data.get("alerts", [])),
        "node_count":              len(node_ids),
        "edge_count":              len(graph_data.get("edges", [])),
        "cross_diagram_edge_count": cross_edge_count,
        "top_candidates":          top_candidates,
        "alerts":                  alerts_data.get("alerts", []),
        "impact_paths":            alerts_data.get("impact_paths", []),
        "enterprise_graph_path":   sc.get("graph_path", ""),
        "alerts_path":             sc.get("alerts_path", ""),
        "model_path":              str(model_path) if model_path else "",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{sc['scenario_id']}_enterprise_gnn_rca_result.json"
    path  = out_dir / fname
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    correct = "[CORRECT]" if result["is_correct"] else "[WRONG]"
    print(f"  Result saved: {path}")
    print(f"  {correct}  predicted={pred_node}  gt={rc_node}  rank={rc_rank}")

    return result
