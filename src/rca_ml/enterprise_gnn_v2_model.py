"""
enterprise_gnn_v2_model.py — Temporal Relation-Aware GNN for Enterprise RCA (V2).

EnterpriseRcaTemporalRelGNN extends V1 by adding relation-aware message passing:
  - One SAGEConv stack for ALL edges            (mirrors V1 baseline)
  - One SAGEConv stack for LOCAL edges only
  - One SAGEConv stack for CROSS_DIAGRAM edges only
  - One SAGEConv stack for VISION_CONNECTOR edges only
  - Concatenate the 4 relation embeddings → MLP → output logit

When edge_type is absent (old graphs.pt without edge_type), local/cross/vision
embeddings default to zeros — effectively falling back to all-edge GraphSAGE.

This is relation-aware GraphSAGE with temporal node features, NOT a full dynamic
temporal heterogeneous graph transformer. V1 remains the safe fallback.

No remediation content. Root cause comes from GNN graph reasoning only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .enterprise_gnn_dataset import FEATURE_NAMES, IN_DIM, EDGE_TYPE_TO_ID

# ── Lazy dependency detection ────────────────────────────────────────────────────

_torch_available = False
_pyg_available   = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _torch_available = True
    try:
        from torch_geometric.nn import SAGEConv
        _pyg_available = True
    except ImportError:
        pass
except ImportError:
    pass


# ── Dependency check ─────────────────────────────────────────────────────────────

def check_torch_geo_v2_requirement() -> None:
    if not _torch_available:
        print("[ERROR] PyTorch is required for Enterprise GNN RCA V2.")
        print("        Install from: https://pytorch.org/get-started/locally/")
        sys.exit(1)
    if not _pyg_available:
        print("[ERROR] torch_geometric is required for Enterprise GNN RCA V2.")
        print("        See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
        sys.exit(1)


# ── Feature indices (for predict_one_v2 evidence strings) ────────────────────────

_FI_IS_ALERTED  = FEATURE_NAMES.index("is_alerted")
_FI_ALERT_COUNT = FEATURE_NAMES.index("alert_count_norm")
_FI_MAX_SEV     = FEATURE_NAMES.index("max_severity_score")
_FI_CROSS_DIAG  = FEATURE_NAMES.index("cross_diagram_degree_norm")
_FI_DIST        = FEATURE_NAMES.index("distance_to_alert_norm")
_FI_SHARED      = FEATURE_NAMES.index("is_shared_entity")
_FI_PROP_CONS   = FEATURE_NAMES.index("propagation_consistency_score")
_FI_SEQ_POS     = FEATURE_NAMES.index("alert_sequence_position_norm")


# ── Model ─────────────────────────────────────────────────────────────────────────

if _pyg_available:
    class EnterpriseRcaTemporalRelGNN(nn.Module):
        """
        Temporal Relation-Aware GraphSAGE for node-level root-cause scoring.

        Architecture:
          x0       = ReLU(input_lin(x))
          h_all    = SAGEConv stack on all edges
          h_local  = SAGEConv stack on local-only edges (or zeros if absent)
          h_cross  = SAGEConv stack on cross-diagram-only edges (or zeros)
          h_vision = SAGEConv stack on vision-connector-only edges (or zeros)
          h        = ReLU(combine_lin(cat([h_all, h_local, h_cross, h_vision])))
          logits   = output_lin(h)   shape [num_nodes]

        When edge_type is None (old graph without type info), h_local/cross/vision
        are all-zeros — the model degrades to all-edge GraphSAGE (like V1).
        """

        def __init__(
            self,
            in_channels:     int   = IN_DIM,
            hidden_channels: int   = 64,
            num_layers:      int   = 2,
            dropout:         float = 0.2,
        ) -> None:
            super().__init__()
            self.in_channels     = in_channels
            self.hidden_channels = hidden_channels
            self.num_layers      = num_layers
            self.dropout_p       = dropout

            self.input_lin = nn.Linear(in_channels, hidden_channels)

            def _sage_stack() -> nn.ModuleList:
                return nn.ModuleList([
                    SAGEConv(hidden_channels, hidden_channels)
                    for _ in range(num_layers)
                ])

            self.convs_all    = _sage_stack()  # all edges
            self.convs_local  = _sage_stack()  # local edges only
            self.convs_cross  = _sage_stack()  # cross_diagram edges only
            self.convs_vision = _sage_stack()  # vision_connector_extraction only

            self.combine_lin = nn.Linear(4 * hidden_channels, hidden_channels)
            self.output_lin  = nn.Linear(hidden_channels, 1)

        def _run_stack(self, convs: nn.ModuleList, h, edge_index):
            for conv in convs:
                h = conv(h, edge_index)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout_p, training=self.training)
            return h

        def _relation_edges(self, edge_index, edge_type, type_id: int):
            """Return edge_index filtered to type_id, or None if empty."""
            if edge_type is None or edge_index.shape[1] == 0:
                return None
            mask = edge_type == type_id
            if not mask.any():
                return None
            return edge_index[:, mask]

        def forward(self, x, edge_index, edge_type=None):
            x0 = F.relu(self.input_lin(x))
            x0 = F.dropout(x0, p=self.dropout_p, training=self.training)

            # All-edge pass (always; equivalent to V1 when edge_type absent)
            h_all = self._run_stack(self.convs_all, x0, edge_index)

            zeros = torch.zeros_like(h_all)

            # Local edges
            ei_loc = self._relation_edges(edge_index, edge_type, EDGE_TYPE_TO_ID["local"])
            h_local = self._run_stack(self.convs_local, x0, ei_loc) if ei_loc is not None else zeros

            # Cross-diagram edges
            ei_cross = self._relation_edges(edge_index, edge_type, EDGE_TYPE_TO_ID["cross_diagram"])
            h_cross  = self._run_stack(self.convs_cross, x0, ei_cross) if ei_cross is not None else zeros

            # Vision-connector edges
            ei_vis = self._relation_edges(edge_index, edge_type, EDGE_TYPE_TO_ID["vision_connector_extraction"])
            h_vision = self._run_stack(self.convs_vision, x0, ei_vis) if ei_vis is not None else zeros

            # Combine: 4 × hidden → hidden → logit
            h = torch.cat([h_all, h_local, h_cross, h_vision], dim=-1)
            h = F.relu(self.combine_lin(h))
            h = F.dropout(h, p=self.dropout_p, training=self.training)

            return self.output_lin(h).squeeze(-1)  # [num_nodes]

else:
    class EnterpriseRcaTemporalRelGNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "EnterpriseRcaTemporalRelGNN requires torch_geometric. "
                "Call check_torch_geo_v2_requirement() before instantiating."
            )


# ── Config ────────────────────────────────────────────────────────────────────────

def build_gnn_v2_config(
    in_channels:     int   = IN_DIM,
    hidden_channels: int   = 64,
    num_layers:      int   = 2,
    dropout:         float = 0.2,
    top_k:           int   = 3,
) -> dict:
    return {
        "in_channels":          in_channels,
        "hidden_channels":      hidden_channels,
        "num_layers":           num_layers,
        "dropout":              dropout,
        "top_k":                top_k,
        "model_type":           "EnterpriseRcaTemporalRelGNN",
        "gnn_architecture":     "RelationAwareTemporalGraphSAGE",
        "feature_dim":          IN_DIM,
        "uses_edge_type":       True,
        "uses_temporal_features": True,
        "relations":            ["local", "cross_diagram", "vision_connector_extraction"],
        "notes": (
            "Relation-aware GraphSAGE with temporal node features. "
            "Not a full dynamic temporal heterogeneous graph transformer. "
            "V1 EnterpriseRcaGNN remains the safe fallback."
        ),
    }


# ── Serialisation ─────────────────────────────────────────────────────────────────

def save_gnn_v2(
    model: "EnterpriseRcaTemporalRelGNN",
    config: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch as _t
    _t.save(model.state_dict(), out_dir / "enterprise_gnn_v2_rca.pt")
    (out_dir / "enterprise_gnn_v2_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "feature_columns.json").write_text(
        json.dumps(FEATURE_NAMES, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_gnn_v2(model_path: Path, config_path: Path) -> "tuple[EnterpriseRcaTemporalRelGNN, dict]":
    import torch as _t
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model  = EnterpriseRcaTemporalRelGNN(
        in_channels=config.get("in_channels", IN_DIM),
        hidden_channels=config.get("hidden_channels", 64),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.2),
    )
    model.load_state_dict(_t.load(str(model_path), map_location="cpu"))
    model.eval()
    return model, config


# ── PyG Data conversion ───────────────────────────────────────────────────────────

def graph_dict_to_pyg_v2(g: dict):
    """Convert graph_dict to PyG Data, always forwarding edge_type when present."""
    from torch_geometric.data import Data
    data = Data(
        x=g["x"],
        edge_index=g["edge_index"],
        y=g["y"],
        num_nodes=g["num_nodes"],
    )
    if g.get("edge_type") is not None:
        data.edge_type = g["edge_type"]
    return data


# ── Scoring ────────────────────────────────────────────────────────────────────────

def score_nodes_v2(model: "EnterpriseRcaTemporalRelGNN", data) -> np.ndarray:
    """Run V2 model, return softmax-normalised per-node probabilities."""
    import torch as _t
    model.eval()
    with _t.no_grad():
        edge_type = getattr(data, "edge_type", None)
        logits    = model(data.x, data.edge_index, edge_type)
        probs     = _t.softmax(logits, dim=0).cpu().numpy()
    return probs


# ── Single-case prediction ────────────────────────────────────────────────────────

def predict_one_v2(
    model: "EnterpriseRcaTemporalRelGNN",
    graph_dict: dict,
    labels_dict: dict | None = None,
    top_k: int = 3,
) -> dict:
    """
    Run V2 GNN inference on one graph_dict.

    Output is schema-compatible with V1 predict_one() but uses V2 source labels
    and includes model_notes. Contains no remediation content.
    """
    data   = graph_dict_to_pyg_v2(graph_dict)
    scores = score_nodes_v2(model, data)

    node_ids    = graph_dict["node_ids"]
    node_types  = graph_dict["node_type_list"]
    diagram_ids = graph_dict["diagram_id_list"]
    num_nodes   = graph_dict["num_nodes"]
    x           = graph_dict["x"].numpy()
    aedm: dict[str, list[str]] = graph_dict.get("alert_event_diagram_map", {})

    _has_edge_type = data.edge_type is not None if hasattr(data, "edge_type") else False
    _edge_type_obj = getattr(data, "edge_type", None)

    # Count relation-type edges for evidence
    _n_local = _n_cross = _n_vision = 0
    if _edge_type_obj is not None and graph_dict["edge_index"].shape[1] > 0:
        import torch as _t
        _n_local  = int((_edge_type_obj == EDGE_TYPE_TO_ID["local"]).sum())
        _n_cross  = int((_edge_type_obj == EDGE_TYPE_TO_ID["cross_diagram"]).sum())
        _n_vision = int((_edge_type_obj == EDGE_TYPE_TO_ID["vision_connector_extraction"]).sum())

    ranked_idx = np.argsort(-scores)

    top_candidates = []
    for rank, idx in enumerate(ranked_idx[:top_k], start=1):
        nid  = node_ids[idx]
        sc   = float(scores[idx])
        observed_diags = aedm.get(nid, [diagram_ids[idx]])
        candidate_diag = observed_diags[0] if observed_diags else diagram_ids[idx]
        top_candidates.append({
            "rank":                      rank,
            "node_id":                   nid,
            "diagram_id":                candidate_diag,
            "node_observed_in_diagrams": observed_diags,
            "node_type":                 node_types[idx],
            "score":                     round(sc, 4),
            "evidence": [
                f"alert_count={round(float(x[idx, _FI_ALERT_COUNT]), 3)}",
                f"cross_diagram_degree={round(float(x[idx, _FI_CROSS_DIAG]), 3)}",
                f"propagation_consistency={round(float(x[idx, _FI_PROP_CONS]), 3)}",
                f"sequence_position={round(float(x[idx, _FI_SEQ_POS]), 3)}",
                f"shared_entity={bool(x[idx, _FI_SHARED] > 0.5)}",
            ],
        })

    predicted_root      = node_ids[ranked_idx[0]]
    confidence          = float(scores[ranked_idx[0]])
    root_observed_diags = aedm.get(predicted_root, [diagram_ids[ranked_idx[0]]])
    predicted_diag      = root_observed_diags[0] if root_observed_diags else diagram_ids[ranked_idx[0]]

    alerted_diagrams: list[str] = []
    for i in range(num_nodes):
        if x[i, _FI_IS_ALERTED] > 0.5:
            for d in aedm.get(node_ids[i], [diagram_ids[i]]):
                if d and d not in alerted_diagrams:
                    alerted_diagrams.append(d)
    impacted_diags = (
        labels_dict.get("impacted_diagrams", alerted_diagrams)
        if labels_dict else alerted_diagrams
    )

    result: dict = {
        "scenario_id":          graph_dict.get("scenario_id", ""),
        "case_id":              graph_dict.get("case_id", ""),
        "mode":                 "enterprise_gnn_v2_temporal_relation_aware",
        "rca_source":           "Enterprise GNN RCA V2 — Temporal Relation-Aware GraphSAGE",
        "predicted_root_cause": predicted_root,
        "root_cause_diagram":   predicted_diag,
        "confidence":           round(confidence, 4),
        "top_candidates":       top_candidates,
        "impacted_diagrams":    impacted_diags,
        "alert_count":          graph_dict.get("event_count", int(x[:, _FI_IS_ALERTED].sum())),
        "model_notes": {
            "uses_temporal_features": True,
            "uses_edge_type":         _has_edge_type,
            "relations":              ["local", "cross_diagram", "vision_connector_extraction"],
            "local_edges":            _n_local,
            "cross_diagram_edges":    _n_cross,
            "vision_edges":           _n_vision,
            "note": (
                "Temporal-aware relation-aware GraphSAGE. "
                "Not a fully dynamic temporal heterogeneous graph transformer."
            ),
        },
    }

    if labels_dict:
        gt           = labels_dict.get("root_cause_node", "")
        top_k_nodes  = [c["node_id"] for c in top_candidates]
        ranked_all   = list(ranked_idx)
        rank_val     = next(
            (i + 1 for i, idx in enumerate(ranked_all) if node_ids[idx] == gt),
            num_nodes,
        )
        result["evaluation"] = {
            "ground_truth_node": gt,
            "correct_top1":      predicted_root == gt,
            "correct_top_k":     gt in top_k_nodes,
            "reciprocal_rank":   round(1.0 / rank_val, 4),
            "rank":              rank_val,
        }

    return result


# ── Dataset evaluation ─────────────────────────────────────────────────────────────

def evaluate_dataset_v2(
    model: "EnterpriseRcaTemporalRelGNN",
    graph_dicts: list[dict],
    index: list[dict],
    top_k: int = 3,
) -> dict:
    """Case-level top-1 / top-k / MRR evaluation using V2 model. No remediation output."""
    top1 = top_k_hits = 0
    rr_sum = 0.0
    n = 0
    per_case:  list[dict] = []
    failed:    list[dict] = []
    per_split: dict[str, dict] = {}

    for g, meta in zip(graph_dicts, index):
        root_cause = meta.get("root_cause_node", "")
        if not root_cause or root_cause not in g["node_ids"]:
            continue

        scores   = score_nodes_v2(model, graph_dict_to_pyg_v2(g))
        node_ids = g["node_ids"]
        ranked   = np.argsort(-scores)

        pred_root   = node_ids[ranked[0]]
        top_k_nodes = [node_ids[i] for i in ranked[:top_k]]

        rank_matches = [i for i, idx in enumerate(ranked) if node_ids[idx] == root_cause]
        rank  = rank_matches[0] + 1 if rank_matches else len(node_ids)
        rr    = 1.0 / rank
        hit1  = pred_root == root_cause
        hitk  = root_cause in top_k_nodes

        top1       += int(hit1)
        top_k_hits += int(hitk)
        rr_sum     += rr
        n          += 1

        sp = meta.get("split", "unknown")
        ps = per_split.setdefault(sp, {"case_count": 0, "top1": 0, "topk": 0, "rr_sum": 0.0})
        ps["case_count"] += 1
        ps["top1"]       += int(hit1)
        ps["topk"]       += int(hitk)
        ps["rr_sum"]     += rr

        per_case.append({
            "case_id":           meta["case_id"],
            "scenario_id":       meta["scenario_id"],
            "split":             sp,
            "predicted_root":    pred_root,
            "expected_root":     root_cause,
            "root_pattern":      meta.get("root_cause_pattern", ""),
            "correct_top1":      hit1,
            f"correct_top{top_k}": hitk,
            "rr":                round(rr, 4),
            "confidence":        round(float(scores[ranked[0]]), 4),
        })
        if not hit1:
            failed.append({"case_id": meta["case_id"], "expected": root_cause, "predicted": pred_root})

    per_split_metrics = {
        sp: {
            "case_count":          v["case_count"],
            "top1_accuracy":       round(v["top1"] / v["case_count"], 4),
            f"top{top_k}_accuracy": round(v["topk"] / v["case_count"], 4),
            "mrr":                 round(v["rr_sum"] / v["case_count"], 4),
        }
        for sp, v in per_split.items()
    }

    return {
        "case_count":               n,
        "top1_accuracy":            round(top1      / n, 4) if n else 0.0,
        f"top{top_k}_accuracy":     round(top_k_hits / n, 4) if n else 0.0,
        "mrr":                      round(rr_sum    / n, 4) if n else 0.0,
        "per_split_metrics":        per_split_metrics,
        "failed_cases":             failed,
        "per_case_predictions":     per_case,
        "model_type":               "EnterpriseRcaTemporalRelGNN",
        "uses_edge_type":           True,
        "uses_temporal_features":   True,
    }
