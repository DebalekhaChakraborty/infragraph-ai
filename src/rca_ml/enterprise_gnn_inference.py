"""
enterprise_gnn_inference.py — Prediction and evaluation for Enterprise GNN RCA.

Output uses label "Enterprise GNN RCA" (rca_source field).
No remediation content is produced here.
"""
from __future__ import annotations

import numpy as np

from .enterprise_gnn_dataset import FEATURE_NAMES
from .enterprise_gnn_model import graph_dict_to_pyg, score_nodes
from .features import NODE_TYPE_PRIORITY

# Pre-compute feature column indices once
_FI_IS_ALERTED    = FEATURE_NAMES.index("is_alerted")
_FI_ALERT_COUNT   = FEATURE_NAMES.index("alert_count_norm")
_FI_MAX_SEV       = FEATURE_NAMES.index("max_severity_score")
_FI_CROSS_DIAG    = FEATURE_NAMES.index("cross_diagram_degree_norm")
_FI_DIST          = FEATURE_NAMES.index("distance_to_alert_norm")
_FI_SHARED        = FEATURE_NAMES.index("is_shared_entity")
_FI_PAGERANK      = FEATURE_NAMES.index("pagerank")

_FORBIDDEN_KEYS = frozenset({
    "recommended_actions", "remediation_steps", "resolution_steps",
    "rollback_steps", "validation_steps", "servicenow_incident_summary",
    "remediation_steps", "resolution", "rollback",
})


def _assert_clean(d: dict) -> None:
    for k in _FORBIDDEN_KEYS:
        if k in d:
            raise ValueError(f"GNN inference output contains forbidden key: {k!r}")


# ── Heuristic baseline (no model required) ─────────────────────────────────────

def baseline_heuristic(graph_dict: dict) -> np.ndarray:
    """
    Simple heuristic for node root-cause ranking used in baseline comparison.

    Combines alert severity, node type priority, and cross-diagram connectivity.
    Stored in reports as 'baseline_topology_score' only — not shown in UI.
    """
    x          = graph_dict["x"].numpy()          # [N, IN_DIM]
    node_types = graph_dict["node_type_list"]
    num_nodes  = graph_dict["num_nodes"]

    def _ntp(t: str) -> float:
        t = t.lower()
        return NODE_TYPE_PRIORITY.get(t, max(
            (v for k, v in NODE_TYPE_PRIORITY.items() if k in t), default=0.1
        ))

    scores = np.zeros(num_nodes, dtype=float)
    for i, ntype in enumerate(node_types):
        is_al   = float(x[i, _FI_IS_ALERTED])
        max_sev = float(x[i, _FI_MAX_SEV])
        cross   = float(x[i, _FI_CROSS_DIAG])
        dist    = float(x[i, _FI_DIST])          # 0 = co-located with alert, 1 = far
        ntp     = _ntp(ntype)
        scores[i] = (
            is_al * max_sev * 0.5
            + cross * 0.2
            + ntp * 0.15
            + (1.0 - dist) * 0.15
        )

    total = scores.sum()
    if total > 0:
        scores /= total
    return scores


# ── Single-case prediction ─────────────────────────────────────────────────────

def predict_one(
    model,
    graph_dict: dict,
    labels_dict: dict | None = None,
    top_k: int = 3,
) -> dict:
    """
    Run GNN inference on one graph_dict and return a result JSON dict.

    Labels_dict is optional; if provided, ground-truth comparison is included.
    Output never contains remediation content.
    """
    data    = graph_dict_to_pyg(graph_dict)
    scores  = score_nodes(model, data)

    node_ids     = graph_dict["node_ids"]
    node_types   = graph_dict["node_type_list"]
    diagram_ids  = graph_dict["diagram_id_list"]
    num_nodes    = graph_dict["num_nodes"]
    x            = graph_dict["x"].numpy()

    ranked_idx   = np.argsort(-scores)

    top_candidates = []
    for rank, idx in enumerate(ranked_idx[:top_k], start=1):
        nid   = node_ids[idx]
        ntype = node_types[idx]
        diag  = diagram_ids[idx]
        sc    = float(scores[idx])
        top_candidates.append({
            "rank":       rank,
            "node_id":    nid,
            "diagram_id": diag,
            "node_type":  ntype,
            "score":      round(sc, 4),
            "evidence": [
                f"alert_count={round(float(x[idx, _FI_ALERT_COUNT]), 3)}",
                f"cross_diagram_degree={round(float(x[idx, _FI_CROSS_DIAG]), 3)}",
                f"distance_to_alert={round(float(x[idx, _FI_DIST]), 3)}",
                f"shared_entity={bool(x[idx, _FI_SHARED] > 0.5)}",
            ],
        })

    predicted_root = node_ids[ranked_idx[0]]
    predicted_diag = diagram_ids[ranked_idx[0]]
    confidence     = float(scores[ranked_idx[0]])

    # Impacted diagrams: from labels if available, else unique alert node diagrams
    alerted_diagrams = list(dict.fromkeys(
        diagram_ids[i] for i in range(num_nodes)
        if x[i, _FI_IS_ALERTED] > 0.5
    ))
    impacted_diags = labels_dict.get("impacted_diagrams", alerted_diagrams) if labels_dict else alerted_diagrams

    result: dict = {
        "scenario_id":          graph_dict.get("scenario_id", ""),
        "case_id":              graph_dict.get("case_id", ""),
        "mode":                 "enterprise_gnn_rca",
        "rca_source":           "Enterprise GNN RCA",
        "predicted_root_cause": predicted_root,
        "root_cause_diagram":   predicted_diag,
        "confidence":           round(confidence, 4),
        "top_candidates":       top_candidates,
        "impacted_diagrams":    impacted_diags,
        "alert_count":          graph_dict.get("event_count", int(x[:, _FI_IS_ALERTED].sum())),
        "remediation":          None,
    }

    if labels_dict:
        gt = labels_dict.get("root_cause_node", "")
        result["expected_root_cause"] = gt
        result["correct"] = (predicted_root == gt) if gt else None
    else:
        result["correct"] = None

    _assert_clean(result)
    return result


# ── Dataset evaluation ─────────────────────────────────────────────────────────

def evaluate_dataset(
    model,
    graph_dicts: list[dict],
    index: list[dict],
    top_k: int = 3,
) -> dict:
    """
    Case-level top-1 / top-k / MRR evaluation across graph_dicts.

    Also computes a heuristic baseline_topology_score for comparison.
    No remediation content in output.
    """
    top1 = top_k_hits = 0
    rr_sum = 0.0
    n = 0
    baseline_top1 = 0
    per_case: list[dict] = []
    failed:   list[dict] = []

    for g, meta in zip(graph_dicts, index):
        root_cause = meta.get("root_cause_node", "")
        if not root_cause or root_cause not in g["node_ids"]:
            continue

        scores    = score_nodes(model, graph_dict_to_pyg(g))
        b_scores  = baseline_heuristic(g)

        node_ids  = g["node_ids"]
        ranked    = np.argsort(-scores)
        b_ranked  = np.argsort(-b_scores)

        pred_root = node_ids[ranked[0]]
        b_pred    = node_ids[b_ranked[0]]
        top_k_nodes = [node_ids[i] for i in ranked[:top_k]]

        rank_matches = [i for i, idx in enumerate(ranked) if node_ids[idx] == root_cause]
        rank  = rank_matches[0] + 1 if rank_matches else len(node_ids)
        rr    = 1.0 / rank

        hit1  = pred_root == root_cause
        hitk  = root_cause in top_k_nodes
        top1     += int(hit1)
        top_k_hits += int(hitk)
        rr_sum   += rr
        n        += 1
        baseline_top1 += int(b_pred == root_cause)

        per_case.append({
            "case_id":         meta["case_id"],
            "scenario_id":     meta["scenario_id"],
            "split":           meta["split"],
            "predicted_root":  pred_root,
            "expected_root":   root_cause,
            "root_pattern":    meta.get("root_cause_pattern", ""),
            "correct_top1":    hit1,
            f"correct_top{top_k}": hitk,
            "rr":              round(rr, 4),
            "confidence":      round(float(scores[ranked[0]]), 4),
            "top_candidates": [
                {
                    "rank":    i + 1,
                    "node_id": node_ids[idx],
                    "score":   round(float(scores[idx]), 4),
                }
                for i, idx in enumerate(ranked[:top_k])
            ],
        })
        if not hit1:
            failed.append({
                "case_id":   meta["case_id"],
                "expected":  root_cause,
                "predicted": pred_root,
            })

    # Per-split breakdown
    per_split: dict[str, dict] = {}
    for pc in per_case:
        sp = pc["split"]
        ps = per_split.setdefault(sp, {"case_count": 0, "top1": 0, "topk": 0, "rr_sum": 0.0})
        ps["case_count"] += 1
        ps["top1"]       += int(pc["correct_top1"])
        ps["topk"]       += int(pc.get(f"correct_top{top_k}", False))
        ps["rr_sum"]     += pc["rr"]

    per_split_metrics = {
        sp: {
            "case_count":    v["case_count"],
            "top1_accuracy": round(v["top1"] / v["case_count"], 4),
            f"top{top_k}_accuracy": round(v["topk"] / v["case_count"], 4),
            "mrr":           round(v["rr_sum"] / v["case_count"], 4),
        }
        for sp, v in per_split.items()
    }

    return {
        "case_count":       n,
        "top1_accuracy":    round(top1      / n, 4) if n else 0.0,
        f"top{top_k}_accuracy": round(top_k_hits / n, 4) if n else 0.0,
        "mrr":              round(rr_sum    / n, 4) if n else 0.0,
        "baseline_topology_score": {
            "top1_accuracy": round(baseline_top1 / n, 4) if n else 0.0,
            "description":   "Heuristic baseline: alert severity + node type priority + cross-diagram connectivity",
        },
        "per_split_metrics": per_split_metrics,
        "failed_cases":      failed,
        "per_case_predictions": per_case,
    }
