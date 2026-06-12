"""
topology_model.py — Pipeline, training, evaluation, and serialisation for topology RCA.

Predicts root-cause node; never produces remediation content.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Column definitions ─────────────────────────────────────────────────────────

CATEGORICAL_COLS: list[str] = ["node_type", "zone", "diagram_id"]

NUMERIC_COLS: list[str] = [
    # base structural + alert summary
    "is_shared_entity", "is_alerted", "alert_count",
    "max_severity_score", "first_alert_time", "mean_alert_time", "min_time_rank",
    "in_degree", "out_degree", "total_degree",
    "pagerank", "betweenness_centrality", "closeness_centrality",
    "is_source_node", "is_sink_node",
    "min_undirected_distance_to_alert", "mean_undirected_distance_to_alert",
    "directed_reachability_to_alert_count", "reverse_reachability_from_alert_count",
    "node_type_priority_score", "severity_weighted_alert_score",
    # alert type counts
    "alert_type_count_cpu",
    "alert_type_count_latency",
    "alert_type_count_packet_drop",
    "alert_type_count_link_errors",
    "alert_type_count_connection_timeout",
    "alert_type_count_auth_errors",
    "alert_type_count_backend_pool_unhealthy",
    "alert_type_count_user_timeout",
    "alert_type_count_other",
    # alert-node compatibility
    "node_alert_compatibility_score",
    # temporal context
    "is_first_alerted_node",
    "is_last_alerted_node",
    "alert_time_span",
    "alert_burst_score",
    "alert_sequence_position_norm",
    # propagation context
    "upstream_alert_count",
    "downstream_alert_count",
    "upstream_critical_alert_count",
    "downstream_warning_alert_count",
    "downstream_after_candidate_count",
    "alerts_reachable_downstream_after_candidate",
    "alerts_reachable_upstream_before_candidate",
    "propagation_consistency_score",
]

ALL_FEATURE_COLS: list[str] = CATEGORICAL_COLS + NUMERIC_COLS
LABEL_COL = "label_is_root"
IDENTIFIER_COLS: list[str] = ["case_id", "split", "scenario_id", "node_id"]


# ── Pipeline factory ───────────────────────────────────────────────────────────

def build_pipeline(model_type: str = "random_forest") -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
    )
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "logistic_regression":
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=42,
            solver="lbfgs",
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return Pipeline([("prep", preprocessor), ("clf", clf)])


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _pos_class_idx(pipeline: Pipeline) -> int:
    classes = list(pipeline.classes_)
    return classes.index(1) if 1 in classes else 0


def score_dataframe(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Add 'prob_is_root' column to df; returns a copy."""
    df = df.copy()
    proba = pipeline.predict_proba(df[ALL_FEATURE_COLS])
    df["prob_is_root"] = proba[:, _pos_class_idx(pipeline)]
    return df


# ── Case-level evaluation ──────────────────────────────────────────────────────

def evaluate_cases(
    pipeline: Pipeline,
    df: pd.DataFrame,
    top_k: int = 3,
) -> dict:
    """
    Group-by-case evaluation: top-1, top-3, MRR, per-split breakdown.

    df must contain only in-scope cases (label_is_root values 0/1).
    """
    df = df[df[LABEL_COL].isin([0, 1])].copy()
    if df.empty:
        return {
            "case_count": 0, "node_row_count": 0,
            "top1_accuracy": 0.0, "top3_accuracy": 0.0, "mrr": 0.0,
        }

    scored = score_dataframe(pipeline, df)

    top1 = top3 = 0
    rr_sum = 0.0
    case_count = 0
    per_case: list[dict] = []
    failed: list[dict] = []

    for case_id, group in scored.groupby("case_id"):
        root_rows = group[group[LABEL_COL] == 1]
        if root_rows.empty:
            continue
        true_root = root_rows.iloc[0]["node_id"]
        split_val = group.iloc[0]["split"]

        ranked = group.sort_values("prob_is_root", ascending=False).reset_index(drop=True)
        pred_root = ranked.iloc[0]["node_id"]
        top_k_nodes = ranked.head(top_k)["node_id"].tolist()

        matches = ranked.index[ranked["node_id"] == true_root].tolist()
        rank = matches[0] + 1 if matches else len(group)
        rr = 1.0 / rank

        hit1 = pred_root == true_root
        hit3 = true_root in top_k_nodes
        top1 += int(hit1)
        top3 += int(hit3)
        rr_sum += rr
        case_count += 1

        top_cands = [
            {
                "rank": i + 1,
                "node_id": r["node_id"],
                "score": round(float(r["prob_is_root"]), 4),
                "node_type": r.get("node_type", ""),
            }
            for i, (_, r) in enumerate(ranked.head(top_k).iterrows())
        ]
        per_case.append({
            "case_id":      str(case_id),
            "split":        split_val,
            "predicted_root": pred_root,
            "expected_root":  true_root,
            "correct_top1":   hit1,
            "correct_top3":   hit3,
            "rr":             round(rr, 4),
            "top_candidates": top_cands,
        })
        if not hit1:
            failed.append({
                "case_id":       str(case_id),
                "expected_root": true_root,
                "predicted_root": pred_root,
                "top_candidates": top_cands,
            })

    # Row-level classification report
    y_true = df[LABEL_COL].values
    y_pred = pipeline.predict(df[ALL_FEATURE_COLS])
    clf_rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Per-split breakdown
    per_split: dict[str, dict] = {}
    for sp in df["split"].unique():
        sp_cases = [c for c in per_case if c["split"] == sp]
        if sp_cases:
            per_split[sp] = {
                "case_count":    len(sp_cases),
                "top1_accuracy": round(sum(c["correct_top1"] for c in sp_cases) / len(sp_cases), 4),
                "top3_accuracy": round(sum(c["correct_top3"] for c in sp_cases) / len(sp_cases), 4),
                "mrr":           round(sum(c["rr"] for c in sp_cases) / len(sp_cases), 4),
            }

    return {
        "case_count":             case_count,
        "node_row_count":         len(df),
        "top1_accuracy":          round(top1 / case_count, 4) if case_count else 0.0,
        "top3_accuracy":          round(top3 / case_count, 4) if case_count else 0.0,
        "mrr":                    round(rr_sum / case_count, 4) if case_count else 0.0,
        "classification_report":  clf_rep,
        "per_split_metrics":      per_split,
        "failed_cases":           failed,
        "per_case_predictions":   per_case,
    }


# ── Feature importance ─────────────────────────────────────────────────────────

def get_feature_importance(pipeline: Pipeline) -> list[dict] | None:
    clf = pipeline.named_steps.get("clf")
    if not hasattr(clf, "feature_importances_"):
        return None
    try:
        cat_names = list(
            pipeline.named_steps["prep"]
            .named_transformers_["cat"]
            .get_feature_names_out(CATEGORICAL_COLS)
        )
    except Exception:
        cat_names = [f"cat_{i}" for i in range(200)]

    all_names = cat_names + NUMERIC_COLS
    imps = clf.feature_importances_
    n = min(len(all_names), len(imps))
    return sorted(
        [{"feature": all_names[i], "importance": round(float(imps[i]), 6)} for i in range(n)],
        key=lambda x: -x["importance"],
    )


# ── Serialisation ─────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, out_dir: Path, feature_cols: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "topology_rca_model.joblib")
    (out_dir / "topology_rca_feature_columns.json").write_text(
        json.dumps(feature_cols, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "topology_rca_label_encoder.json").write_text(
        json.dumps({"classes": [0, 1], "label": "is_root_node",
                    "note": "binary classification — 1 means this node is the root cause"},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_model(model_path: Path, feature_cols_path: Path) -> tuple[Pipeline, list[str]]:
    pipeline = joblib.load(str(model_path))
    feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))
    return pipeline, feature_cols
