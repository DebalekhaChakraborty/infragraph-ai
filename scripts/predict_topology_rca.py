#!/usr/bin/env python3
"""
predict_topology_rca.py — Run root-cause prediction for one topology RCA case.

Reads:
  scenario_library/topology_rca/<case_id>/events.json
  scenario_library/topology_rca/<case_id>/labels.json  (for ground-truth comparison)
  scenario_library/topology_rca/<case_id>/graph_ref.json
  model_artifacts/topology_rca/topology_rca_model.joblib

Writes:
  assets/preloaded/topology_rca_results/<case_id>.json

Output format contains only root-cause node ranking — no remediation content.

Usage:
  python scripts/predict_topology_rca.py --case-id topo_enterprise_v3_0000_datacenter_topology
  python scripts/predict_topology_rca.py --case-id topo_enterprise_v3_0000_datacenter_topology \\
      --top-k 5 --no-eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rca_ml.features import compute_case_features, normalize_repo_path  # noqa: E402
from rca_ml.topology_model import (  # noqa: E402
    ALL_FEATURE_COLS,
    load_model,
    score_dataframe,
)

_FORBIDDEN_KEYS = frozenset({
    "recommended_actions", "remediation_steps", "resolution_steps",
    "rollback_steps", "validation_steps", "servicenow_ticket",
    "remediation", "resolution", "rollback",
})


def _assert_clean(obj: dict) -> None:
    for key in _FORBIDDEN_KEYS:
        if key in obj:
            raise ValueError(f"Output contains forbidden key: {key!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict root-cause node for one topology RCA case.",
    )
    parser.add_argument("--case-id", required=True, help="Case ID from manifest")
    parser.add_argument("--scenario-library", default="scenario_library")
    parser.add_argument("--model-dir",         default="model_artifacts/topology_rca")
    parser.add_argument("--out-dir",           default="assets/preloaded/topology_rca_results")
    parser.add_argument("--top-k", type=int,   default=3)
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip ground-truth comparison (labels.json not read)")
    args = parser.parse_args()

    repo_root   = REPO_ROOT
    lib_root    = (repo_root / args.scenario_library).resolve()
    model_dir   = (repo_root / args.model_dir).resolve()
    out_dir     = (repo_root / args.out_dir).resolve()
    case_dir    = lib_root / "topology_rca" / args.case_id

    if not case_dir.exists():
        print(f"[ERROR] Case directory not found: {case_dir}")
        sys.exit(1)

    model_path    = model_dir / "topology_rca_model.joblib"
    feat_col_path = model_dir / "topology_rca_feature_columns.json"
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("        Run scripts/train_topology_rca_model.py first.")
        sys.exit(1)

    # Load model
    pipeline, _ = load_model(model_path, feat_col_path)

    # Load case inputs
    def _r(name: str) -> dict:
        return json.loads((case_dir / name).read_text(encoding="utf-8"))

    events_doc = _r("events.json")
    graph_ref  = _r("graph_ref.json")
    events     = events_doc.get("events", [])

    lg_path    = normalize_repo_path(repo_root, graph_ref["local_graph_path"])
    local_graph = json.loads(lg_path.read_text(encoding="utf-8"))

    # Labels (ground truth) if available and not suppressed
    ground_truth_node: str | None = None
    in_scope = False
    if not args.no_eval:
        label_path = case_dir / "labels.json"
        if label_path.exists():
            labels    = json.loads(label_path.read_text(encoding="utf-8"))
            in_scope  = bool(labels.get("root_cause_in_scope", False))
            ground_truth_node = labels.get("root_cause_node") if in_scope else None

    # Build features
    manifest_row_meta = graph_ref.get("case_id", args.case_id)
    split       = graph_ref.get("split", "infer")
    scenario_id = graph_ref.get("scenario_id", "")
    diagram_id  = graph_ref.get("diagram_id", "")

    feature_rows = compute_case_features(
        case_id=args.case_id,
        split=split,
        scenario_id=scenario_id,
        diagram_id=diagram_id,
        events=events,
        local_graph=local_graph,
        root_cause_node=None,   # don't leak ground truth into features
    )

    if not feature_rows:
        print(f"[ERROR] No nodes found in local_graph for case {args.case_id}")
        sys.exit(1)

    df = pd.DataFrame(feature_rows)
    missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing feature columns: {missing}")
        sys.exit(1)

    # Score
    scored = score_dataframe(pipeline, df)
    ranked = scored.sort_values("prob_is_root", ascending=False).reset_index(drop=True)

    top_candidates = []
    for i, (_, row) in enumerate(ranked.head(args.top_k).iterrows()):
        top_candidates.append({
            "rank":      i + 1,
            "node_id":   row["node_id"],
            "score":     round(float(row["prob_is_root"]), 4),
            "node_type": row.get("node_type", ""),
            "zone":      row.get("zone", ""),
            "is_alerted":bool(row.get("is_alerted", False)),
        })

    predicted_root = ranked.iloc[0]["node_id"]

    # Evaluation block (no remediation)
    eval_block: dict = {}
    if ground_truth_node:
        top_k_nodes = [c["node_id"] for c in top_candidates]
        ranks  = ranked.index[ranked["node_id"] == ground_truth_node].tolist()
        rank   = ranks[0] + 1 if ranks else len(ranked)
        eval_block = {
            "ground_truth_node": ground_truth_node,
            "correct_top1":      predicted_root == ground_truth_node,
            "correct_top_k":     ground_truth_node in top_k_nodes,
            "reciprocal_rank":   round(1.0 / rank, 4),
            "rank":              rank,
        }

    result = {
        "case_id":        args.case_id,
        "model":          "topology_rca_random_forest_v1",
        "top_k":          args.top_k,
        "predicted_root": predicted_root,
        "top_candidates": top_candidates,
        "total_nodes":    len(ranked),
        "alert_count":    len(events),
    }
    if eval_block:
        result["evaluation"] = eval_block

    _assert_clean(result)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.case_id}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Human-readable summary
    print(f"Case             : {args.case_id}")
    print(f"Predicted root   : {predicted_root}")
    print(f"Top-{args.top_k} candidates :")
    for c in top_candidates:
        gt_marker = " <-- ground truth" if c["node_id"] == ground_truth_node else ""
        print(f"  #{c['rank']:>2}  {c['node_id']:<40}  score={c['score']:.4f}  type={c['node_type']}{gt_marker}")
    if eval_block:
        print(f"Top-1 correct    : {eval_block['correct_top1']}")
        print(f"Top-{args.top_k} correct    : {eval_block['correct_top_k']}")
        print(f"Reciprocal rank  : {eval_block['reciprocal_rank']}")
    print(f"\nResult written   : {out_path}")


if __name__ == "__main__":
    main()
