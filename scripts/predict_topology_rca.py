#!/usr/bin/env python3
"""
predict_topology_rca.py — Run root-cause prediction for one topology RCA case.

Reads:
  scenario_library/topology_rca/<case_id>/events.json
  scenario_library/topology_rca/<case_id>/graph_ref.json
  model_artifacts/topology_rca/topology_rca_model.joblib

Default writes to:
  assets/preloaded/topology_rca_results/<case_id>.json   (demo-safe, no eval fields)

With --with-eval (unless --out-dir is overridden) writes to:
  reports/topology_rca/manual_eval/<case_id>.json

No remediation content is produced here.

Usage:
  python scripts/predict_topology_rca.py --case-id topo_enterprise_v3_0000_datacenter_topology
  python scripts/predict_topology_rca.py --case-id topo_enterprise_v3_0000_datacenter_topology \\
      --top-k 5 --with-eval
  python scripts/predict_topology_rca.py --case-id topo_enterprise_v3_0000_datacenter_topology \\
      --hybrid-score
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
    "rollback_steps", "validation_steps", "itsm_ticket",
    "remediation", "resolution", "rollback",
    # top-level evaluation leakage (must be under "evaluation" key only)
    "expected_root_cause", "ground_truth_node", "correct",
    "correct_top1", "correct_top_k", "reciprocal_rank",
})


def _enrich_with_cluster(result: dict, cluster_file: str, case_id: str, scenario_id: str, repo_root: Path) -> None:
    """Load cluster file and merge primary cluster fields into result."""
    cf_path = Path(cluster_file) if Path(cluster_file).is_absolute() else (repo_root / cluster_file).resolve()
    if not cf_path.exists():
        print(f"[WARN] Cluster file not found: {cf_path}")
        return
    try:
        cluster_data = json.loads(cf_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not load cluster file: {exc}")
        return

    if cluster_data.get("case_id") != case_id and cluster_data.get("scenario_id") != scenario_id:
        print(f"[WARN] Cluster file case_id/scenario_id does not match {case_id!r}")
        return

    clusters = cluster_data.get("clusters", [])
    if not clusters:
        print("[WARN] Cluster file contains no clusters.")
        return

    primary = clusters[0]
    result["cluster_id"]           = primary.get("cluster_id", "")
    result["cluster_score"]        = primary.get("cluster_score", 0.0)
    result["correlation_reasons"]  = primary.get("correlation_reasons", [])
    result["causal_evidence"]      = primary.get("causal_evidence", [])
    print(f"Cluster enriched : {result['cluster_id']} (score={result['cluster_score']:.4f})")


def _assert_clean(obj: dict) -> None:
    for key in _FORBIDDEN_KEYS:
        if key in obj:
            raise ValueError(f"Output contains forbidden key: {key!r}")


def _compute_alert_context_score(df: pd.DataFrame) -> pd.Series:
    """Normalized alert-context score for hybrid ranking (0–1)."""
    prop   = df.get("propagation_consistency_score",    pd.Series(0.0, index=df.index))
    compat = df.get("node_alert_compatibility_score",   pd.Series(0.0, index=df.index))
    seq    = df.get("alert_sequence_position_norm",     pd.Series(1.0, index=df.index))
    first  = df.get("is_first_alerted_node",            pd.Series(0.0, index=df.index)).astype(float)

    raw = (
        prop   * 0.40
        + compat * 0.25
        + (1.0 - seq) * 0.25
        + first * 0.10
    )
    mx = raw.max()
    return raw / mx if mx > 0 else raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict root-cause node for one topology RCA case.",
    )
    parser.add_argument("--case-id", required=True, help="Case ID from manifest")
    parser.add_argument("--scenario-library", default="scenario_library")
    parser.add_argument("--model-dir",  default="model_artifacts/topology_rca")
    parser.add_argument("--out-dir",    default=None,
                        help="Output directory override.  Default: assets/preloaded/topology_rca_results "
                             "(or reports/topology_rca/manual_eval when --with-eval is set).")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--with-eval", action="store_true",
                        help="Include ground-truth comparison (reads labels.json).  "
                             "Output is written to reports/topology_rca/manual_eval by default.")
    parser.add_argument("--hybrid-score", action="store_true",
                        help="Combine model probability with alert-context score "
                             "(0.75 x model + 0.25 x context)")
    parser.add_argument(
        "--cluster-file", default=None,
        help="Path to event correlation cluster file for this case.  "
             "Enriches RCA output with cluster_id, cluster_score, "
             "correlation_reasons, and causal_evidence.",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    lib_root  = (repo_root / args.scenario_library).resolve()
    model_dir = (repo_root / args.model_dir).resolve()
    case_dir  = lib_root / "topology_rca" / args.case_id

    # Resolve output directory
    if args.out_dir:
        out_dir = (repo_root / args.out_dir).resolve()
    elif args.with_eval:
        out_dir = repo_root / "reports" / "topology_rca" / "manual_eval"
        print("[INFO] --with-eval enabled; writing evaluation output under "
              "reports/topology_rca/manual_eval, not assets/preloaded/.")
    else:
        out_dir = repo_root / "assets" / "preloaded" / "topology_rca_results"

    if not case_dir.exists():
        print(f"[ERROR] Case directory not found: {case_dir}")
        sys.exit(1)

    model_path    = model_dir / "topology_rca_model.joblib"
    feat_col_path = model_dir / "topology_rca_feature_columns.json"
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("        Run scripts/train_topology_rca_model.py first.")
        sys.exit(1)

    # Load model and check feature column compatibility (Part A)
    pipeline, saved_feature_cols = load_model(model_path, feat_col_path)
    if saved_feature_cols and saved_feature_cols != ALL_FEATURE_COLS:
        print("[ERROR] Feature column mismatch: topology model was trained with a different feature set.")
        print("        Rebuild dataset and retrain:")
        print("          python scripts/build_topology_rca_dataset.py")
        print("          python scripts/train_topology_rca_model.py")
        sys.exit(1)

    # Load case inputs
    def _r(name: str) -> dict:
        return json.loads((case_dir / name).read_text(encoding="utf-8"))

    events_doc  = _r("events.json")
    graph_ref   = _r("graph_ref.json")
    events      = events_doc.get("events", [])

    lg_path     = normalize_repo_path(repo_root, graph_ref["local_graph_path"])
    local_graph = json.loads(lg_path.read_text(encoding="utf-8"))

    # Ground truth — only read when --with-eval is explicitly requested
    ground_truth_node: str | None = None
    if args.with_eval:
        label_path = case_dir / "labels.json"
        if label_path.exists():
            labels    = json.loads(label_path.read_text(encoding="utf-8"))
            in_scope  = bool(labels.get("root_cause_in_scope", False))
            ground_truth_node = labels.get("root_cause_node") if in_scope else None

    # Build features
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
        root_cause_node=None,   # never leak ground truth into features
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

    # Hybrid scoring
    if args.hybrid_score:
        ctx_score = _compute_alert_context_score(scored)
        scored = scored.copy()
        scored["_hybrid"] = 0.75 * scored["prob_is_root"] + 0.25 * ctx_score.values
        score_col    = "_hybrid"
        scoring_mode = "hybrid_alert_context"
    else:
        score_col    = "prob_is_root"
        scoring_mode = "ml_only"

    ranked = scored.sort_values(score_col, ascending=False).reset_index(drop=True)

    top_candidates = []
    for i, (_, row) in enumerate(ranked.head(args.top_k).iterrows()):
        top_candidates.append({
            "rank":       i + 1,
            "node_id":    row["node_id"],
            "score":      round(float(row[score_col]), 4),
            "node_type":  row.get("node_type", ""),
            "zone":       row.get("zone", ""),
            "is_alerted": bool(row.get("is_alerted", False)),
        })

    predicted_root = ranked.iloc[0]["node_id"]

    result: dict = {
        "case_id":        args.case_id,
        "model":          "topology_rca_random_forest_v1",
        "top_k":          args.top_k,
        "predicted_root": predicted_root,
        "top_candidates": top_candidates,
        "total_nodes":    len(ranked),
        "alert_count":    len(events),
        "scoring_mode":   scoring_mode,
    }

    # Evaluation block — only when --with-eval
    if args.with_eval and ground_truth_node:
        top_k_nodes = [c["node_id"] for c in top_candidates]
        ranks  = ranked.index[ranked["node_id"] == ground_truth_node].tolist()
        rank   = ranks[0] + 1 if ranks else len(ranked)
        result["evaluation"] = {
            "ground_truth_node": ground_truth_node,
            "correct_top1":      predicted_root == ground_truth_node,
            "correct_top_k":     ground_truth_node in top_k_nodes,
            "reciprocal_rank":   round(1.0 / rank, 4),
            "rank":              rank,
        }

    # Enrich with event correlation cluster data when --cluster-file is provided
    if args.cluster_file:
        _enrich_with_cluster(result, args.cluster_file, args.case_id, "", repo_root)

    # Guard: clean outputs going to preloaded must have no eval keys
    if not args.with_eval:
        _assert_clean(result)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.case_id}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Human-readable summary
    print(f"Case             : {args.case_id}")
    print(f"Predicted root   : {predicted_root}")
    print(f"Scoring mode     : {scoring_mode}")
    print(f"Top-{args.top_k} candidates :")
    for c in top_candidates:
        gt_marker = " <-- ground truth" if c["node_id"] == ground_truth_node else ""
        print(f"  #{c['rank']:>2}  {c['node_id']:<40}  score={c['score']:.4f}  "
              f"type={c['node_type']}{gt_marker}")
    if "evaluation" in result:
        ev = result["evaluation"]
        print(f"Top-1 correct    : {ev['correct_top1']}")
        print(f"Top-{args.top_k} correct    : {ev['correct_top_k']}")
        print(f"Reciprocal rank  : {ev['reciprocal_rank']}")
    print(f"\nResult written   : {out_path}")


if __name__ == "__main__":
    main()
