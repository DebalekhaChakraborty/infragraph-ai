#!/usr/bin/env python3
"""
train_topology_rca_model.py — Train and evaluate the topology RCA classifier.

Reads:
  data/rca/topology/topology_node_dataset.csv
  data/rca/topology/topology_case_index.json

Writes:
  model_artifacts/topology_rca/topology_rca_model.joblib
  model_artifacts/topology_rca/topology_rca_feature_columns.json
  model_artifacts/topology_rca/topology_rca_label_encoder.json
  reports/topology_rca/eval_metrics.json
  reports/topology_rca/per_case_predictions.json
  reports/topology_rca/feature_importance.json

No remediation content is produced here.

Usage:
  python scripts/train_topology_rca_model.py
  python scripts/train_topology_rca_model.py --model logistic_regression
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

from rca_ml.topology_model import (  # noqa: E402
    ALL_FEATURE_COLS,
    LABEL_COL,
    build_pipeline,
    evaluate_cases,
    get_feature_importance,
    save_model,
)


def _print_metrics(metrics: dict) -> None:
    print(f"  cases evaluated  : {metrics['case_count']}")
    print(f"  node rows        : {metrics['node_row_count']}")
    print(f"  top-1 accuracy   : {metrics['top1_accuracy']:.4f}")
    print(f"  top-3 accuracy   : {metrics['top3_accuracy']:.4f}")
    print(f"  MRR              : {metrics['mrr']:.4f}")
    per_split = metrics.get("per_split_metrics", {})
    if per_split:
        print("  per-split:")
        for sp, sm in sorted(per_split.items()):
            print(
                f"    {sp:<10} cases={sm['case_count']}  "
                f"top1={sm['top1_accuracy']:.4f}  "
                f"top3={sm['top3_accuracy']:.4f}  "
                f"mrr={sm['mrr']:.4f}"
            )
    if metrics.get("failed_cases"):
        print(f"  failed cases     : {len(metrics['failed_cases'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train topology RCA classifier from pre-built feature dataset."
    )
    parser.add_argument("--data-dir",    default="data/rca/topology")
    parser.add_argument("--model-dir",   default="model_artifacts/topology_rca")
    parser.add_argument("--reports-dir", default="reports/topology_rca")
    parser.add_argument(
        "--model", default="random_forest",
        choices=["random_forest", "logistic_regression"],
    )
    parser.add_argument(
        "--train-splits", nargs="+", default=["train"],
        help="Splits used for training (default: train)",
    )
    parser.add_argument(
        "--eval-splits", nargs="+", default=["test", "val"],
        help="Splits used for evaluation (default: test val)",
    )
    args = parser.parse_args()

    repo_root   = REPO_ROOT
    data_dir    = (repo_root / args.data_dir).resolve()
    model_dir   = (repo_root / args.model_dir).resolve()
    reports_dir = (repo_root / args.reports_dir).resolve()

    csv_path   = data_dir / "topology_node_dataset.csv"
    index_path = data_dir / "topology_case_index.json"

    if not csv_path.exists():
        print(f"[ERROR] Dataset not found: {csv_path}")
        print("        Run scripts/build_topology_rca_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset   : {len(df)} rows from {csv_path}")

    # Keep only in-scope cases for training and evaluation
    case_index = json.loads(index_path.read_text(encoding="utf-8"))
    in_scope_ids = {c["case_id"] for c in case_index if c["root_cause_in_scope"]}
    df_scope = df[df["case_id"].isin(in_scope_ids)].copy()
    print(f"In-scope rows    : {len(df_scope)} (out of {len(df)})")

    train_mask = df_scope["split"].isin(args.train_splits)
    df_train   = df_scope[train_mask]
    df_eval    = df_scope[df_scope["split"].isin(args.eval_splits)]

    print(f"Train rows       : {len(df_train)} (splits: {args.train_splits})")
    print(f"Eval rows        : {len(df_eval)}  (splits: {args.eval_splits})")

    if df_train.empty:
        print("[ERROR] No training rows found.")
        sys.exit(1)

    missing = [c for c in ALL_FEATURE_COLS if c not in df_train.columns]
    if missing:
        print(f"[ERROR] Missing feature columns: {missing}")
        sys.exit(1)

    # Train
    print(f"\nTraining {args.model} model...")
    pipeline = build_pipeline(model_type=args.model)
    X_train  = df_train[ALL_FEATURE_COLS]
    y_train  = df_train[LABEL_COL].values
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate on train split too (sanity check)
    print("\n--- Train-set case metrics ---")
    train_metrics = evaluate_cases(pipeline, df_train)
    _print_metrics(train_metrics)

    # Evaluate on held-out splits
    if not df_eval.empty:
        print("\n--- Eval-set case metrics ---")
        eval_metrics = evaluate_cases(pipeline, df_eval)
        _print_metrics(eval_metrics)
        primary_metrics = eval_metrics
    else:
        print("\n[NOTE] No eval split rows; reporting train metrics only.")
        primary_metrics = train_metrics

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    save_model(pipeline, model_dir, ALL_FEATURE_COLS)

    # Feature importance
    reports_dir.mkdir(parents=True, exist_ok=True)
    fi = get_feature_importance(pipeline)
    if fi:
        (reports_dir / "feature_importance.json").write_text(
            json.dumps(fi[:30], indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Save eval report (no remediation keys)
    safe_metrics = {
        k: v for k, v in primary_metrics.items()
        if k not in {"per_case_predictions", "failed_cases"}
    }
    (reports_dir / "eval_metrics.json").write_text(
        json.dumps(safe_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (reports_dir / "per_case_predictions.json").write_text(
        json.dumps(primary_metrics.get("per_case_predictions", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- Output ---")
    print(f"  Model     : {model_dir / 'topology_rca_model.joblib'}")
    print(f"  Features  : {model_dir / 'topology_rca_feature_columns.json'}")
    print(f"  Metrics   : {reports_dir / 'eval_metrics.json'}")
    print(f"  Per-case  : {reports_dir / 'per_case_predictions.json'}")
    if fi:
        print(f"  Feat imp. : {reports_dir / 'feature_importance.json'}")


if __name__ == "__main__":
    main()
