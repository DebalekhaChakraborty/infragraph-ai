#!/usr/bin/env python3
"""
train_enterprise_gnn_v2_rca.py — Train Enterprise GNN RCA V2 (Temporal Relation-Aware).

Reads:
  data/rca/enterprise_gnn/graphs.pt          (same source as V1)
  data/rca/enterprise_gnn/graph_index.json

Writes:
  model_artifacts/enterprise_gnn_rca_v2/enterprise_gnn_v2_rca.pt
  model_artifacts/enterprise_gnn_rca_v2/enterprise_gnn_v2_config.json
  model_artifacts/enterprise_gnn_rca_v2/feature_columns.json
  model_artifacts/enterprise_gnn_rca_v2/training_report.json
  reports/enterprise_gnn_rca_v2/training_history.json
  reports/enterprise_gnn_rca_v2/evaluation.json
  reports/enterprise_gnn_rca_v2/predictions_test.json

V1 artifacts in model_artifacts/enterprise_gnn_rca/ are NOT touched.
No remediation content is produced here.

Usage:
  python scripts/train_enterprise_gnn_v2_rca.py
  python scripts/train_enterprise_gnn_v2_rca.py --epochs 80 --hidden-dim 64 --lr 0.001
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rca_ml.enterprise_gnn_v2_model import (
    EnterpriseRcaTemporalRelGNN,
    build_gnn_v2_config,
    check_torch_geo_v2_requirement,
    evaluate_dataset_v2,
    graph_dict_to_pyg_v2,
    save_gnn_v2,
)
from rca_ml.enterprise_gnn_dataset import IN_DIM


def _print_split_metrics(metrics: dict) -> None:
    topk_key = next(
        (k for k in metrics if k.startswith("top") and "accuracy" in k and k != "top1_accuracy"),
        None,
    )
    print(f"  cases={metrics['case_count']}  "
          f"top1={metrics['top1_accuracy']:.4f}  "
          + (f"{topk_key}={metrics[topk_key]:.4f}  " if topk_key else "")
          + f"mrr={metrics['mrr']:.4f}")
    for sp, sm in sorted(metrics.get("per_split_metrics", {}).items()):
        print(f"    {sp:<12} cases={sm['case_count']}  "
              f"top1={sm['top1_accuracy']:.4f}  mrr={sm['mrr']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Enterprise GNN RCA V2 (Temporal Relation-Aware GraphSAGE)."
    )
    parser.add_argument("--graphs",      default="data/rca/enterprise_gnn/graphs.pt")
    parser.add_argument("--index",       default="data/rca/enterprise_gnn/graph_index.json")
    parser.add_argument("--out-dir",     default="model_artifacts/enterprise_gnn_rca_v2")
    parser.add_argument("--report-dir",  default="reports/enterprise_gnn_rca_v2")
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--hidden-dim",  type=int,   default=64)
    parser.add_argument("--num-layers",  type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--top-k",       type=int,   default=3)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--eval-every",  type=int,   default=10)
    args = parser.parse_args()

    check_torch_geo_v2_requirement()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)

    repo_root   = REPO_ROOT
    graphs_path = (repo_root / args.graphs).resolve()
    index_path  = (repo_root / args.index).resolve()
    out_dir     = (repo_root / args.out_dir).resolve()
    report_dir  = (repo_root / args.report_dir).resolve()

    if not graphs_path.exists():
        print(f"[ERROR] Graphs file not found: {graphs_path}")
        print("        Run scripts/build_enterprise_gnn_dataset.py first.")
        print("        V2 uses the same graphs.pt as V1 (edge_type included when rebuilt).")
        sys.exit(1)

    if not index_path.exists():
        print(f"[ERROR] Graph index not found: {index_path}")
        sys.exit(1)

    print(f"Loading graphs    : {graphs_path}")
    all_graphs = torch.load(str(graphs_path))
    case_index = json.loads(index_path.read_text(encoding="utf-8"))
    print(f"Graphs loaded     : {len(all_graphs)}")

    # Check whether edge_type is present (graphs built after PART 1)
    _has_edge_type = any(g.get("edge_type") is not None for g in all_graphs)
    print(f"Edge type present : {_has_edge_type}")
    if not _has_edge_type:
        print("  [note] edge_type not found in graphs.pt — V2 falls back to all-edge SAGEConv.")
        print("         Rebuild graphs with build_enterprise_gnn_dataset.py for full V2 benefit.")

    train_graphs = [g for g in all_graphs if g.get("split") == "train"]
    val_graphs   = [g for g in all_graphs if g.get("split") == "val"]
    test_graphs  = [g for g in all_graphs if g.get("split") == "test"]
    train_index  = [c for c in case_index if c.get("split") == "train"]
    val_index    = [c for c in case_index if c.get("split") == "val"]
    test_index   = [c for c in case_index if c.get("split") == "test"]

    print(f"Train/val/test    : {len(train_graphs)} / {len(val_graphs)} / {len(test_graphs)}")

    if not train_graphs:
        print("[ERROR] No training graphs found.")
        sys.exit(1)

    config = build_gnn_v2_config(
        in_channels=IN_DIM,
        hidden_channels=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        top_k=args.top_k,
    )
    model = EnterpriseRcaTemporalRelGNN(**{
        k: v for k, v in config.items()
        if k in ("in_channels", "hidden_channels", "num_layers", "dropout")
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nModel             : EnterpriseRcaTemporalRelGNN (RelationAwareTemporalGraphSAGE)")
    print(f"Config            : in={IN_DIM}  hidden={args.hidden_dim}  layers={args.num_layers}  dropout={args.dropout}")
    print(f"Training          : epochs={args.epochs}  lr={args.lr}  seed={args.seed}")
    print()

    best_val_mrr = -1.0
    best_epoch   = 0
    best_state   = None
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for g in train_graphs:
            if g["y"].item() < 0:
                continue
            data      = graph_dict_to_pyg_v2(g)
            edge_type = getattr(data, "edge_type", None)
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index, edge_type)
            loss   = -F.log_softmax(logits, dim=0)[data.y]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(train_graphs))
        entry: dict = {"epoch": epoch, "train_loss": round(avg_loss, 6)}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            if val_graphs:
                vm = evaluate_dataset_v2(model, val_graphs, val_index, top_k=args.top_k)
                entry["val_top1"] = vm["top1_accuracy"]
                entry["val_mrr"]  = vm["mrr"]
                if vm["mrr"] > best_val_mrr:
                    best_val_mrr = vm["mrr"]
                    best_epoch   = epoch
                    best_state   = copy.deepcopy(model.state_dict())
                print(
                    f"  Epoch {epoch:>4}/{args.epochs}  "
                    f"loss={avg_loss:.4f}  "
                    f"val_top1={vm['top1_accuracy']:.4f}  "
                    f"val_mrr={vm['mrr']:.4f}"
                )
            else:
                print(f"  Epoch {epoch:>4}/{args.epochs}  loss={avg_loss:.4f}")
        history.append(entry)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored checkpoint from epoch {best_epoch}  val_mrr={best_val_mrr:.4f}")

    model.eval()

    print("\n--- Train-set metrics ---")
    train_metrics = evaluate_dataset_v2(model, train_graphs, train_index, top_k=args.top_k)
    _print_split_metrics(train_metrics)

    eval_metrics: dict = {}
    if val_graphs:
        print("\n--- Val-set metrics ---")
        eval_metrics = evaluate_dataset_v2(model, val_graphs, val_index, top_k=args.top_k)
        _print_split_metrics(eval_metrics)

    test_metrics: dict = {}
    if test_graphs:
        print("\n--- Test-set metrics ---")
        test_metrics = evaluate_dataset_v2(model, test_graphs, test_index, top_k=args.top_k)
        _print_split_metrics(test_metrics)

    primary_metrics = (
        test_metrics if test_graphs
        else eval_metrics if eval_metrics
        else train_metrics
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    save_gnn_v2(model, config, out_dir)

    def _safe(m: dict) -> dict:
        return {k: v for k, v in m.items() if k not in ("per_case_predictions", "failed_cases")}

    # Training report (spec: training_report.json with extra V2 metadata)
    training_report = {
        "model_type":             "EnterpriseRcaTemporalRelGNN",
        "gnn_architecture":       "RelationAwareTemporalGraphSAGE",
        "uses_edge_type":         True,
        "uses_temporal_features": True,
        "edge_type_present_in_graphs": _has_edge_type,
        "num_graphs":             len(all_graphs),
        "train_count":            len(train_graphs),
        "val_count":              len(val_graphs),
        "test_count":             len(test_graphs),
        "epochs":                 args.epochs,
        "best_epoch":             best_epoch,
        "best_val_mrr":           round(best_val_mrr, 4),
        "train_metrics":          _safe(train_metrics),
        "val_metrics":            _safe(eval_metrics) if eval_metrics else {},
        "test_metrics":           _safe(test_metrics) if test_metrics else {},
        "primary_metrics":        _safe(primary_metrics),
        "relations":              ["local", "cross_diagram", "vision_connector_extraction"],
        "note_remediation":       "Remediation is NOT part of this model. Root cause from GNN only.",
        "note_architecture": (
            "Temporal-aware relation-aware GraphSAGE. "
            "Not a fully dynamic temporal heterogeneous graph transformer."
        ),
    }
    (out_dir / "training_report.json").write_text(
        json.dumps(training_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (report_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "evaluation.json").write_text(
        json.dumps({
            "train": _safe(train_metrics),
            "val":   _safe(eval_metrics) if eval_metrics else {},
            "test":  _safe(test_metrics) if test_metrics else {},
            "best_epoch":   best_epoch,
            "best_val_mrr": round(best_val_mrr, 4),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / "predictions_test.json").write_text(
        json.dumps(primary_metrics.get("per_case_predictions", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- Output ---")
    print(f"  Model   : {out_dir / 'enterprise_gnn_v2_rca.pt'}")
    print(f"  Config  : {out_dir / 'enterprise_gnn_v2_config.json'}")
    print(f"  Report  : {out_dir / 'training_report.json'}")
    print(f"  History : {report_dir / 'training_history.json'}")
    print(f"  Eval    : {report_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
