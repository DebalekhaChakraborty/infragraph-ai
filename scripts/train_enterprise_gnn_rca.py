#!/usr/bin/env python3
"""
train_enterprise_gnn_rca.py — Train Enterprise GNN RCA model.

Reads:
  data/rca/enterprise_gnn/graphs.pt
  data/rca/enterprise_gnn/graph_index.json

Writes:
  model_artifacts/enterprise_gnn_rca/enterprise_gnn_rca.pt
  model_artifacts/enterprise_gnn_rca/enterprise_gnn_config.json
  model_artifacts/enterprise_gnn_rca/feature_columns.json
  reports/enterprise_gnn_rca/training_history.json
  reports/enterprise_gnn_rca/evaluation.json
  reports/enterprise_gnn_rca/predictions_test.json

No remediation content is produced here.

Usage:
  python scripts/train_enterprise_gnn_rca.py
  python scripts/train_enterprise_gnn_rca.py --epochs 100 --hidden-dim 64 --lr 0.001
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

from rca_ml.enterprise_gnn_model import (  # noqa: E402
    EnterpriseRcaGNN,
    build_gnn_config,
    check_torch_geo_requirement,
    graph_dict_to_pyg,
    save_gnn,
)
from rca_ml.enterprise_gnn_inference import evaluate_dataset  # noqa: E402
from rca_ml.enterprise_gnn_dataset import IN_DIM              # noqa: E402


def _print_split_metrics(metrics: dict) -> None:
    topk_key = next(
        (k for k in metrics if k.startswith("top") and "accuracy" in k and k != "top1_accuracy"),
        None,
    )
    print(f"  cases={metrics['case_count']}  "
          f"top1={metrics['top1_accuracy']:.4f}  "
          + (f"{topk_key}={metrics[topk_key]:.4f}  " if topk_key else "")
          + f"mrr={metrics['mrr']:.4f}")
    bl = metrics.get("baseline_topology_score", {})
    if bl:
        print(f"  baseline top1 = {bl.get('top1_accuracy', 'n/a')}")
    for sp, sm in sorted(metrics.get("per_split_metrics", {}).items()):
        print(f"    {sp:<12} cases={sm['case_count']}  "
              f"top1={sm['top1_accuracy']:.4f}  mrr={sm['mrr']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Enterprise GNN RCA from pre-built graph dataset."
    )
    parser.add_argument("--graphs",      default="data/rca/enterprise_gnn/graphs.pt")
    parser.add_argument("--index",       default="data/rca/enterprise_gnn/graph_index.json")
    parser.add_argument("--out-dir",     default="model_artifacts/enterprise_gnn_rca")
    parser.add_argument("--report-dir",  default="reports/enterprise_gnn_rca")
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--hidden-dim",  type=int,   default=64)
    parser.add_argument("--num-layers",  type=int,   default=3)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--top-k",       type=int,   default=3)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--eval-every",  type=int,   default=10)
    args = parser.parse_args()

    check_torch_geo_requirement()

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
        sys.exit(1)

    print(f"Loading graphs    : {graphs_path}")
    all_graphs = torch.load(str(graphs_path))
    case_index = json.loads(index_path.read_text(encoding="utf-8"))
    print(f"Graphs loaded     : {len(all_graphs)}")

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

    config = build_gnn_config(
        in_channels=IN_DIM,
        hidden_channels=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        top_k=args.top_k,
    )
    model     = EnterpriseRcaGNN(**{
        k: v for k, v in config.items()
        if k in ("in_channels", "hidden_channels", "num_layers", "dropout")
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nModel             : EnterpriseRcaGNN (GraphSAGE)")
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
            data = graph_dict_to_pyg(g)
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss   = -F.log_softmax(logits, dim=0)[data.y]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(train_graphs))
        entry: dict = {"epoch": epoch, "train_loss": round(avg_loss, 6)}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            vm: dict = {}
            if val_graphs:
                vm = evaluate_dataset(model, val_graphs, val_index, top_k=args.top_k)
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

    print("\n--- Train-set case metrics ---")
    train_metrics = evaluate_dataset(model, train_graphs, train_index, top_k=args.top_k)
    _print_split_metrics(train_metrics)

    eval_metrics: dict = {}
    if val_graphs:
        print("\n--- Val-set case metrics ---")
        eval_metrics = evaluate_dataset(model, val_graphs, val_index, top_k=args.top_k)
        _print_split_metrics(eval_metrics)

    test_metrics: dict = {}
    if test_graphs:
        print("\n--- Test-set case metrics ---")
        test_metrics = evaluate_dataset(model, test_graphs, test_index, top_k=args.top_k)
        _print_split_metrics(test_metrics)

    primary_metrics = (
        test_metrics if test_graphs
        else eval_metrics if eval_metrics
        else train_metrics
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    save_gnn(model, config, out_dir)

    def _safe(m: dict) -> dict:
        return {k: v for k, v in m.items() if k not in ("per_case_predictions", "failed_cases")}

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
    print(f"  Model   : {out_dir / 'enterprise_gnn_rca.pt'}")
    print(f"  Config  : {out_dir / 'enterprise_gnn_config.json'}")
    print(f"  History : {report_dir / 'training_history.json'}")
    print(f"  Eval    : {report_dir / 'evaluation.json'}")
    print(f"  Preds   : {report_dir / 'predictions_test.json'}")


if __name__ == "__main__":
    main()
