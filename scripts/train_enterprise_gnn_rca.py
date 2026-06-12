#!/usr/bin/env python3
"""
train_enterprise_gnn_rca.py -- Enterprise GNN Root Cause Analysis

Trains a 3-layer Graph Convolutional Network on stitched multi-diagram
enterprise topology graphs.  Each training sample is one enterprise scenario
containing multiple local diagram clusters unified into one graph.

The model ranks ALL nodes across the enterprise graph and predicts the
root-cause node that triggered cross-diagram alert propagation.

Backend: PyTorch required.

Usage
-----
python scripts/train_enterprise_gnn_rca.py \
    --dataset-root ./datasets/infragraph_v1/enterprise_graph \
    --out ./outputs/enterprise_gnn_rca \
    --epochs 80 \
    --presentation-scenario enterprise_0000 \
    --presentation-split test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

# ── Shared GNN utilities ──────────────────────────────────────────────────────
_REPO_ROOT_EARLY = Path(__file__).resolve().parent.parent
_src_dir = str(_REPO_ROOT_EARLY / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from enterprise_gnn_rca import (   # type: ignore  # noqa: E402
    NODE_TYPES, DIAGRAM_TYPES, FEATURE_NAMES, IN_FEAT, SEV_SCORE,
    _NODE_T_IDX, _DIAG_T_IDX,
    _GCNLayer, EnterpriseGCN,
    _bfs, _extract_features, _build_adj_norm,
    _load_scenario, _score_scenario,
    run_inference_for_scenario,
)

# ── PyTorch (imported via enterprise_gnn_rca, re-import for training code) ───
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print(
        "[ERROR] PyTorch is not installed in the current environment.\n"
        "        Install with:  pip install torch\n"
        "        This script requires torch and does not use a heuristic."
    )
    sys.exit(1)

# ── Optional visualisation ────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False


NODE_VIZ_COLOR = {
    "router":        "#60a5fa",
    "switch":        "#818cf8",
    "firewall":      "#ef4444",
    "server":        "#22d3ee",
    "database":      "#10b981",
    "load_balancer": "#f59e0b",
    "cloud_or_wan":  "#94a3b8",
    "service":       "#a78bfa",
    "unknown":       "#475569",
}

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_enterprise_graph_v1_root(repo_root: Path) -> Path:
    """V1 root for backward compatibility — V3 is the preferred dataset."""
    preferred = repo_root / "datasets" / "infragraph_v1" / "enterprise_graph"
    legacy = repo_root / "datasets" / "enterprise_graph_v1"
    if preferred.exists():
        return preferred
    return legacy


def get_default_dataset_root(repo_root: Path) -> Path:
    """Return V3 dataset root if available, else fall back to V1."""
    v3 = repo_root / "datasets" / "infragraph_v3"
    if v3.exists():
        return v3
    return get_enterprise_graph_v1_root(repo_root)


def load_split(dataset_root: Path, split: str) -> list[dict]:
    split_dir = dataset_root / "scenarios" / split
    if not split_dir.exists():
        return []
    scenarios = []
    for sc_dir in sorted(split_dir.iterdir()):
        if not sc_dir.is_dir():
            continue
        sc = _load_scenario(sc_dir)
        if sc is not None:
            scenarios.append(sc)
    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model: EnterpriseGCN,
    scenarios: list[dict],
    device: torch.device,
) -> dict[str, Any]:
    """
    Scenario-level ranking evaluation.

    Returns top1, top3, mrr, scenario_count.
    """
    if not scenarios:
        return {"top1": 0.0, "top3": 0.0, "mrr": 0.0, "scenario_count": 0}

    model.eval()
    top1_n = 0
    top3_n = 0
    mrr    = 0.0

    with torch.no_grad():
        for sc in scenarios:
            X  = torch.from_numpy(sc["X"]).to(device)
            A  = torch.from_numpy(sc["A"]).to(device)
            logits = model(X, A)
            scores = torch.sigmoid(logits).cpu().numpy()

            ranked = np.argsort(-scores)       # descending
            rank   = int(np.where(ranked == sc["rc_idx"])[0][0]) + 1   # 1-based

            if rank == 1:
                top1_n += 1
            if rank <= 3:
                top3_n += 1
            mrr += 1.0 / rank

    total = len(scenarios)
    return {
        "top1":           round(top1_n / total, 4),
        "top3":           round(top3_n / total, 4),
        "mrr":            round(mrr    / total, 4),
        "scenario_count": total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(
    model: EnterpriseGCN,
    train_scenarios: list[dict],
    val_scenarios:   list[dict],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[dict, dict, int]:
    """
    Train the model.

    Returns (history, best_val_metrics, best_val_epoch).
    history keys: train_loss, train_top1, val_top1, val_top3, val_mrr  (list per epoch)
    """
    random.seed(seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    history: dict[str, list[float]] = {
        "train_loss": [], "train_top1": [],
        "val_top1": [], "val_top3": [], "val_mrr": [],
    }

    best_val_mrr   = -1.0
    best_val_epoch = 1
    best_val_state: dict | None = None
    best_val_metrics: dict = {}

    print(f"\n  Training EnterpriseGCN  |  {len(train_scenarios)} train  "
          f"{len(val_scenarios)} val  |  {epochs} epochs\n"
          f"  {'Epoch':>6}  {'Loss':>8}  {'Tr-Top1':>8}  "
          f"{'V-Top1':>7}  {'V-Top3':>7}  {'V-MRR':>7}")
    print("  " + "-" * 55)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        order = list(range(len(train_scenarios)))
        random.shuffle(order)

        for idx in order:
            sc = train_scenarios[idx]
            X      = torch.from_numpy(sc["X"]).to(device)
            A      = torch.from_numpy(sc["A"]).to(device)
            labels = torch.from_numpy(sc["labels"]).to(device)

            n         = labels.shape[0]
            pos_w     = torch.tensor([float(n - 1)], dtype=torch.float32, device=device)
            loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_w)

            optimizer.zero_grad()
            logits = model(X, A)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_scenarios), 1)

        # Evaluate every epoch (needed for best_val_epoch tracking)
        tr_m  = evaluate(model, train_scenarios, device)
        val_m = evaluate(model, val_scenarios,   device)

        history["train_loss"].append(avg_loss)
        history["train_top1"].append(tr_m["top1"])
        history["val_top1"].append(val_m["top1"])
        history["val_top3"].append(val_m["top3"])
        history["val_mrr"].append(val_m["mrr"])

        # Track best by val MRR (then val top1)
        if (val_m["mrr"] > best_val_mrr
                or (val_m["mrr"] == best_val_mrr and val_m["top1"] > best_val_metrics.get("top1", 0))):
            best_val_mrr    = val_m["mrr"]
            best_val_epoch  = epoch
            best_val_metrics = dict(val_m)
            best_val_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {avg_loss:>8.4f}  {tr_m['top1']:>8.2f}  "
                  f"{val_m['top1']:>7.2f}  {val_m['top3']:>7.2f}  {val_m['mrr']:>7.4f}")

    # Restore best weights
    if best_val_state is not None:
        model.load_state_dict(best_val_state)
    print(f"\n  Best val epoch: {best_val_epoch}  |  "
          f"val top1={best_val_metrics.get('top1',0):.2f}  "
          f"val MRR={best_val_metrics.get('mrr',0):.4f}")

    return history, best_val_metrics, best_val_epoch


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_model(
    model: EnterpriseGCN,
    out_dir: Path,
    hidden1: int,
    hidden2: int,
) -> None:
    ckpt = {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "feature_names":    FEATURE_NAMES,
        "architecture": {
            "in_dim":  IN_FEAT,
            "hidden1": hidden1,
            "hidden2": hidden2,
            "out_dim": 1,
        },
        "node_type_vocab":    NODE_TYPES,
        "diagram_type_vocab": DIAGRAM_TYPES,
    }
    path = out_dir / "enterprise_gnn_model.pt"
    torch.save(ckpt, str(path))
    print(f"  Model saved: {path}")


def save_metrics(
    out_dir:         Path,
    history:         dict,
    train_metrics:   dict,
    val_metrics:     dict,
    test_metrics:    dict,
    split_counts:    dict,
    epochs_trained:  int,
    best_val_epoch:  int,
    hidden1:         int,
    hidden2:         int,
    device_info:     "dict | None" = None,
) -> None:
    metrics = {
        "backend":       "torch",
        "model_type":    "Enterprise GCN RCA",
        "architecture":  f"GCN({IN_FEAT}-{hidden1}-{hidden2}-1)",
        "epochs_trained": epochs_trained,
        "best_val_epoch": best_val_epoch,
        "feature_names":  FEATURE_NAMES,
        "feature_dim":    IN_FEAT,
        "dataset_sizes":  split_counts,
        "train_metrics":  train_metrics,
        "val_metrics":    val_metrics,
        "test_metrics":   test_metrics,
        "note": (
            "This model is trained on stitched multi-diagram enterprise topology graphs. "
            "It predicts root cause across the unified enterprise graph, "
            "not just inside a single diagram."
        ),
    }
    if device_info:
        metrics.update(device_info)
    path = out_dir / "enterprise_gnn_metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  Metrics saved: {path}")


def save_training_curve(history: dict, out_dir: Path) -> None:
    if not HAS_MPL:
        print("  [warn] matplotlib not available -- training curve skipped")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.patch.set_facecolor("#0b0f1c")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1220")
        ax.spines[:].set_color("#334155")
        ax.tick_params(colors="#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")

    # Loss
    ax1.plot(epochs, history["train_loss"], color="#60a5fa", linewidth=1.8, label="Train Loss")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Enterprise GCN RCA -- Training Curve", color="#e2e8f0", fontsize=11)
    ax1.legend(facecolor="#0d1220", edgecolor="#334155", labelcolor="white", fontsize=8)

    # Ranking metrics
    ax2.plot(epochs, history["train_top1"], color="#22d3ee", linewidth=1.5, label="Train Top-1")
    ax2.plot(epochs, history["val_top1"],   color="#10b981", linewidth=1.8, label="Val Top-1")
    ax2.plot(epochs, history["val_top3"],   color="#f59e0b", linewidth=1.5, label="Val Top-3", linestyle="--")
    ax2.plot(epochs, history["val_mrr"],    color="#a78bfa", linewidth=1.5, label="Val MRR",   linestyle=":")
    ax2.set_xlabel("Epoch", color="#94a3b8")
    ax2.set_ylabel("Score")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(facecolor="#0d1220", edgecolor="#334155", labelcolor="white", fontsize=8)

    fig.tight_layout()
    path = out_dir / "enterprise_gnn_training_curve.png"
    fig.savefig(str(path), dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Training curve: {path}")



def save_presentation_result(
    model: EnterpriseGCN,
    sc: dict,
    out_dir: Path,
    device: torch.device,
) -> None:
    model_path = out_dir / "enterprise_gnn_model.pt"
    run_inference_for_scenario(model, sc, out_dir, device, model_path=model_path)
    print(f"  Presentation result saved for: {sc['scenario_id']}")


def save_prediction_viz(
    model: EnterpriseGCN,
    sc: dict,
    out_dir: Path,
    device: torch.device,
) -> None:
    if not (HAS_MPL and HAS_NX):
        print("  [warn] matplotlib/networkx not available -- prediction viz skipped")
        return

    scores, ranked = _score_scenario(model, sc, device)
    graph_data  = sc["graph_data"]
    alerts_data = sc["alerts_data"]
    node_ids    = sc["node_ids"]
    node_data   = sc["node_data"]

    rc_node   = alerts_data["root_cause"]
    pred_node = node_ids[ranked[0][0]]
    impacted  = set(alerts_data.get("impacted_nodes", []))
    alerting  = {a["node"] for a in alerts_data.get("alerts", [])}

    # Build networkx graph
    G = nx.DiGraph()
    for nd in node_data:
        G.add_node(nd["id"], ntype=nd.get("type", "unknown"),
                   diag=nd.get("diagram_id", ""))

    local_edges  = []
    cross_edges  = []
    for edge in graph_data["edges"]:
        s, t = edge["source"], edge["target"]
        if G.has_node(s) and G.has_node(t):
            G.add_edge(s, t)
            if edge.get("edge_scope") == "cross_diagram":
                cross_edges.append((s, t))
            else:
                local_edges.append((s, t))

    # Position: separate clusters by diagram_id
    diag_nodes: dict[str, list[str]] = defaultdict(list)
    for nd in node_data:
        diag_nodes[nd.get("diagram_id", "unknown")].append(nd["id"])

    pos: dict[str, tuple[float, float]] = {}
    diag_ids = sorted(diag_nodes.keys())
    n_diag   = max(len(diag_ids), 1)
    for di, diag in enumerate(diag_ids):
        sub_nodes = diag_nodes[diag]
        cx = di * 4.0
        cy = 0.0
        sub = nx.path_graph(len(sub_nodes))
        sub_pos = nx.spring_layout(sub, seed=42, k=1.5)
        for j, nid in enumerate(sub_nodes):
            sx, sy = sub_pos[j]
            pos[nid] = (cx + sx * 1.5, cy + sy * 1.5)

    # Node colours by role
    node_colors = []
    for nid in G.nodes:
        if nid == pred_node and nid == rc_node:
            node_colors.append("#00ff88")     # correct prediction
        elif nid == pred_node:
            node_colors.append("#f59e0b")     # wrong prediction (predicted)
        elif nid == rc_node:
            node_colors.append("#00d4ff")     # ground truth (not predicted)
        elif nid in alerting:
            node_colors.append("#ef4444")     # alerting
        elif nid in impacted:
            node_colors.append("#fbbf24")     # impacted
        else:
            node_colors.append("#334155")     # normal

    # Node sizes by score
    score_map = {node_ids[i]: float(scores[i]) for i in range(len(node_ids))}
    node_sizes = [600 + 1200 * score_map.get(nid, 0.0) for nid in G.nodes]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor("#0b0f1c")
    fig.patch.set_facecolor("#0b0f1c")

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: n for n in G.nodes},
                            font_size=6, font_color="white")
    nx.draw_networkx_edges(G, pos, edgelist=local_edges, ax=ax,
                           edge_color="#475569", arrows=True, arrowsize=12,
                           width=1.2, connectionstyle="arc3,rad=0.05")
    if cross_edges:
        nx.draw_networkx_edges(G, pos, edgelist=cross_edges, ax=ax,
                               edge_color="#60a5fa", arrows=True, arrowsize=14,
                               width=2.0, style="dashed",
                               connectionstyle="arc3,rad=0.12")

    legend = [
        mpatches.Patch(color="#00ff88", label="Correct prediction (root cause)"),
        mpatches.Patch(color="#00d4ff", label="Ground truth (not top-1)"),
        mpatches.Patch(color="#f59e0b", label="Wrong prediction"),
        mpatches.Patch(color="#ef4444", label="Alerting node"),
        mpatches.Patch(color="#fbbf24", label="Impacted node"),
        mpatches.Patch(color="#334155", label="Normal node"),
        mpatches.Patch(color="#60a5fa", label="Cross-diagram edge (dashed)"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=7,
              facecolor="#0d1220", edgecolor="#334155", labelcolor="white")

    scenario_id = sc["scenario_id"]
    ax.set_title(
        f"Enterprise GNN RCA -- {scenario_id} | "
        f"pred={pred_node} | gt={rc_node} | nodes={len(node_ids)}",
        color="#e2e8f0", fontsize=10,
    )
    ax.axis("off")
    fig.tight_layout()

    fname = f"{scenario_id}_enterprise_gnn_prediction.png"
    path  = out_dir / fname
    fig.savefig(str(path), dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Prediction viz: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enterprise GNN RCA -- train on stitched multi-diagram graphs"
    )
    p.add_argument("--dataset-root", default=str(get_default_dataset_root(REPO_ROOT)),
                   help="Dataset root: V3 (datasets/infragraph_v3) preferred; "
                        "V1 (datasets/infragraph_v1/enterprise_graph) supported for backward compat")
    p.add_argument("--out",  default="assets/preloaded/enterprise_gnn_rca",
                   help="Output directory")
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--lr",           type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--hidden1",      type=int,   default=96)
    p.add_argument("--hidden2",      type=int,   default=48)
    p.add_argument("--presentation-scenario", default="enterprise_v3_0000",
                   help="Scenario ID to use for inference visualisation output")
    p.add_argument("--presentation-split",    default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Training device. 'auto' uses CUDA if available, else CPU.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    out_dir      = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device selection ─────────────────────────────────────────────────────
    cuda_available = torch.cuda.is_available()
    if args.device == "auto":
        device = torch.device("cuda") if cuda_available else torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device_name = ""
    try:
        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(device)
    except Exception:
        pass

    device_info = {
        "selected_device": str(device),
        "torch_version":   torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "cuda_available":  cuda_available,
        "device_name":     device_name,
    }

    print("InfraGraph AI -- Enterprise GNN RCA")
    print(f"  Dataset : {dataset_root}")
    print(f"  Output  : {out_dir}")
    print(f"  Device  : {device}  (cuda_available={cuda_available}"
          + (f", device_name={device_name}" if device_name else "") + ")")
    print(f"  Epochs  : {args.epochs}  LR={args.lr}  WD={args.weight_decay}")
    print(f"  Hidden  : {args.hidden1} -> {args.hidden2}  Features: {IN_FEAT}")
    if not HAS_MPL:
        print("  [warn] matplotlib not available -- visualisations will be skipped")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading dataset splits...")
    train_sc = load_split(dataset_root, "train")
    val_sc   = load_split(dataset_root, "val")
    test_sc  = load_split(dataset_root, "test")

    split_counts = {
        "train": len(train_sc),
        "val":   len(val_sc),
        "test":  len(test_sc),
    }
    print(f"  train={len(train_sc)}  val={len(val_sc)}  test={len(test_sc)}")

    if not train_sc:
        print("[ERROR] No valid training scenarios found. Check --dataset-root.")
        sys.exit(1)

    # ── Build model ──────────────────────────────────────────────────────────
    model = EnterpriseGCN(
        in_dim=IN_FEAT,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── Train ────────────────────────────────────────────────────────────────
    t0 = time.time()
    history, best_val_metrics, best_val_epoch = train(
        model, train_sc, val_sc, device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed:.1f}s")

    # ── Final evaluation ─────────────────────────────────────────────────────
    print("\nFinal evaluation (best-epoch weights)...")
    train_m = evaluate(model, train_sc, device)
    val_m   = evaluate(model, val_sc,   device)
    test_m  = evaluate(model, test_sc,  device)

    for split, m in [("train", train_m), ("val", val_m), ("test", test_m)]:
        print(f"  {split:5s}  top1={m['top1']:.2f}  top3={m['top3']:.2f}  "
              f"mrr={m['mrr']:.4f}  n={m['scenario_count']}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    print("\nSaving outputs...")
    save_model(model, out_dir, args.hidden1, args.hidden2)
    save_metrics(
        out_dir, history, train_m, val_m, test_m,
        split_counts, args.epochs, best_val_epoch, args.hidden1, args.hidden2,
        device_info=device_info,
    )
    save_training_curve(history, out_dir)

    # ── Presentation scenario ────────────────────────────────────────────────
    print(f"\nPresentation scenario: {args.presentation_scenario} (split={args.presentation_split})")
    presentation_pool = {"train": train_sc, "val": val_sc, "test": test_sc}[args.presentation_split]
    presentation_sc   = next((s for s in presentation_pool if s["scenario_id"] == args.presentation_scenario), None)

    if presentation_sc is None:
        # Use the first available scenario when the requested one is absent.
        for pool in [test_sc, val_sc, train_sc]:
            if pool:
                presentation_sc = pool[0]
                print(f"  [warn] '{args.presentation_scenario}' not found in '{args.presentation_split}' "
                      f"-- using {presentation_sc['scenario_id']}")
                break

    if presentation_sc is not None:
        save_presentation_result(model, presentation_sc, out_dir, device)
        save_prediction_viz(model, presentation_sc, out_dir, device)
    else:
        print("  [warn] No presentation scenario available")

    print("\nDone.")


if __name__ == "__main__":
    main()

