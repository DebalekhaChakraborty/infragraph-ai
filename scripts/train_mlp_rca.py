"""
Learned MLP node-ranker for root cause analysis -- InfraGraph AI.

This model scores each node independently from engineered features with no
graph message-passing.  It is a supervised learned baseline that sits between
the rule-based heuristic scorer (Stage 2) and the topology-aware GNN (Stage 3).

Backend: torch (if installed) -> pure-numpy MLP Alternate path.
"""

import argparse
import json
import math
import os
import time

import numpy as np

# ── Torch detection ──────────────────────────────────────────────────────────
USE_TORCH = False
try:
    import torch
    import torch.nn as nn
    USE_TORCH = True
except ImportError:
    pass

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ── Constants ────────────────────────────────────────────────────────────────
DEVICE_TYPES = [
    "router", "switch", "firewall", "server",
    "database", "load_balancer", "cloud_or_wan",
]
DEVICE_TO_IDX = {d: i for i, d in enumerate(DEVICE_TYPES)}
SEV_SCORE = {"critical": 4, "major": 3, "minor": 2, "warning": 1, "info": 1}

FEATURE_NAMES = [
    # one-hot device type [0-6]
    "device_router", "device_switch", "device_firewall", "device_server",
    "device_database", "device_load_balancer", "device_cloud_or_wan",
    # alert features [7-10]
    "has_alert", "max_severity_score", "earliest_time_score", "alert_count_norm",
    # topology features [11-15]
    "in_degree_norm", "out_degree_norm", "total_degree_norm",
    "downstream_reach_norm", "upstream_reach_norm",
    # boolean device-type flags [16-22]
    "is_firewall", "is_router", "is_switch", "is_server",
    "is_database", "is_load_balancer", "is_cloud_or_wan",
]
IN_FEAT = len(FEATURE_NAMES)   # 23
HIDDEN1, HIDDEN2 = 64, 32


# ── Feature extraction ────────────────────────────────────────────────────────

def _bfs_count(adj_list, start):
    visited, q = set(), [start]
    while q:
        cur = q.pop()
        for nb in adj_list[cur]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return len(visited)


def build_sample(graph_data, alert_data):
    """
    Build per-node feature matrix and metadata for one graph.

    Returns a dict with keys:
      X          : np.array (n, IN_FEAT)  float32
      y          : np.array (n,)          float32  (1 at root_idx, 0 elsewhere)
      root_idx   : int
      node_ids   : list[str]
      node_types : list[str]
      node_meta  : list[dict]  -- has_alert, max_severity per node
      diagram_id : str
    Returns None if root cause node is missing from the graph.
    """
    nodes = graph_data["nodes"]
    n = len(nodes)
    node_ids = [nd["id"] for nd in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    root_cause = alert_data.get("root_cause", "")
    if root_cause not in id_to_idx:
        return None
    root_idx = id_to_idx[root_cause]

    # Alert lookup
    alerts_by_node = {}
    for al in alert_data.get("alerts", []):
        alerts_by_node.setdefault(al["node"], []).append(al)
    max_alert_count = max((len(v) for v in alerts_by_node.values()), default=1)

    # Degree and adjacency lists
    in_deg = [0] * n
    out_deg = [0] * n
    fwd_adj = [[] for _ in range(n)]   # directed forward edges
    rev_adj = [[] for _ in range(n)]   # directed backward edges
    for edge in graph_data.get("edges", []):
        s, d = edge.get("source"), edge.get("target")
        if s in id_to_idx and d in id_to_idx:
            si, di = id_to_idx[s], id_to_idx[d]
            out_deg[si] += 1
            in_deg[di] += 1
            fwd_adj[si].append(di)
            rev_adj[di].append(si)

    X = np.zeros((n, IN_FEAT), dtype=np.float32)
    node_meta = []
    for i, nd in enumerate(nodes):
        ntype = nd.get("type", "server")
        type_idx = DEVICE_TO_IDX.get(ntype, 3)

        # [0-6] one-hot
        X[i, type_idx] = 1.0

        # [7-10] alert features
        node_alerts = alerts_by_node.get(nd["id"], [])
        has_alert = len(node_alerts) > 0
        if has_alert:
            max_sev = max(SEV_SCORE.get(a["severity"], 1) for a in node_alerts)
            earliest = min(a.get("time_offset_min", 0) for a in node_alerts)
            max_sev_str = max(node_alerts, key=lambda a: SEV_SCORE.get(a["severity"], 1))["severity"]
            X[i, 7] = 1.0
            X[i, 8] = max_sev / 4.0
            X[i, 9] = 1.0 / (1.0 + earliest)
            X[i, 10] = len(node_alerts) / max_alert_count
        else:
            max_sev_str = None

        # [11-15] topology features
        denom = max(1, n - 1)
        X[i, 11] = in_deg[i] / denom
        X[i, 12] = out_deg[i] / denom
        X[i, 13] = (in_deg[i] + out_deg[i]) / max(1, 2 * (n - 1))
        X[i, 14] = _bfs_count(fwd_adj, i) / denom
        X[i, 15] = _bfs_count(rev_adj, i) / denom

        # [16-22] boolean device-type flags
        X[i, 16] = 1.0 if ntype == "firewall" else 0.0
        X[i, 17] = 1.0 if ntype == "router" else 0.0
        X[i, 18] = 1.0 if ntype == "switch" else 0.0
        X[i, 19] = 1.0 if ntype == "server" else 0.0
        X[i, 20] = 1.0 if ntype == "database" else 0.0
        X[i, 21] = 1.0 if ntype == "load_balancer" else 0.0
        X[i, 22] = 1.0 if ntype == "cloud_or_wan" else 0.0

        node_meta.append({
            "has_alert": has_alert,
            "max_severity": max_sev_str,
        })

    y = np.zeros(n, dtype=np.float32)
    y[root_idx] = 1.0

    return {
        "X": X,
        "y": y,
        "root_idx": root_idx,
        "node_ids": node_ids,
        "node_types": [nd.get("type", "server") for nd in nodes],
        "node_meta": node_meta,
        "diagram_id": graph_data["diagram_id"],
    }


def load_split(dataset_root, split):
    graphs_dir = os.path.join(dataset_root, "graphs", split)
    alerts_dir = os.path.join(dataset_root, "alerts", split)
    samples = []
    for fname in sorted(os.listdir(graphs_dir)):
        if not fname.endswith(".json"):
            continue
        gpath = os.path.join(graphs_dir, fname)
        apath = os.path.join(alerts_dir, fname)
        if not os.path.isfile(apath):
            continue
        with open(gpath) as f:
            gdata = json.load(f)
        with open(apath) as f:
            adata = json.load(f)
        s = build_sample(gdata, adata)
        if s is not None:
            samples.append(s)
    return samples


def flatten_split(samples):
    """Stack all node features and labels into flat arrays for batch training."""
    X_parts, y_parts = [], []
    for s in samples:
        X_parts.append(s["X"])
        y_parts.append(s["y"])
    return np.vstack(X_parts), np.concatenate(y_parts)


# ── Torch model ───────────────────────────────────────────────────────────────
if USE_TORCH:
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(IN_FEAT, HIDDEN1),
                nn.ReLU(),
                nn.Linear(HIDDEN1, HIDDEN2),
                nn.ReLU(),
                nn.Linear(HIDDEN2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)   # (n,)

    def _create_model():
        return MLP()

    def _create_optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    def _train_epoch(model, optimizer, loss_fn, X_t, y_t):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    def _infer(model, X):
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(X)).numpy()

    def _save_model(model, path):
        torch.save(model.state_dict(), path)

    def _load_model(path):
        m = MLP()
        m.load_state_dict(torch.load(path, weights_only=True))
        return m

    def _n_params(model):
        return sum(p.numel() for p in model.parameters())

    def _model_ext():
        return ".pt"

# ── Pure-numpy MLP Alternate path ───────────────────────────────────────────────────
else:
    def _sigmoid(x):
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    class NpMLP:
        def __init__(self):
            rng = np.random.default_rng(42)
            self.W1 = rng.normal(0, math.sqrt(2.0 / IN_FEAT),
                                 (IN_FEAT, HIDDEN1)).astype(np.float32)
            self.b1 = np.zeros(HIDDEN1, dtype=np.float32)
            self.W2 = rng.normal(0, math.sqrt(2.0 / HIDDEN1),
                                 (HIDDEN1, HIDDEN2)).astype(np.float32)
            self.b2 = np.zeros(HIDDEN2, dtype=np.float32)
            self.W3 = rng.normal(0, math.sqrt(2.0 / HIDDEN2),
                                 (HIDDEN2, 1)).astype(np.float32)
            self.b3 = np.zeros(1, dtype=np.float32)
            self._params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
            self._m = [np.zeros_like(p) for p in self._params]
            self._v = [np.zeros_like(p) for p in self._params]
            self._t = 0
            self.lr = 1e-3
            self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

        def forward(self, X):
            self._X = X
            self._Z1 = X @ self.W1 + self.b1
            self._H1 = np.maximum(0, self._Z1)
            self._Z2 = self._H1 @ self.W2 + self.b2
            self._H2 = np.maximum(0, self._Z2)
            logits = (self._H2 @ self.W3 + self.b3).squeeze(-1)
            self._logits = logits
            return logits

        def _adam_step(self, grads):
            self._t += 1
            for i, (p, g) in enumerate(zip(self._params, grads)):
                gnorm = np.linalg.norm(g)
                if gnorm > 1.0:
                    g = g * (1.0 / gnorm)
                self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
                self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * g * g
                m_hat = self._m[i] / (1 - self.beta1 ** self._t)
                v_hat = self._v[i] / (1 - self.beta2 ** self._t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        def backward(self, y, pos_weight):
            sig = _sigmoid(self._logits)
            n = len(y)
            # BCE with logits + pos_weight gradient
            # d_loss/d_logit = (sigma*(1+(w-1)*y) - w*y) / n
            d_logits = (sig * (1.0 + (pos_weight - 1.0) * y) - pos_weight * y) / n
            # loss for logging
            loss = -np.mean(
                pos_weight * y * np.log(sig + 1e-9)
                + (1 - y) * np.log(1 - sig + 1e-9)
            )

            d_logits_col = d_logits[:, np.newaxis]     # (n, 1)
            dW3 = self._H2.T @ d_logits_col
            db3 = d_logits_col.sum(axis=0)

            d_H2 = d_logits_col @ self.W3.T            # (n, HIDDEN2)
            d_Z2 = d_H2 * (self._Z2 > 0)

            dW2 = self._H1.T @ d_Z2
            db2 = d_Z2.sum(axis=0)

            d_H1 = d_Z2 @ self.W2.T                    # (n, HIDDEN1)
            d_Z1 = d_H1 * (self._Z1 > 0)

            dW1 = self._X.T @ d_Z1
            db1 = d_Z1.sum(axis=0)

            self._adam_step([dW1, db1, dW2, db2, dW3, db3])
            return float(loss)

    def _create_model():
        return NpMLP()

    def _create_optimizer(_model):
        return None

    def _train_epoch(model, _optimizer, _loss_fn, X_t, y_t, pos_weight=None):
        # pos_weight passed separately for numpy path
        model.forward(X_t)
        return model.backward(y_t, pos_weight)

    def _infer(model, X):
        return model.forward(X)

    def _save_model(model, path):
        np.savez(
            path.replace(".pt", ".npz"),
            W1=model.W1, b1=model.b1,
            W2=model.W2, b2=model.b2,
            W3=model.W3, b3=model.b3,
        )

    def _load_model(path):
        return None  # numpy: re-use live model

    def _n_params(model):
        return sum(p.size for p in [model.W1, model.b1,
                                    model.W2, model.b2,
                                    model.W3, model.b3])

    def _model_ext():
        return ".npz"


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, samples):
    top1 = top3 = mrr_sum = 0
    for s in samples:
        scores = _infer(model, s["X"])
        ranked = np.argsort(-scores)
        rank = int(np.where(ranked == s["root_idx"])[0][0])
        top1 += rank == 0
        top3 += rank < 3
        mrr_sum += 1.0 / (rank + 1)
    n = len(samples)
    return {"top1": top1 / n, "top3": top3 / n, "mrr": mrr_sum / n, "n": n}


# ── Training curve ────────────────────────────────────────────────────────────

def plot_curve(history, out_path):
    if not HAS_MPL:
        print("  [skip] matplotlib not available -- training curve not saved")
        return
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE loss")
    ax1.set_title("Training loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [h["train_top1"] for h in history], label="train top-1")
    ax2.plot(epochs, [h["val_top1"] for h in history], label="val top-1")
    ax2.plot(epochs, [h["val_top3"] for h in history], label="val top-3", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Top-1 / Top-3 accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"MLP RCA training -- backend: {'torch' if USE_TORCH else 'numpy_mlp'}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved training curve -> {out_path}")


# ── Presentation inference ────────────────────────────────────────────────────────────

def infer_diagram(model, dataset_root, split, diagram_id):
    gpath = os.path.join(dataset_root, "graphs", split, f"{diagram_id}.json")
    apath = os.path.join(dataset_root, "alerts", split, f"{diagram_id}.json")
    with open(gpath) as f:
        gdata = json.load(f)
    with open(apath) as f:
        adata = json.load(f)

    s = build_sample(gdata, adata)
    if s is None:
        raise ValueError(f"Root cause node missing in {diagram_id}")

    scores = _infer(model, s["X"])
    ranked = list(np.argsort(-scores))
    rank = int(np.where(np.array(ranked) == s["root_idx"])[0][0])

    top_candidates = [
        {
            "rank": r + 1,
            "node_id": s["node_ids"][idx],
            "score": round(float(scores[idx]), 4),
            "type": s["node_types"][idx],
            "has_alert": s["node_meta"][idx]["has_alert"],
            "severity": s["node_meta"][idx]["max_severity"],
        }
        for r, idx in enumerate(ranked[:5])
    ]

    return {
        "diagram_id": diagram_id,
        "backend": "torch" if USE_TORCH else "numpy_mlp",
        "model_type": "MLP node ranker",
        "predicted_root_cause": s["node_ids"][ranked[0]],
        "ground_truth_root_cause": adata["root_cause"],
        "is_correct": bool(ranked[0] == s["root_idx"]),
        "ground_truth_rank": rank + 1,
        "mrr": round(1.0 / (rank + 1), 4),
        "top_candidates": top_candidates,
        "n_nodes": len(s["node_ids"]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train MLP node-ranker RCA model on the InfraGraph v2 dataset."
    )
    parser.add_argument("--dataset-root", default="datasets/infragraph_v2")
    parser.add_argument("--out", default="demo_assets/mlp_rca")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--Presentation-diagram", default="diagram_0373")
    parser.add_argument("--Presentation-split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if USE_TORCH:
        torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print("  InfraGraph AI -- MLP RCA Node Ranker")
    print(f"  Backend : {'torch ' + torch.__version__ if USE_TORCH else 'pure-numpy MLP'}")
    print(f"  Dataset : {args.dataset_root}")
    print(f"  Epochs  : {args.epochs}")
    print(f"{'='*60}\n")

    os.makedirs(args.out, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("[1/5] Loading dataset ...")
    t0 = time.time()
    train_samples = load_split(args.dataset_root, "train")
    val_samples   = load_split(args.dataset_root, "val")
    test_samples  = load_split(args.dataset_root, "test")
    print(f"  train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}"
          f"  ({time.time()-t0:.1f}s)")

    X_train_np, y_train_np = flatten_split(train_samples)
    n_total = len(y_train_np)
    n_pos   = int(y_train_np.sum())
    pos_weight_val = (n_total - n_pos) / max(n_pos, 1)
    print(f"  Total training nodes: {n_total}  root-cause nodes: {n_pos}"
          f"  pos_weight: {pos_weight_val:.1f}")

    # ── Create model ───────────────────────────────────────────────────────
    print(f"\n[2/5] Building model ...")
    model = _create_model()
    optimizer = _create_optimizer(model)
    n_params = _n_params(model)
    print(f"  MLP({IN_FEAT}->{HIDDEN1}->{HIDDEN2}->1)  params={n_params:,}")

    if USE_TORCH:
        X_t = torch.tensor(X_train_np)
        y_t = torch.tensor(y_train_np)
        pw_t = torch.tensor([pos_weight_val])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_t)
    else:
        X_t = X_train_np
        y_t = y_train_np

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n[3/5] Training for {args.epochs} epochs ...")
    history = []
    best_val_top1 = -1.0
    best_epoch = 0
    model_path = os.path.join(args.out, "mlp_rca_model" + _model_ext())
    best_np_weights = None

    for epoch in range(1, args.epochs + 1):
        if USE_TORCH:
            loss_val = _train_epoch(model, optimizer, loss_fn, X_t, y_t)
        else:
            loss_val = _train_epoch(model, None, None, X_t, y_t,
                                    pos_weight=pos_weight_val)

        train_metrics = evaluate(model, train_samples)
        val_metrics   = evaluate(model, val_samples)

        rec = {
            "epoch": epoch,
            "train_loss": round(loss_val, 4),
            "train_top1": round(train_metrics["top1"], 4),
            "val_top1": round(val_metrics["top1"], 4),
            "val_top3": round(val_metrics["top3"], 4),
            "val_mrr": round(val_metrics["mrr"], 4),
        }
        history.append(rec)

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_epoch = epoch
            _save_model(model, model_path)
            if not USE_TORCH:
                best_np_weights = {
                    k: v.copy() for k, v in
                    {"W1": model.W1, "b1": model.b1,
                     "W2": model.W2, "b2": model.b2,
                     "W3": model.W3, "b3": model.b3}.items()
                }

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d}/{args.epochs}  "
                f"loss={loss_val:.4f}  "
                f"tr@1={train_metrics['top1']:.3f}  "
                f"val@1={val_metrics['top1']:.3f}  "
                f"val@3={val_metrics['top3']:.3f}  "
                f"val_mrr={val_metrics['mrr']:.3f}"
            )

    print(f"  Best val top-1={best_val_top1:.3f} at epoch {best_epoch}")

    # Restore best weights
    if USE_TORCH:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    elif best_np_weights is not None:
        for k, v in best_np_weights.items():
            getattr(model, k)[:] = v

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/5] Final evaluation ...")
    train_m = evaluate(model, train_samples)
    val_m   = evaluate(model, val_samples)
    test_m  = evaluate(model, test_samples)

    def _fmt(m):
        return f"top1={m['top1']:.3f}  top3={m['top3']:.3f}  mrr={m['mrr']:.3f}"

    print(f"  TRAIN ({train_m['n']} samples): {_fmt(train_m)}")
    print(f"  VAL   ({val_m['n']} samples): {_fmt(val_m)}")
    print(f"  TEST  ({test_m['n']} samples): {_fmt(test_m)}")

    metrics = {
        "backend": "torch" if USE_TORCH else "numpy_mlp",
        "model_type": "MLP node ranker",
        "architecture": f"MLP({IN_FEAT}->{HIDDEN1}->{HIDDEN2}->1)",
        "n_params": n_params,
        "epochs_trained": args.epochs,
        "best_val_epoch": best_epoch,
        "feature_names": FEATURE_NAMES,
        "dataset_sizes": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "train": {k: round(v, 4) for k, v in train_m.items() if k != "n"},
        "val":   {k: round(v, 4) for k, v in val_m.items()   if k != "n"},
        "test":  {k: round(v, 4) for k, v in test_m.items()  if k != "n"},
        "note": (
            "This is a learned node-level RCA model without graph message passing. "
            "It is used as a learned baseline against the topology-aware GNN RCA model."
        ),
        "training_history": history,
    }
    metrics_path = os.path.join(args.out, "mlp_rca_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics -> {metrics_path}")

    curve_path = os.path.join(args.out, "mlp_training_curve.png")
    plot_curve(history, curve_path)

    # ── Presentation inference ─────────────────────────────────────────────────────
    print(f"\n[5/5] Presentation inference on {args.demo_diagram} ...")
    Presentation = infer_diagram(model, args.dataset_root, args.demo_split, args.demo_diagram)
    Presentation["model_path"] = model_path
    Presentation["test_metrics"] = {k: round(v, 4) for k, v in test_m.items() if k != "n"}
    demo_path = os.path.join(args.out, f"{args.demo_diagram}_mlp_rca_result.json")
    with open(demo_path, "w") as f:
        json.dump(Presentation, f, indent=2)

    correct_str = "CORRECT" if Presentation["is_correct"] else "WRONG"
    print(
        f"  Predicted: {Presentation['predicted_root_cause']}  "
        f"GT: {Presentation['ground_truth_root_cause']}  [{correct_str}]"
    )
    print(f"  GT rank={Presentation['ground_truth_rank']}  MRR={Presentation['mrr']}")
    print(f"  Saved Presentation result -> {demo_path}")

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"  Model  : {model_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Presentation   : {demo_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

