"""
Stage 3 of InfraGraph AI: GNN-based root cause ranking.

Trains a 2-layer Graph Convolutional Network on the infragraph_v2 dataset.
The GNN learns propagation direction from topology structure and temporal alert
features -- resolving the ambiguity that the heuristic scorer (Stage 2) cannot.

Backend: torch (if installed in the active venv) -> pure-numpy GCN Alternate path.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# ── Torch detection ──────────────────────────────────────────────────────────
USE_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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
IN_FEAT = 16
HIDDEN1 = 64
HIDDEN2 = 32


# ── Feature engineering ──────────────────────────────────────────────────────

def _build_adj(n, edges_src_dst):
    """Symmetrised + self-loop + D^{-1/2} A D^{-1/2} normalisation."""
    A = np.zeros((n, n), dtype=np.float32)
    for s, d in edges_src_dst:
        A[s, d] = 1.0
        A[d, s] = 1.0
    A += np.eye(n, dtype=np.float32)
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    return (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]


def _bfs_count(adj_list, start):
    """Number of nodes reachable from start (directed, not counting start)."""
    visited, q = set(), [start]
    while q:
        cur = q.pop()
        for nb in adj_list[cur]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return len(visited)


def build_features(graph_data, alert_data):
    """
    Returns (X, A_norm, root_idx, node_ids) or None if root node missing.

    Node feature layout (16 dims):
      [0-6]  one-hot device type
      [7]    has_alert
      [8]    max_severity / 4.0
      [9]    1 / (1 + earliest_alert_time)
      [10]   alert_count / max_alert_count_in_graph
      [11]   in_degree_norm
      [12]   out_degree_norm
      [13]   total_degree_norm
      [14]   downstream_reach_norm
      [15]   is_priority_device (firewall or router)
    """
    nodes = graph_data["nodes"]
    n = len(nodes)
    node_ids = [nd["id"] for nd in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    root_cause = alert_data.get("root_cause", "")
    if root_cause not in id_to_idx:
        return None
    root_idx = id_to_idx[root_cause]

    alerts_by_node = {}
    for al in alert_data.get("alerts", []):
        nid = al["node"]
        alerts_by_node.setdefault(nid, []).append(al)
    max_alert_count = max((len(v) for v in alerts_by_node.values()), default=1)

    in_deg = [0] * n
    out_deg = [0] * n
    edges_src_dst = []
    adj_list = [[] for _ in range(n)]
    for edge in graph_data.get("edges", []):
        s, d = edge.get("source"), edge.get("target")
        if s in id_to_idx and d in id_to_idx:
            si, di = id_to_idx[s], id_to_idx[d]
            out_deg[si] += 1
            in_deg[di] += 1
            edges_src_dst.append((si, di))
            adj_list[si].append(di)

    A_norm = _build_adj(n, edges_src_dst)

    X = np.zeros((n, IN_FEAT), dtype=np.float32)
    for i, nd in enumerate(nodes):
        ntype = nd.get("type", "server")
        X[i, DEVICE_TO_IDX.get(ntype, 3)] = 1.0

        node_alerts = alerts_by_node.get(nd["id"], [])
        if node_alerts:
            X[i, 7] = 1.0
            max_sev = max(SEV_SCORE.get(a["severity"], 1) for a in node_alerts)
            earliest = min(a.get("time_offset_min", 0) for a in node_alerts)
            X[i, 8] = max_sev / 4.0
            X[i, 9] = 1.0 / (1.0 + earliest)
            X[i, 10] = len(node_alerts) / max_alert_count

        denom = max(1, n - 1)
        X[i, 11] = in_deg[i] / denom
        X[i, 12] = out_deg[i] / denom
        X[i, 13] = (in_deg[i] + out_deg[i]) / max(1, 2 * (n - 1))
        X[i, 14] = _bfs_count(adj_list, i) / denom
        X[i, 15] = 1.0 if ntype in ("firewall", "router") else 0.0

    return X, A_norm, root_idx, node_ids


def load_split(dataset_root, split):
    """Return list of (X, A_norm, root_idx, node_ids) for every valid sample."""
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
        result = build_features(gdata, adata)
        if result is not None:
            samples.append(result)
    return samples


# ── Torch model ───────────────────────────────────────────────────────────────
if USE_TORCH:
    class GCN(nn.Module):
        """Two-layer GCN with per-node scoring head."""
        def __init__(self):
            super().__init__()
            self.W1 = nn.Linear(IN_FEAT, HIDDEN1, bias=False)
            self.W2 = nn.Linear(HIDDEN1, HIDDEN2, bias=False)
            self.out = nn.Linear(HIDDEN2, 1, bias=True)

        def forward(self, A_norm, X):
            H1 = torch.relu(A_norm @ self.W1(X))
            H2 = torch.relu(A_norm @ self.W2(H1))
            return self.out(H2).squeeze(-1)   # (n,)

    def _create_model():
        return GCN()

    def _create_optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    def _train_step(model, optimizer, X, A_norm, root_idx):
        model.train()
        optimizer.zero_grad()
        Xt = torch.tensor(X)
        At = torch.tensor(A_norm)
        scores = model(At, Xt)
        loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([root_idx]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.item(), scores.detach().numpy()

    def _infer(model, X, A_norm):
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(A_norm), torch.tensor(X)).numpy()

    def _save_model(model, path):
        torch.save(model.state_dict(), path)

    def _model_ext():
        return ".pt"

# ── Pure-numpy GCN (no-torch Alternate path) ───────────────────────────────────────
else:
    class NpGCN:
        """
        Two-layer GCN implemented entirely in NumPy.
        Backprop and Adam are computed analytically.
        """
        def __init__(self):
            rng = np.random.default_rng(42)
            self.W1 = rng.normal(0, math.sqrt(2.0 / IN_FEAT),
                                 (IN_FEAT, HIDDEN1)).astype(np.float32)
            self.W2 = rng.normal(0, math.sqrt(2.0 / HIDDEN1),
                                 (HIDDEN1, HIDDEN2)).astype(np.float32)
            self.W_out = rng.normal(0, math.sqrt(2.0 / HIDDEN2),
                                    (HIDDEN2, 1)).astype(np.float32)
            self.b_out = np.zeros((1,), dtype=np.float32)
            self._params = [self.W1, self.W2, self.W_out, self.b_out]
            self._m = [np.zeros_like(p) for p in self._params]
            self._v = [np.zeros_like(p) for p in self._params]
            self._t = 0
            self.lr = 1e-3
            self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

        def forward(self, A_norm, X):
            self._A = A_norm
            self._Y1 = A_norm @ X
            self._Z1 = self._Y1 @ self.W1
            self._H1 = np.maximum(0, self._Z1)
            self._Y2 = A_norm @ self._H1
            self._Z2 = self._Y2 @ self.W2
            self._H2 = np.maximum(0, self._Z2)
            logits = self._H2 @ self.W_out + self.b_out  # (n, 1)
            return logits.squeeze(-1)                     # (n,)

        def _adam_update(self, grads):
            self._t += 1
            for i, (p, g) in enumerate(zip(self._params, grads)):
                g_norm = np.linalg.norm(g)
                if g_norm > 1.0:
                    g = g * (1.0 / g_norm)
                self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
                self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * g * g
                m_hat = self._m[i] / (1 - self.beta1 ** self._t)
                v_hat = self._v[i] / (1 - self.beta2 ** self._t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        def backward(self, scores, root_idx):
            """Compute loss, run Adam step, return scalar loss."""
            ex = np.exp(scores - scores.max())
            probs = ex / ex.sum()
            loss = -math.log(max(float(probs[root_idx]), 1e-9))

            d_scores = probs.copy()
            d_scores[root_idx] -= 1.0
            d_logits = d_scores[:, np.newaxis]   # (n, 1)

            dW_out = self._H2.T @ d_logits        # (h2, 1)
            db_out = d_logits.sum(axis=0)         # (1,)

            d_H2 = d_logits @ self.W_out.T        # (n, h2)
            d_Z2 = d_H2 * (self._Z2 > 0)

            dW2 = self._Y2.T @ d_Z2              # (h1, h2)
            d_Y2 = d_Z2 @ self.W2.T              # (n, h1)

            d_H1 = self._A.T @ d_Y2              # (n, h1)
            d_Z1 = d_H1 * (self._Z1 > 0)

            dW1 = self._Y1.T @ d_Z1              # (in_feat, h1)

            self._adam_update([dW1, dW2, dW_out, db_out])
            return loss

    def _create_model():
        return NpGCN()

    def _create_optimizer(_model):
        return None  # Adam is baked into NpGCN

    def _train_step(model, _optimizer, X, A_norm, root_idx):
        scores = model.forward(A_norm, X)
        loss = model.backward(scores, root_idx)
        # Re-forward to get post-update scores for accuracy tracking
        post_scores = model.forward(A_norm, X)
        return loss, post_scores

    def _infer(model, X, A_norm):
        return model.forward(A_norm, X)

    def _save_model(model, path):
        np.savez(
            path.replace(".pt", ".npz"),
            W1=model.W1, W2=model.W2,
            W_out=model.W_out, b_out=model.b_out,
        )

    def _model_ext():
        return ".npz"


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_epoch(model, optimizer, samples):
    total_loss, correct = 0.0, 0
    for X, A_norm, root_idx, _ in samples:
        loss_val, scores = _train_step(model, optimizer, X, A_norm, root_idx)
        total_loss += loss_val
        if int(np.argmax(scores)) == root_idx:
            correct += 1
    return total_loss / len(samples), correct / len(samples)


def evaluate(model, samples):
    top1 = top3 = mrr_sum = 0
    for X, A_norm, root_idx, _ in samples:
        scores = _infer(model, X, A_norm)
        ranked = np.argsort(-scores)
        rank = int(np.where(ranked == root_idx)[0][0])
        top1 += rank == 0
        top3 += rank < 3
        mrr_sum += 1.0 / (rank + 1)
    n = len(samples)
    return {"top1": top1 / n, "top3": top3 / n, "mrr": mrr_sum / n, "n": n}


# ── Diagram-level inference ───────────────────────────────────────────────────

def infer_diagram(model, dataset_root, split, diagram_id):
    gpath = os.path.join(dataset_root, "graphs", split, f"{diagram_id}.json")
    apath = os.path.join(dataset_root, "alerts", split, f"{diagram_id}.json")
    with open(gpath) as f:
        gdata = json.load(f)
    with open(apath) as f:
        adata = json.load(f)

    result = build_features(gdata, adata)
    if result is None:
        raise ValueError(f"Root cause node not found in graph for {diagram_id}")
    X, A_norm, root_idx, node_ids = result

    scores = _infer(model, X, A_norm)
    ranked = list(np.argsort(-scores))
    rank = int(np.where(np.array(ranked) == root_idx)[0][0])

    node_types = {nd["id"]: nd.get("type", "server") for nd in gdata["nodes"]}
    top_candidates = [
        {
            "rank": r + 1,
            "node": node_ids[idx],
            "score": float(scores[idx]),
            "type": node_types.get(node_ids[idx], "server"),
        }
        for r, idx in enumerate(ranked[:5])
    ]

    return {
        "diagram_id": diagram_id,
        "method": "gnn_rca",
        "backend": "torch" if USE_TORCH else "numpy_gcn",
        "predicted_root_cause": node_ids[ranked[0]],
        "ground_truth_root_cause": adata["root_cause"],
        "is_correct": bool(ranked[0] == root_idx),
        "ground_truth_rank": rank + 1,
        "mrr": round(1.0 / (rank + 1), 4),
        "node_scores": {node_ids[i]: round(float(scores[i]), 4) for i in range(len(node_ids))},
        "top_candidates": top_candidates,
        "n_nodes": len(node_ids),
    }


# ── Training curve ────────────────────────────────────────────────────────────

def plot_curve(history, out_path):
    if not HAS_MPL:
        print("  [skip] matplotlib not available — training curve not saved")
        return
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
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
        f"GNN RCA training — backend: {'torch' if USE_TORCH else 'numpy_gcn'}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved training curve ->{out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train GNN for root cause ranking on the InfraGraph v2 dataset."
    )
    parser.add_argument("--dataset-root", default="datasets/infragraph_v2")
    parser.add_argument("--out", default="demo_assets/gnn_rca",
                        help="Output directory for model, metrics, and curve")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--Presentation-diagram", default="diagram_0373",
                        help="Diagram ID to run inference on and save Presentation JSON")
    parser.add_argument("--Presentation-split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if USE_TORCH:
        torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print("  InfraGraph AI — GNN Root Cause Ranking (Stage 3)")
    print(f"  Backend : {'torch ' + torch.__version__ if USE_TORCH else 'pure-numpy GCN'}")
    print(f"  Dataset : {args.dataset_root}")
    print(f"  Epochs  : {args.epochs}")
    print(f"{'='*60}\n")

    os.makedirs(args.out, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("[1/5] Loading dataset …")
    t0 = time.time()
    train_data = load_split(args.dataset_root, "train")
    val_data = load_split(args.dataset_root, "val")
    test_data = load_split(args.dataset_root, "test")
    print(f"  train={len(train_data)}  val={len(val_data)}  test={len(test_data)}"
          f"  ({time.time()-t0:.1f}s)")

    # ── Create model ───────────────────────────────────────────────────────
    print("\n[2/5] Building model …")
    model = _create_model()
    optimizer = _create_optimizer(model)
    n_params = (
        sum(p.numel() for p in model.parameters())
        if USE_TORCH
        else sum(p.size for p in [model.W1, model.W2, model.W_out, model.b_out])
    )
    print(f"  GCN({IN_FEAT}->{HIDDEN1}->{HIDDEN2}->1)  params={n_params:,}")

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n[3/5] Training for {args.epochs} epochs …")
    history = []
    best_val_top1 = -1.0
    best_epoch = 0
    model_path = os.path.join(args.out, "gnn_rca_model" + _model_ext())

    # Store best numpy model weights separately
    best_np_weights = None

    for epoch in range(1, args.epochs + 1):
        loss, tr_acc = train_epoch(model, optimizer, train_data)
        val_metrics = evaluate(model, val_data)

        rec = {
            "epoch": epoch,
            "train_loss": round(loss, 4),
            "train_top1": round(tr_acc, 4),
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
                    "W1": model.W1.copy(), "W2": model.W2.copy(),
                    "W_out": model.W_out.copy(), "b_out": model.b_out.copy(),
                }

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d}/{args.epochs}  "
                f"loss={loss:.4f}  "
                f"tr@1={tr_acc:.3f}  "
                f"val@1={val_metrics['top1']:.3f}  "
                f"val@3={val_metrics['top3']:.3f}  "
                f"val_mrr={val_metrics['mrr']:.3f}"
            )

    print(f"  Best val top-1={best_val_top1:.3f} at epoch {best_epoch}")

    # Restore best weights for test evaluation
    if USE_TORCH:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    elif best_np_weights is not None:
        model.W1[:] = best_np_weights["W1"]
        model.W2[:] = best_np_weights["W2"]
        model.W_out[:] = best_np_weights["W_out"]
        model.b_out[:] = best_np_weights["b_out"]

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/5] Final evaluation …")
    train_metrics = evaluate(model, train_data)
    val_metrics = evaluate(model, val_data)
    test_metrics = evaluate(model, test_data)

    def _fmt(m):
        return f"top1={m['top1']:.3f}  top3={m['top3']:.3f}  mrr={m['mrr']:.3f}"

    print(f"  TRAIN ({train_metrics['n']} samples): {_fmt(train_metrics)}")
    print(f"  VAL   ({val_metrics['n']} samples): {_fmt(val_metrics)}")
    print(f"  TEST  ({test_metrics['n']} samples): {_fmt(test_metrics)}")

    metrics = {
        "backend": "torch" if USE_TORCH else "numpy_gcn",
        "torch_version": str(torch.__version__) if USE_TORCH else None,
        "architecture": f"GCN({IN_FEAT}-{HIDDEN1}-{HIDDEN2}-1)",
        "n_params": n_params,
        "epochs_trained": args.epochs,
        "best_val_epoch": best_epoch,
        "train": {k: round(v, 4) for k, v in train_metrics.items() if k != "n"},
        "val": {k: round(v, 4) for k, v in val_metrics.items() if k != "n"},
        "test": {k: round(v, 4) for k, v in test_metrics.items() if k != "n"},
        "training_history": history,
    }
    metrics_path = os.path.join(args.out, "gnn_rca_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics ->{metrics_path}")

    # ── Training curve ─────────────────────────────────────────────────────
    curve_path = os.path.join(args.out, "gnn_training_curve.png")
    plot_curve(history, curve_path)

    # ── Presentation inference ─────────────────────────────────────────────────────
    print(f"\n[5/5] Presentation inference on {args.demo_diagram} …")
    demo_result = infer_diagram(
        model, args.dataset_root, args.demo_split, args.demo_diagram
    )
    demo_result["model_path"] = model_path
    demo_result["test_metrics"] = {
        k: round(v, 4) for k, v in test_metrics.items() if k != "n"
    }
    demo_path = os.path.join(args.out, f"{args.demo_diagram}_gnn_rca_result.json")
    with open(demo_path, "w") as f:
        json.dump(demo_result, f, indent=2)

    correct_str = "CORRECT" if demo_result["is_correct"] else "WRONG"
    print(
        f"  Predicted: {demo_result['predicted_root_cause']}  "
        f"GT: {demo_result['ground_truth_root_cause']}  [{correct_str}]"
    )
    print(f"  GT rank={demo_result['ground_truth_rank']}  "
          f"MRR={demo_result['mrr']}")
    print(f"  Saved Presentation result ->{demo_path}")

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"  Model  : {model_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Presentation   : {demo_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

