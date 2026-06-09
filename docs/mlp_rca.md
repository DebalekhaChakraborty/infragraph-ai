# MLP RCA Node Ranker

`scripts/train_mlp_rca.py` trains a supervised MLP that scores each node
independently and ranks them to identify the root cause of a network incident.

This model sits between the rule-based heuristic scorer and the topology-aware
GNN in the InfraGraph AI pipeline.  It answers the question: **how much of the
GNN's accuracy comes from graph structure vs. node features alone?**

---

## Why this model exists

The InfraGraph AI pipeline has three RCA approaches:

| Approach | Uses graph structure? | Learned? |
|----------|-----------------------|----------|
| Heuristic scorer (Stage 2) | Yes (downstream reach, degree) | No |
| **MLP node ranker** | No | Yes |
| GNN (Stage 3) | Yes (message passing) | Yes |

The MLP is a controlled ablation.  Both the MLP and GNN use the same 23-dim
node feature vector.  The difference is that the GNN additionally aggregates
neighbour information through two rounds of graph convolution.

If the MLP achieves similar accuracy to the GNN, it suggests the node features
alone are sufficient and the graph structure adds limited signal on this dataset.
If the GNN outperforms the MLP, it demonstrates the value of topology-aware
message passing.

---

## How it differs from the heuristic scorer

The heuristic scorer uses hand-crafted rules:

```
score = severity_weight * 2
      + (1 / (1 + time_offset)) * 10
      + (downstream_count / total_nodes) * 3
      + device_type_bonus
```

The MLP learns these weights and interactions end-to-end from 400 labelled
training scenarios.  It can discover non-linear combinations the hand-crafted
formula cannot express.

---

## How it differs from GNN RCA

The GNN applies two rounds of neighbourhood aggregation:

```
H1 = ReLU(A_norm @ X @ W1)
H2 = ReLU(A_norm @ H1 @ W2)
```

Each node's embedding is updated by its neighbours' features.  The MLP skips
this step entirely — it maps each node's own feature vector to a score without
any cross-node communication:

```
H1 = ReLU(X @ W1 + b1)
H2 = ReLU(H1 @ W2 + b2)
score = H2 @ W3 + b3
```

---

## Node feature vector (23 dimensions)

| Dims | Feature | Description |
|------|---------|-------------|
| 0–6 | one-hot device type | router / switch / firewall / server / database / load_balancer / cloud_or_wan |
| 7 | `has_alert` | 1 if node has any alert |
| 8 | `max_severity_score` | max severity / 4.0 (critical=4, major=3, minor=2, warning/info=1) |
| 9 | `earliest_time_score` | 1 / (1 + earliest_alert_time_min) |
| 10 | `alert_count_norm` | alert count / max alert count in graph |
| 11 | `in_degree_norm` | in-degree / (n_nodes − 1) |
| 12 | `out_degree_norm` | out-degree / (n_nodes − 1) |
| 13 | `total_degree_norm` | (in + out) / (2 × (n_nodes − 1)) |
| 14 | `downstream_reach_norm` | BFS-reachable descendants / (n_nodes − 1) |
| 15 | `upstream_reach_norm` | BFS-reachable ancestors / (n_nodes − 1) |
| 16–22 | boolean device-type flags | `is_firewall`, `is_router`, `is_switch`, `is_server`, `is_database`, `is_load_balancer`, `is_cloud_or_wan` |

The boolean type flags (dims 16–22) duplicate the one-hot encoding but are
retained as separate features so the model can learn different weights for type
identity in different feature contexts.

---

## Training

```
Loss: BCEWithLogitsLoss with pos_weight ≈ 9.3
      (each graph has 1 root-cause node out of ~10 nodes on average)
Optimizer: Adam (lr=1e-3)
Gradient clipping: max norm 1.0
Epochs: 80
Batch: full dataset (3,283 training nodes, all graphs pooled)
```

All training nodes from all graphs are pooled into a single flat tensor.  The
MLP is trained as a standard binary classifier, then evaluated at graph level
by ranking all nodes within each graph and measuring top-1/top-3/MRR.

---

## How to train

```bash
# Default: infragraph_v2, 80 epochs, outputs to outputs/mlp_rca/
python scripts/train_mlp_rca.py

# Full options
python scripts/train_mlp_rca.py \
    --dataset-root datasets/infragraph_v2 \
    --out          outputs/mlp_rca \
    --epochs       80 \
    --demo-diagram diagram_0373 \
    --demo-split   test \
    --seed         42
```

---

## Results on infragraph_v2

| Split | n | Top-1 | Top-3 | MRR |
|-------|---|-------|-------|-----|
| Train | 320 | 0.997 | 1.000 | 0.998 |
| Val   | 52  | 1.000 | 1.000 | 1.000 |
| Test  | 28  | **1.000** | 1.000 | 1.000 |

Best val accuracy reached at epoch 56 (vs. epoch 6 for the GNN).

### Comparison: MLP vs GNN on infragraph_v2 test set

| Model | Top-1 | Top-3 | MRR | Convergence |
|-------|-------|-------|-----|-------------|
| Heuristic | ~60% | ~90% | ~0.75 | N/A |
| **MLP (no graph)** | 100% | 100% | 1.000 | epoch 56 |
| **GNN (graph MP)** | 100% | 100% | 1.000 | epoch 6 |

Both learned models reach 100% on test.  The GNN converges ~9x faster (6 vs 56
epochs) because graph message-passing propagates supervision signal more
efficiently across connected nodes.  On synthetic data with clean topology
signals, the node features alone are sufficient for the MLP to eventually learn
to rank correctly — but the GNN does it with far fewer parameter updates.

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/mlp_rca/mlp_rca_model.pt` | Best-epoch PyTorch state dict |
| `outputs/mlp_rca/mlp_rca_model.npz` | Best-epoch numpy weights (fallback) |
| `outputs/mlp_rca/mlp_rca_metrics.json` | Full metrics + training history |
| `outputs/mlp_rca/mlp_training_curve.png` | Loss and accuracy curves |
| `outputs/mlp_rca/<id>_mlp_rca_result.json` | Per-diagram inference result |

### `<id>_mlp_rca_result.json` schema

```json
{
  "diagram_id": "diagram_0373",
  "backend": "torch",
  "model_type": "MLP node ranker",
  "predicted_root_cause": "FW-01",
  "ground_truth_root_cause": "FW-01",
  "is_correct": true,
  "ground_truth_rank": 1,
  "mrr": 1.0,
  "top_candidates": [
    {
      "rank": 1,
      "node_id": "FW-01",
      "score": 3.4521,
      "type": "firewall",
      "has_alert": true,
      "severity": "critical"
    }
  ],
  "n_nodes": 17
}
```

---

## How to interpret results

A high MLP top-1 accuracy means the node-level features are discriminative
enough to identify root causes without graph structure.  On synthetic data this
is expected because the alert simulation directly sets has_alert and severity
based on the root cause scenario.

In a real deployment, partial observability (missing alerts, noisy telemetry)
would likely widen the gap between MLP and GNN — the GNN can use graph
neighbourhood to infer that a silent upstream node is the probable origin.
