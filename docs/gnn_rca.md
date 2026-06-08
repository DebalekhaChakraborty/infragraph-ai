# GNN-based Root Cause Ranking

`scripts/train_gnn_rca.py` is Stage 3 of InfraGraph AI. It trains a Graph
Convolutional Network (GCN) on the infragraph_v2 dataset to rank nodes by their
probability of being the root cause of an alert scenario.

---

## Why a GNN?

The heuristic scorer (Stage 2) ranks nodes by a weighted combination of
severity, timing, and downstream-reach.  This can confuse a **downstream
aggregation node** (e.g. `SW-CORE` receiving cascaded alerts from many
children) with the true **upstream origin** (e.g. `FW-01` that initiated the
fault).

A GCN can learn propagation direction from:

1. **Graph structure** — which nodes sit upstream vs. downstream in the
   directed topology.
2. **Temporal alert features** — which node alerted first and with what
   severity.
3. **Device-type signals** — firewalls and routers tend to be chokepoints
   whose failure cascades broadly.

The GCN is trained end-to-end with cross-entropy loss, directly optimising for
correct root-cause identification.

---

## How to run

```bash
# Default: infragraph_v2 dataset, 80 epochs, outputs to outputs/gnn_rca/
python scripts/train_gnn_rca.py

# Custom options
python scripts/train_gnn_rca.py \
    --dataset-root datasets/infragraph_v2 \
    --epochs       120 \
    --out          outputs/gnn_rca \
    --demo-diagram diagram_0373 \
    --demo-split   test \
    --seed         42
```

### Backend selection

The script auto-detects torch and falls back to a **pure-numpy GCN** if torch
is not installed in the active environment:

| Backend | How to trigger | Notes |
|---------|----------------|-------|
| `torch` | `pip install torch` in the active venv | Adam + autograd, GPU-ready |
| `numpy_gcn` | torch not installed | Manual forward/backward + Adam, CPU only |

---

## Node feature vector (16 dimensions)

| Dim | Feature | Description |
|-----|---------|-------------|
| 0–6 | one-hot device type | router / switch / firewall / server / database / load_balancer / cloud_or_wan |
| 7 | `has_alert` | 1 if node has any alert in the scenario |
| 8 | `max_severity` | max severity score / 4.0 (critical=4, major=3, minor=2, warning/info=1) |
| 9 | `earliest_time` | 1 / (1 + earliest_alert_time_min) — earlier = higher |
| 10 | `alert_count_norm` | alert count / max alert count in graph |
| 11 | `in_degree_norm` | in-degree / (n_nodes − 1) |
| 12 | `out_degree_norm` | out-degree / (n_nodes − 1) |
| 13 | `total_degree_norm` | (in + out) / (2 × (n_nodes − 1)) |
| 14 | `downstream_reach` | BFS reachable descendants / (n_nodes − 1) |
| 15 | `is_priority_device` | 1 if firewall or router |

---

## Model architecture

```
Input X: (n_nodes × 16)
Adjacency A_norm: (n_nodes × n_nodes)  [symmetrised + self-loop + D^{-1/2} A D^{-1/2}]

Layer 1:  H1 = ReLU( A_norm @ X  @ W1 )    W1: 16 x 64
Layer 2:  H2 = ReLU( A_norm @ H1 @ W2 )    W2: 64 x 32
Scoring:  s  = H2 @ W_out + b_out           W_out: 32 x 1, b_out: 1

scores shape: (n_nodes,)
loss = cross_entropy(scores, root_cause_node_index)
```

Total trainable parameters: **3,105**

The output layer produces a single scalar per node (not a fixed vocabulary
size), so the model handles graphs of any size without padding.

---

## Training details

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 80 |
| Optimiser | Adam (lr=1e-3, β₁=0.9, β₂=0.999) |
| Gradient clipping | max norm 1.0 |
| Batch size | 1 graph per step (variable node count) |
| Loss | Cross-entropy over all nodes |
| Best-model criterion | Highest val top-1 accuracy |

---

## Results on infragraph_v2

| Split | n | Top-1 | Top-3 | MRR |
|-------|---|-------|-------|-----|
| Train | 320 | 0.994 | 1.000 | 0.997 |
| Val   | 52  | 1.000 | 1.000 | 1.000 |
| Test  | 28  | 1.000 | 1.000 | 1.000 |

Best validation accuracy reached at **epoch 6**.

### Stage 2 vs Stage 3 on diagram_0373

| Stage | Predicted root cause | GT | Correct? |
|-------|---------------------|----|----------|
| Heuristic (Stage 2) | `SW-CORE` | `FW-01` | No |
| GNN (Stage 3) | `FW-01` | `FW-01` | Yes |

The heuristic scored SW-CORE higher because it received two correlated major
alerts.  The GNN learned that FW-01's earlier critical alert, combined with its
position as the upstream chokepoint (high out-degree, firewall priority flag),
is the stronger signal.

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/gnn_rca/gnn_rca_model.pt` | Best-epoch PyTorch state dict |
| `outputs/gnn_rca/gnn_rca_model.npz` | Best-epoch numpy weights (fallback backend) |
| `outputs/gnn_rca/gnn_rca_metrics.json` | Full metrics + per-epoch training history |
| `outputs/gnn_rca/gnn_training_curve.png` | Loss and accuracy curves (requires matplotlib) |
| `outputs/gnn_rca/<id>_gnn_rca_result.json` | Per-diagram inference result |

### `<id>_gnn_rca_result.json` schema

```json
{
  "diagram_id": "diagram_0373",
  "method": "gnn_rca",
  "backend": "torch",
  "predicted_root_cause": "FW-01",
  "ground_truth_root_cause": "FW-01",
  "is_correct": true,
  "ground_truth_rank": 1,
  "mrr": 1.0,
  "node_scores": {
    "FW-01": 30.775,
    "FW-02": 22.655,
    "SW-CORE": 11.435
  },
  "top_candidates": [
    {"rank": 1, "node": "FW-01", "score": 30.775, "type": "firewall"}
  ],
  "n_nodes": 17,
  "model_path": "outputs/gnn_rca/gnn_rca_model.pt",
  "test_metrics": {"top1": 1.0, "top3": 1.0, "mrr": 1.0}
}
```

---

## How this fits the pipeline

```
[Stage 2: Heuristic RCA]
    build_topology_rca_demo.py
        -> outputs/topology_demo/<id>_rca_result.json   (heuristic scores)

[Stage 3: GNN RCA]  <-- this script
    train_gnn_rca.py
        -> outputs/gnn_rca/gnn_rca_model.pt             (trained weights)
        -> outputs/gnn_rca/<id>_gnn_rca_result.json     (GNN scores)

[Stage 4: LLM explanation layer]
    gnn_rca_result.json + heuristic_rca_result.json
        -> Qwen/open-LLM prompt
        -> plain-language incident explanation
```

The `node_scores` dict from the GNN result feeds directly into the Qwen prompt
to ground the explanation in learned graph signals rather than rule-based
scores.
