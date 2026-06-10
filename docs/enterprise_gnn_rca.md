# Enterprise GNN Root Cause Analysis

## Why enterprise GNN RCA is needed

Single-diagram RCA assumes all devices live in one topology diagram.
In a real enterprise, an alert on a branch office router might cascade
to WAN routers, then to datacenter firewalls, then to application servers
— each in a completely different diagram drawn by a different team.

A single-diagram model cannot trace causality across those boundaries.
The enterprise GNN works on a *stitched* graph that merges multiple local
diagrams into one unified topology, with explicit cross-diagram edges
between them. This allows the model to rank root cause candidates across
the entire enterprise in one pass.

---

## Single-diagram GNN vs Enterprise GNN

| Property | Single-diagram GNN | Enterprise GNN |
|---|---|---|
| Input graph | One diagram's topology | Stitched multi-diagram enterprise graph |
| Nodes | ~17 per diagram | 18–50+ across 3–5 diagrams |
| Edges | Local only | Local + explicit cross-diagram |
| Alert scope | Single diagram | Cross-diagram propagation |
| Node features | 16-dim | 34-dim (adds diagram-type, shared-entity, bridge, reachability) |
| Architecture | GCN(16-64-32-1) | GCN(34-96-48-1) |
| Training data | infragraph_v2 per-diagram graphs | enterprise_graph_v1 stitched graphs |
| Backend | torch or numpy alternate path | torch required |

---

## Local diagram graph vs stitched enterprise graph

A **local diagram graph** is extracted from one network diagram PNG:
- Nodes: devices detected by YOLO in that diagram
- Edges: connections inferred from layout (layout_inference_v1)
- Scope: one team's view of one infrastructure layer

A **stitched enterprise graph** (`enterprise_graph.json`) merges several local diagrams:
- All nodes from all diagrams appear in a single `nodes` list
- Each node carries a `diagram_id` and `diagram_type`
- `edges` contains both local edges and `cross_diagram` edges
- `cross_diagram_edges` lists only the inter-diagram connections
- `diagram_clusters` groups nodes by their source diagram
- `shared_entities` records nodes that appear in multiple diagrams under different local aliases

The GNN operates on this unified graph, so message passing propagates
information across diagram boundaries during training and inference.

---

## Node feature vector (34 dimensions)

| Group | Dimensions | Features |
|---|---|---|
| Node type one-hot | 9 | router, switch, firewall, server, database, load_balancer, cloud_or_wan, service, unknown |
| Diagram type one-hot | 6 | branch_topology, wan_topology, datacenter_topology, app_db_topology, shared_services_topology, unknown |
| Alert features | 4 | has_alert, max_severity_score, earliest_alert_score, alert_count_norm |
| Topology status | 3 | is_impacted_node, is_shared_entity, is_cross_diagram_bridge |
| Graph structure | 7 | in_degree_norm, out_degree_norm, total_degree_norm, downstream_reach_norm, upstream_reach_norm, cross_diag_degree_norm, cluster_size_norm |
| RC-prior indicators | 5 | rc_prior_firewall, rc_prior_router, rc_prior_lb, rc_prior_database, rc_prior_service |

**Alert severity scores:** critical=1.0, high=0.8, warning=0.6, medium=0.5, low=0.3, info=0.1

**Reachability:** computed by BFS on the directed enterprise graph.
- `downstream_reach_norm` — number of nodes reachable *from* this node
- `upstream_reach_norm` — number of nodes *from which* this node is reachable

**Cross-diagram degree** — count of cross-diagram edges incident to this node, normalised by N−1.

**Cluster size** — number of nodes in the same diagram cluster, normalised by N−1.

---

## Model architecture

```
EnterpriseGCN(34 -> 96 -> 48 -> 1)

H1     = ReLU( A_norm @ X  @ W1 )    # shape (N, 96)
H2     = ReLU( A_norm @ H1 @ W2 )    # shape (N, 48)
logits =       A_norm @ H2 @ W3      # shape (N,  1) -> squeezed to (N,)
```

`A_norm` is the symmetric D^{-1/2} A D^{-1/2} normalised adjacency with
self-loops and bidirectional edges.

`BCEWithLogitsLoss` with `pos_weight = N − 1` counters the extreme class
imbalance (1 root-cause node vs N−1 normal nodes per scenario).

---

## Root-cause labels

```
label[i] = 1   if node_ids[i] == alerts["root_cause"]
label[i] = 0   otherwise
```

One positive node per scenario. The model learns to assign a high score
to the root-cause node relative to all other nodes in the same graph.

---

## Training

```bash
python scripts/train_enterprise_gnn_rca.py \
    --dataset-root ./datasets/enterprise_graph_v1 \
    --out ./outputs/enterprise_gnn_rca \
    --epochs 80 \
    --presentation-scenario enterprise_0000 \
    --presentation-split test
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--dataset-root` | `datasets/enterprise_graph_v1` | Dataset root |
| `--out` | `outputs/enterprise_gnn_rca` | Output directory |
| `--epochs` | 80 | Training epochs |
| `--lr` | 0.001 | Adam learning rate |
| `--weight-decay` | 0.0001 | Adam L2 regularisation |
| `--hidden1` | 96 | First hidden layer width |
| `--hidden2` | 48 | Second hidden layer width |
| `--presentation-scenario` | enterprise_0000 | Scenario ID for detailed output |
| `--presentation-split` | test | Split to find the presentation scenario in |
| `--seed` | 2026 | Random seed |

### Training loop

- Adam optimizer with per-scenario gradient steps
- Dropout 0.3 applied after each hidden GCN layer
- Best-epoch weights selected by highest val MRR (ties broken by val top-1)
- Progress printed every 10 epochs: loss, train top-1, val top-1, val top-3, val MRR

---

## Evaluation (scenario-level ranking)

For each scenario the model scores all N nodes. Nodes are ranked descending by score.
The ground-truth root-cause node's rank determines:

| Metric | Formula |
|---|---|
| Top-1 | root-cause node is rank 1 |
| Top-3 | root-cause node is in top 3 |
| MRR | 1 / rank of root-cause node |

Results are averaged across all scenarios in the split.

---

## Output files

| File | Description |
|---|---|
| `enterprise_gnn_model.pt` | PyTorch checkpoint (state_dict + feature vocab + arch metadata) |
| `enterprise_gnn_metrics.json` | Full training metrics, feature names, split results |
| `enterprise_gnn_training_curve.png` | Loss and ranking metric curves (requires matplotlib) |
| `{scenario_id}_enterprise_gnn_rca_result.json` | Detailed result for the presentation scenario |
| `{scenario_id}_enterprise_gnn_prediction.png` | Enterprise graph visualisation (requires matplotlib + networkx) |

### rca_result.json schema

```json
{
  "scenario_id": "enterprise_0000",
  "model_type": "Enterprise GCN RCA",
  "backend": "torch",
  "predicted_root_cause": "DC-FW-01",
  "ground_truth_root_cause": "DC-FW-01",
  "is_correct": true,
  "ground_truth_rank": 1,
  "root_cause_diagram": "datacenter_topology",
  "impacted_diagrams": ["datacenter_topology", "wan_topology"],
  "alert_count": 4,
  "node_count": 18,
  "edge_count": 17,
  "cross_diagram_edge_count": 2,
  "top_candidates": [
    {
      "rank": 1,
      "node_id": "DC-FW-01",
      "score": 0.931,
      "type": "firewall",
      "diagram_id": "datacenter_topology",
      "diagram_type": "datacenter_topology",
      "has_alert": true,
      "is_impacted_node": false,
      "is_shared_entity": true,
      "is_cross_diagram_bridge": true
    }
  ],
  "alerts": [...],
  "impact_paths": [...]
}
```

### Prediction visualisation

The PNG shows the enterprise graph with:
- **Bright green** — correctly predicted root cause
- **Cyan** — ground truth if not predicted top-1
- **Orange** — wrong top-1 prediction
- **Red** — alerting nodes
- **Yellow** — impacted nodes
- **Steel blue** — normal nodes
- **Dashed blue edges** — cross-diagram connections
- Node size scaled by GNN score

---

## Limitations

**Synthetic benchmark only.** The enterprise dataset is generated procedurally.
Node types, edge patterns, and alert propagation follow fixed templates.
Production use requires:

1. Real network diagrams with accurate device detection (YOLO fine-tuned on production topologies)
2. Validated cross-diagram stitch maps derived from actual CMDB or network management systems
3. Real-time alert enrichment from monitoring systems (SNMP traps, syslog, Prometheus, etc.)
4. Ongoing re-labelling as new incident root causes are confirmed by operators

**Small dataset.** `enterprise_graph_v1` has 16 train / 2 val / 2 test scenarios.
This is sufficient to demonstrate the architecture and verify learning, but the model
will require a much larger labelled dataset for production deployment.

**No line detection or OCR.** Cross-diagram edges in the training data are generated
from a stitch map, not extracted from diagram images. A production system would need
an automated pipeline to detect cables, labels, and CMDB references across diagram boundaries.

