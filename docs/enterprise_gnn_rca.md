# Enterprise GNN RCA Pipeline

Graph neural network that predicts the root-cause node in an enterprise
multi-diagram topology from observable event streams.

Output: ranked node list with confidence scores.
No remediation, no validation steps, no rollback, no ServiceNow output.
Those are generated later by the AI Resolution Agent.

---

## What it does

Given an enterprise scenario (events + stitched multi-diagram graph), the GNN
ranks every node across all diagrams by its estimated probability of being
the root cause.  A downstream Streamlit tab or AI Resolution Agent then takes
`predicted_root_cause` as input to generate remediation.

---

## Architecture

```
scenario_library/enterprise_gnn_rca/
  <case_id>/events.json          (observable events — no labels)
  <case_id>/labels.json          (ground truth — never used as input at inference)
  <case_id>/graph_ref.json
<referenced enterprise_graph.json + stitch_map.json>

        ▼  scripts/build_enterprise_gnn_dataset.py

data/rca/enterprise_gnn/
  graphs.pt              (list of graph dicts with torch tensors)
  graph_index.json       (per-case metadata + node_id_to_index maps)
  feature_columns.json   (34 feature names)
  label_stats.json       (split distribution, graph sizes)

        ▼  scripts/train_enterprise_gnn_rca.py

model_artifacts/enterprise_gnn_rca/
  enterprise_gnn_rca.pt          (best-checkpoint weights)
  enterprise_gnn_config.json     (model hyperparameters)
  feature_columns.json

reports/enterprise_gnn_rca/
  evaluation.json        (train/val/test top-1/top-3/MRR + baseline comparison)
  predictions_test.json  (per-case ranked predictions, test split)
  training_history.json  (loss + val metrics per epoch)

        ▼  scripts/predict_enterprise_gnn_rca.py --scenario-id <id>

assets/preloaded/enterprise_gnn_rca/
  <scenario_id>.json     (prediction result consumed by Streamlit / AI Agent)
```

---

## Input data

### Enterprise case inputs

| File | Purpose |
|------|---------|
| `events.json` | Observable alert events (node, severity, time_offset_min, diagram_id) |
| `labels.json` | Ground truth: `root_cause_node`, `root_cause_diagram`, `impacted_diagrams` — private |
| `graph_ref.json` | Paths to `enterprise_graph.json` and `stitch_map.json` |
| `enterprise_graph.json` | Nodes + edges + cross_diagram_edges for the full enterprise graph |
| `stitch_map.json` | Shared entity canonicalisation across diagrams (provenance only — see note below) |

> **stitch_map.json provenance note**: `stitch_map.json` is currently retained as
> provenance for how scenario diagrams were stitched into the enterprise graph.
> The Enterprise GNN consumes `enterprise_graph.json`, which is already pre-stitched
> and contains local plus cross-diagram edges.  Future versions may use
> `stitch_map.json` directly to reconstruct `enterprise_graph.json` at inference time.

### Public vs private

`events.json` is observable.  `labels.json` is never read as input at inference —
only for optional evaluation comparison.

---

## Graph construction

One PyG `Data` object per scenario:

- **Nodes**: all nodes from `enterprise_graph["nodes"]`
- **Edges**: all `edges` + `cross_diagram_edges` from the enterprise graph, bidirectional
- **Node features**: 54-dimensional vector (see below)
- **Label** (`y`): integer index of `labels["root_cause_node"]` in the node list
- **Cross-diagram degree**: deduplicated from both `cross_diagram_edges[]` and `edges[]` with `edge_scope=="cross_diagram"`, self-loops excluded

---

## Node features (54 total)

| Group | Features | Dim |
|-------|----------|-----|
| Node type one-hot | router, switch, firewall, server, database, load_balancer, cloud, wan, service, unknown | 10 |
| Diagram type one-hot | branch_topology, wan_topology, datacenter_topology, app_db_topology, shared_services_topology, unknown | 6 |
| Alert activity | is_shared_entity, is_alerted, alert_count_norm, max_severity_score, first_alert_time_norm, mean_alert_time_norm, min_time_rank_norm | 7 |
| Graph degree | degree_norm, in_degree_norm, out_degree_norm, cross_diagram_degree_norm | 4 |
| Centrality | pagerank, betweenness_centrality, closeness_centrality | 3 |
| Reachability | distance_to_alert_norm, reverse_reachability_norm | 2 |
| Role | source_like_score, sink_like_score | 2 |
| Alert type multi-hot | at_cpu, at_latency, at_packet_drop, at_link_errors, at_connection_timeout, at_auth_errors, at_backend_pool_unhealthy, at_user_timeout, at_route_flap, at_dependency_error, at_other | 11 |
| Temporal context | is_first_alerted_node, is_last_alerted_node, alert_sequence_position_norm | 3 |
| Propagation context | upstream_alert_count_norm, downstream_alert_count_norm, upstream_critical_count_norm, downstream_warning_count_norm, propagation_consistency_score | 5 |
| Compatibility | node_alert_compatibility_score | 1 |
| **Total** | | **54** |

**Severity scores**: critical=1.0, high=0.8, warning=0.6, medium=0.5, low=0.3, info=0.1

**Node type priority** (used in baseline): firewall=0.95, router=0.90, cloud/wan=0.85, load_balancer=0.80, switch=0.70, database=0.65, server=0.55, service=0.50

### Feature dimension mismatch

If the model was trained on 34-dim features and the current code produces 54-dim,
the predict script exits with:

```
[ERROR] Feature dimension mismatch: model expects 34 features, but dataset produces 54 features.
        Rebuild the dataset and retrain the Enterprise GNN RCA model:
          python scripts/build_enterprise_gnn_dataset.py
          python scripts/train_enterprise_gnn_rca.py
```

---

## Model

`EnterpriseRcaGNN` — 3-layer GraphSAGE:

```
Linear(54 → hidden)       input projection
SAGEConv(hidden → hidden)  x3 message-passing layers
Linear(hidden → 1)         per-node logit
softmax(logits)            → node probability scores
```

**Loss**: Negative log-likelihood of the root node: `-log_softmax(logits)[root_idx]`

**Best checkpoint**: Selected by highest MRR on the validation split.

---

## Dependencies

```
torch       >= 2.0
torch_geometric
networkx    >= 3.0
```

If `torch_geometric` is not installed, all scripts exit with:
```
[ERROR] torch_geometric is required for enterprise GNN RCA.
        Install with the correct torch/ROCm/CUDA wheel.
```

---

## Ephemeral AMD GPU setup

When the Jupyter environment resets (pip installs lost), run these two scripts
in order to restore the full RCA + GNN stack:

```bash
# Step 1 — restore ROCm torch + vERL/GRPO stack (skips if already present)
bash scripts/amd_rocm/bootstrap_grpo_env.sh

# Step 2 — restore RCA / GNN dependencies
bash scripts/amd_rocm/bootstrap_rca_gnn_env.sh
```

Verify the result:

```bash
python -c "
import torch, torch_geometric
print(torch.__version__, torch.cuda.is_available(), torch_geometric.__version__)
"
```

`bootstrap_rca_gnn_env.sh` is safe to re-run and will exit early with a clear
message if torch is missing rather than silently installing a mismatched wheel.

---

## Step-by-step usage

### 1. Build the graph dataset

```bash
python scripts/build_enterprise_gnn_dataset.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--scenario-library` | `scenario_library` | Path to scenario library root |
| `--out-dir` | `data/rca/enterprise_gnn` | Where to write graphs.pt + index |

### 2. Train

```bash
python scripts/train_enterprise_gnn_rca.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 80 | Training epochs |
| `--lr` | 0.001 | Adam learning rate |
| `--hidden-dim` | 64 | GNN hidden dimension |
| `--num-layers` | 3 | SAGEConv layers |
| `--dropout` | 0.2 | Dropout rate |
| `--top-k` | 3 | Top-k for evaluation |
| `--seed` | 42 | Random seed |
| `--eval-every` | 10 | Val evaluation frequency |

### 3. Predict one scenario

```bash
python scripts/predict_enterprise_gnn_rca.py --scenario-id enterprise_v3_0000
python scripts/predict_enterprise_gnn_rca.py --case-id ent_enterprise_v3_0000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--scenario-id` | — | scenario_id from manifest |
| `--case-id` | — | case_id from manifest |
| `--top-k` | 3 | Number of ranked candidates |
| `--with-eval` | off | Include ground-truth comparison (reads labels.json) |
| `--out` | _(auto)_ | Override output directory |
| `--cluster-file` | (none) | Path to event correlation cluster file — enriches output with `cluster_id`, `cluster_score`, `correlation_reasons`, `causal_evidence` |

**Output routing:**

- Default (no `--with-eval`): writes demo-safe output to `assets/preloaded/enterprise_gnn_rca/`.
- With `--with-eval` (and no `--out`): writes to `reports/enterprise_gnn_rca/manual_eval/`.
- Do not use `--with-eval` when generating preloaded demo assets.

---

## Evaluation metrics

All metrics are **case-level**:

| Metric | Definition |
|--------|-----------|
| **Top-1 accuracy** | Fraction of scenarios where #1 ranked node == ground-truth root |
| **Top-3 accuracy** | Fraction where ground-truth root appears in top 3 |
| **MRR** | Mean Reciprocal Rank = mean(1 / rank_of_true_root) |
| **baseline_topology_score** | Heuristic: alert severity + node type priority + cross-diagram connectivity |

---

## Output format (`<scenario_id>.json`)

```json
{
  "scenario_id":          "enterprise_v3_0000",
  "case_id":              "ent_enterprise_v3_0000",
  "mode":                 "enterprise_gnn_rca",
  "rca_source":           "Enterprise GNN RCA",
  "predicted_root_cause": "DC-FW-01",
  "root_cause_diagram":   "datacenter_topology",
  "confidence":           0.9999,
  "top_candidates": [
    {
      "rank": 1,
      "node_id": "DC-FW-01",
      "diagram_id": "datacenter_topology",
      "node_observed_in_diagrams": ["datacenter_topology"],
      "node_type": "firewall",
      "score": 0.9999,
      "evidence": ["alert_count=1.0", "cross_diagram_degree=0.0",
                   "distance_to_alert=0.0", "shared_entity=True"]
    }
  ],
  "impacted_diagrams": ["branch_topology", "datacenter_topology", "app_db_topology", "shared_services_topology"],
  "alert_count": 7,
  "cluster_id":           "CLU-ent_enterprise_v3_0000-001",
  "cluster_score":        0.8273,
  "correlation_reasons":  ["7 event(s) span 18 min (t=0..18)", "..."],
  "causal_evidence":      [...]
}
```

With `--with-eval`, an `"evaluation"` block is appended:

```json
  "evaluation": {
    "ground_truth_node": "DC-FW-01",
    "correct_top1": true,
    "correct_top_k": true,
    "reciprocal_rank": 1.0,
    "rank": 1
  }
```

Each `top_candidates` entry includes `"node_observed_in_diagrams"` — the list of
diagrams where this node was actually observed in events (handles shared entities
that appear in multiple diagrams).  `"diagram_id"` in the result uses the
earliest-event diagram, not the canonical node diagram.

The output never contains: `remediation`, `recommended_actions`, `remediation_steps`,
`resolution_steps`, `rollback_steps`, `validation_steps`, `servicenow_incident_summary`.
Remediation is generated later by the AI Resolution Agent, not pre-computed here.

`"rca_source": "Enterprise GNN RCA"` is applied only when a trained `.pt` model
produces the prediction.  When torch/GNN is unavailable and a scenario-grounded
heuristic is used instead, `rca_source` is `"Scenario-grounded RCA simulation"`.

Default output (no `--with-eval`) contains **no evaluation fields** — safe for
production and Streamlit use.  Use `scripts/validate_rca_outputs.py` to verify.

---

## Generating clean RCA assets

All four demo scenarios must produce real Enterprise GNN outputs
(`rca_source = "Enterprise GNN RCA"`).  Scenario-grounded simulation is not the
default and must not be used for the final demo unless explicitly labelled.

Requirements before running:
- `model_artifacts/enterprise_gnn_rca/enterprise_gnn_rca.pt`
- `model_artifacts/enterprise_gnn_rca/enterprise_gnn_config.json`

If these are missing the script exits with a clear error and train command.

```bash
python scripts/generate_enterprise_rca_demo_assets.py
python scripts/validate_rca_outputs.py --verbose
```

For each scenario the script runs in order:

```bash
python scripts/build_event_correlation_clusters.py --case-id ent_<scenario_id>
python scripts/predict_enterprise_gnn_rca.py \
    --scenario-id <scenario_id> \
    --cluster-file assets/preloaded/event_correlation/<scenario_id>.json
```

Additional flags:

| Flag | Description |
|------|-------------|
| `--scenarios <id> ...` | Process specific scenario IDs only |
| `--dry-run` | Print commands without executing |
| `--allow-simulation-fallback` | Allow simulation if GNN fails (clearly labelled, not default) |

```
outputs written:
  assets/preloaded/event_correlation/enterprise_v3_000{0,2,3,4}.json
  assets/preloaded/enterprise_gnn_rca/enterprise_v3_000{0,2,3,4}.json
```

All four RCA files must contain `"rca_source": "Enterprise GNN RCA"`.
Run `validate_rca_outputs.py --verbose` to confirm.

---

## Consumed by

- **Streamlit demo** reads `assets/preloaded/enterprise_gnn_rca/<scenario_id>.json`.
- **AI Resolution Agent** receives `predicted_root_cause` + `root_cause_diagram` +
  `top_candidates` as context to generate remediation.

---

## Integrity constraints

- `labels.json` is never read at inference.
- The `"remediation"` key is never written to any preloaded output file.
  Remediation is generated by the AI Resolution Agent in a separate pipeline step.
- Model checkpoints and generated datasets are not committed to Git.
- `rca_source: "Enterprise GNN RCA"` only when a trained `.pt` model produces the result.
- `rca_source: "Scenario-grounded RCA simulation"` when torch/GNN is unavailable.

---

## Directory layout

```
src/rca_ml/
  enterprise_gnn_dataset.py    graph construction, feature engineering
  enterprise_gnn_model.py      GraphSAGE model, save/load
  enterprise_gnn_inference.py  predict_one(), evaluate_dataset(), heuristic baseline

src/event_correlation/           event correlation & causal evidence package

scripts/
  build_enterprise_gnn_dataset.py
  train_enterprise_gnn_rca.py
  predict_enterprise_gnn_rca.py
  generate_enterprise_rca_demo_assets.py  ← regenerate all 4 clean preloaded outputs
  validate_rca_outputs.py

data/rca/enterprise_gnn/             (gitignored)
model_artifacts/enterprise_gnn_rca/  (gitignored)
reports/enterprise_gnn_rca/          (gitignored)
assets/preloaded/enterprise_gnn_rca/ (committed — clean, no remediation/eval keys)
assets/preloaded/event_correlation/  (committed — cluster output per scenario)
```
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
| Training data | infragraph_v2 per-diagram graphs | infragraph_v1_enterprise_graph stitched graphs |
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
    --dataset-root ./datasets/infragraph_v1/enterprise_graph \
    --out ./outputs/enterprise_gnn_rca \
    --epochs 80 \
    --presentation-scenario enterprise_0000 \
    --presentation-split test
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--dataset-root` | `datasets/infragraph_v1/enterprise_graph` | Dataset root |
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

**Small dataset.** `infragraph_v1_enterprise_graph` has 16 train / 2 val / 2 test scenarios.
This is sufficient to demonstrate the architecture and verify learning, but the model
will require a much larger labelled dataset for production deployment.

**No line detection or OCR.** Cross-diagram edges in the training data are generated
from a stitch map, not extracted from diagram images. A production system would need
an automated pipeline to detect cables, labels, and CMDB references across diagram boundaries.


