# Topology RCA ML Pipeline

Classical ML pipeline that predicts the root-cause node in a network topology
diagram from observable event streams.  No remediation content is produced.

---

## What it does

Given a topology case (events + local graph), the model ranks every node by
its estimated probability of being the root cause.  Output is a ranked list of
node IDs with confidence scores — nothing else.

---

## Architecture

```
scenario_library/topology_rca/
  <case_id>/events.json        (observable events only)
  <case_id>/labels.json        (ground truth — never used as input at inference)
  <case_id>/graph_ref.json

        ▼  scripts/build_topology_rca_dataset.py

data/rca/topology/
  topology_node_dataset.csv    (one row per node per case, 28 features + label)
  topology_case_index.json

        ▼  scripts/train_topology_rca_model.py

model_artifacts/topology_rca/
  topology_rca_model.joblib
  topology_rca_feature_columns.json
  topology_rca_label_encoder.json

reports/topology_rca/
  eval_metrics.json
  per_case_predictions.json
  feature_importance.json

        ▼  scripts/predict_topology_rca.py --case-id <id>

assets/preloaded/topology_rca_results/<case_id>.json
```

---

## Features (28 per node)

| Group | Features |
|-------|----------|
| Node identity | `node_type`, `zone`, `is_shared_entity` |
| Alert activity | `is_alerted`, `alert_count`, `max_severity_score`, `first_alert_time`, `mean_alert_time`, `min_time_rank` |
| Graph degree | `in_degree`, `out_degree`, `total_degree`, `is_source_node`, `is_sink_node` |
| Centrality | `pagerank`, `betweenness_centrality`, `closeness_centrality` |
| Reachability | `min_undirected_distance_to_alert`, `mean_undirected_distance_to_alert`, `directed_reachability_to_alert_count`, `reverse_reachability_from_alert_count` |
| Composite | `node_type_priority_score`, `severity_weighted_alert_score` |

**Severity scores**: critical=1.0, high=0.8, warning=0.6, medium=0.5, low=0.3, info=0.1

**Node type priority**: firewall=0.95, router=0.90, cloud/wan=0.85, load_balancer=0.80, switch=0.70, database=0.65, server=0.55, service=0.50

---

## Model

`RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)`

`class_weight="balanced"` compensates for heavy label imbalance (~1 positive per 9 nodes).

Preprocessing: `OneHotEncoder` on categoricals (`node_type`, `zone`, `diagram_id`) +
`StandardScaler` on numerics — all in a single `sklearn.Pipeline`.

---

## Step-by-step usage

### 1. Build the feature dataset

```bash
cd infragraph-ai
python scripts/build_topology_rca_dataset.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--scenario-library` | `scenario_library` | Path to scenario library root |
| `--out-dir` | `data/rca/topology` | Output directory |
| `--include-out-of-scope` | off | Include cases where root cause is outside the diagram |

### 2. Train

```bash
python scripts/train_topology_rca_model.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `random_forest` | `random_forest` or `logistic_regression` |
| `--train-splits` | `train` | Splits used for fitting |
| `--eval-splits` | `test val` | Splits used for case-level evaluation |
| `--data-dir` | `data/rca/topology` | Where to read the CSV |
| `--model-dir` | `model_artifacts/topology_rca` | Where to write the model |
| `--reports-dir` | `reports/topology_rca` | Where to write eval reports |

### 3. Predict one case

```bash
python scripts/predict_topology_rca.py --case-id topo_enterprise_v3_0000_datacenter_topology
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--case-id` | (required) | Case ID from `scenario_library/manifest.json` |
| `--top-k` | `3` | How many ranked candidates to include |
| `--no-eval` | off | Skip ground-truth comparison |

---

## Evaluation metrics

Evaluation is **case-level**, not row-level:

| Metric | Definition |
|--------|-----------|
| **Top-1 accuracy** | Fraction of cases where the #1 ranked node is the ground-truth root |
| **Top-3 accuracy** | Fraction of cases where the ground-truth root appears in the top 3 |
| **MRR** | Mean Reciprocal Rank — `mean(1/rank_of_true_root)` |

These are reported per split (train / val / test) in `eval_metrics.json`.

---

## Output format (`<case_id>.json`)

```json
{
  "case_id": "topo_enterprise_v3_0000_datacenter_topology",
  "model":   "topology_rca_random_forest_v1",
  "top_k":   3,
  "predicted_root": "router_core",
  "top_candidates": [
    {"rank": 1, "node_id": "router_core", "score": 0.74, "node_type": "router", "zone": "core", "is_alerted": true},
    {"rank": 2, "node_id": "fw_main",     "score": 0.18, "node_type": "firewall", "zone": "edge", "is_alerted": false},
    {"rank": 3, "node_id": "sw_dist_01",  "score": 0.05, "node_type": "switch", "zone": "distribution", "is_alerted": true}
  ],
  "total_nodes": 9,
  "alert_count": 4,
  "evaluation": {
    "ground_truth_node": "router_core",
    "correct_top1": true,
    "correct_top_k": true,
    "reciprocal_rank": 1.0,
    "rank": 1
  }
}
```

The output contains **no** `recommended_actions`, `remediation_steps`,
`resolution_steps`, `rollback_steps`, `validation_steps`, or `servicenow_ticket` fields.

---

## Integrity constraints

- Labels (`labels.json`) are **never read during inference** — only for optional eval comparison.
- No remediation content is read from or written to any file in this pipeline.
- Out-of-scope cases (`root_cause_in_scope=false`) have `label_is_root=0` for all nodes and are excluded from training by default.
- Model checkpoints and generated datasets are not committed to Git.

---

## Directory layout

```
src/rca_ml/
  __init__.py
  features.py           28-feature builder (per node per case)
  topology_dataset.py   scenario_library loader + DataFrame builder
  topology_model.py     RandomForest pipeline, evaluation, serialisation

scripts/
  build_topology_rca_dataset.py
  train_topology_rca_model.py
  predict_topology_rca.py

data/rca/topology/          (gitignored — generated)
model_artifacts/topology_rca/  (gitignored — generated)
reports/topology_rca/          (gitignored — generated)
assets/preloaded/topology_rca_results/  (committed for Streamlit demo)
```
