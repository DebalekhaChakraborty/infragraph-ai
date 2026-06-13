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

## Features (47 per node)

| Group | Count | Features |
|-------|-------|----------|
| Node identity | 3 | `node_type`, `zone`, `is_shared_entity` |
| Alert activity | 6 | `is_alerted`, `alert_count`, `max_severity_score`, `first_alert_time`, `mean_alert_time`, `min_time_rank` |
| Graph degree | 5 | `in_degree`, `out_degree`, `total_degree`, `is_source_node`, `is_sink_node` |
| Centrality | 3 | `pagerank`, `betweenness_centrality`, `closeness_centrality` |
| Reachability | 4 | `min_undirected_distance_to_alert`, `mean_undirected_distance_to_alert`, `directed_reachability_to_alert_count`, `reverse_reachability_from_alert_count` |
| Composite | 2 | `node_type_priority_score`, `severity_weighted_alert_score` |
| Alert type counts | 9 | `alert_type_count_cpu`, `_latency`, `_packet_drop`, `_link_errors`, `_connection_timeout`, `_auth_errors`, `_backend_pool_unhealthy`, `_user_timeout`, `_other` |
| Compatibility | 1 | `node_alert_compatibility_score` |
| Temporal context | 5 | `is_first_alerted_node`, `is_last_alerted_node`, `alert_time_span`, `alert_burst_score`, `alert_sequence_position_norm` |
| Propagation context | 8 | `upstream_alert_count`, `downstream_alert_count`, `upstream_critical_alert_count`, `downstream_warning_alert_count`, `downstream_after_candidate_count`, `alerts_reachable_downstream_after_candidate`, `alerts_reachable_upstream_before_candidate`, `propagation_consistency_score` |

**Severity scores**: critical=1.0, high=0.8, warning=0.6, medium=0.5, low=0.3, info=0.1

**Node type priority**: firewall=0.95, router=0.90, cloud/wan=0.85, load_balancer=0.80, switch=0.70, database=0.65, server=0.55, service=0.50

**Alert type buckets** map raw `alert_type` strings to 9 canonical buckets:

| Bucket | Matched patterns |
|--------|-----------------|
| `cpu` | cpu_spike, cpu_high, high_cpu |
| `latency` | latency, api_latency, high_latency, slow_response |
| `packet_drop` | packet_drop, packet_loss |
| `link_errors` | link_errors, link_down, interface_errors, interface_flap, port_flap |
| `connection_timeout` | connection_timeout, connection_refused, connection_drop |
| `auth_errors` | auth_errors, auth_failure, authentication_failed |
| `backend_pool_unhealthy` | backend_pool_unhealthy, unhealthy_backend, pool_error |
| `user_timeout` | user_timeout, session_timeout, request_timeout |
| `other` | anything not matched above |

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
| `--with-eval` | off | Include ground-truth comparison (reads labels.json) |
| `--hybrid-score` | off | Combine model probability with alert-context score |
| `--cluster-file` | (none) | Path to event correlation cluster file — enriches output with `cluster_id`, `cluster_score`, `correlation_reasons`, `causal_evidence` |

**Output routing:**

- Default (no `--with-eval`): writes demo-safe output to `assets/preloaded/topology_rca_results/`.
- With `--with-eval` (and no `--out-dir`): writes to `reports/topology_rca/manual_eval/`.
- Do not use `--with-eval` when generating preloaded demo assets.

**Hybrid scoring** (`--hybrid-score`):

```
final_score = 0.75 × model_probability + 0.25 × alert_context_score
```

`alert_context_score` is a normalised combination of `propagation_consistency_score`,
`node_alert_compatibility_score`, and temporal position.  The `scoring_mode` field in
the output shows `"ml_only"` or `"hybrid_alert_context"`.

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
  "scoring_mode": "ml_only"
}
```

With `--with-eval`, an `"evaluation"` block is appended:

```json
  "evaluation": {
    "ground_truth_node": "router_core",
    "correct_top1": true,
    "correct_top_k": true,
    "reciprocal_rank": 1.0,
    "rank": 1
  }
```

With `--cluster-file`, four additional fields are appended:

```json
  "cluster_id":          "CLU-topo_enterprise_v3_0000_datacenter_topology-001",
  "cluster_score":       0.7950,
  "correlation_reasons": ["2 event(s) span 3 min (t=0..3)", "..."],
  "causal_evidence":     [{"evidence_id": "CE-001", "stage": "temporal_correlation", ...}]
```

These are allowed in `assets/preloaded/` and pass `validate_rca_outputs.py`.

The output contains **no** `recommended_actions`, `remediation_steps`,
`resolution_steps`, `rollback_steps`, `validation_steps`, or `servicenow_ticket` fields.

Default output (no `--with-eval`) contains **no evaluation fields** — safe for
production and Streamlit use.  Use `scripts/validate_rca_outputs.py` to verify.

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
  features.py           47-feature builder (per node per case)
  topology_dataset.py   scenario_library loader + DataFrame builder
  topology_model.py     RandomForest pipeline, evaluation, serialisation

src/event_correlation/
  __init__.py, schema.py, correlator.py, evidence.py, io.py
  (pre-RCA event clustering — see docs/event_correlation_and_causal_evidence.md)

scripts/
  build_topology_rca_dataset.py
  train_topology_rca_model.py
  predict_topology_rca.py
  build_event_correlation_clusters.py
  validate_rca_outputs.py      demo-safety checker for assets/preloaded/

data/rca/topology/          (gitignored — generated)
model_artifacts/topology_rca/  (gitignored — generated)
reports/topology_rca/          (gitignored — generated)
assets/preloaded/topology_rca_results/  (committed for Streamlit demo)
assets/preloaded/event_correlation/     (committed — cluster files)
```
