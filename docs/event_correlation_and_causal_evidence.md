# Event Correlation & Causal Evidence

Pre-RCA layer that groups observable alert events into coherent clusters
and annotates each cluster with a deterministic causal evidence trail.

Event correlation runs **before** RCA scoring — its output can optionally
enrich topology RCA and enterprise GNN RCA results with cluster context.

---

## What it does

Given a set of observable alert events for a case, the correlator:

1. **Deduplicates** events by `(node, alert_type, severity, time_offset_min, diagram_id)`.
2. **Forms temporal windows** — events within 60 minutes of a window's first event form one cluster.
3. **Scores each cluster** across five dimensions (see below).
4. **Assigns a correlation role** to each event within its cluster.
5. **Builds a causal evidence trail** of deterministic, hypothesis-labelled items.

No root-cause labels, remediation steps, or evaluation fields are produced.

---

## Scoring dimensions

| Dimension | Weight (topology) | Weight (enterprise) | Description |
|-----------|:-----------------:|:-------------------:|-------------|
| `temporal` | 0.30 | 0.25 | Time span of events within the cluster |
| `topology` | 0.35 | 0.30 | Shortest-path proximity between alerted nodes |
| `alert_type_seq` | 0.20 | 0.20 | Match against known propagation patterns |
| `source_peer` | 0.15 | 0.10 | Events sharing the same diagram |
| `cross_diagram` | 0.00 | 0.15 | Events spanning connected diagram boundaries |

**Temporal score**:

| Time span | Score |
|-----------|-------|
| ≤ 15 min | 1.00 |
| ≤ 30 min | 0.85 |
| ≤ 60 min | 0.65 |
| > 60 min | 0.40 |

**Topology score** (when a local graph or enterprise graph is available):

| Shortest path | Score |
|---------------|-------|
| 1 hop | 1.00 |
| 2 hops | 0.85 |
| 3 hops | 0.65 |
| 4 hops | 0.45 |
| > 4 hops | 0.00 |

**Composite cluster score**:

```
cluster_score = sum(weight_k × dim_k for each dimension k)
```

---

## Propagation patterns

The following alert-type sequences are treated as known propagation chains
when scoring the `alert_type_seq` dimension:

| Pattern |
|---------|
| `link_errors → packet_drop` |
| `link_errors → packet_drop → connection_timeout` |
| `link_errors → packet_drop → latency` |
| `packet_drop → connection_timeout` |
| `packet_drop → latency` |
| `packet_drop → connection_timeout → user_timeout` |
| `cpu → latency` |
| `cpu → latency → user_timeout` |
| `cpu → connection_timeout` |
| `auth_errors → connection_timeout` |
| `auth_errors → user_timeout` |
| `backend_pool_unhealthy → latency` |
| `backend_pool_unhealthy → connection_timeout` |
| `backend_pool_unhealthy → user_timeout` |
| `latency → user_timeout` |
| `connection_timeout → user_timeout` |
| `route_flap → packet_drop` |
| `route_flap → link_errors` |
| `dependency_error → latency` |
| `dependency_error → connection_timeout` |

---

## Correlation roles

Each event within a cluster is assigned one `correlation_role`:

| Role | Meaning |
|------|---------|
| `cluster_seed` | First event (lowest `time_offset_min`) — candidate fault origin |
| `propagation_signal` | Alert type continues a known propagation chain from the seed |
| `peer_signal` | Same diagram as seed, but not part of a propagation chain |
| `noise_candidate` | Does not clearly connect to others via pattern or peer context |

---

## Causal evidence trail

Each cluster contains a `causal_evidence` list.  Every item covers one
evidence stage and is grounded exclusively in observable event data:

| Stage | Description |
|-------|-------------|
| `temporal_correlation` | Time span and co-occurrence analysis |
| `alert_sequence` | Best-matching propagation pattern |
| `topology_proximity` | Shortest-path reasoning between alerted nodes |
| `cross_diagram_correlation` | Multi-diagram spanning evidence (enterprise only) |
| `propagation_hypothesis` | Synthesis hypothesis from seed to downstream nodes |

Evidence items are tagged with `evidence_id` (e.g. `CE-001`) so the AI
remediation prompt can reference them.

---

## Step-by-step usage

### 1. Build clusters for one case

```bash
python scripts/build_event_correlation_clusters.py \
    --case-id ent_enterprise_v3_0000
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--case-id` | (required) | Topology or enterprise case ID |
| `--mode` | (inferred) | `topology_rca` or `enterprise_gnn_rca` |
| `--scenario-library` | `scenario_library` | Path to scenario library root |
| `--out-dir` | (see below) | Output directory override |
| `--with-eval` | off | Route output to `reports/` instead of `assets/preloaded/` |

**Output routing**:

| Flags | Output location |
|-------|----------------|
| Default | `assets/preloaded/event_correlation/<stem>.json` |
| `--with-eval` | `reports/event_correlation/manual_eval/<stem>.json` |

`<stem>` is the `scenario_id` for enterprise cases or the `case_id` otherwise.

### 2. Enrich RCA output with cluster data

Pass `--cluster-file` to either predict script:

```bash
# Topology RCA — enrich with cluster
python scripts/predict_topology_rca.py \
    --case-id topo_enterprise_v3_0000_datacenter_topology \
    --cluster-file assets/preloaded/event_correlation/enterprise_v3_0000.json

# Enterprise GNN RCA — enrich with cluster
python scripts/predict_enterprise_gnn_rca.py \
    --scenario-id enterprise_v3_0000 \
    --cluster-file assets/preloaded/event_correlation/enterprise_v3_0000.json
```

When a cluster file is provided, the RCA output gains four additional fields:

```json
{
  "cluster_id":           "CLU-ent_enterprise_v3_0000-001",
  "cluster_score":        0.7812,
  "correlation_reasons":  ["5 event(s) span 24 min (t=0..24)", "Propagation pattern detected: link_errors -> packet_drop"],
  "causal_evidence":      [...]
}
```

These fields are **allowed** in `assets/preloaded/` outputs — they are
observable correlation data, not root-cause labels or remediation content.

### 3. Validate outputs

The default scan in `validate_rca_outputs.py` now includes the event
correlation directory:

```bash
python scripts/validate_rca_outputs.py
```

Cluster files pass validation because they contain no forbidden keys
(remediation or evaluation leakage fields).

---

## Output format (`<stem>.json`)

```json
{
  "case_id":       "ent_enterprise_v3_0000",
  "scenario_id":   "enterprise_v3_0000",
  "mode":          "enterprise_gnn_rca",
  "cluster_count": 1,
  "clusters": [
    {
      "cluster_id":    "CLU-ent_enterprise_v3_0000-001",
      "case_id":       "ent_enterprise_v3_0000",
      "scenario_id":   "enterprise_v3_0000",
      "mode":          "enterprise_gnn_rca",
      "diagram_scope": ["datacenter_topology", "app_db_topology"],
      "event_count":   5,
      "time_window":   {"start_offset_min": 0, "end_offset_min": 24},
      "cluster_score": 0.7812,
      "correlation_dimensions": {
        "temporal":       0.85,
        "topology":       0.70,
        "alert_type_seq": 0.70,
        "source_peer":    0.60,
        "cross_diagram":  0.60
      },
      "correlation_reasons": [
        "5 event(s) span 24 min (t=0..24)",
        "Propagation pattern detected: link_errors -> packet_drop -> connection_timeout",
        "Events span 2 diagrams: app_db_topology, datacenter_topology"
      ],
      "events": [
        {
          "event_id":         "EVT-0001",
          "node":             "DC-FW-01",
          "alert_type":       "packet_drop",
          "severity":         "critical",
          "time_offset_min":  0,
          "diagram_id":       "datacenter_topology",
          "correlation_role": "cluster_seed"
        }
      ],
      "cluster_fingerprint": "a3f1b9c2d4e7",
      "causal_evidence": [
        {
          "evidence_id":       "CE-001",
          "stage":             "temporal_correlation",
          "summary":           "5 alert(s) within a 24-minute window ...",
          "supporting_events": ["EVT-0001", "EVT-0002"],
          "supporting_nodes":  ["DC-FW-01", "DC-CORE-SW-01"],
          "confidence":        0.85
        }
      ]
    }
  ]
}
```

The output contains **no** `root_cause`, `root_cause_node`, `expected_root_cause`,
`remediation_steps`, `validation_steps`, `rollback`, or evaluation fields.

---

## Integrity constraints

- Labels (`labels.json`) are **never read** by the correlator.
- No remediation, resolution, or rollback content is produced.
- No root-cause labels or ground-truth fields appear in any cluster output.
- `validate_rca_outputs.py` scans `assets/preloaded/event_correlation/` by default
  and enforces the same forbidden-key rules as the other RCA preloaded directories.

---

## Directory layout

```
src/event_correlation/
  __init__.py         public API exports
  _patterns.py        alert classification + propagation patterns (internal)
  schema.py           cluster dict builders + FORBIDDEN_KEYS enforcement
  correlator.py       deterministic correlation engine
  evidence.py         causal evidence trail builder
  io.py               case loading + output writing helpers

scripts/
  build_event_correlation_clusters.py

assets/preloaded/event_correlation/   (committed for Streamlit / API use)
reports/event_correlation/manual_eval/ (gitignored — evaluation runs)
```
