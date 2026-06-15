# Enterprise GNN RCA V2 — Temporal Relation-Aware GraphSAGE

## Overview

InfraGraph AI ships with two Enterprise GNN RCA variants:

| | V1 | V2 |
|---|---|---|
| Architecture | HomogeneousGraphSAGE | Relation-Aware GraphSAGE |
| Edge types used | No | Yes (local, cross-diagram, vision) |
| Temporal node features | Yes (via feature vector) | Yes (via same feature vector) |
| Status | Default, stable | Optional, additive |

V2 adds relation-awareness to V1's established temporal node features. V1 remains the safe default fallback.

## What V1 Does

`EnterpriseRcaGNN` (V1) is a 3-layer GraphSAGE model that operates on a homogeneous enterprise graph. All edges are treated equally. Node features are 54-dimensional and already include temporal and propagation signals:

- First alert time, mean alert time, alert sequence position
- Upstream/downstream alert counts (propagation depth)
- Propagation consistency score
- Node–alert type compatibility score

Root cause is determined solely by GNN node scoring. No LLM or remediation content is involved.

## What V2 Adds

`EnterpriseRcaTemporalRelGNN` (V2) runs four parallel SAGEConv stacks — one per relation type:

- **All edges** — mirrors V1 baseline
- **Local edges** — intra-diagram connectivity only
- **Cross-diagram edges** — inter-diagram stitch connections
- **Vision-connector edges** — edges extracted by the vision connector pipeline

The four relation embeddings are concatenated and projected through an MLP to produce per-node logits.

### Why this matters

Cross-diagram root causes propagate across stitch boundaries. Giving the model explicit awareness of which edges cross diagram boundaries allows it to weight cross-diagram paths differently from local paths — consistent with how human engineers triage cross-diagram faults.

Vision-extracted edges, when present, provide additional physical connectivity evidence that is not always captured in annotation metadata.

## Why This is Safer Than Replacing V1

1. **Separate artifacts** — V2 writes to `model_artifacts/enterprise_gnn_rca_v2/` and `outputs/enterprise_gnn_rca_v2/`. V1 is untouched.
2. **Graceful fallback** — If V2 output is absent, the orchestrator automatically falls back to V1, then to graph-grounded heuristics.
3. **Backward-compatible graph format** — Old `graphs.pt` files without `edge_type` still load correctly. V2 detects the absence and falls back to all-edge SAGEConv internally.
4. **Same root-cause contract** — Root cause still comes from GNN graph scoring, never from an LLM.

## Temporal Signals Used

These are the same 9 temporal and propagation features present in V1's 54-dim vector, now also leveraged by V2:

| Feature | Description |
|---|---|
| `is_first_alerted_node` | Binary: was this node the first to alert? |
| `is_last_alerted_node` | Binary: was this node the last to alert? |
| `alert_sequence_position_norm` | Normalised rank in alert timeline |
| `upstream_alert_count_norm` | Fraction of upstream ancestors that are alerted |
| `downstream_alert_count_norm` | Fraction of downstream descendants that are alerted |
| `upstream_critical_count_norm` | Upstream alerted nodes with critical severity |
| `downstream_warning_count_norm` | Downstream alerted nodes with warning severity |
| `propagation_consistency_score` | Composite: earliness × downstream fraction × upstream cleanliness |
| `node_alert_compatibility_score` | Fraction of alert types compatible with this node type |

## Relation / Heterogeneous Signals Used

| Edge type | ID | Source |
|---|---|---|
| `local` | 0 | Intra-diagram edges from enterprise_graph |
| `cross_diagram` | 1 | Stitch map cross-diagram edges |
| `vision_connector_extraction` | 2 | Vision pipeline detected connectors |
| `annotation_connector` | 3 | Annotation metadata connectors |
| `local_graph` | 4 | Local graph fallback edges |
| `unknown` | 5 | Unclassified edges |

V2 uses separate SAGEConv stacks for types 0, 1, and 2. Types 3–5 contribute to the all-edge stack only.

## Fallback Behaviour

| Condition | V2 Behaviour |
|---|---|
| `edge_type` absent in graph | `h_local`, `h_cross`, `h_vision` are zeros; degrades to all-edge GraphSAGE |
| No local edges | `h_local` is zeros |
| No cross-diagram edges | `h_cross` is zeros |
| No vision edges | `h_vision` is zeros |
| V2 result file absent | Orchestrator loads V1 result instead |
| V1 result also absent | Orchestrator uses graph-grounded heuristic fallback |

## Honest Architecture Statement

V2 is a **temporal-aware relation-aware GraphSAGE** model. It is **not** a fully dynamic temporal heterogeneous graph transformer (such as TGN, HTGNN, or a Transformer-based temporal GNN). The temporal awareness comes entirely from hand-engineered node features in the 54-dim vector, not from learned temporal attention across time steps.

This is by design: the goal is to add relation-awareness incrementally without replacing a working V1 system.

## Known Limitations

- Edge type IDs are assigned at graph build time; edge types not listed in `EDGE_TYPE_TO_ID` fall back to `unknown` (type 5)
- V2 trains on the same graphs.pt as V1; full V2 benefit requires rebuilding graphs.pt after adding `edge_type` support (PART 1)
- Vision-connector edges are sparse in current datasets; `h_vision` will often be zeros in practice until the vision pipeline is more widely run
