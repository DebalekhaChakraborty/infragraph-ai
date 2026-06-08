# Topology Graph & RCA Demo

`scripts/build_topology_rca_demo.py` is the bridge between the YOLO vision
model and the graph intelligence layer of InfraGraph AI.  Given a single
diagram ID it pulls together YOLO predicted bboxes, the ground-truth topology
graph, and the synthetic alert scenario, then runs a heuristic RCA and produces
a visualisation.

---

## How to run

```bash
# Default: V2 dataset, V2 predictions, output to outputs/topology_demo/
python scripts/build_topology_rca_demo.py --diagram-id diagram_0373

# Override paths
python scripts/build_topology_rca_demo.py \
    --diagram-id    diagram_0381 \
    --dataset-root  ./datasets/infragraph_v2 \
    --pred-root     ./outputs/v2_test_predictions_cpu \
    --out           ./outputs/topology_demo
```

---

## Inputs

| Input | Default path |
|-------|-------------|
| Source image | `datasets/infragraph_v2/images/test/<id>.png` |
| YOLO predicted labels | `outputs/v2_test_predictions_cpu/labels/<id>.txt` |
| Topology graph JSON | `datasets/infragraph_v2/graphs/test/<id>.json` |
| Alert / RCA scenario JSON | `datasets/infragraph_v2/alerts/test/<id>.json` |

**YOLO label format** (one detection per line):
```
<class_id> <cx> <cy> <w> <h> <confidence>
```
Coordinates are normalized to `[0, 1]`.  The script converts them to pixel
bboxes using the PNG header dimensions (no PIL dependency needed).

**Graph JSON schema** — top-level keys: `diagram_id`, `template`, `metadata`,
`nodes` (list with `id`, `type`, `zone`, optional `bbox`), `edges` (list with
`source`, `target`, `label`, `relationship`).

**Alert JSON schema** — top-level keys: `scenario_id`, `root_cause`,
`root_cause_type`, `alerts` (list with `node`, `alert_type`, `severity`,
`time_offset_min`), `expected_impacted_nodes`.

---

## Outputs (all under `--out`)

| File | Description |
|------|-------------|
| `<id>_detected_nodes.json` | YOLO predictions with pixel bboxes and class labels |
| `<id>_rca_result.json` | RCA ranking, predicted root cause, impact paths, reasoning |
| `<id>_topology.png` | NetworkX topology visualisation coloured by device type, highlighting root cause and alert/impact nodes |
| `<id>_graph_summary.json` | Compact summary: node/edge counts, per-type counts, alert count, root cause |

### `<id>_detected_nodes.json` schema
```json
[
  {
    "predicted_id": "pred_0",
    "class_id": 0,
    "type": "router",
    "confidence": 0.9945,
    "bbox_normalized": [0.162, 0.661, 0.033, 0.051],
    "bbox_pixel": [204, 574, 250, 620]
  }
]
```

### `<id>_rca_result.json` schema
```json
{
  "diagram_id": "diagram_0373",
  "predicted_root_cause": "SW-CORE",
  "ground_truth_root_cause": "FW-01",
  "confidence_score": 0.503,
  "top_candidates": [{"node": "SW-CORE", "score": 20.86, "type": "switch"}],
  "alerting_nodes": ["FW-01", "SW-CORE"],
  "impacted_nodes": ["APP-01", "APP-02", "..."],
  "impact_paths": {
    "predicted_root_cause": [
      {
        "source": "SW-CORE",
        "target": "FW-01",
        "target_reason": "alerting_node",
        "path": ["FW-01", "SW-CORE"],
        "path_length": 2,
        "method": "directed_target_to_root",
        "truncated": false
      }
    ],
    "ground_truth_root_cause": [
      {
        "source": "FW-01",
        "target": "SW-CORE",
        "target_reason": "alerting_node",
        "path": ["FW-01", "SW-CORE"],
        "path_length": 2,
        "method": "directed_root_to_target",
        "truncated": false
      }
    ]
  },
  "impact_path_summary": {
    "predicted_root_cause_path_count": 10,
    "ground_truth_root_cause_path_count": 10,
    "shortest_predicted_path": ["FW-01", "SW-CORE"],
    "shortest_ground_truth_path": ["FW-01", "SW-CORE"]
  },
  "reasoning_summary": "The heuristic selected 'SW-CORE' because ..."
}
```

---

## RCA algorithm

### Heuristic scoring

Each alerting node receives a weighted score:

```
score = severity_weight × 2
      + (1 / (1 + time_offset_min)) × 10   # earlier = higher
      + (downstream_count / total_nodes) × 3
      + device_type_bonus(node_type, alert_type)
```

Severity weights: `critical=4`, `major=3`, `minor=2`, `warning/info=1`.

Device-type bonuses (+0.5):
- `firewall` when alert mentions packet/drop/deny/policy
- `router` when alert mentions unreachable/route/BGP/OSPF
- `database` when alert mentions database/query/SQL/slow
- `load_balancer` when alert mentions load/pool/upstream/502
- `server` when alert mentions CPU/memory/app/crash

The top 5 ranked nodes are saved as `top_candidates`.  Ground-truth root cause
(from the alert JSON) is recorded alongside the prediction so accuracy can be
measured across the test set.

### Impact paths

For each of the two root candidates (predicted and ground-truth), the script
finds paths to up to 10 target nodes, capped at 8 hops each.

Target priority order:
1. `alerting_node` — other nodes that raised alerts
2. `expected_impacted` — nodes listed in the alert scenario (GT root only)
3. `impacted_node` — all descendants in the directed topology graph

Path discovery uses three methods in order:

| Method | Description |
|--------|-------------|
| `directed_root_to_target` | `nx.shortest_path(G, root, target)` |
| `directed_target_to_root` | `nx.shortest_path(G, target, root)` — used when the root is *downstream* of the target (upstream dependency) |
| `undirected` | `nx.shortest_path(G.to_undirected(), root, target)` — fallback for cross-zone connections with no directed path |

The `method` field on each path entry records which approach succeeded.  When
a path exceeds 8 nodes it is truncated and marked `"truncated": true`.

#### Why impact paths matter for RCA

Impact paths provide the *evidence chain* that explains how a fault at the root
propagates to the affected devices.  This has three uses downstream:

1. **Topology visualisation** — the shortest predicted and GT paths are drawn
   in contrasting colours (red = predicted, blue = GT) on the topology PNG so
   that a human reviewer can immediately see whether the predicted path makes
   topological sense.

2. **Qwen explanation layer** — the LLM prompt will include the path list as
   structured context, enabling it to generate a sentence like *"The firewall
   FW-01 dropped packets, causing SW-CORE and the downstream app servers to
   become unreachable via the path FW-01 → SW-CORE → SW-APP → LB-01."*

3. **GNN training signal** — the propagation direction recorded in each path
   entry (`directed_root_to_target` vs `directed_target_to_root`) surfaces
   the ambiguity that the heuristic cannot resolve but a GNN trained on full
   alert-propagation sequences can learn.

#### Heuristic limitations and motivation for the GNN stage

The heuristic may confuse a downstream aggregation node (e.g. `SW-CORE`
receiving cascaded alerts from many children) with the true upstream origin
(`FW-01`).  The `reasoning_summary` field in the RCA JSON explicitly
flags this mismatch and explains it.  The GNN stage (notebook
`05_rca_graph_demo.ipynb`) addresses this by learning propagation direction
from graph structure and temporal alert features.

---

## How this connects the pipeline

```
[Data generator]
    generate_infragraph_dataset.py
        └─ PNG diagrams  ──────────────────────────────► YOLO training
        └─ graphs/test/<id>.json  ──────────────────────► build_topology_rca_demo.py
        └─ alerts/test/<id>.json  ──────────────────────► build_topology_rca_demo.py

[Vision model]
    training_runs/infragraph_yolo_v1/ (or v2/)
        └─ weights/best.pt  ──► yolo detect predict  ──► outputs/v2_test_predictions_cpu/
                                                              └─ labels/<id>.txt ──► build_topology_rca_demo.py

[Topology & RCA]  ◄── this script
    outputs/topology_demo/<id>_*.{json,png}
        └─ _detected_nodes.json  ──► future: matcher against ground-truth graph
        └─ _rca_result.json      ──► future: GNN RCA (notebooks/05_rca_graph_demo.ipynb)
        └─ _topology.png         ──► demo / submission visualisation
        └─ _graph_summary.json   ──► aggregate metrics across test set

[Next: Qwen / LLM explanation layer]
    _rca_result.json  ──► Qwen prompt ──► plain-language incident explanation
```

---

## Dependencies

Only `networkx` and `matplotlib` are required beyond the standard library.
Both are already listed in `requirements.txt`.  PNG dimensions are read from
the file header using `struct`, so `Pillow` is not required for this script.
