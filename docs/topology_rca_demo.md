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
  "predicted_root_cause": "FW-01",
  "ground_truth_root_cause": "FW-01",
  "confidence_score": 0.8,
  "top_candidates": [{"node": "FW-01", "score": 12.5, "type": "firewall"}],
  "alerting_nodes": ["FW-01", "SW-CORE"],
  "impacted_nodes": ["SW-CORE", "SW-APP", "LB-01", "..."],
  "impact_paths": [["FW-01", "SW-CORE", "..."]],
  "reasoning_summary": "Node 'FW-01' ranked highest: ..."
}
```

---

## RCA algorithm

The heuristic scorer assigns each alerting node a weighted score:

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
