# Diagram Onboarding

Onboard a new topology diagram into InfraGraph AI graph memory with a single command.

---

## What it does

| Step | Output |
|------|--------|
| 1. Receive image | `original.png` copied to output directory |
| 2. YOLO detection | `detected.png` — annotated image; uses mock for `diagram_0373` |
| 3. Node table | `detected_nodes.json` — canonical names, zones, confidences |
| 4. Graph build | `local_graph.json` — nodes + layout-inferred edges |
| 5. Graph preview | `graph_preview.png` — NetworkX + matplotlib visualization |
| 6. Memory update | `graph_memory/index.json` — registry of all onboarded diagrams |
| 7. Summary | `onboarding_summary.json` — run metadata and stats |

---

## CLI usage

```bash
python scripts/onboard_diagram.py \
    --image  datasets/infragraph_v2/images/test/diagram_0373.png \
    --diagram-id demo_onboard_0373 \
    --out    outputs/onboarded_diagrams
```

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | (required) | Path to input PNG or JPG |
| `--diagram-id` | (required) | Unique identifier for this diagram |
| `--model` | `training_runs/infragraph_yolo_v2/weights/best.pt` | YOLO weights; v1 used as Alternate path |
| `--out` | `outputs/onboarded_diagrams` | Root output directory |

---

## Output layout

```
outputs/onboarded_diagrams/<diagram_id>/
    original.png           — unmodified input image
    detected.png           — YOLO-annotated or mock detection image
    detected_nodes.json    — node table (see schema below)
    local_graph.json       — graph with layout-inferred edges
    graph_preview.png      — matplotlib/networkx visualization
    onboarding_summary.json — run metadata and stats

graph_memory/
    index.json             — registry of all onboarded diagrams
```

---

## detected_nodes.json schema

```json
[
  {
    "node_id":        "FW-001",
    "detected_type":  "firewall",
    "confidence":     0.9231,
    "bbox":           {"x1": 312, "y1": 220, "x2": 455, "y2": 310},
    "bbox_center":    {"x": 383, "y": 265},
    "zone":           "perimeter",
    "canonical_name": "FW-001",
    "graph_status":   "added"
  }
]
```

> **Note:** Node names are generated during MVP onboarding unless metadata/OCR is available.
> Canonical names follow the `TYPE-NNN` format (e.g., `FW-001`, `SW-002`, `DB-001`).

---

## Node naming

| Detected type | Canonical prefix | Example |
|---------------|-----------------|---------|
| `firewall` | `FW` | `FW-001`, `FW-002` |
| `router` | `RTR` | `RTR-001` |
| `switch` | `SW` | `SW-001`, `SW-002` |
| `server` | `APP` | `APP-001` |
| `database` | `DB` | `DB-001` |
| `load_balancer` | `LB` | `LB-001` |
| `cloud_or_wan` | `WAN` | `WAN-001` |
| `service` | `SVC` | `SVC-001` |

---

## Zone inference

Zones are inferred from the vertical position of each device's bounding-box center:

| Image position (fraction of height) | Zone |
|--------------------------------------|------|
| 0 – 25 % | `wan` |
| 25 – 50 % | `perimeter` |
| 50 – 75 % | `core` |
| 75 – 100 % | `server` |

---

## Edge inference (layout_inference_v1)

Edges are **not** fully connected. They are inferred from detected device types and
spatial proximity (nearest-neighbour within each downstream type group):

| Source type | Can connect to |
|-------------|----------------|
| `cloud_or_wan` | `router`, `firewall` |
| `router` | `firewall`, `switch` |
| `firewall` | `switch`, `load_balancer` |
| `switch` | `load_balancer`, `server`, `database` |
| `load_balancer` | `server` (up to 3) |
| `server` | `database` |

Each non-LB source connects to its 1–2 spatially nearest downstream nodes.
Bidirectional duplicates are suppressed.

---

## local_graph.json schema

```json
{
  "diagram_id": "demo_onboard_0373",
  "nodes": [
    {
      "id": "FW-001",
      "label": "Firewall FW-001",
      "type": "firewall",
      "zone": "perimeter",
      "confidence": 0.9231,
      "bbox_center": {"x": 383, "y": 265}
    }
  ],
  "edges": [
    {
      "source": "WAN-001",
      "target": "FW-001",
      "relationship": "routes_to"
    }
  ],
  "graph_build_method": "layout_inference_v1",
  "notes": "..."
}
```

---

## graph_memory/index.json schema

```json
[
  {
    "diagram_id": "demo_onboard_0373",
    "source_image": "datasets/infragraph_v2/images/test/diagram_0373.png",
    "detected_nodes_path": "outputs/onboarded_diagrams/demo_onboard_0373/detected_nodes.json",
    "local_graph_path": "outputs/onboarded_diagrams/demo_onboard_0373/local_graph.json",
    "graph_preview_path": "outputs/onboarded_diagrams/demo_onboard_0373/graph_preview.png",
    "node_count": 17,
    "edge_count": 14,
    "detection_method": "mock_fallback",
    "timestamp": "2026-06-10T12:00:00+00:00",
    "status": "processed"
  }
]
```

---

## YOLO Alternate path

When `ultralytics` is unavailable or the model file does not exist:

1. Script tries `infragraph_yolo_v2/weights/best.pt`
2. uses `infragraph_yolo_v1/weights/best.pt`
3. If no model is found and the image is `diagram_0373`, uses the pre-existing
   `outputs/topology_demo/diagram_0373_detected_nodes.json` as `mock_fallback`
4. For any other image without a model, the script exits with a clear error message

---

## Streamlit UI

The **Diagram Intelligence** workspace exposes the onboarding flow at the top of the page:

- **Upload new topology diagram** — file uploader for any PNG/JPG
- **Use Presentation diagram_0373** — triggers onboarding on the reference diagram without upload

Progress is shown live via Streamlit's `st.status` widget:

```
→ Image received
→ Running vision detection...
→ Vision detection complete
→ Node table generated
→ Local graph created
→ Graph memory updated
✓ Diagram Onboarding complete
```

Results displayed:
- Original vs detection image pair
- Detected node table (node_id, type, confidence, zone, status)
- Local graph preview (PNG)
- Graph Memory Updated card (type distribution + diagram_id)

---

## Limitations (MVP)

- Edge inference uses bounding-box spatial layout only — no line detection or OCR
- Node names are canonical placeholders, not extracted from diagram labels
- Multi-page or very large diagrams may require resizing before upload
- Graph preview requires `matplotlib` and `networkx` (installed via `requirements.txt`)


