# Diagram Intelligence V3 Dataset

Diagram Intelligence V3 starts the scenario-native dataset track for InfraGraph AI.
It is designed for the demo story where separate topology diagrams are onboarded,
each diagram becomes a local graph, and those local graphs are stitched into one
enterprise graph memory for cross-diagram RCA.

## Why V3 Exists

V1 and V2 are useful for single-diagram detection, graph extraction, and RCA. They
do not prove that several independent diagrams belong to the same enterprise
environment.

V3 fixes that gap. One scenario is one enterprise environment. Each scenario
contains 3 to 5 related topology diagrams, and the same diagram annotations are
used for:

- RF-DETR object detection training.
- YOLO-compatible fallback training.
- OCR validation for names, IPs, zones, and connector labels.
- Connector and edge extraction validation.
- Local graph creation.
- Multi-diagram enterprise graph stitching.
- Future enterprise GNN RCA training.

V3 diagrams are not isolated object-detection images. They are local views inside
one shared scenario.

## Scenario Structure

The dataset root is:

```bash
datasets/diagram_v3_enterprise/
```

Each scenario is written under:

```bash
datasets/diagram_v3_enterprise/scenarios/{train,val,test}/enterprise_v3_0000/
```

A scenario includes:

- `diagrams/*.png`: source topology diagrams.
- `annotations/*.json`: single source of truth for objects, text, and connectors.
- `labels_yolo/*.txt`: YOLO labels derived from annotations.
- `local_graphs/*.json`: one graph per diagram.
- `stitch_map.json`: shared entities and cross-diagram dependencies.
- `enterprise_graph.json`: stitched graph generated from the local graphs.
- `alerts.json`: root cause and symptoms across diagrams.
- Preview PNGs for contact sheets, stitching story, hero stitching story, enterprise graph, and RCA overlay.

## Diagram Types

The generator reserves `enterprise_v3_0000` as the demo hero scenario. It always
contains all five diagram types in one enterprise chain: Branch, WAN/MPLS,
Datacenter, App/DB, and Shared Services. This scenario also writes
`preview_stitching_story_hero.png`, a large three-stage visual showing the same
source diagrams becoming local graphs and then a stitched enterprise graph with
RCA context.

Scenarios use 3 to 5 diagrams from:

- `branch_topology`
- `wan_topology`
- `datacenter_topology`
- `app_db_topology`
- `shared_services_topology`

The diagrams are visibly separate architecture views, but shared entities and
cross-diagram dependencies prove they are part of one enterprise.

Example:

- `branch_topology` includes `BR-RTR-01`.
- `wan_topology` also shows `BR-RTR-01` and connects it to WAN PE nodes.
- `datacenter_topology` includes `DC-FW-01` and core switching.
- `app_db_topology` depends on the datacenter aggregation switch.
- `shared_services_topology` provides DNS, identity, monitoring, and logging services.

## Annotation Contract

Each `annotations/<diagram_id>.json` file contains:

- `objects`: device boxes with class, label, IP, zone, and canonical ID.
- `text_blocks`: node labels, IP addresses, zones, and connector text for OCR validation.
- `connectors`: source, target, relationship, label, points, bbox, and style.

This annotation is the source for detector exports, OCR validation, connector
validation, local graph generation, and stitching.

## Local Graphs And Stitching

Each local graph is generated from the objects and connectors visible in that
diagram. Shared entities keep a `canonical_id`, so the enterprise graph can merge
the same logical node across diagrams.

`stitch_map.json` records:

- `shared_entities`: where canonical entities appear.
- `cross_diagram_edges`: dependencies between diagrams.

`enterprise_graph.json` is built from the local graphs and stitch map. It is not a
separately random graph. This keeps RF-DETR images, OCR labels, connector labels,
local graphs, alerts, and enterprise RCA labels tied to the same scenario.

## RCA Labels

`alerts.json` creates incidents where the root cause and symptoms may span
diagrams. Examples include WAN failures affecting branch and app views,
datacenter firewall drops affecting app and database nodes, shared identity
failures affecting branch and app users, load balancer failures, and database
failures.

These labels are intended for future enterprise GNN RCA training over the stitched
graph.

## Commands

Generate the dataset:

```bash
python scripts/generate_diagram_v3_enterprise_dataset.py \
  --num-scenarios 100 \
  --out ./datasets/diagram_v3_enterprise \
  --seed 2026 \
  --clean
```

Prepare the RF-DETR COCO export:

```bash
python scripts/prepare_rfdetr_dataset.py \
  --dataset-root ./datasets/diagram_v3_enterprise \
  --out ./datasets/diagram_v3_enterprise/rfdetr
```

The generator also writes a YOLO-compatible export:

```bash
datasets/diagram_v3_enterprise/yolo/dataset.yaml
```

## Limitations

- The images are synthetic, so icon styles and document noise are controlled.
- OCR boxes are generated from known text placement, not from a real OCR engine.
- Connector annotations are generated from known topology edges, not from a line detector.
- RF-DETR training depends on the external RF-DETR package and local hardware setup.
- The enterprise RCA graph is ready for future GNN work, but V3 GNN training is not part of this generator.
