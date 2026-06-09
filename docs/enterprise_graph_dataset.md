# Enterprise Graph Dataset

Multi-diagram graph stitching for cross-diagram GNN root-cause analysis (RCA).

---

## Why Cross-Diagram RCA

### The Single-Diagram Blind Spot

Traditional network-diagram datasets treat each diagram as an isolated universe.
A GNN trained on single diagrams can propagate alert signals within one diagram
but has no mechanism to reason about failures that originate in a diagram it
has never seen in relation to the one showing the symptom.

Real enterprises never fit on a single diagram:

- A branch office diagram shows workstations and a local switch — but the
  WAN router that connects the branch to HQ lives in a separate WAN topology diagram.
- The database master lives in the app/DB diagram; the firewall that dropped
  its packets lives in the datacenter diagram.
- The identity service lives in a shared-services diagram but serves every
  application in every other diagram.

When the WAN MPLS core fails, the symptom (connection timeouts) appears in
the branch and app diagrams — but the root cause only exists in the WAN diagram.
A single-diagram model will misattribute the fault every time.

### What We Need

A model that can follow the causal chain *across diagram boundaries* needs:

1. A unified graph where nodes from multiple diagrams coexist.
2. Explicit *cross-diagram edges* that connect the diagrams at shared infrastructure points.
3. Training labels that mark the true root-cause node even when it lives in a
   different diagram from the alerting nodes.

The enterprise graph dataset provides all three.

---

## Solar System vs Galaxy Analogy

Each **local diagram** is like a **solar system**: a self-contained set of
nodes orbiting a core device (the edge router, the datacenter firewall, the
load balancer).  The local graph captures topology within that system perfectly.

The **enterprise graph** is the **galaxy**: all solar systems laid out in
space, connected by interstellar highways — the cross-diagram edges.
A packet traveling from a branch workstation to a database replica crosses
multiple solar systems.  The GNN must understand the galaxy to find where
the supernova (root cause) actually happened.

The **stitch map** is the **star-chart**: it declares which solar systems are
connected, which objects appear in more than one system (shared entities), and
where the interstellar highways (cross-diagram edges) run.

---

## Dataset Structure

```
datasets/enterprise_graph_v1/
├── dataset_summary.json                    # Aggregate statistics across all scenarios
├── previews/
│   └── enterprise_contact_sheet.png        # Tiled preview of enterprise graphs
└── scenarios/
    ├── train/
    │   └── enterprise_0000/
    │       ├── metadata.json               # Per-scenario key metrics
    │       ├── stitch_map.json             # Cross-diagram wiring
    │       ├── enterprise_graph.json       # Unified merged graph
    │       ├── alerts.json                 # Root cause + secondary alerts
    │       ├── local_graphs/
    │       │   ├── branch_topology.json
    │       │   ├── wan_topology.json
    │       │   ├── datacenter_topology.json
    │       │   ├── app_db_topology.json
    │       │   └── shared_services_topology.json
    │       └── diagrams/
    │           ├── branch_topology.png
    │           ├── wan_topology.png
    │           ├── datacenter_topology.png
    │           ├── app_db_topology.png
    │           ├── shared_services_topology.png
    │           ├── preview_enterprise_graph.png  # Galaxy view
    │           └── preview_contact_sheet.png     # Local diagrams tiled
    ├── val/
    └── test/
```

Each enterprise scenario contains **3 to 5** local infrastructure diagrams
stitched into one galaxy-scale enterprise graph.

---

## Local Graph JSON Schema

Each file in `local_graphs/` describes one infrastructure diagram as a
directed graph.

```json
{
  "diagram_id":   "branch_topology",
  "diagram_type": "branch_topology",
  "nodes": [
    {
      "id":             "BR-RTR-01",
      "type":           "router",
      "zone":           "branch-edge",
      "diagram_id":     "branch_topology",
      "is_shared_entity": false
    },
    {
      "id":             "BR-FW-01",
      "type":           "firewall",
      "zone":           "branch-edge",
      "diagram_id":     "branch_topology",
      "is_shared_entity": false
    }
  ],
  "edges": [
    {
      "source":   "BR-RTR-01",
      "target":   "BR-FW-01",
      "relation": "routes_to"
    }
  ]
}
```

### Node Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Canonical node identifier (stable across the enterprise graph) |
| `type` | string | Device class: router, switch, firewall, server, database, load_balancer, cloud_or_wan, service |
| `zone` | string | Logical segment within the diagram (e.g. branch-edge, dc-core) |
| `diagram_id` | string | Which local diagram this node belongs to |
| `is_shared_entity` | bool | True when this node appears in more than one diagram |

### Edge Fields

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Source node ID |
| `target` | string | Target node ID |
| `relation` | string | Semantic relation: routes_to, secured_by, connected_to, depends_on, serves, wan_dependency |

---

## Stitch Map Schema

`stitch_map.json` declares the cross-diagram wiring for one enterprise scenario.

```json
{
  "shared_entities": [
    {
      "canonical_id": "DC-FW-01",
      "appears_in": [
        {"diagram_id": "wan_topology",        "local_alias": "WAN-DC-FW-01"},
        {"diagram_id": "datacenter_topology", "local_alias": "DC-FW-01"}
      ]
    },
    {
      "canonical_id": "SVC-DNS-01",
      "appears_in": [
        {"diagram_id": "shared_services_topology", "local_alias": "SVC-DNS-01"},
        {"diagram_id": "app_db_topology",          "local_alias": "APP-DNS-REF"}
      ]
    }
  ],
  "cross_diagram_edges": [
    {
      "source":         "BR-RTR-01",
      "target":         "WAN-PE-01",
      "relation":       "wan_dependency",
      "source_diagram": "branch_topology",
      "target_diagram": "wan_topology"
    },
    {
      "source":         "WAN-MPLS-CORE",
      "target":         "DC-FW-01",
      "relation":       "routes_to",
      "source_diagram": "wan_topology",
      "target_diagram": "datacenter_topology"
    },
    {
      "source":         "DC-CORE-SW-01",
      "target":         "APP-LB-01",
      "relation":       "connected_to",
      "source_diagram": "datacenter_topology",
      "target_diagram": "app_db_topology"
    }
  ]
}
```

### Cross-Diagram Edge Rules

| Diagram pair | Cross-diagram edge | Relation |
|---|---|---|
| branch + wan | BR-RTR-01 → WAN-PE-01 | wan_dependency |
| wan + datacenter | WAN-MPLS-CORE → DC-FW-01 | routes_to |
| branch + datacenter (no wan) | BR-RTR-01 → DC-FW-01 | wan_dependency |
| datacenter + app_db | DC-CORE-SW-01 → APP-LB-01 | connected_to |
| wan + app_db (no datacenter) | WAN-MPLS-CORE → APP-LB-01 | routes_to |
| shared_services + app_db | SVC-IDENTITY-01 → APP-01/02 | serves |
| shared_services + branch | SVC-DNS-01 → BR-SW-01 | serves |

---

## Enterprise Graph Schema

`enterprise_graph.json` is the unified galaxy-scale graph.

```json
{
  "scenario_id": "enterprise_0000",
  "nodes": [
    {
      "id":                    "BR-RTR-01",
      "type":                  "router",
      "zone":                  "branch-edge",
      "diagram_id":            "branch_topology",
      "diagram_type":          "branch_topology",
      "is_shared_entity":      false,
      "is_cross_diagram_bridge": true
    }
  ],
  "edges": [
    {
      "source":     "BR-RTR-01",
      "target":     "BR-FW-01",
      "relation":   "routes_to",
      "edge_scope": "local",
      "diagram_id": "branch_topology"
    },
    {
      "source":         "BR-RTR-01",
      "target":         "WAN-PE-01",
      "relation":       "wan_dependency",
      "edge_scope":     "cross_diagram",
      "source_diagram": "branch_topology",
      "target_diagram": "wan_topology"
    }
  ],
  "diagram_clusters": [
    {"diagram_id": "branch_topology",     "node_ids": ["BR-RTR-01", "BR-FW-01", "BR-SW-01", "BR-WRK-01"]},
    {"diagram_id": "wan_topology",        "node_ids": ["WAN-MPLS-CORE", "WAN-ISP", "WAN-PE-01", "WAN-PE-02"]},
    {"diagram_id": "datacenter_topology", "node_ids": ["DC-FW-01", "DC-FW-02", "DC-CORE-SW-01"]}
  ],
  "cross_diagram_edges": [...],
  "shared_entities":     [...]
}
```

### Key Node Fields in Enterprise Graph

| Field | Description |
|-------|-------------|
| `diagram_id` | The local diagram this node was merged from |
| `diagram_type` | Same as diagram_id (for GNN feature engineering) |
| `is_shared_entity` | True when the node appears in multiple diagrams (shared infrastructure) |
| `is_cross_diagram_bridge` | True when the node is a source or target of a cross-diagram edge |

### Edge Scope

| `edge_scope` | Meaning |
|---|---|
| `local` | Edge exists within a single diagram; `diagram_id` field identifies which |
| `cross_diagram` | Edge crosses diagram boundaries; `source_diagram` and `target_diagram` identify both sides |

---

## Alert Scenarios

### Why Root Cause is Often in a Different Diagram

In real enterprise incidents the observable symptoms propagate *downstream*
from the failure point.  The failure point (root cause) is upstream and often
in a different functional diagram:

- WAN MPLS core fails → packets drop → branch workers lose connectivity (branch diagram shows alerts)
- Datacenter firewall fails → app servers can't reach DB → DB write errors (app diagram shows alerts)
- Identity service fails → app login calls timeout → apps report 503s (app diagram shows alerts)

A model that only sees the alerting diagram will blame the first alerting
node it finds — which is never the root cause.  The enterprise graph exposes
the causal chain so the GNN can trace backward through cross-diagram edges
to the actual origin.

### Root Cause Patterns

| Pattern | Required Diagram | Root Node | Default Severity |
|---|---|---|---|
| `wan_mpls_failure` | wan_topology | WAN-MPLS-CORE | critical |
| `dc_fw_failure` | datacenter_topology | DC-FW-01 | critical |
| `db_master_failure` | app_db_topology | DB-MASTER | high |
| `lb_failure` | app_db_topology | APP-LB-01 | critical |
| `identity_svc_failure` | shared_services_topology | SVC-IDENTITY-01 | high |
| `wan_pe_failure` | wan_topology | WAN-PE-01 | high |
| `branch_rtr_failure` | branch_topology | BR-RTR-01 | medium |
| `dc_core_sw_failure` | datacenter_topology | DC-CORE-SW-01 | high |

### Alert JSON Schema

```json
{
  "scenario_id":   "enterprise_0000",
  "root_cause_pattern": "wan_mpls_failure",
  "root_node_id":  "WAN-MPLS-CORE",
  "severity":      "critical",
  "alert_type":    "packet_drop",
  "primary_alert": {
    "alert_id":        "enterprise_0000-PRIMARY",
    "node_id":         "WAN-MPLS-CORE",
    "alert_type":      "packet_drop",
    "severity":        "critical",
    "time_offset_min": 0,
    "is_root_cause":   true,
    "diagram_id":      "wan_topology"
  },
  "secondary_alerts": [
    {
      "alert_id":        "enterprise_0000-SEC-01",
      "node_id":         "BR-RTR-01",
      "alert_type":      "connection_timeout",
      "severity":        "high",
      "time_offset_min": 3,
      "is_root_cause":   false,
      "diagram_id":      "branch_topology"
    }
  ],
  "impacted_nodes":  ["WAN-PE-01", "WAN-PE-02", "DC-FW-01", "BR-RTR-01"],
  "impact_paths":    [
    ["WAN-MPLS-CORE", "WAN-PE-01"],
    ["WAN-MPLS-CORE", "DC-FW-01", "DC-CORE-SW-01", "DC-AGG-SW-01"]
  ],
  "cross_diagram_propagation": true
}
```

The field `cross_diagram_propagation` is `true` when at least one secondary
alert comes from a different diagram than the primary root-cause alert.
This flag filters training samples that specifically exercise cross-diagram
reasoning.

---

## How This Enables Cross-Diagram GNN RCA

### Standard Single-Diagram GNN

A 2-layer GCN on a single diagram propagates messages at depth 2.
The receptive field is limited to 2-hop neighbours within the diagram.
If the root cause is 3 hops away or in another diagram, the signal cannot reach it.

### Enterprise GNN with Cross-Diagram Edges

By merging all local graphs and inserting `cross_diagram_edges`:

1. **Layer 1** aggregates messages within each local cluster (same as before).
2. **Layer 2** propagates across `cross_diagram_edges` — bridging
   the WAN diagram into the datacenter diagram, the datacenter into app/DB, etc.
3. The root-cause node, even if buried in a leaf diagram, receives a boosted
   signal from its downstream alerting neighbours *across diagram boundaries*.

The `is_cross_diagram_bridge` node feature explicitly tells the GNN which
nodes are galaxy-highway on-ramps — allowing it to weight cross-diagram
message passing more heavily during training.

The `diagram_type` feature (one-hot encoded) tells the GNN which functional
domain each node belongs to, enabling domain-specific attention weights.

### Training Recommendation

Use the `enterprise_graph.json` as the input graph and `alerts.json` as
the supervision signal.  Node features:

| Feature | Encoding |
|---|---|
| `type` | One-hot (8 classes) |
| `diagram_type` | One-hot (5 classes) |
| `is_shared_entity` | Binary |
| `is_cross_diagram_bridge` | Binary |
| `has_alert` | Binary (from alerts.json) |
| `alert_severity` | Ordinal 0–3 |
| `time_offset_min` | Normalised float |

Label: binary `is_root_cause` per node.

---

## Generate Command

```bash
# Full 120-scenario dataset
python scripts/generate_enterprise_scenarios.py \
    --num 120 \
    --out ./datasets/enterprise_graph_v1 \
    --seed 2026 \
    --clean

# Quick 20-scenario smoke test
python scripts/generate_enterprise_scenarios.py \
    --num 20 \
    --out ./datasets/enterprise_graph_v1 \
    --seed 2026 \
    --clean
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num N` | 120 | Number of enterprise scenarios to generate |
| `--out PATH` | `./datasets/enterprise_graph_v1` | Output root directory |
| `--seed INT` | 2026 | Global random seed (reproducible) |
| `--clean` | false | Wipe output directory before generating |
| `--min-diagrams INT` | 3 | Minimum local diagrams per scenario |
| `--max-diagrams INT` | 5 | Maximum local diagrams per scenario |
| `--preview-count INT` | 12 | Enterprise graphs included in contact sheet |

### Optional Dependencies

| Package | Effect if missing |
|---------|-------------------|
| `matplotlib` | PNG diagram visualisations skipped |
| `networkx` | Graph layout skipped (no PNGs) |
| `Pillow` | Contact sheets skipped |

All three are optional — the JSON dataset is generated regardless.

---

## Dataset Statistics (120 scenarios, seed 2026)

| Metric | Expected value |
|--------|----------------|
| Total scenarios | 120 |
| Train / val / test | ~96 / 12 / 12 |
| Diagram configs: 3-diagram | ~42 scenarios (35%) |
| Diagram configs: 4-diagram | ~54 scenarios (45%) |
| Diagram configs: 5-diagram | ~24 scenarios (20%) |
| Avg diagrams per scenario | ~3.9 |
| Avg nodes per enterprise graph | ~22–32 |
| Avg cross-diagram edges | 2–4 |
| Scenarios with cross-diagram propagation | ~70% |
| Root cause patterns | 8 (balanced by diagram coverage) |

Files generated per scenario:

| File | Description |
|------|-------------|
| `metadata.json` | Key metrics and split assignment |
| `stitch_map.json` | Cross-diagram wiring |
| `enterprise_graph.json` | Unified topology (nodes + edges + clusters) |
| `alerts.json` | Root cause + secondary alert scenario |
| `local_graphs/*.json` | Per-diagram local graph (3–5 files) |
| `diagrams/*.png` | Per-diagram dark-theme visualisation |
| `preview_enterprise_graph.png` | Galaxy-view enterprise graph |
| `preview_contact_sheet.png` | Local diagrams tiled |

Total file count for 120 scenarios: approximately 1,800–2,400 files.
