# Scenario Library

## Why scenario_library/ exists

The InfraGraph AI dataset (`datasets/infragraph_v3/`) stores raw scenario artifacts
including ground-truth labels mixed together with alert streams in `alerts.json`.
This is fine for dataset storage, but it creates a leakage risk when the same files
are used to drive simulation UIs or train/evaluate models:

- Passing `alerts.json` directly to a simulation shows the model its own answer.
- Rendering `recommended_actions` before the AI agent runs makes the output look
  hardcoded.
- Evaluation code that reads labels from the same object that feeds events cannot
  test generalisation.

`scenario_library/` solves this by **separating the public event stream from the
private label/answer key** at build time.

---

## Directory structure

```
scenario_library/
  manifest.json                          # index of all cases
  topology_rca/{case_id}/
    events.json                          # public  — simulation observable only
    labels.json                          # private — ground truth
    metadata.json                        # bookkeeping (split, counts, provenance)
    graph_ref.json                       # repo-relative paths to graph files
  enterprise_gnn_rca/{case_id}/
    events.json
    labels.json
    metadata.json
    graph_ref.json
```

Build or rebuild with:

```bash
python scripts/build_scenario_library.py
```

Validate without writing:

```bash
python scripts/build_scenario_library.py --dry-run
```

---

## events.json — public, simulation-visible only

Contains only what an operator/observer would see in real time:

| Field | Description |
|-------|-------------|
| `case_id` | Unique case identifier |
| `mode` | `topology_rca` or `enterprise_gnn_rca` |
| `scenario_id` | Source scenario from infragraph_v3 |
| `diagram_id` | (topology_rca only) which diagram this case covers |
| `events[]` | List of alert events |

Each event contains: `event_id`, `time_offset_min`, `node`, `diagram_id`,
`alert_type`, `severity`.

**The following keys must NEVER appear in events.json:**
`root_cause`, `root_cause_diagram`, `root_cause_pattern`, `recommended_actions`,
`remediation_steps`, `triage_steps`, `rollback`, `rollback_or_safety_notes`,
`itsm`, `impacted_nodes`, `impacted_diagrams`, `impact_paths`,
`validation_steps`, `post_checks`, `pre_checks`.

The build script enforces this with an explicit validation step and will exit
non-zero if any forbidden key is found.

---

## labels.json — private, ground truth

Used only by training and evaluation code.  Never fed to a live simulation UI.

### topology_rca labels

| Field | Description |
|-------|-------------|
| `root_cause_in_scope` | `true` if root cause node is in this diagram |
| `severity` | Overall incident severity |
| `impacted_nodes` | Nodes alerted in this diagram |
| `root_cause_node` | (if in_scope) root cause node ID |
| `root_cause_diagram` | (if in_scope) diagram containing root cause |
| `root_cause_pattern` | (if in_scope) pattern label |
| `impact_paths` | (if in_scope) propagation paths |
| `expected_behavior` | (if not in_scope) `"escalate_or_unknown"` |

### enterprise_gnn_rca labels

| Field | Description |
|-------|-------------|
| `root_cause_node` | Root cause node ID |
| `root_cause_diagram` | Diagram containing root cause |
| `root_cause_pattern` | Pattern label |
| `impacted_nodes` | All nodes affected across diagrams |
| `impacted_diagrams` | All diagrams affected |
| `impact_paths` | Cross-diagram propagation paths |
| `severity` | Overall incident severity |

---

## Topology RCA vs Enterprise GNN RCA cases

### topology_rca

One case is generated per diagram that appears in the alert stream AND has a
`local_graphs/<diagram_id>.json` file.  The case for the `root_cause_diagram`
is always included.

- `root_cause_in_scope = true`: root cause is in this diagram.  A topology RCA
  model should identify it.
- `root_cause_in_scope = false`: root cause is in a different diagram.  A
  topology RCA model should return `escalate_or_unknown` and hand off to
  enterprise RCA.

### enterprise_gnn_rca

One case per scenario, using all alerts across all diagrams and the stitched
enterprise graph.  The GNN is expected to rank nodes across diagram boundaries
and identify the cross-diagram root cause.

---

## graph_ref.json

Contains repo-relative paths to the graph files.  Code should resolve them
from `REPO_ROOT`:

```python
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
graph = json.loads((REPO_ROOT / graph_ref["local_graph_path"]).read_text())
```

This keeps `scenario_library/` lightweight (no duplicated graph data) while
making the source location explicit.

---

## No remediation content in simulation files

`scenario_library/` is a clean boundary.  Rules:

1. `events.json` feeds the simulation UI — it must contain only observable events.
2. `labels.json` feeds ML training and evaluation — it must stay out of the UI.
3. Remediation actions, rollback notes, ITSM summaries, and resolution plans
   come **only** from the AI Resolution Agent (Qwen3/vLLM) at inference time.
4. If a model or script needs to evaluate remediation quality, it reads
   `labels.json` and compares against the agent's output — it does not pre-load
   remediation steps into the event stream.

---

## How future ML/GNN scripts should consume scenario_library/

```python
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIB_ROOT  = REPO_ROOT / "scenario_library"

manifest = json.loads((LIB_ROOT / "manifest.json").read_text())

for row in manifest["enterprise_gnn_rca"]:
    # Public events (feed to model)
    events = json.loads((LIB_ROOT / row["events_path"]).read_text())

    # Private labels (use for loss / evaluation only)
    labels = json.loads((LIB_ROOT / row["labels_path"]).read_text())

    # Graph (load from dataset via ref)
    graph_ref  = json.loads((LIB_ROOT / row["graph_ref_path"]).read_text())
    graph_data = json.loads((REPO_ROOT / graph_ref["enterprise_graph_path"]).read_text())

    # --- train / evaluate here ---
    # events["events"]  -> alert timeline fed to model
    # labels["root_cause_node"]  -> ground truth for loss
    # graph_data["nodes"], graph_data["edges"] -> GNN input
```

Topology RCA works the same way using `manifest["topology_rca"]` and
`graph_ref["local_graph_path"]`.

---

## Rebuilding after new dataset generation

Whenever `datasets/infragraph_v3/` is regenerated, rebuild the library:

```bash
python datasets/infragraph_v3/generate_infragraph_v3_dataset.py ...
python scripts/build_scenario_library.py
```

The library is a derived artifact — it is committed to git because the JSON
files are small and enable offline evaluation without re-reading the full
dataset.  Model checkpoints and large binaries are never stored here.
