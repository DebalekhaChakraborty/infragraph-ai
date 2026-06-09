# Qwen Explanation Layer

`scripts/generate_qwen_rca_explanation.py` is Stage 4 of InfraGraph AI.
It takes the structured evidence produced by the heuristic RCA (Stage 2) and
GNN RCA (Stage 3) and generates a human-readable incident report targeted at an
L1/L2 operations team.

The script runs locally in **mock mode** (no LLM required) and calls
**Qwen via vLLM** when the AMD Jupyter environment is available.

---

## Purpose of the LLM layer

The earlier stages produce structured JSON:

| Stage | Output |
|-------|--------|
| 2 — Heuristic RCA | predicted root cause, impact paths, confidence score |
| 3 — GNN RCA | re-ranked root cause, node scores, correction over heuristic |

An L1 engineer still has to read and interpret those JSON files.
The Qwen layer bridges the gap: it turns the structured evidence into
a narrative incident report with an executive summary, propagation explanation,
ServiceNow ticket fields, and actionable L1/L2 remediation steps.

---

## Local development — mock mode

Mock mode generates a **deterministic** report from the evidence using Python
string templates.  No internet, no GPU, no Qwen installation needed.

```powershell
python scripts/generate_qwen_rca_explanation.py `
    --diagram-id diagram_0373 `
    --mode mock
```

The output is identical every time for the same input files.  Use this mode
to develop, test, and iterate on the prompt structure and output format.

---

## AMD Jupyter — vLLM mode

On the AMD GPU node, start a Qwen vLLM server first:

```bash
# Start vLLM with Qwen3-4B (adjust GPU count / model as needed)
vllm serve Qwen/Qwen3-4B \
    --port 8000 \
    --dtype auto \
    --max-model-len 4096
```

Then run the explanation script pointing at the server:

```bash
python scripts/generate_qwen_rca_explanation.py \
    --diagram-id  diagram_0373 \
    --mode        vllm \
    --model       Qwen/Qwen3-4B \
    --base-url    http://localhost:8000/v1
```

If the vLLM call fails for any reason the script automatically falls back to
mock mode and marks `provider` as `"mock_fallback_after_vllm_error"` in the
output JSON.

### Prompt design

**System prompt**
```
You are an enterprise AIOps root cause analysis assistant. Use only the
provided topology, alert, heuristic RCA, and GNN RCA evidence. Do not invent
nodes, alerts, tools, or remediation actions. Be concise, operational, and
suitable for an L1/L2 incident response team.
```

The system prompt constrains Qwen to the evidence JSON so it cannot hallucinate
node names or actions that are not grounded in the actual topology.

**User prompt** includes the full evidence JSON and asks for eight sections in
order: Executive Summary, What Happened, Root Cause Conclusion, Heuristic vs
GNN Comparison, Impacted Nodes/Services, Recommended Next Actions,
ServiceNow Incident Summary, Confidence and Limitations.

---

## CLI reference

```
python scripts/generate_qwen_rca_explanation.py [options]

  --diagram-id  DIAGRAM_ID   e.g. diagram_0373 (default)
  --mode        mock|vllm    mock: no LLM; vllm: call OpenAI-compat endpoint
  --model       MODEL        Qwen model name passed to vLLM (vllm mode only)
  --base-url    URL          vLLM API base URL (default: http://localhost:8000/v1)
  --topo-dir    PATH         heuristic RCA output dir (default: outputs/topology_demo)
  --gnn-dir     PATH         GNN RCA output dir (default: outputs/gnn_rca)
  --out         PATH         explanation output dir (default: outputs/qwen_explanation)
```

---

## Inputs

| File | Stage |
|------|-------|
| `outputs/topology_demo/<id>_rca_result.json` | Stage 2: heuristic RCA |
| `outputs/topology_demo/<id>_graph_summary.json` | Stage 2: graph stats |
| `outputs/gnn_rca/<id>_gnn_rca_result.json` | Stage 3: GNN RCA |

---

## Outputs

All files written to `outputs/qwen_explanation/`:

| File | Description |
|------|-------------|
| `<id>_explanation.md` | Human-readable Markdown report |
| `<id>_explanation.json` | Full result: evidence + prompt + markdown + metadata |
| `<id>_prompt.json` | System and user prompts (for debugging / prompt tuning) |

### `<id>_explanation.json` schema

```json
{
  "diagram_id": "diagram_0373",
  "provider": "mock",
  "model": "mock-template",
  "evidence": {
    "diagram_id": "...",
    "graph": { "node_count": 17, "edge_count": 17, "alert_count": 3, ... },
    "ground_truth": { "root_cause": "FW-01", "root_cause_type": "firewall" },
    "heuristic_rca": { "predicted_root_cause": "SW-CORE", "is_correct": false, ... },
    "gnn_rca": { "predicted_root_cause": "FW-01", "is_correct": true, ... },
    "gnn_improved_over_heuristic": true,
    "alerting_nodes": ["FW-01", "SW-CORE"],
    "impacted_nodes": ["APP-01", "..."],
    "impact_paths": { ... },
    "impact_path_summary": { ... }
  },
  "prompt": { "system": "...", "user": "..." },
  "explanation_markdown": "# InfraGraph AI RCA Explanation ...",
  "output_markdown_path": "outputs/qwen_explanation/diagram_0373_explanation.md"
}
```

---

## How this completes the InfraGraph AI story

```
[Stage 1: Data]
    generate_infragraph_dataset.py
        -> PNG diagrams + YOLO labels + graph JSON + alert JSON

[Stage 2: Vision + Heuristic RCA]
    YOLOv8 detector  ->  build_topology_rca_demo.py
        -> outputs/topology_demo/<id>_rca_result.json  (heuristic prediction)

[Stage 3: GNN RCA]
    train_gnn_rca.py
        -> outputs/gnn_rca/<id>_gnn_rca_result.json   (GNN prediction)

[Stage 4: LLM Explanation]  <-- this script
    generate_qwen_rca_explanation.py
        -> outputs/qwen_explanation/<id>_explanation.md   (human report)
        -> outputs/qwen_explanation/<id>_explanation.json (full artifact)
        -> outputs/qwen_explanation/<id>_prompt.json      (prompt for tuning)
```

The end-to-end loop: a synthetic network diagram enters Stage 1, propagates
through vision detection (Stage 2), graph intelligence (Stages 2-3), and
surfaces as a natural-language incident report with ServiceNow-ready fields
(Stage 4) — all without any manual analysis.

---

## diagram_0373 example output

For the reference scenario where the heuristic incorrectly chose `SW-CORE`
and the GNN correctly identified `FW-01`:

```markdown
## Executive Summary

A network incident triggered 3 alerts across 2 nodes, impacting 10 downstream
services. The heuristic RCA incorrectly identified SW-CORE as root cause. The
GNN-based RCA correctly identified FW-01 (firewall), demonstrating the value
of learned propagation-direction signals over rule-based scoring.

## Root Cause Conclusion

Root cause: FW-01 (firewall)
Confidence: HIGH (GNN score: 30.73, margin over 2nd-ranked: 8.12)

## ServiceNow Incident Summary

Short description: Network fault on FW-01 causing 10-service outage
Affected CI: FW-01 (firewall)
Priority: P1 -- 10 downstream nodes impacted
Assignment group: Network Operations
...
```
