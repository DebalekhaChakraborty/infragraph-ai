# Remediation Pipeline

Generates remediation plans from clean Enterprise GNN RCA outputs and event
correlation evidence.

Remediation outputs are kept **separate** from RCA outputs.  RCA files remain
clean — no remediation fields are written to `assets/preloaded/enterprise_gnn_rca/`.

---

## Architecture

```
RCA JSON                      Event Correlation Clusters
  (assets/preloaded/           (assets/preloaded/
   enterprise_gnn_rca/          event_correlation/
   <scenario_id>.json)          <scenario_id>.json)
         │                             │
         └──────────────┬──────────────┘
                        ▼
          src/ai_remediation/context_builder.py
          build_enterprise_remediation_context()
                        │
                        ▼  remediation input dict
          ┌─────────────┴──────────────┐
          │                            │
          ▼                            ▼
 Qwen/vLLM (preferred)       Template mode (deterministic)
 src/ai_remediation/         src/ai_remediation/
   qwen_client.py              template_mode.py
          │                            │
          └─────────────┬──────────────┘
                        ▼
          assets/preloaded/remediation/<scenario_id>.json
```

---

## What the context builder does

`build_enterprise_remediation_context()` in `src/ai_remediation/context_builder.py`:

1. Loads the clean RCA JSON from `assets/preloaded/enterprise_gnn_rca/`.
2. Loads observable events from `scenario_library/enterprise_gnn_rca/<case_id>/events.json`.
3. **Never reads `labels.json`** and never adds ground_truth or evaluation fields.
4. Derives impacted nodes from causal evidence (supporting_nodes) or top_candidates.
5. Derives impact path from the first causal evidence item's supporting_nodes.
6. Passes cluster fields (`cluster_id`, `cluster_score`, `correlation_reasons`,
   `causal_evidence`) directly into the remediation context.

---

## Commands

### Generate remediation outputs

```bash
# Template mode (deterministic; no vLLM required)
python scripts/generate_remediation_demo_assets.py --template-only

# Prefer Qwen/vLLM (falls back to template if vLLM unavailable)
python scripts/generate_remediation_demo_assets.py --prefer-qwen

# Strict Qwen — fail if vLLM unavailable, no template fallback
python scripts/generate_remediation_demo_assets.py --strict-qwen
```

| Flag | Description |
|------|-------------|
| `--scenarios <id> ...` | Process specific scenario IDs only |
| `--template-only` | Deterministic template mode, no vLLM call |
| `--prefer-qwen` | Try Qwen first, fall back to template |
| `--strict-qwen` | Require vLLM; exit 1 if unavailable |
| `--out-dir <dir>` | Override output directory (default: `assets/preloaded/remediation`) |
| `--include-raw` | Include `raw_model_output` field in envelope |

### Validate remediation outputs

```bash
python scripts/validate_remediation_outputs.py --verbose
```

### Full pipeline

```bash
# With vLLM running
bash scripts/run_final_demo_pipeline.sh

# Template-only (no vLLM required)
INFRAGRAPH_TEMPLATE_ONLY=1 bash scripts/run_final_demo_pipeline.sh
```

---

## Output file format

`assets/preloaded/remediation/<scenario_id>.json`:

```json
{
  "scenario_id":        "enterprise_v3_0000",
  "case_id":            "ent_enterprise_v3_0000",
  "incident_id":        "INC-enterprise_v3_0000",
  "scope":              "enterprise",
  "rca_source":         "Enterprise GNN RCA",
  "cluster_id":         "CLU-ent_enterprise_v3_0000-001",
  "cluster_score":      0.8273,
  "remediation_source": "template",
  "model":              "—",
  "ok":                 true,
  "error":              "",
  "input_context_summary": {
    "root_cause":              "DC-FW-01",
    "root_cause_diagram":      "datacenter_topology",
    "candidate_count":         3,
    "causal_evidence_count":   5,
    "correlation_reason_count": 4
  },
  "remediation": {
    "executive_summary":           "...",
    "probable_root_cause":         "DC-FW-01 (in datacenter_topology)",
    "scope":                       "enterprise",
    "risk_level":                  "critical",
    "automation_eligibility":      "manual_only",
    "blast_radius":                "enterprise_wide",
    "evidence_ids_used":           ["CE-001", "CE-002", "CE-003", "CE-004", "CE-005"],
    "evidence_from_graph":         ["..."],
    "pre_checks":                  ["..."],
    "triage_steps":                ["..."],
    "validation_steps":            ["..."],
    "remediation_steps":           ["..."],
    "post_checks":                 ["..."],
    "do_not_execute_if":           ["..."],
    "rollback_or_safety_notes":    ["..."],
    "escalation_recommendation":   "...",
    "servicenow_incident_summary": {
      "short_description": "...",
      "description":       "...",
      "affected_ci":       "DC-FW-01",
      "priority":          "1-Critical",
      "assignment_group":  "Network Engineering — Enterprise Operations"
    },
    "audit_summary":    "...",
    "confidence_notes": "..."
  }
}
```

When `remediation_source` is `"template"`, the output is clearly deterministic
and must not be presented as model-generated or AI output.

---

## Integrity constraints

- **RCA outputs are never modified.** Only `assets/preloaded/remediation/` is written.
- **`labels.json` is never read** at remediation generation time.
- **No ground_truth or evaluation fields** appear in remediation outputs.
- **Template mode** is labelled `"remediation_source": "template"` — do not present
  as AI-generated.
- **Qwen output** is labelled `"remediation_source": "qwen_vllm"` — only when an actual
  vLLM response was received and parsed successfully.
- **Causal evidence** is treated as supporting evidence, not absolute proof.  If causal
  evidence conflicts with the RCA result, the output must say human validation is required.
- The LLM is never allowed to invent devices, teams, commands, IPs, or services not
  referenced in the context.

---

## Validation rules

`validate_remediation_outputs.py` checks each file for:

**Required envelope keys**: `scenario_id`, `case_id`, `incident_id`, `scope`,
`rca_source`, `cluster_id`, `remediation_source`, `ok`, `remediation`.

**Required remediation keys**: `executive_summary`, `probable_root_cause`, `scope`,
`risk_level`, `automation_eligibility`, `blast_radius`, `evidence_from_graph`,
`pre_checks`, `triage_steps`, `validation_steps`, `remediation_steps`, `post_checks`,
`do_not_execute_if`, `rollback_or_safety_notes`, `escalation_recommendation`,
`servicenow_incident_summary`, `audit_summary`, `confidence_notes`.

**Non-empty lists**: `remediation_steps`, `validation_steps`, `rollback_or_safety_notes`.

**Non-empty**: `servicenow_incident_summary.short_description`.

**Forbidden anywhere**: `expected_root_cause`, `ground_truth_node`, `correct_top1`,
`correct_top_k`, `reciprocal_rank`, `evaluation`.

---

## Directory layout

```
src/ai_remediation/
  context_builder.py     build remediation input from RCA JSON + events
  prompt_builder.py      Qwen3 prompt construction (local + enterprise)
  template_mode.py       deterministic fallback plan generator
  qwen_client.py         vLLM/OpenAI-compatible HTTP client
  response_schema.py     input/output schema definitions

scripts/
  generate_remediation_demo_assets.py   generate per-scenario remediation JSONs
  validate_remediation_outputs.py       validate remediation output files
  run_final_demo_pipeline.sh            full RCA → validate → remediation → validate

assets/preloaded/
  enterprise_gnn_rca/    RCA outputs only (no remediation fields)
  event_correlation/     cluster outputs (causal evidence)
  remediation/           remediation outputs (separate from RCA)
```
