# SOP/KB-Grounded Remediation RAG

Retrieval-augmented generation for remediation plans backed by SOP, runbook, and known-resolution documents.

When the KB vector index is built and retrieval is enabled, every remediation output is grounded in structured SOP/KB evidence — not invented text. KB evidence IDs (``KB-*``) appear alongside causal evidence IDs (``CE-*``) in every output.

---

## Architecture

```
assets/kb/                          ChromaDB vector index
  sops/                 ──loader──▶ runtime_state/kb_index/
  runbooks/             ──chunk──▶  (embeddings via all-MiniLM-L6-v2)
  known_resolutions/                         │
                                             │ query
                                             ▼
Enterprise GNN RCA output          src/kb_retrieval/retriever.py
  +                       ────▶    retrieve_kb_evidence()
Event correlation clusters                   │
                                             ▼
                          src/ai_remediation/context_builder.py
                          build_enterprise_remediation_context()
                          (injects KB evidence into context)
                                             │
                          ┌──────────────────┴──────────────────┐
                          ▼                                      ▼
               Qwen/vLLM prompt                    Template mode
               (KB-* IDs in grounding rules)       (KB-* IDs in evidence_from_graph,
                                                    remediation_steps, validation_steps,
                                                    rollback_or_safety_notes)
                          │                                      │
                          └──────────────────┬──────────────────┘
                                             ▼
                          assets/preloaded/remediation/<scenario_id>.json
```

---

## KB Document Structure

KB documents live in `assets/kb/` with three subdirectories:

```
assets/kb/
  sops/               Standard Operating Procedures
  runbooks/           Operational runbooks (cross-cutting)
  known_resolutions/  Confirmed past resolution patterns
```

Each file is a markdown document with YAML-like frontmatter:

```markdown
---
kb_id: SOP-DC-FW-001
title: "Datacenter Firewall Packet Drop and Link Error Response"
doc_type: sop
version: "1.4"
owner_group: "Network Engineering — Datacenter Operations"
applies_to_node_types:
  - firewall
applies_to_diagrams:
  - datacenter_topology
applies_to_alert_types:
  - packet_drop
  - link_errors
rca_patterns:
  - datacenter_firewall_policy_block
last_reviewed: "2026-03-15"
evidence_tags:
  - DC-FW
---

## Purpose
## Trigger Symptoms
## Applicable RCA Patterns
## Pre-Checks
## Triage Steps
## Remediation Steps
## Validation Steps
## Rollback / Safety Notes
## Do Not Execute If
## ITSM Routing
## Evidence Tags
```

---

## Evidence ID Format

KB evidence chunks are assigned IDs of the form:

```
KB-<kb_id>-<chunk_index>
```

Examples:
- `KB-SOP-DC-FW-001-000` — first chunk of SOP-DC-FW-001
- `KB-SOP-APP-LB-001-002` — third chunk of SOP-APP-LB-001

These IDs appear in `evidence_ids_used`, `evidence_from_graph`, `audit_summary`, and `confidence_notes` in remediation outputs.

---

## Commands

### 1. Build the KB index

```bash
# First-time build
python scripts/build_kb_index.py

# Force rebuild (clear existing index)
python scripts/build_kb_index.py --reset

# Custom paths
python scripts/build_kb_index.py --kb-root assets/kb --index-dir runtime_state/kb_index
```

### 2. Generate remediation with KB grounding

```bash
# Template mode + KB (default: KB retrieval enabled)
python scripts/generate_remediation_demo_assets.py --template-only

# Build KB index first, then generate
python scripts/generate_remediation_demo_assets.py --build-kb-index --template-only

# Strict KB (fail if no KB evidence retrieved)
python scripts/generate_remediation_demo_assets.py --template-only --strict-kb

# Disable KB retrieval
python scripts/generate_remediation_demo_assets.py --template-only --no-kb

# Override top-k
python scripts/generate_remediation_demo_assets.py --template-only --kb-top-k 8
```

### 3. Validate remediation outputs

```bash
python scripts/validate_remediation_outputs.py --verbose
```

The validator enforces:
- `evidence_from_graph` is non-empty
- If `kb_evidence_count > 0`, at least one `KB-*` ID appears in the output

### 4. Generate training data

```bash
# With KB grounding
python scripts/build_sop_grounded_remediation_training_data.py

# Without KB (RCA context only)
python scripts/build_sop_grounded_remediation_training_data.py --no-kb
```

Output: `data/remediation_training/sop_grounded_remediation_sft.jsonl`

---

## Embedding Model

Default: `sentence-transformers/all-MiniLM-L6-v2`

Override with the `INFRAGRAPH_EMBED_MODEL` environment variable:

```bash
INFRAGRAPH_EMBED_MODEL="sentence-transformers/all-mpnet-base-v2" \
  python scripts/build_kb_index.py --reset
```

**Important**: After changing the embedding model, always rebuild the index with `--reset`. Mixing embeddings from different models in the same collection produces incorrect retrieval results.

---

## Updating KB Documents

To add or update a SOP/runbook/known_resolution:

1. Edit (or create) the markdown file in `assets/kb/sops/`, `assets/kb/runbooks/`, or `assets/kb/known_resolutions/`.
2. Rebuild the index: `python scripts/build_kb_index.py --reset`
3. Regenerate remediation outputs: `python scripts/generate_remediation_demo_assets.py --template-only`
4. Validate: `python scripts/validate_remediation_outputs.py --verbose`

No model retraining is required — only the vector index needs to be rebuilt.

---

## Integrity Constraints

- **Never read `labels.json`** — the KB retrieval pipeline reads only RCA JSON + events.json.
- **Never add evaluation fields** — KB evidence appears only in remediation context, never in RCA outputs.
- **Template mode is deterministic** — KB evidence adds structured references, but output is always labelled `"remediation_source": "template"`.
- **Do not invent SOP names or commands** — only KB-* IDs retrieved from the index may be cited; template_mode.py does not hallucinate SOP content.
- **KB-* IDs are traceable** — every KB-* ID can be resolved to a specific chunk of a specific markdown file via `chunk_id = f"{kb_id}::chunk-{i:03d}"`.

---

## Source Package

```
src/kb_retrieval/
  __init__.py    public API: build_kb_index, retrieve_kb_evidence, build_query_from_rca_context
  schema.py      constants: DEFAULT_KB_ROOT, DEFAULT_INDEX_DIR, DEFAULT_COLLECTION
  loader.py      parse_frontmatter, load_kb_documents
  chunker.py     chunk_document (section-level + sliding window)
  indexer.py     build_kb_index (ChromaDB PersistentClient + sentence-transformers)
  retriever.py   build_query_from_rca_context, retrieve_kb_evidence (with reranking)
```

---

## Training Data

`scripts/build_sop_grounded_remediation_training_data.py` generates supervised fine-tuning data from:
- Enterprise GNN RCA outputs
- KB-enriched remediation contexts
- Template remediation outputs (as target)

Output: `data/remediation_training/sop_grounded_remediation_sft.jsonl`

Each row format:
```json
{
  "scenario_id": "enterprise_v3_0000",
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "... RCA + KB evidence ..."},
    {"role": "assistant", "content": "... remediation JSON ..."}
  ],
  "metadata": {
    "root_cause":        "DC-FW-01",
    "rca_source":        "Enterprise GNN RCA",
    "kb_ids":            ["SOP-DC-FW-001"],
    "evidence_ids":      ["KB-SOP-DC-FW-001-000", "CE-001"],
    "kb_evidence_count": 5
  }
}
```

This data is for later fine-tuning only. No training is run by this script.
