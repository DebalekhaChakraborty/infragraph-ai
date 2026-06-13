# SOP-Grounded Qwen Dataset Expansion

Synthetic expansion of the 4-scenario base dataset to 80–120 training records via safe context variation.

---

## Why 4 records are insufficient

4 records cover 4 distinct root-cause nodes (DC-FW-01, APP-LB-01, DB-MASTER-01, WAN-PE-01) with identical incident contexts for each. A model trained on 4 samples:

- Memorises surface patterns rather than learning schema discipline
- Cannot generalise alert presentation variation (minor severity shifts, paraphrased correlation language)
- Cannot generalise KB evidence subset sensitivity (fewer chunks available should still produce valid citations)
- Has no exposure to confidence score variation even within the same root-cause class

This expansion creates 25 synthetic variants per scenario (100 total) by varying only **presentation-level** fields. The root cause, causal evidence chain, and domain-specific remediation class remain identical in every variant.

---

## What varies across synthetic records

| Field | Variation type | Constraint |
|-------|---------------|-----------|
| `cluster_score` | ±0.03 uniform nudge, clamped to [0.50, 0.99] | Never changes root-cause alignment |
| `cluster_id` | Suffix: `CLU-...-var_NNN` | Structural format preserved |
| `correlation_reasons` | 0–2 word substitutions from `_PARAPHRASE_PAIRS` | Meaning preserved; no new facts introduced |
| `alert_timeline[*].severity` | Domain-safe downstep (e.g., critical→high) | `alert_type` and `node` are never touched |
| `candidate_ranking[*].score` | ±0.02 (top), ±0.05 (others), rank order enforced | `node_id`, `rank`, `evidence` list unchanged |
| `retrieved_kb_evidence` | Subset of 3–kb_top_k from ranked pool | Top-ranked chunks preferred; domain alignment preserved |
| `incident_id` / `scenario_id` | Synthetic suffix `sop_grounded_<base>_var_NNN` | Traceability: `base_scenario_id` in metadata |

---

## What never varies

Unchanged across all variants of a given base scenario:

- `root_cause` — the predicted root-cause node ID
- `root_cause_diagram` — which topology diagram contains the root cause
- `rca_source` — "Enterprise GNN RCA" or "Scenario-grounded RCA simulation"
- `impacted_nodes` / `impacted_diagrams` — structural facts from the correlation graph
- `causal_evidence` (CE-* items) — evidence chain, summaries, stage labels
- Domain class — a WAN scenario never receives load_balancer remediation steps
- KB domain alignment — KB chunks ranked for the base root cause anchor all variants

---

## What this dataset is NOT

- Not a knowledge base. SOP facts come from retrieved KB chunks at inference time, not from model weights.
- Not a replacement for the KB index. New SOPs require KB re-indexing (`scripts/build_kb_index.py --reset`), not model retraining.
- Not an evaluation dataset. No labels, ground truth, or evaluation fields appear anywhere.

---

## When to re-index vs when to retrain

| Change | Action |
|--------|--------|
| New SOP or runbook added to `assets/kb/` | Rebuild KB index only |
| Existing SOP content revised | Rebuild KB index only |
| New alert type / node type added | Rebuild KB index if SOPs cover it |
| Model outputs wrong schema fields | Retrain with updated training data |
| Model ignores KB-*/CE-* citation discipline | Retrain |
| Model invents device names or commands | Retrain |
| Coverage of more root-cause classes needed | Add new scenarios, rebuild dataset, retrain |
| Model too sensitive to presentation variation | Expand synthetic dataset, retrain |

---

## Dataset scale

**Current state:** 4 base scenarios x 25 variants = 100 synthetic records, scaled for pipeline validation (not production alignment).

| Split | Records | Notes |
|-------|---------|-------|
| train | ~85 | Shuffled with `seed=42`, split at `round(100 * 0.85)` |
| val   | ~15 | Remainder after split |

Each base scenario contributes roughly 21–22 train / 3–4 val records after the global shuffle.

This scale is intentional for the current phase. The goal is to verify the pipeline, schema, and citation discipline — not to train a production model.

---

## Synthetic generation integrity

- `synthetic_generation_policy: "safe_context_variation_no_ground_truth"`
- `target_source: "template_sop_grounded_synthetic"`
- Each record carries `base_scenario_id` for traceability to the originating scenario
- `labels.json` is never read at any point in the generation pipeline
- No `expected_root_cause`, `ground_truth_node`, `correct_top1`, `correct_top_k`, `reciprocal_rank`, or `evaluation` fields appear anywhere
- Validation rejects any record where the top candidate is not the root cause node
- Validation rejects any record missing KB-* or CE-* citations in `evidence_ids_used`

---

## Commands

### Expand the dataset

```bash
# Requires KB index to be built first
python scripts/expand_sop_grounded_qwen_training_data.py --strict-kb --pretty

# Custom records per scenario
python scripts/expand_sop_grounded_qwen_training_data.py --records-per-scenario 50 --strict-kb

# Custom KB top-k
python scripts/expand_sop_grounded_qwen_training_data.py --kb-top-k 8 --strict-kb

# Custom output directory
python scripts/expand_sop_grounded_qwen_training_data.py \
    --out-dir data/qwen_sop_grounded_expanded_v2 \
    --strict-kb
```

### Inspect the expanded dataset

```bash
python scripts/inspect_qwen_sop_dataset.py --data-dir data/qwen_sop_grounded_expanded
python scripts/inspect_qwen_sop_dataset.py --data-dir data/qwen_sop_grounded_expanded --max-records 20
```

### Validate output

```python
import json, sys
from pathlib import Path
from collections import Counter

data_dir = Path("data/qwen_sop_grounded_expanded")

records = []
for p in [data_dir / "train.jsonl", data_dir / "val.jsonl"]:
    lines = p.read_text().strip().splitlines()
    assert lines, f"{p} is empty"
    for line in lines:
        rec = json.loads(line)
        assert len(rec["messages"]) == 3
        json.loads(rec["messages"][1]["content"])
        out = json.loads(rec["messages"][2]["content"])
        assert "evidence_ids_used" in out, f"missing evidence_ids_used in {rec['id']}"
        assert any(str(x).startswith("KB-") for x in out["evidence_ids_used"]), f"no KB-* in {rec['id']}"
        assert any(str(x).startswith("CE-") for x in out["evidence_ids_used"]), f"no CE-* in {rec['id']}"
        records.append(rec)

counts = Counter(rec["metadata"]["base_scenario_id"] for rec in records)
for scenario, count in counts.items():
    assert count >= 15, f"{scenario} has only {count} records"

print(f"PASS: {len(records)} records, per-base: {dict(counts)}")
```

---

## Output files

```
data/qwen_sop_grounded_expanded/
  train.jsonl               ~85 synthetic SFT training records (shuffled)
  val.jsonl                 ~15 synthetic validation records
  dataset_summary.json      Counts, base scenarios, generation config
  previews/                 (--pretty only)
    sop_grounded_<base>_var_NNN_input.json   User message content
    sop_grounded_<base>_var_NNN_target.json  Assistant target output
```

---

## Scaling guidance

For production-scale alignment:

1. Add new enterprise scenarios to `scenario_library/enterprise_gnn_rca/`
2. Add corresponding RCA outputs to `assets/preloaded/enterprise_gnn_rca/`
3. Add domain-specific SOPs to `assets/kb/`
4. Rebuild the KB index: `python scripts/build_kb_index.py --reset`
5. Run the base dataset generator: `python scripts/build_sop_grounded_qwen_training_data.py --strict-kb`
6. Run the expansion script: `python scripts/expand_sop_grounded_qwen_training_data.py --records-per-scenario 100 --strict-kb`

For production alignment, aim for:
- 50–200 base scenarios per domain
- Multiple SOP variants per domain
- Coverage of edge cases: partial evidence, multi-domain incidents, cascading failure chains
- Records per scenario: 50–100 to give the model robust exposure to presentation variation
