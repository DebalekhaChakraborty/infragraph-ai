#!/usr/bin/env python3
"""
build_sop_grounded_qwen_training_data.py
Generate Qwen-compatible SFT training data from SOP/KB-grounded remediation.

For each scenario:
  1. Loads clean RCA from assets/preloaded/enterprise_gnn_rca/<scenario_id>.json
  2. Builds context via build_enterprise_remediation_context() (with KB retrieval)
  3. Generates deterministic template remediation as the training target
  4. Produces a 3-message record: system / user (compact JSON) / assistant (strict JSON)

Output:
  data/qwen_sop_grounded/train.jsonl
  data/qwen_sop_grounded/val.jsonl
  data/qwen_sop_grounded/dataset_summary.json

Optional (--pretty):
  data/qwen_sop_grounded/previews/<scenario_id>_input.json
  data/qwen_sop_grounded/previews/<scenario_id>_target.json

Integrity:
  Never reads labels.json.
  Never includes evaluation or ground-truth fields.
  Validates each record before writing.

Usage:
  python scripts/build_sop_grounded_qwen_training_data.py --strict-kb --pretty
  python scripts/build_sop_grounded_qwen_training_data.py --scenarios enterprise_v3_0000
  python scripts/build_sop_grounded_qwen_training_data.py --kb-top-k 8 --train-ratio 0.75
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_remediation.context_builder import build_enterprise_remediation_context  # noqa: E402
from ai_remediation.template_mode import generate_template_remediation            # noqa: E402
from kb_retrieval.schema import DEFAULT_INDEX_DIR                                 # noqa: E402

_DEFAULT_SCENARIOS = [
    "enterprise_v3_0000",
    "enterprise_v3_0072",
    "enterprise_v3_0073",
    "enterprise_v3_0074",
]

_DEFAULT_OUT_DIR = "data/qwen_sop_grounded"

_SYSTEM_PROMPT = (
    "You are InfraGraph AI Remediation Agent. "
    "Generate a safe, SOP-grounded enterprise remediation plan. "
    "Use only the supplied RCA, event-correlation, causal evidence, and retrieved SOP/KB evidence. "
    "Cite KB-* and CE-* evidence IDs. "
    "Do not invent commands, IP addresses, owners, or device facts. "
    "Return strict JSON only."
)

_FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "expected_root_cause",
    "ground_truth_node",
    "correct_top1",
    "correct_top_k",
    "reciprocal_rank",
    "evaluation",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slim_kb_evidence(kb_evidence: list[dict]) -> list[dict]:
    """Return KB evidence items slimmed to only the fields needed in the prompt."""
    out: list[dict] = []
    for item in kb_evidence:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata") or {}
        out.append({
            "evidence_id": item.get("evidence_id", ""),
            "score":       item.get("score", 0.0),
            "metadata": {
                "kb_id":    meta.get("kb_id", ""),
                "title":    meta.get("title", ""),
                "doc_type": meta.get("doc_type", ""),
                "section":  meta.get("section", ""),
            },
            "text": item.get("text", ""),
        })
    return out


def _build_user_obj(context: dict, kb_evidence: list[dict]) -> dict:
    """Build the compact incident context object for the user message."""
    return {
        "incident_id":          context.get("incident_id", ""),
        "scenario_id":          context.get("scenario_id", ""),
        "scope":                context.get("scope", "enterprise"),
        "root_cause":           context.get("root_cause", ""),
        "root_cause_diagram":   context.get("root_cause_diagram", ""),
        "impacted_nodes":       context.get("impacted_nodes", []),
        "impacted_diagrams":    context.get("impacted_diagrams", []),
        "alert_timeline":       context.get("alert_timeline", []),
        "candidate_ranking":    context.get("candidate_ranking", []),
        "cluster_id":           context.get("cluster_id", ""),
        "cluster_score":        context.get("cluster_score"),
        "correlation_reasons":  context.get("correlation_reasons", []),
        "causal_evidence":      context.get("causal_evidence", []),
        "retrieved_kb_evidence": _slim_kb_evidence(kb_evidence),
    }


def _collect_all_keys(obj: object, depth: int = 0) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            if depth < 20:
                keys.update(_collect_all_keys(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_all_keys(item, depth + 1))
    return keys


def _validate_record(record: dict, strict_kb: bool) -> list[str]:
    """Return violation messages (empty list = clean)."""
    violations: list[str] = []

    msgs = record.get("messages", [])
    if len(msgs) != 3:
        violations.append(f"messages length must be 3, got {len(msgs)}")
        return violations

    if msgs[0].get("role") != "system":
        violations.append("messages[0].role must be 'system'")
    if msgs[1].get("role") != "user":
        violations.append("messages[1].role must be 'user'")
    if msgs[2].get("role") != "assistant":
        violations.append("messages[2].role must be 'assistant'")

    try:
        json.loads(msgs[1]["content"])
    except Exception as exc:
        violations.append(f"user content is not valid JSON: {exc}")

    asst_obj: dict = {}
    try:
        asst_obj = json.loads(msgs[2]["content"])
    except Exception as exc:
        violations.append(f"assistant content is not valid JSON: {exc}")

    if "evidence_ids_used" not in asst_obj:
        violations.append("assistant output missing 'evidence_ids_used'")

    if strict_kb:
        ev_ids = asst_obj.get("evidence_ids_used", [])
        if not any(str(x).startswith("KB-") for x in ev_ids):
            violations.append(
                "strict_kb: no KB-* evidence ID in assistant output's evidence_ids_used"
            )

    all_keys = _collect_all_keys(record)
    for fk in _FORBIDDEN_KEYS:
        if fk in all_keys:
            violations.append(f"forbidden key found: {fk!r}")

    return violations


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SOP-grounded Qwen SFT training data for InfraGraph AI."
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=_DEFAULT_SCENARIOS, metavar="ID",
        help="Scenario IDs to process (default: all four enterprise_v3 scenarios).",
    )
    parser.add_argument(
        "--out-dir", default=_DEFAULT_OUT_DIR, metavar="DIR",
        help=f"Output directory (default: {_DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.75, metavar="RATIO",
        help="Train split ratio (default: 0.75 -> first 3 of 4 scenarios to train).",
    )
    parser.add_argument(
        "--kb-index-dir", default=DEFAULT_INDEX_DIR, metavar="DIR",
        help=f"KB vector index directory (default: {DEFAULT_INDEX_DIR}).",
    )
    parser.add_argument(
        "--kb-top-k", type=int, default=5, metavar="N",
        help="KB evidence chunks to retrieve per scenario (default: 5).",
    )
    parser.add_argument(
        "--strict-kb", action="store_true",
        help="Exit with error if no KB evidence is retrieved for any scenario.",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Also write pretty-printed JSON preview files per scenario.",
    )
    args = parser.parse_args()

    out_dir      = (REPO_ROOT / args.out_dir).resolve()
    kb_index_dir = (REPO_ROOT / args.kb_index_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" InfraGraph AI -- SOP-Grounded Qwen Training Data")
    print("=" * 60)
    print(f"Scenarios  : {', '.join(args.scenarios)}")
    print(f"KB index   : {kb_index_dir.relative_to(REPO_ROOT)}")
    print(f"KB top-k   : {args.kb_top_k}")
    print(f"strict-kb  : {args.strict_kb}")
    print(f"train-ratio: {args.train_ratio}")
    print(f"Output dir : {out_dir.relative_to(REPO_ROOT)}")
    print()

    records: list[dict] = []
    skipped: list[str] = []

    for scenario_id in args.scenarios:
        rca_path = (
            REPO_ROOT / "assets" / "preloaded" / "enterprise_gnn_rca"
            / f"{scenario_id}.json"
        )
        if not rca_path.exists():
            print(f"  [SKIP] RCA file not found: {rca_path.relative_to(REPO_ROOT)}")
            skipped.append(scenario_id)
            continue

        print(f"--- {scenario_id} ---")

        try:
            context = build_enterprise_remediation_context(
                repo_root=REPO_ROOT,
                scenario_id=scenario_id,
                rca_path=rca_path,
                kb_index_dir=kb_index_dir,
                retrieve_kb=True,
                kb_top_k=args.kb_top_k,
                strict_kb=args.strict_kb,
            )
        except RuntimeError as exc:
            print(f"  [ERROR] {exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"  [ERROR] Could not build context: {exc}")
            skipped.append(scenario_id)
            continue

        kb_evidence    = context.get("retrieved_kb_evidence", []) or []
        causal_evidence = context.get("causal_evidence", []) or []
        print(f"  KB chunks      : {len(kb_evidence)}")
        print(f"  Causal evidence: {len(causal_evidence)}")

        # Build compact user message content
        user_obj = _build_user_obj(context, kb_evidence)

        # Generate deterministic remediation as assistant target
        remediation_out = generate_template_remediation(context)
        remediation_out.pop("source", None)  # remove internal label, not part of schema

        # Serialise to compact JSON strings
        user_content = json.dumps(user_obj, ensure_ascii=False, separators=(",", ":"))
        asst_content = json.dumps(remediation_out, ensure_ascii=False, separators=(",", ":"))

        # Collect KB IDs for metadata
        kb_ids: list[str] = []
        for item in kb_evidence:
            kb_id = (item.get("metadata") or {}).get("kb_id", "")
            if kb_id and kb_id not in kb_ids:
                kb_ids.append(kb_id)

        record: dict = {
            "id": f"sop_grounded_{scenario_id}",
            "messages": [
                {"role": "system",    "content": _SYSTEM_PROMPT},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": asst_content},
            ],
            "metadata": {
                "scenario_id":           scenario_id,
                "root_cause":            context.get("root_cause", ""),
                "root_cause_diagram":    context.get("root_cause_diagram", ""),
                "rca_source":            context.get("rca_source", ""),
                "cluster_id":            context.get("cluster_id", ""),
                "cluster_score":         context.get("cluster_score"),
                "kb_evidence_count":     len(kb_evidence),
                "causal_evidence_count": len(causal_evidence),
                "target_source":         "template_sop_grounded",
            },
        }

        violations = _validate_record(record, strict_kb=args.strict_kb)
        if violations:
            print(f"  [ERROR] Validation failures for {scenario_id}:")
            for v in violations:
                print(f"    - {v}")
            skipped.append(scenario_id)
            continue

        records.append(record)

        ev_ids    = remediation_out.get("evidence_ids_used", [])
        kb_ev_ids = [e for e in ev_ids if str(e).startswith("KB-")]
        print(f"  evidence_ids_used: {len(ev_ids)} total, {len(kb_ev_ids)} KB-*")
        print(f"  KB IDs: {kb_ids}")

        if args.pretty:
            preview_dir = out_dir / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            (preview_dir / f"{scenario_id}_input.json").write_text(
                json.dumps(user_obj, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (preview_dir / f"{scenario_id}_target.json").write_text(
                json.dumps(remediation_out, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  Previews: previews/{scenario_id}_input.json, _target.json")

        print()

    if not records:
        print("[ERROR] No training records generated.")
        sys.exit(1)

    # Deterministic split
    n_total = len(records)
    n_train = max(1, round(n_total * args.train_ratio))
    if n_total > 1:
        n_train = min(n_train, n_total - 1)  # always keep at least 1 for val

    train_records = records[:n_train]
    val_records   = records[n_train:]

    if not val_records:
        print("[ERROR] Not enough records for a val split (need at least 2 scenarios).")
        sys.exit(1)

    # Write JSONL files
    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"

    train_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_records) + "\n",
        encoding="utf-8",
    )
    val_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_records) + "\n",
        encoding="utf-8",
    )

    # Dataset summary
    summary = {
        "total_records":    n_total,
        "train_records":    len(train_records),
        "val_records":      len(val_records),
        "scenarios":        [r["metadata"]["scenario_id"] for r in records],
        "train_scenarios":  [r["metadata"]["scenario_id"] for r in train_records],
        "val_scenarios":    [r["metadata"]["scenario_id"] for r in val_records],
        "skipped":          skipped,
        "kb_top_k":         args.kb_top_k,
        "strict_kb":        args.strict_kb,
        "target_source":    "template_sop_grounded",
        "output_dir":       str(out_dir.relative_to(REPO_ROOT)),
    }
    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"[PASS] {n_total} record(s) generated.")
    print(f"  train : {len(train_records)} record(s) -> {train_path.relative_to(REPO_ROOT)}")
    print(f"  val   : {len(val_records)} record(s)  -> {val_path.relative_to(REPO_ROOT)}")
    print(f"  summary -> {summary_path.relative_to(REPO_ROOT)}")
    if skipped:
        print(f"  skipped: {skipped}")
    if args.pretty:
        print(f"  previews -> {(out_dir / 'previews').relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
