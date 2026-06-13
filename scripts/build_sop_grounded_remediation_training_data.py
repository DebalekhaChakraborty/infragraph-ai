#!/usr/bin/env python3
"""
build_sop_grounded_remediation_training_data.py
Generate SFT training data from SOP/KB-grounded template remediation outputs.

Reads:
  assets/preloaded/enterprise_gnn_rca/<scenario_id>.json  (RCA outputs)
  assets/preloaded/remediation/<scenario_id>.json          (template outputs)
  runtime_state/kb_index/                                  (KB vector index)

Writes:
  data/remediation_training/sop_grounded_remediation_sft.jsonl

Each JSONL row:
  {
    "scenario_id": "...",
    "messages": [
      {"role": "system",    "content": "<system prompt>"},
      {"role": "user",      "content": "<RCA + SOP evidence>"},
      {"role": "assistant", "content": "<template remediation JSON>"}
    ],
    "metadata": {
      "root_cause": "...",
      "rca_source": "...",
      "kb_ids":     [...],
      "evidence_ids": [...]
    }
  }

This file generates training data only.  No training is run.
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
from ai_remediation.prompt_builder import build_remediation_prompt                # noqa: E402
from ai_remediation.template_mode import generate_template_remediation            # noqa: E402
from kb_retrieval.schema import DEFAULT_COLLECTION, DEFAULT_INDEX_DIR             # noqa: E402

_DEFAULT_SCENARIOS = [
    "enterprise_v3_0000",
    "enterprise_v3_0072",
    "enterprise_v3_0073",
    "enterprise_v3_0074",
]

_OUT_FILE = "data/remediation_training/sop_grounded_remediation_sft.jsonl"


def _try_retrieve_kb(context: dict, index_dir: Path, top_k: int) -> list[dict]:
    try:
        from kb_retrieval.retriever import retrieve_kb_evidence
        return retrieve_kb_evidence(
            context=context,
            index_dir=index_dir,
            collection_name=DEFAULT_COLLECTION,
            top_k=top_k,
        )
    except Exception:
        return []


def _inject_kb_evidence(context: dict, kb_evidence: list[dict]) -> dict:
    """Return a copy of context with KB evidence injected into retrieved_graph_memory_evidence."""
    ctx = dict(context)
    existing = list(ctx.get("retrieved_graph_memory_evidence", []) or [])
    ctx["retrieved_graph_memory_evidence"] = existing + kb_evidence
    ctx["retrieved_kb_evidence"] = kb_evidence
    return ctx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SOP-grounded SFT training data from template remediation outputs."
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=_DEFAULT_SCENARIOS, metavar="SCENARIO_ID",
        help="Scenario IDs to include (default: all four).",
    )
    parser.add_argument(
        "--index-dir", default=DEFAULT_INDEX_DIR, metavar="DIR",
        help=f"KB index directory (default: {DEFAULT_INDEX_DIR}).",
    )
    parser.add_argument(
        "--kb-top-k", type=int, default=5, metavar="N",
        help="Number of KB evidence chunks to retrieve per scenario (default: 5).",
    )
    parser.add_argument(
        "--out-file", default=_OUT_FILE, metavar="PATH",
        help=f"Output JSONL path (default: {_OUT_FILE}).",
    )
    parser.add_argument(
        "--no-kb", action="store_true",
        help="Skip KB retrieval — use only RCA context in the training prompts.",
    )
    args = parser.parse_args()

    index_dir = (REPO_ROOT / args.index_dir).resolve()
    out_path  = (REPO_ROOT / args.out_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("====================================================")
    print(" InfraGraph AI — SOP-Grounded Remediation Training Data")
    print("====================================================")
    print(f"Scenarios : {', '.join(args.scenarios)}")
    print(f"Index dir : {index_dir.relative_to(REPO_ROOT)}")
    print(f"KB top-k  : {args.kb_top_k}")
    print(f"KB mode   : {'disabled (--no-kb)' if args.no_kb else 'enabled'}")
    print(f"Output    : {out_path.relative_to(REPO_ROOT)}")
    print()

    rows: list[dict] = []

    for scenario_id in args.scenarios:
        rca_path = REPO_ROOT / "assets/preloaded/enterprise_gnn_rca" / f"{scenario_id}.json"
        if not rca_path.exists():
            print(f"  [SKIP] RCA output not found: {rca_path.relative_to(REPO_ROOT)}")
            continue

        print(f"--- {scenario_id} ---")

        try:
            context = build_enterprise_remediation_context(
                repo_root=REPO_ROOT,
                scenario_id=scenario_id,
                rca_path=rca_path,
            )
        except Exception as exc:
            print(f"  [ERROR] Could not build context: {exc}")
            continue

        # Retrieve KB evidence if enabled
        kb_evidence: list[dict] = []
        if not args.no_kb:
            kb_evidence = _try_retrieve_kb(context, index_dir, args.kb_top_k)
            print(f"  KB evidence : {len(kb_evidence)} chunk(s) retrieved")

        enriched_context = _inject_kb_evidence(context, kb_evidence)

        # Build prompt messages
        messages = build_remediation_prompt(enriched_context)

        # Generate template remediation as training target
        template_out = generate_template_remediation(enriched_context)
        assistant_content = json.dumps(template_out, ensure_ascii=False)

        # Collect evidence IDs and KB IDs for metadata
        evidence_ids: list[str] = []
        kb_ids: list[str] = []
        for item in kb_evidence:
            eid = item.get("evidence_id", "")
            if eid:
                evidence_ids.append(eid)
            kb_id = (item.get("metadata") or {}).get("kb_id", "")
            if kb_id and kb_id not in kb_ids:
                kb_ids.append(kb_id)

        for item in (context.get("causal_evidence", []) or []):
            eid = item.get("evidence_id", "")
            if eid and eid not in evidence_ids:
                evidence_ids.append(eid)

        row = {
            "scenario_id": scenario_id,
            "messages": [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ] + [{"role": "assistant", "content": assistant_content}],
            "metadata": {
                "root_cause":   context.get("root_cause", ""),
                "rca_source":   context.get("rca_source", ""),
                "kb_ids":       kb_ids,
                "evidence_ids": evidence_ids,
                "kb_evidence_count": len(kb_evidence),
            },
        }
        rows.append(row)
        print(f"  Evidence IDs: {evidence_ids}")
        print(f"  KB IDs used : {kb_ids}")
        print()

    if not rows:
        print("[WARN] No training rows generated.")
        sys.exit(0)

    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[PASS] {len(rows)} training row(s) written to "
          f"{out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
