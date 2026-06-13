#!/usr/bin/env python3
"""
expand_sop_grounded_qwen_training_data.py

Generate an expanded SOP-grounded Qwen SFT dataset by producing safe
synthetic variants of the 4 base enterprise RCA scenarios.

Each variant modifies only presentation-level or non-contradictory fields:
  - Incident ID and cluster ID suffixes
  - Cluster score (±0.03, clamped to 0.50–0.99)
  - Correlation reasons (word-level paraphrase, meaning preserved)
  - Alert event severity (within domain-safe downstep bounds)
  - Candidate confidence scores (nudged, rank order preserved)
  - KB evidence subset size (3 to --kb-top-k from ranked pool)

Unchanged across all variants:
  root_cause, root_cause_diagram, rca_source, impacted_nodes,
  impacted_diagrams, impact_path, causal_evidence (CE-* IDs),
  domain-specific remediation class, KB domain alignment

Never reads: labels.json, ground-truth files, evaluation outputs.
Never introduces: fake IPs, fake commands, unsupported device facts.

Output:
  data/qwen_sop_grounded_expanded/train.jsonl
  data/qwen_sop_grounded_expanded/val.jsonl
  data/qwen_sop_grounded_expanded/dataset_summary.json
  data/qwen_sop_grounded_expanded/previews/  (--pretty)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_remediation.context_builder import build_enterprise_remediation_context  # noqa: E402
from ai_remediation.template_mode import generate_template_remediation            # noqa: E402
from kb_retrieval.evidence_ordering import (                                      # noqa: E402
    DOMAIN_EXPECTED_FIRST_KB,
    apply_domain_first_ordering,
    infer_domain,
)
from kb_retrieval.schema import DEFAULT_INDEX_DIR                                 # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_SCENARIOS = [
    "enterprise_v3_0000",
    "enterprise_v3_0072",
    "enterprise_v3_0073",
    "enterprise_v3_0074",
]

_DEFAULT_OUT_DIR       = "data/qwen_sop_grounded_expanded"
_RECORDS_PER_SCENARIO  = 25

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

# ── Paraphrase helpers ────────────────────────────────────────────────────────

# Word-pair substitutions for correlation reason paraphrasing.
# Bidirectional: applying the same list twice round-trips back to the original.
_PARAPHRASE_PAIRS: list[tuple[str, str]] = [
    ("correlated",    "co-occurring"),
    ("co-occurring",  "correlated"),
    ("detected",      "observed"),
    ("observed",      "detected"),
    ("spike",         "surge"),
    ("surge",         "spike"),
    ("simultaneously", "concurrently"),
    ("concurrently",  "simultaneously"),
    ("failure",       "anomaly"),
    ("anomaly",       "event"),
    ("exceeded",      "breached"),
    ("breached",      "exceeded"),
    ("elevated",      "increased"),
    ("increased",     "elevated"),
    ("indicates",     "suggests"),
    ("suggests",      "indicates"),
    ("degraded",      "impacted"),
    ("impacted",      "affected"),
    ("affected",      "degraded"),
]

# Safe severity variants: weighted toward original, allow one-step downgrade.
_SEVERITY_VARIANTS: dict[str, list[str]] = {
    "critical": ["critical", "critical", "critical", "high"],
    "high":     ["high", "high", "medium"],
    "warning":  ["warning", "medium"],
    "medium":   ["medium", "low"],
    "low":      ["low"],
}


def _paraphrase_reason(reason: str, rng: random.Random) -> str:
    """Apply 0–2 word substitutions to paraphrase a correlation reason."""
    result = reason
    applicable = [
        (old, new) for old, new in _PARAPHRASE_PAIRS
        if old.lower() in result.lower()
    ]
    rng.shuffle(applicable)
    n_changes = rng.randint(0, min(2, len(applicable)))
    for old, new in applicable[:n_changes]:
        idx = result.lower().find(old.lower())
        if idx < 0:
            continue
        if result[idx].isupper():
            replacement = new[0].upper() + new[1:]
        else:
            replacement = new
        result = result[:idx] + replacement + result[idx + len(old):]
    return result


def _vary_severity(severity: str, rng: random.Random) -> str:
    variants = _SEVERITY_VARIANTS.get(severity.lower(), [severity])
    return rng.choice(variants)


def _vary_alert_events(events: list[dict], rng: random.Random) -> list[dict]:
    """Return safe copies with severity lightly varied; alert_type and node preserved."""
    result = []
    for ev in events:
        vev = dict(ev)
        if "severity" in vev:
            vev["severity"] = _vary_severity(str(vev["severity"]), rng)
        result.append(vev)
    return result


def _vary_candidate_scores(candidates: list[dict], rng: random.Random) -> list[dict]:
    """Nudge top-level score values; all other fields (node_id, rank, evidence) unchanged."""
    if not candidates:
        return candidates
    varied = []
    for i, c in enumerate(candidates):
        vc = dict(c)
        orig = float(vc.get("score", 0.5))
        # Top candidate: small nudge; others: wider range
        delta = rng.uniform(-0.02, 0.02) if i == 0 else rng.uniform(-0.05, 0.05)
        vc["score"] = round(max(0.01, min(0.99, orig + delta)), 4)
        varied.append(vc)
    # Enforce: top candidate must retain the highest score
    if len(varied) > 1:
        max_rest = max(c.get("score", 0.0) for c in varied[1:])
        if varied[0].get("score", 0.0) <= max_rest:
            varied[0]["score"] = round(max_rest + rng.uniform(0.03, 0.08), 4)
    return varied


# ── Context variation ─────────────────────────────────────────────────────────

def _make_variant_context(
    base_context: dict,
    base_kb_evidence: list[dict],
    base_scenario_id: str,
    variant_index: int,
    seed: int,
    kb_top_k: int,
) -> dict:
    """
    Build a safe synthetic variant of the base context.

    Returns a new dict with modified presentation-level fields only.
    The base_context is never mutated.
    """
    rng = random.Random(seed + variant_index * 31337)

    var_tag         = f"var_{variant_index:03d}"
    synthetic_sid   = f"{base_scenario_id}_{var_tag}"

    # Cluster score: ±0.03, clamped to [0.50, 0.99]
    base_score   = base_context.get("cluster_score") or 0.75
    cluster_score = round(
        max(0.50, min(0.99, base_score + rng.uniform(-0.03, 0.03))), 4
    )
    cluster_id = f"{base_context.get('cluster_id', 'CLU-unknown')}-{var_tag}"

    # Paraphrase correlation reasons (meaning preserved)
    correlation_reasons = [
        _paraphrase_reason(r, rng)
        for r in (base_context.get("correlation_reasons", []) or [])
    ]

    # Vary alert severity within safe bounds; alert_type stays exactly the same
    alert_timeline = _vary_alert_events(
        base_context.get("alert_timeline", []) or [], rng
    )

    # Nudge candidate scores; rank order and node_id preserved
    candidate_ranking = _vary_candidate_scores(
        base_context.get("candidate_ranking", []) or [], rng
    )

    # KB evidence: select 3..kb_top_k items from the ranked pool
    min_k    = max(1, kb_top_k - 2)
    subset_k = rng.choice(list(range(min_k, kb_top_k + 1)))
    kb_subset = base_kb_evidence[:min(subset_k, len(base_kb_evidence))]

    # Build variant — start from base, replace only safe fields
    varied = dict(base_context)
    varied.update({
        "incident_id":         f"INC-{synthetic_sid}",
        "scenario_id":         synthetic_sid,
        "cluster_id":          cluster_id,
        "cluster_score":       cluster_score,
        "correlation_reasons": correlation_reasons,
        "alert_timeline":      alert_timeline,
        "candidate_ranking":   candidate_ranking,
    })

    # Replace KB evidence with the (possibly smaller) subset.
    # retrieved_graph_memory_evidence contains only KB items in this pipeline.
    varied["retrieved_graph_memory_evidence"] = kb_subset
    varied["retrieved_kb_evidence"]           = kb_subset

    return varied


# ── User / assistant content ──────────────────────────────────────────────────

def _slim_kb_evidence(kb_evidence: list[dict]) -> list[dict]:
    """Return KB evidence with only the fields needed in the training prompt."""
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


def _build_user_obj(
    context: dict,
    base_scenario_id: str,
    variant_id: str,
) -> dict:
    kb_evidence = context.get("retrieved_kb_evidence", []) or []
    return {
        "incident_id":          context.get("incident_id", ""),
        "scenario_id":          context.get("scenario_id", ""),
        "base_scenario_id":     base_scenario_id,
        "synthetic_variant_id": variant_id,
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


# ── Validation ────────────────────────────────────────────────────────────────

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


def _validate_record(
    record: dict,
    base_root_cause: str,
    strict_kb: bool,
) -> list[str]:
    """Return violation messages (empty list = clean)."""
    violations: list[str] = []

    msgs = record.get("messages", [])
    if len(msgs) != 3:
        violations.append(f"messages length must be 3, got {len(msgs)}")
        return violations

    if [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
        violations.append("message roles must be system / user / assistant")

    user_obj: dict = {}
    try:
        user_obj = json.loads(msgs[1]["content"])
    except Exception as exc:
        violations.append(f"user content is not valid JSON: {exc}")

    asst_obj: dict = {}
    try:
        asst_obj = json.loads(msgs[2]["content"])
    except Exception as exc:
        violations.append(f"assistant content is not valid JSON: {exc}")

    # Root cause preserved
    if user_obj.get("root_cause") != base_root_cause:
        violations.append(
            f"root_cause mismatch: got {user_obj.get('root_cause')!r}, "
            f"expected {base_root_cause!r}"
        )

    # Top candidate must be root cause (if candidates exist and have node_id)
    candidates = user_obj.get("candidate_ranking", [])
    if candidates:
        top_id = (
            candidates[0].get("node_id")
            or candidates[0].get("node", "")
        )
        if top_id and top_id != base_root_cause:
            violations.append(
                f"top candidate {top_id!r} != root_cause {base_root_cause!r}"
            )

    # evidence_ids_used
    ev_ids: list[str] = []
    if "evidence_ids_used" not in asst_obj:
        violations.append("assistant output missing 'evidence_ids_used'")
    else:
        ev_ids = [str(x) for x in asst_obj["evidence_ids_used"]]
        if strict_kb and not any(x.startswith("KB-") for x in ev_ids):
            violations.append("strict_kb: no KB-* ID in evidence_ids_used")
        if not any(x.startswith("CE-") for x in ev_ids):
            violations.append("no CE-* ID in evidence_ids_used")
        # Domain-first KB ordering check (only when KB evidence is present)
        kb_ids = [x for x in ev_ids if x.startswith("KB-")]
        if kb_ids:
            domain = infer_domain(base_root_cause)
            expected = DOMAIN_EXPECTED_FIRST_KB.get(domain, ())
            if expected and not kb_ids[0].startswith(expected):
                violations.append(
                    f"domain-first ordering: first KB ID {kb_ids[0]!r} "
                    f"does not match expected prefixes {expected} "
                    f"for root_cause {base_root_cause!r} (domain={domain!r})"
                )

    # Non-empty required list fields
    for field in ("remediation_steps", "validation_steps", "rollback_or_safety_notes"):
        if not asst_obj.get(field):
            violations.append(f"assistant output '{field}' is empty or missing")

    snow = asst_obj.get("servicenow_incident_summary", {})
    if not (snow and snow.get("short_description", "").strip()):
        violations.append("servicenow_incident_summary.short_description is empty")

    # No forbidden leakage keys anywhere in the record
    all_keys = _collect_all_keys(record)
    for fk in _FORBIDDEN_KEYS:
        if fk in all_keys:
            violations.append(f"forbidden key found: {fk!r}")

    return violations


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate expanded SOP-grounded Qwen SFT training data from "
            "safe synthetic variants of the 4 base enterprise scenarios."
        )
    )
    parser.add_argument(
        "--base-scenarios", nargs="+", default=_DEFAULT_SCENARIOS, metavar="ID",
        help="Base scenario IDs (default: all four enterprise_v3 scenarios).",
    )
    parser.add_argument(
        "--records-per-scenario", type=int, default=_RECORDS_PER_SCENARIO, metavar="N",
        help=f"Synthetic variants per scenario (default: {_RECORDS_PER_SCENARIO}).",
    )
    parser.add_argument(
        "--out-dir", default=_DEFAULT_OUT_DIR, metavar="DIR",
        help=f"Output directory (default: {_DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--kb-index-dir", default=DEFAULT_INDEX_DIR, metavar="DIR",
        help=f"KB vector index directory (default: {DEFAULT_INDEX_DIR}).",
    )
    parser.add_argument(
        "--kb-top-k", type=int, default=5, metavar="N",
        help="KB chunks to retrieve per scenario (default: 5).",
    )
    parser.add_argument(
        "--strict-kb", action="store_true",
        help="Fail if no KB evidence is retrieved for any scenario.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed for reproducible variation (default: 42).",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Write per-record pretty-printed JSON preview files.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.85, metavar="RATIO",
        help="Train split ratio (default: 0.85).",
    )
    args = parser.parse_args()

    out_dir      = (REPO_ROOT / args.out_dir).resolve()
    kb_index_dir = (REPO_ROOT / args.kb_index_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_expected = len(args.base_scenarios) * args.records_per_scenario
    print("=" * 60)
    print(" InfraGraph AI -- Expanded SOP-Grounded Qwen Training Data")
    print("=" * 60)
    print(f"Base scenarios : {', '.join(args.base_scenarios)}")
    print(f"Records/scenario: {args.records_per_scenario}")
    print(f"Expected total : {n_expected}")
    print(f"KB index       : {kb_index_dir.relative_to(REPO_ROOT)}")
    print(f"KB top-k       : {args.kb_top_k}")
    print(f"strict-kb      : {args.strict_kb}")
    print(f"seed           : {args.seed}")
    print(f"train-ratio    : {args.train_ratio}")
    print(f"Output dir     : {out_dir.relative_to(REPO_ROOT)}")
    print()

    if args.pretty:
        preview_dir = out_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    skipped_scenarios: list[str] = []

    for scenario_id in args.base_scenarios:
        rca_path = (
            REPO_ROOT / "assets" / "preloaded" / "enterprise_gnn_rca"
            / f"{scenario_id}.json"
        )
        if not rca_path.exists():
            print(f"[SKIP] RCA file not found: {rca_path.relative_to(REPO_ROOT)}")
            skipped_scenarios.append(scenario_id)
            continue

        print(f"=== {scenario_id} ===")

        # Load base context once per scenario
        try:
            base_context = build_enterprise_remediation_context(
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
            skipped_scenarios.append(scenario_id)
            continue

        base_root_cause  = base_context.get("root_cause", "")
        base_kb_evidence = base_context.get("retrieved_kb_evidence", []) or []
        base_ce_count    = len(base_context.get("causal_evidence", []) or [])
        print(f"  root_cause     : {base_root_cause}")
        print(f"  KB chunks      : {len(base_kb_evidence)}")
        print(f"  causal evidence: {base_ce_count}")

        scenario_records: list[dict] = []
        n_valid = 0
        n_invalid = 0

        for vi in range(args.records_per_scenario):
            var_tag       = f"var_{vi:03d}"
            synthetic_sid = f"{scenario_id}_{var_tag}"
            record_id     = f"sop_grounded_{synthetic_sid}"

            varied_context = _make_variant_context(
                base_context=base_context,
                base_kb_evidence=base_kb_evidence,
                base_scenario_id=scenario_id,
                variant_index=vi,
                seed=args.seed,
                kb_top_k=args.kb_top_k,
            )

            # Apply domain-first KB evidence ordering before building content
            apply_domain_first_ordering(varied_context, base_root_cause)

            # Build user message content
            user_obj = _build_user_obj(varied_context, scenario_id, var_tag)

            # Generate deterministic target remediation
            remediation_out = generate_template_remediation(varied_context)
            remediation_out.pop("source", None)  # strip internal label

            user_content = json.dumps(user_obj, ensure_ascii=False, separators=(",", ":"))
            asst_content = json.dumps(remediation_out, ensure_ascii=False, separators=(",", ":"))

            kb_ids: list[str] = []
            for item in (varied_context.get("retrieved_kb_evidence", []) or []):
                kb_id = (item.get("metadata") or {}).get("kb_id", "")
                if kb_id and kb_id not in kb_ids:
                    kb_ids.append(kb_id)

            record: dict = {
                "id": record_id,
                "messages": [
                    {"role": "system",    "content": _SYSTEM_PROMPT},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": asst_content},
                ],
                "metadata": {
                    "base_scenario_id":    scenario_id,
                    "synthetic_scenario_id": synthetic_sid,
                    "variant_id":          var_tag,
                    "root_cause":          base_root_cause,
                    "root_cause_diagram":  base_context.get("root_cause_diagram", ""),
                    "rca_source":          base_context.get("rca_source", ""),
                    "cluster_id":          varied_context.get("cluster_id", ""),
                    "cluster_score":       varied_context.get("cluster_score"),
                    "kb_evidence_count":   len(varied_context.get("retrieved_kb_evidence", []) or []),
                    "causal_evidence_count": base_ce_count,
                    "target_source":       "template_sop_grounded_synthetic",
                    "synthetic_generation_policy": "safe_context_variation_no_ground_truth",
                },
            }

            violations = _validate_record(record, base_root_cause, args.strict_kb)
            if violations:
                n_invalid += 1
                if n_invalid <= 3:  # only print first few failures
                    print(f"  [WARN] {record_id}: {violations[0]}")
                continue

            scenario_records.append(record)
            n_valid += 1

            if args.pretty:
                (preview_dir / f"{record_id}_input.json").write_text(
                    json.dumps(user_obj, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (preview_dir / f"{record_id}_target.json").write_text(
                    json.dumps(remediation_out, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        print(f"  Generated      : {n_valid} valid, {n_invalid} invalid")
        all_records.extend(scenario_records)
        print()

    if not all_records:
        print("[ERROR] No training records generated.")
        sys.exit(1)

    # Shuffle then split
    rng_split = random.Random(args.seed)
    rng_split.shuffle(all_records)

    n_total = len(all_records)
    n_train = max(1, round(n_total * args.train_ratio))
    if n_total > 1:
        n_train = min(n_train, n_total - 1)

    train_records = all_records[:n_train]
    val_records   = all_records[n_train:]

    if not val_records:
        print("[ERROR] Not enough records for a val split.")
        sys.exit(1)

    # Write JSONL
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
    from collections import Counter
    base_counts = Counter(r["metadata"]["base_scenario_id"] for r in all_records)
    summary = {
        "total_records":      n_total,
        "train_records":      len(train_records),
        "val_records":        len(val_records),
        "records_per_base":   dict(base_counts),
        "base_scenarios":     args.base_scenarios,
        "skipped_scenarios":  skipped_scenarios,
        "records_per_scenario": args.records_per_scenario,
        "kb_top_k":           args.kb_top_k,
        "strict_kb":          args.strict_kb,
        "seed":               args.seed,
        "train_ratio":        args.train_ratio,
        "target_source":      "template_sop_grounded_synthetic",
        "synthetic_generation_policy": "safe_context_variation_no_ground_truth",
        "output_dir":         str(out_dir.relative_to(REPO_ROOT)),
    }
    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"[PASS] {n_total} total records generated.")
    print(f"  train : {len(train_records)} -> {train_path.relative_to(REPO_ROOT)}")
    print(f"  val   : {len(val_records)}  -> {val_path.relative_to(REPO_ROOT)}")
    print(f"  per base scenario: {dict(base_counts)}")
    print(f"  summary -> {summary_path.relative_to(REPO_ROOT)}")
    if args.pretty:
        print(f"  previews -> {preview_dir.relative_to(REPO_ROOT)}/")
    if skipped_scenarios:
        print(f"  skipped: {skipped_scenarios}")


if __name__ == "__main__":
    main()
