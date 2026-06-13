#!/usr/bin/env python3
"""
validate_remediation_outputs.py — Verify remediation output files are correct.

Default scan:
  assets/preloaded/remediation/

Validates each JSON for:
  - Required top-level envelope keys
  - Required nested remediation keys
  - Non-empty mandatory list fields
  - Non-empty ServiceNow short_description
  - No forbidden evaluation-leakage keys anywhere

Does NOT scan RCA output directories — those are checked by validate_rca_outputs.py.

Exits 0 on pass, 1 on failure.

Usage:
  python scripts/validate_remediation_outputs.py
  python scripts/validate_remediation_outputs.py --verbose
  python scripts/validate_remediation_outputs.py --scan-dir assets/preloaded/remediation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_SCAN_DIR = "assets/preloaded/remediation"

# Required top-level envelope keys
_REQUIRED_ENVELOPE_KEYS: frozenset[str] = frozenset({
    "scenario_id",
    "case_id",
    "incident_id",
    "scope",
    "rca_source",
    "cluster_id",
    "remediation_source",
    "ok",
    "remediation",
})

# Required keys inside the "remediation" dict
_REQUIRED_REMEDIATION_KEYS: frozenset[str] = frozenset({
    "executive_summary",
    "probable_root_cause",
    "scope",
    "risk_level",
    "automation_eligibility",
    "blast_radius",
    "evidence_from_graph",
    "pre_checks",
    "triage_steps",
    "validation_steps",
    "remediation_steps",
    "post_checks",
    "do_not_execute_if",
    "rollback_or_safety_notes",
    "escalation_recommendation",
    "servicenow_incident_summary",
    "audit_summary",
    "confidence_notes",
})

# Evaluation-leakage keys that must never appear anywhere
_FORBIDDEN_KEYS: frozenset[str] = frozenset({
    "expected_root_cause",
    "ground_truth_node",
    "correct_top1",
    "correct_top_k",
    "reciprocal_rank",
    "evaluation",
})

# List fields inside "remediation" that must be non-empty
_REQUIRED_NON_EMPTY_LISTS: frozenset[str] = frozenset({
    "remediation_steps",
    "validation_steps",
    "rollback_or_safety_notes",
})


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


def _check_file(path: Path) -> list[str]:
    """Return list of violation messages for this file (empty = clean)."""
    violations: list[str] = []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Could not parse JSON: {exc}"]

    if not isinstance(data, dict):
        return ["Top-level object is not a dict"]

    # Required envelope keys
    for key in _REQUIRED_ENVELOPE_KEYS:
        if key not in data:
            violations.append(f"missing required envelope key: {key!r}")

    # Forbidden keys anywhere
    all_keys = _collect_all_keys(data)
    for key in _FORBIDDEN_KEYS:
        if key in all_keys:
            violations.append(f"forbidden key found: {key!r}")

    # Validate nested remediation dict
    rem = data.get("remediation")
    if not isinstance(rem, dict):
        violations.append("'remediation' must be a non-null dict")
        return violations

    for key in _REQUIRED_REMEDIATION_KEYS:
        if key not in rem:
            violations.append(f"remediation missing required key: {key!r}")

    for key in _REQUIRED_NON_EMPTY_LISTS:
        val = rem.get(key, [])
        if not isinstance(val, list) or len(val) == 0:
            violations.append(f"remediation[{key!r}] must be a non-empty list")

    # ServiceNow short_description must be non-empty
    snow = rem.get("servicenow_incident_summary", {})
    if not isinstance(snow, dict) or not snow.get("short_description", "").strip():
        violations.append(
            "remediation.servicenow_incident_summary.short_description must be non-empty"
        )

    # evidence_from_graph must be non-empty
    efg = rem.get("evidence_from_graph", [])
    if not isinstance(efg, list) or len(efg) == 0:
        violations.append("remediation.evidence_from_graph must be a non-empty list")

    # KB evidence check: if kb_evidence_count > 0, at least one KB-* ID must appear somewhere
    input_summary = data.get("input_context_summary", {})
    kb_count = input_summary.get("kb_evidence_count", data.get("kb_evidence_count", 0))
    if isinstance(kb_count, int) and kb_count > 0:
        kb_id_found = _has_kb_evidence_id(rem)
        if not kb_id_found:
            violations.append(
                f"kb_evidence_count={kb_count} but no KB-* evidence ID found in "
                "evidence_ids_used, evidence_from_graph, audit_summary, or confidence_notes"
            )

    return violations


def _has_kb_evidence_id(rem: dict) -> bool:
    """Return True if any KB-* evidence ID appears in key remediation fields."""
    def _contains_kb(value) -> bool:
        if isinstance(value, str):
            return "KB-" in value
        if isinstance(value, list):
            return any(_contains_kb(v) for v in value)
        return False

    for field in ("evidence_ids_used", "evidence_from_graph", "audit_summary", "confidence_notes"):
        if _contains_kb(rem.get(field)):
            return True
    return False


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate remediation preloaded output files."
    )
    parser.add_argument(
        "--scan-dir", default=_DEFAULT_SCAN_DIR, metavar="DIR",
        help=f"Directory to scan (default: {_DEFAULT_SCAN_DIR}).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print each file checked (pass or fail).",
    )
    args = parser.parse_args()

    scan_dir = (REPO_ROOT / args.scan_dir).resolve()
    if not scan_dir.exists():
        print(f"[INFO] Directory does not exist, skipping: {args.scan_dir}")
        print(f"[PASS] 0 file(s) checked (directory not present).")
        sys.exit(0)

    json_files = sorted(scan_dir.rglob("*.json"))
    if not json_files:
        print(f"[INFO] No JSON files found in: {args.scan_dir}")
        print("[PASS] 0 file(s) checked.")
        sys.exit(0)

    failures: dict[str, list[str]] = {}

    for path in json_files:
        rel = _rel(path)
        viols = _check_file(path)
        if args.verbose:
            if viols:
                print(f"  FAIL  {rel}")
            else:
                print(f"  ok    {rel}")
        if viols:
            failures[rel] = viols

    if failures:
        print(f"\n[FAIL] {len(failures)} file(s) have violations "
              f"(checked {len(json_files)} file(s) in '{args.scan_dir}')")
        print()
        for rel_path, viols in sorted(failures.items()):
            print(f"  {rel_path}")
            for v in viols:
                print(f"    - {v}")
        print()
        print("Fix: re-run generate_remediation_demo_assets.py to regenerate clean outputs.")
        sys.exit(1)
    else:
        print(f"[PASS] All {len(json_files)} file(s) in '{args.scan_dir}' are valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()
