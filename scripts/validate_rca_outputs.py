#!/usr/bin/env python3
"""
validate_rca_outputs.py — Verify RCA output files are demo-safe.

Scans:
  assets/preloaded/topology_rca_results/*.json
  assets/preloaded/enterprise_gnn_rca/*.json

Exits 0 if all checks pass.  Exits 1 and prints a failure report if any
output file contains forbidden remediation or evaluation-leakage keys.

Reports in reports/ are NOT scanned here — they are allowed to contain
evaluation fields.

Usage:
  python scripts/validate_rca_outputs.py
  python scripts/validate_rca_outputs.py --scan-dir assets/preloaded
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Keys that must NEVER appear anywhere in a preloaded output file (recursive check)
_FORBIDDEN_REMEDIATION: frozenset[str] = frozenset({
    "recommended_actions",
    "remediation_steps",
    "resolution_steps",
    "rollback_steps",
    "validation_steps",
    "servicenow_incident_summary",
    "resolution",
    "rollback",
})

# Evaluation leakage keys that must not appear in preloaded files.
# (They are allowed in reports/ evaluation JSONs.)
_FORBIDDEN_EVALUATION: frozenset[str] = frozenset({
    "expected_root_cause",
    "ground_truth_node",
    "correct_top1",
    "correct_top_k",
    "reciprocal_rank",
    "evaluation",   # entire eval block must not appear in preloaded output
})

# For "correct" specifically, check top-level only (it's too generic a word)
_FORBIDDEN_TOP_LEVEL: frozenset[str] = frozenset({"correct"})

_ALL_FORBIDDEN: frozenset[str] = _FORBIDDEN_REMEDIATION | _FORBIDDEN_EVALUATION


def _collect_all_keys(obj: object, depth: int = 0) -> set[str]:
    """Recursively collect every dict key in a JSON structure."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            if depth < 20:  # guard against pathological nesting
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

    all_keys    = _collect_all_keys(data)
    top_keys    = set(data.keys()) if isinstance(data, dict) else set()

    for key in _ALL_FORBIDDEN:
        if key in all_keys:
            violations.append(f"forbidden key found: {key!r}")

    for key in _FORBIDDEN_TOP_LEVEL:
        if key in top_keys:
            violations.append(f"forbidden top-level key: {key!r}")

    return violations


def _scan_directory(scan_dir: Path) -> dict[str, list[str]]:
    """Scan all *.json files under scan_dir and return {path: [violations]}."""
    results: dict[str, list[str]] = {}
    for json_file in sorted(scan_dir.rglob("*.json")):
        try:
            rel = str(json_file.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(json_file.relative_to(scan_dir.parent))
        viol = _check_file(json_file)
        if viol:
            results[rel] = viol
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RCA output files for demo safety."
    )
    parser.add_argument(
        "--scan-dir",
        default="assets/preloaded",
        help="Directory to scan (default: assets/preloaded)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print each file checked (pass or fail)",
    )
    args = parser.parse_args()

    scan_dir = (REPO_ROOT / args.scan_dir).resolve()
    if not scan_dir.exists():
        print(f"[INFO] Scan directory does not exist: {scan_dir}")
        print("       Nothing to validate.")
        sys.exit(0)

    failures = _scan_directory(scan_dir)

    # Count total files checked
    total = sum(1 for _ in scan_dir.rglob("*.json"))

    if args.verbose:
        for json_file in sorted(scan_dir.rglob("*.json")):
            try:
                rel = str(json_file.relative_to(REPO_ROOT))
            except ValueError:
                rel = str(json_file.relative_to(scan_dir.parent))
            if rel in failures:
                print(f"  FAIL  {rel}")
            else:
                print(f"  ok    {rel}")

    if failures:
        print(f"\n[FAIL] {len(failures)} file(s) contain forbidden keys "
              f"(checked {total} file(s) in {args.scan_dir})")
        print()
        for path, viols in sorted(failures.items()):
            print(f"  {path}")
            for v in viols:
                print(f"    - {v}")
        print()
        print("Fix: re-run the predict scripts without --with-eval to regenerate "
              "clean output files.  Evaluation data belongs in reports/, not "
              "assets/preloaded/.")
        sys.exit(1)
    else:
        print(f"[PASS] All {total} file(s) in {args.scan_dir!r} are demo-safe.")
        sys.exit(0)


if __name__ == "__main__":
    main()
