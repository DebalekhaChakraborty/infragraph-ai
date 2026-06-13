#!/usr/bin/env python3
"""
validate_rca_outputs.py — Verify RCA preloaded output files are demo-safe.

By default, scans only the three RCA output directories:
  assets/preloaded/topology_rca_results/
  assets/preloaded/enterprise_gnn_rca/
  assets/preloaded/event_correlation/

Reports under reports/ are NOT scanned — evaluation fields are allowed there.
Event correlation cluster files are checked for the same forbidden keys
(remediation and evaluation leakage); cluster-specific fields such as
cluster_id, cluster_score, correlation_reasons, and causal_evidence are
explicitly allowed.

Exits 0 if all checks pass.  Exits 1 and prints a failure report if any
file contains forbidden remediation or evaluation-leakage keys.

Usage:
  python scripts/validate_rca_outputs.py
  python scripts/validate_rca_outputs.py --verbose
  python scripts/validate_rca_outputs.py \\
      --scan-dir assets/preloaded/topology_rca_results \\
      --scan-dir assets/preloaded/enterprise_gnn_rca \\
      --scan-dir assets/preloaded/event_correlation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_SCAN_DIRS: list[str] = [
    "assets/preloaded/topology_rca_results",
    "assets/preloaded/enterprise_gnn_rca",
    "assets/preloaded/event_correlation",
]

# Keys that must NEVER appear anywhere in a preloaded RCA output file
_FORBIDDEN_REMEDIATION: frozenset[str] = frozenset({
    "remediation",
    "recommended_actions",
    "remediation_steps",
    "resolution_steps",
    "rollback_steps",
    "validation_steps",
    "servicenow_incident_summary",
    "resolution",
    "rollback",
})

# Evaluation leakage keys that must not appear in preloaded files
_FORBIDDEN_EVALUATION: frozenset[str] = frozenset({
    "expected_root_cause",
    "ground_truth_node",
    "correct_top1",
    "correct_top_k",
    "reciprocal_rank",
    "evaluation",   # entire eval block must not appear in preloaded output
})

_ALL_FORBIDDEN: frozenset[str] = _FORBIDDEN_REMEDIATION | _FORBIDDEN_EVALUATION

# "correct" is too generic to check recursively — only block at top level
_FORBIDDEN_TOP_LEVEL: frozenset[str] = frozenset({"correct"})


def _collect_all_keys(obj: object, depth: int = 0) -> set[str]:
    """Recursively collect every dict key in a JSON structure."""
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

    all_keys = _collect_all_keys(data)
    top_keys = set(data.keys()) if isinstance(data, dict) else set()

    for key in _ALL_FORBIDDEN:
        if key in all_keys:
            violations.append(f"forbidden key found: {key!r}")

    for key in _FORBIDDEN_TOP_LEVEL:
        if key in top_keys:
            violations.append(f"forbidden top-level key: {key!r}")

    return violations


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        try:
            return str(path.relative_to(base.parent))
        except ValueError:
            return str(path)


def _scan_directory(scan_dir: Path) -> dict[str, list[str]]:
    """Scan all *.json files under scan_dir and return {rel_path: [violations]}."""
    results: dict[str, list[str]] = {}
    for json_file in sorted(scan_dir.rglob("*.json")):
        rel = _rel(json_file, scan_dir)
        viol = _check_file(json_file)
        if viol:
            results[rel] = viol
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RCA preloaded output files for demo safety."
    )
    parser.add_argument(
        "--scan-dir", dest="scan_dirs", action="append", default=None,
        metavar="DIR",
        help="Directory to scan (repeatable).  Default: "
             + " and ".join(_DEFAULT_SCAN_DIRS),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print each file checked (pass or fail)",
    )
    args = parser.parse_args()

    target_dirs: list[str] = args.scan_dirs if args.scan_dirs else _DEFAULT_SCAN_DIRS

    all_failures: dict[str, list[str]] = {}
    total_files = 0

    for dir_str in target_dirs:
        scan_dir = (REPO_ROOT / dir_str).resolve()
        if not scan_dir.exists():
            print(f"[INFO] Directory does not exist, skipping: {dir_str}")
            continue

        dir_files = list(scan_dir.rglob("*.json"))
        total_files += len(dir_files)

        if args.verbose:
            for json_file in sorted(dir_files):
                rel = _rel(json_file, scan_dir)
                viol = _check_file(json_file)
                if viol:
                    print(f"  FAIL  {rel}")
                else:
                    print(f"  ok    {rel}")

        failures = _scan_directory(scan_dir)
        all_failures.update(failures)

    if all_failures:
        scanned_label = ", ".join(f"'{d}'" for d in target_dirs)
        print(f"\n[FAIL] {len(all_failures)} file(s) contain forbidden keys "
              f"(checked {total_files} file(s) across {scanned_label})")
        print()
        for path, viols in sorted(all_failures.items()):
            print(f"  {path}")
            for v in viols:
                print(f"    - {v}")
        print()
        print("Fix: re-run the predict scripts without --with-eval to regenerate "
              "clean output files.  Evaluation data belongs in reports/, not "
              "assets/preloaded/.")
        sys.exit(1)
    else:
        scanned_label = ", ".join(f"'{d}'" for d in target_dirs)
        print(f"[PASS] All {total_files} file(s) across {scanned_label} are demo-safe.")
        sys.exit(0)


if __name__ == "__main__":
    main()
