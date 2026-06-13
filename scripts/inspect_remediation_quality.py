#!/usr/bin/env python3
"""
inspect_remediation_quality.py — Print per-scenario remediation quality summary.

For each remediation JSON in assets/preloaded/remediation/, prints:
  - scenario_id
  - root_cause
  - Top KB evidence IDs and titles
  - First 3 remediation steps
  - First 3 validation steps

Usage:
  python scripts/inspect_remediation_quality.py
  python scripts/inspect_remediation_quality.py --scan-dir assets/preloaded/remediation
  python scripts/inspect_remediation_quality.py --scenarios enterprise_v3_0072 enterprise_v3_0073
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SCAN_DIR = "assets/preloaded/remediation"


def _kb_items(rem: dict) -> list[dict]:
    """Extract KB-* evidence entries from evidence_from_graph."""
    out: list[dict] = []
    for entry in rem.get("evidence_from_graph", []):
        if isinstance(entry, str) and "KB-" in entry:
            out.append({"raw": entry})
    return out


def _kb_ids_from_evidence(rem: dict) -> list[str]:
    """Collect all KB-* IDs from evidence_ids_used."""
    return [
        eid for eid in (rem.get("evidence_ids_used") or [])
        if isinstance(eid, str) and eid.startswith("KB-")
    ]


def _print_scenario(data: dict, verbose: bool) -> None:
    sid = data.get("scenario_id", "—")
    summary = data.get("input_context_summary", {})
    root_cause = summary.get("root_cause", data.get("root_cause", "—"))
    rc_diagram = summary.get("root_cause_diagram", "—")
    kb_count = summary.get("kb_evidence_count", 0)
    rca_source = data.get("rca_source", "—")

    rem = data.get("remediation", {})
    kb_ids = _kb_ids_from_evidence(rem)
    kb_entries = _kb_items(rem)

    print("-" * 70)
    print(f"  scenario     : {sid}")
    print(f"  root_cause   : {root_cause}  (diagram: {rc_diagram})")
    print(f"  rca_source   : {rca_source}")
    print(f"  KB chunks    : {kb_count}")

    if kb_ids:
        print(f"  KB IDs used  : {', '.join(kb_ids[:5])}")
    else:
        print("  KB IDs used  : (none in evidence_ids_used)")

    if kb_entries:
        print(f"  KB evidence in graph ({min(len(kb_entries), 3)} shown):")
        for entry in kb_entries[:3]:
            print(f"    - {entry['raw'][:100]}")
    else:
        print("  KB evidence in graph: (none)")

    rem_steps = rem.get("remediation_steps", [])
    print(f"  remediation_steps ({len(rem_steps)} total, first 3):")
    for step in rem_steps[:3]:
        print(f"    - {step[:110]}")

    val_steps = rem.get("validation_steps", [])
    print(f"  validation_steps ({len(val_steps)} total, first 3):")
    for step in val_steps[:3]:
        print(f"    - {step[:110]}")

    if verbose:
        rollback = rem.get("rollback_or_safety_notes", [])
        print(f"  rollback_notes ({len(rollback)} total, first 2):")
        for note in rollback[:2]:
            print(f"    - {note[:110]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect remediation output quality per scenario."
    )
    parser.add_argument(
        "--scan-dir", default=_DEFAULT_SCAN_DIR, metavar="DIR",
        help=f"Directory to scan (default: {_DEFAULT_SCAN_DIR}).",
    )
    parser.add_argument(
        "--scenarios", nargs="+", metavar="ID",
        help="Limit to specific scenario IDs.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Also print rollback notes.",
    )
    args = parser.parse_args()

    scan_dir = (REPO_ROOT / args.scan_dir).resolve()
    if not scan_dir.exists():
        print(f"[ERROR] Directory not found: {args.scan_dir}")
        sys.exit(1)

    json_files = sorted(scan_dir.rglob("*.json"))
    if not json_files:
        print(f"[INFO] No JSON files found in: {args.scan_dir}")
        sys.exit(0)

    shown = 0
    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Could not parse {path.name}: {exc}")
            continue

        sid = data.get("scenario_id", path.stem)
        if args.scenarios and sid not in args.scenarios:
            continue

        _print_scenario(data, args.verbose)
        shown += 1

    if shown == 0:
        print("[INFO] No matching scenarios found.")
    else:
        print("-" * 70)
        print(f"[DONE] Inspected {shown} scenario(s).")


if __name__ == "__main__":
    main()
