#!/usr/bin/env python3
"""
inspect_qwen_sop_dataset.py

Inspect a SOP-grounded Qwen training dataset.

Reads train.jsonl and val.jsonl from a dataset directory and prints:
  - Total / train / val record counts
  - Per-base-scenario distribution (from metadata.base_scenario_id if present,
    else metadata.scenario_id)
  - Root causes covered (unique, de-duplicated)
  - Sample record IDs
  - Per-record: evidence IDs used, first 2 remediation steps,
    KB-*/CE-* presence in evidence_ids_used

Usage:
  python scripts/inspect_qwen_sop_dataset.py --data-dir data/qwen_sop_grounded
  python scripts/inspect_qwen_sop_dataset.py --data-dir data/qwen_sop_grounded_expanded --max-records 8
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"  [WARN] Could not parse line: {exc}")
    return records


def _base_scenario(record: dict) -> str:
    meta = record.get("metadata") or {}
    return meta.get("base_scenario_id") or meta.get("scenario_id") or "unknown"


def _root_cause(record: dict) -> str:
    meta = record.get("metadata") or {}
    rc = meta.get("root_cause", "")
    if rc:
        return rc
    try:
        user_obj = json.loads(record["messages"][1]["content"])
        return user_obj.get("root_cause", "—")
    except Exception:
        return "—"


def _evidence_ids(record: dict) -> list[str]:
    try:
        asst_obj = json.loads(record["messages"][2]["content"])
        return [str(x) for x in (asst_obj.get("evidence_ids_used") or [])]
    except Exception:
        return []


def _remediation_steps(record: dict) -> list[str]:
    try:
        asst_obj = json.loads(record["messages"][2]["content"])
        return [str(s) for s in (asst_obj.get("remediation_steps") or [])]
    except Exception:
        return []


def _print_record(record: dict, split_label: str) -> None:
    rid = record.get("id", "—")
    rc  = _root_cause(record)
    ev  = _evidence_ids(record)
    rem = _remediation_steps(record)

    kb_ids = [e for e in ev if e.startswith("KB-")]
    ce_ids = [e for e in ev if e.startswith("CE-")]

    print(f"  [{split_label}] {rid}")
    print(f"    root_cause : {rc}")
    print(f"    KB-* IDs   : {kb_ids[:5] if kb_ids else '(none)'}")
    print(f"    CE-* IDs   : {ce_ids[:4] if ce_ids else '(none)'}")
    if rem:
        print(f"    rem[0]     : {rem[0][:100]}")
    if len(rem) > 1:
        print(f"    rem[1]     : {rem[1][:100]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a SOP-grounded Qwen SFT dataset."
    )
    parser.add_argument(
        "--data-dir", required=True, metavar="DIR",
        help="Directory containing train.jsonl and val.jsonl.",
    )
    parser.add_argument(
        "--max-records", type=int, default=8, metavar="N",
        help="Maximum per-record details to print (default: 8).",
    )
    args = parser.parse_args()

    data_dir = (REPO_ROOT / args.data_dir).resolve()
    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {args.data_dir}")
        sys.exit(1)

    train = _load_jsonl(data_dir / "train.jsonl")
    val   = _load_jsonl(data_dir / "val.jsonl")
    all_records = train + val

    print("=" * 70)
    print(f" Dataset: {data_dir.relative_to(REPO_ROOT)}")
    print("=" * 70)

    if not all_records:
        print("[INFO] No records found.")
        sys.exit(0)

    # Summary counts
    print(f"  Total   : {len(all_records)}")
    print(f"  train   : {len(train)}")
    print(f"  val     : {len(val)}")
    print()

    # Per-base-scenario distribution
    base_counts = Counter(_base_scenario(r) for r in all_records)
    print("  Per-base-scenario distribution:")
    for sid, cnt in sorted(base_counts.items()):
        print(f"    {sid:40s} : {cnt}")
    print()

    # Root causes covered
    root_causes = sorted(set(_root_cause(r) for r in all_records))
    print(f"  Root causes covered ({len(root_causes)}):")
    for rc in root_causes:
        print(f"    - {rc}")
    print()

    # KB / CE coverage
    all_ev = [e for r in all_records for e in _evidence_ids(r)]
    kb_ev  = [e for e in all_ev if e.startswith("KB-")]
    ce_ev  = [e for e in all_ev if e.startswith("CE-")]
    records_with_kb = sum(
        1 for r in all_records
        if any(e.startswith("KB-") for e in _evidence_ids(r))
    )
    records_with_ce = sum(
        1 for r in all_records
        if any(e.startswith("CE-") for e in _evidence_ids(r))
    )
    print(f"  Evidence coverage across all records:")
    print(f"    Records with KB-* IDs : {records_with_kb} / {len(all_records)}")
    print(f"    Records with CE-* IDs : {records_with_ce} / {len(all_records)}")
    print(f"    Total KB-* citations  : {len(kb_ev)}")
    print(f"    Total CE-* citations  : {len(ce_ev)}")
    print()

    # Per-record sample details
    n_shown = 0
    print(f"  Sample record details (max {args.max_records}):")
    for split_label, records in [("train", train), ("val", val)]:
        for record in records:
            if n_shown >= args.max_records:
                break
            _print_record(record, split_label)
            n_shown += 1
        if n_shown >= args.max_records:
            break

    if len(all_records) > args.max_records:
        print(f"  ... {len(all_records) - args.max_records} more records (use --max-records to show more)")

    print()

    # Validation summary
    bad_kb = [r.get("id") for r in all_records if not any(e.startswith("KB-") for e in _evidence_ids(r))]
    bad_ce = [r.get("id") for r in all_records if not any(e.startswith("CE-") for e in _evidence_ids(r))]
    if bad_kb:
        print(f"  [WARN] Records missing KB-* citations ({len(bad_kb)}): {bad_kb[:5]}")
    if bad_ce:
        print(f"  [WARN] Records missing CE-* citations ({len(bad_ce)}): {bad_ce[:5]}")
    if not bad_kb and not bad_ce:
        print("  [PASS] All records have both KB-* and CE-* evidence citations.")

    # Dataset summary JSON if present
    summary_path = data_dir / "dataset_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            policy  = summary.get("synthetic_generation_policy", "—")
            src     = summary.get("target_source", "—")
            print()
            print(f"  dataset_summary.json:")
            print(f"    target_source              : {src}")
            print(f"    synthetic_generation_policy: {policy}")
        except Exception:
            pass

    print("=" * 70)


if __name__ == "__main__":
    main()
