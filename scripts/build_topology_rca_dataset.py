#!/usr/bin/env python3
"""
build_topology_rca_dataset.py — Build node-level feature dataset for topology RCA ML.

Reads scenario_library/topology_rca cases and writes:
  data/rca/topology/topology_node_dataset.csv
  data/rca/topology/topology_case_index.json

No remediation content is read or written here.

Usage:
  python scripts/build_topology_rca_dataset.py
  python scripts/build_topology_rca_dataset.py --include-out-of-scope
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rca_ml.topology_dataset import build_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build topology RCA node-level feature dataset.",
    )
    parser.add_argument("--repo-root",         default=str(REPO_ROOT))
    parser.add_argument("--scenario-library",  default="scenario_library")
    parser.add_argument("--out-dir",           default="data/rca/topology")
    parser.add_argument("--include-out-of-scope", action="store_true",
                        help="Include out-of-scope cases (label_is_root always 0)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    lib_root  = (repo_root / args.scenario_library).resolve()
    out_dir   = (repo_root / args.out_dir).resolve()

    manifest_path = lib_root / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] manifest.json not found: {manifest_path}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("topology_rca", [])

    print(f"Scenario library : {lib_root}")
    print(f"Manifest rows    : {len(rows)}")
    print(f"Include out-of-scope: {args.include_out_of_scope}")
    print()

    df, case_index = build_dataset(
        manifest_rows=rows,
        lib_root=lib_root,
        repo_root=repo_root,
        include_out_of_scope=args.include_out_of_scope,
    )

    if df.empty:
        print("[ERROR] No data built — check scenario_library paths.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "topology_node_dataset.csv", index=False)
    (out_dir / "topology_case_index.json").write_text(
        json.dumps(case_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    in_scope    = [c for c in case_index if c["root_cause_in_scope"]]
    out_scope   = [c for c in case_index if not c["root_cause_in_scope"]]
    root_rows   = int((df["label_is_root"] == 1).sum()) if "label_is_root" in df.columns else 0
    split_dist: dict[str, int] = {}
    for c in case_index:
        split_dist[c["split"]] = split_dist.get(c["split"], 0) + 1

    print("=" * 46)
    print(" Dataset Summary")
    print("=" * 46)
    print(f"  cases total          : {len(case_index)}")
    print(f"  cases in scope       : {len(in_scope)}")
    print(f"  cases out of scope   : {len(out_scope)}")
    print(f"  total node rows      : {len(df)}")
    print(f"  positive root rows   : {root_rows}")
    print(f"  split distribution   : {split_dist}")
    print(f"\n  Output:")
    print(f"    {out_dir / 'topology_node_dataset.csv'}")
    print(f"    {out_dir / 'topology_case_index.json'}")


if __name__ == "__main__":
    main()
