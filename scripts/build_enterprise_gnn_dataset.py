#!/usr/bin/env python3
"""
build_enterprise_gnn_dataset.py — Build PyTorch graph dataset for Enterprise RCA GNN.

Reads:
  scenario_library/manifest.json  (enterprise_gnn_rca section)
  scenario_library/enterprise_gnn_rca/*/events.json
  scenario_library/enterprise_gnn_rca/*/labels.json
  scenario_library/enterprise_gnn_rca/*/graph_ref.json
  <referenced enterprise_graph.json and stitch_map.json>

Writes:
  data/rca/enterprise_gnn/graphs.pt
  data/rca/enterprise_gnn/graph_index.json
  data/rca/enterprise_gnn/feature_columns.json
  data/rca/enterprise_gnn/label_stats.json

No remediation content is produced here.

Usage:
  python scripts/build_enterprise_gnn_dataset.py
  python scripts/build_enterprise_gnn_dataset.py --out-dir data/rca/enterprise_gnn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rca_ml.enterprise_gnn_dataset import (  # noqa: E402
    FEATURE_NAMES,
    build_graph_dataset,
    check_torch_requirement,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Enterprise GNN RCA graph dataset from scenario_library."
    )
    parser.add_argument("--repo-root",        default=str(REPO_ROOT))
    parser.add_argument("--scenario-library", default="scenario_library")
    parser.add_argument("--out-dir",          default="data/rca/enterprise_gnn")
    args = parser.parse_args()

    check_torch_requirement()
    import torch

    repo_root = Path(args.repo_root).resolve()
    lib_root  = (repo_root / args.scenario_library).resolve()
    out_dir   = (repo_root / args.out_dir).resolve()

    manifest_path = lib_root / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] manifest.json not found: {manifest_path}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("enterprise_gnn_rca", [])

    if not rows:
        print("[ERROR] scenario_library manifest has no enterprise_gnn_rca cases.")
        print("        Re-run scripts/build_scenario_library.py with enterprise mode enabled.")
        sys.exit(1)

    in_scope = [r for r in rows if r.get("root_cause_in_scope", False)]
    print(f"Scenario library  : {lib_root}")
    print(f"Manifest rows     : {len(rows)}  (in-scope: {len(in_scope)})")
    print()

    graphs, case_index = build_graph_dataset(
        manifest_rows=rows,
        lib_root=lib_root,
        repo_root=repo_root,
    )

    if not graphs:
        print("[ERROR] No graph data built. Check scenario_library paths and enterprise_graph_path values.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save graphs.pt
    torch.save(graphs, out_dir / "graphs.pt")

    # Save case index
    (out_dir / "graph_index.json").write_text(
        json.dumps(case_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Save feature columns
    (out_dir / "feature_columns.json").write_text(
        json.dumps(FEATURE_NAMES, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Label stats
    split_dist: dict[str, int] = {}
    pattern_dist: dict[str, int] = {}
    node_counts: list[int] = []
    edge_counts: list[int] = []
    event_counts: list[int] = []

    for c in case_index:
        sp = c["split"]
        split_dist[sp] = split_dist.get(sp, 0) + 1
        pat = c.get("root_cause_pattern", "")
        if pat:
            pattern_dist[pat] = pattern_dist.get(pat, 0) + 1
        node_counts.append(c["node_count"])
        edge_counts.append(c["edge_count"])
        event_counts.append(c["event_count"])

    def _avg(lst: list[int]) -> float:
        return round(sum(lst) / len(lst), 1) if lst else 0.0

    label_stats = {
        "total_graphs":       len(graphs),
        "feature_dim":        len(FEATURE_NAMES),
        "split_distribution": split_dist,
        "pattern_distribution": pattern_dist,
        "node_count_avg":     _avg(node_counts),
        "node_count_min":     min(node_counts) if node_counts else 0,
        "node_count_max":     max(node_counts) if node_counts else 0,
        "edge_count_avg":     _avg(edge_counts),
        "event_count_avg":    _avg(event_counts),
    }
    (out_dir / "label_stats.json").write_text(
        json.dumps(label_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    in_scope_idx  = [c for c in case_index if c["split"] in ("train", "val", "test")]

    print("=" * 50)
    print(" Enterprise GNN Dataset Summary")
    print("=" * 50)
    print(f"  graphs built       : {len(graphs)}")
    print(f"  feature dim        : {len(FEATURE_NAMES)}")
    print(f"  split distribution : {split_dist}")
    print(f"  avg nodes/graph    : {label_stats['node_count_avg']}")
    print(f"  avg edges/graph    : {label_stats['edge_count_avg']}")
    print(f"  avg events/graph   : {label_stats['event_count_avg']}")
    if pattern_dist:
        top_pattern = max(pattern_dist, key=lambda k: pattern_dist[k])
        print(f"  dominant pattern   : {top_pattern} ({pattern_dist[top_pattern]} cases)")
    print(f"\n  Output:")
    print(f"    {out_dir / 'graphs.pt'}")
    print(f"    {out_dir / 'graph_index.json'}")
    print(f"    {out_dir / 'feature_columns.json'}")
    print(f"    {out_dir / 'label_stats.json'}")


if __name__ == "__main__":
    main()
