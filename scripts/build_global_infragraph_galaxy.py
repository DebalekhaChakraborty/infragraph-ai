#!/usr/bin/env python3
"""
build_global_infragraph_galaxy.py

Builds the Global InfraGraph Galaxy — a combined graph-memory index across
all V3 enterprise scenarios.  Each scenario enterprise_graph.json is read and
its nodes/edges are merged into one global namespace using:

    global_node_id = scenario_id + "::" + node_id

This avoids cross-scenario ID collisions without attempting semantic dedup
(canonical_id dedup is a separate, future step once canonical IDs are
globally validated).

Usage
-----
python scripts/build_global_infragraph_galaxy.py \\
    --dataset-root ./datasets/infragraph_v3 \\
    --out ./outputs/global_graph_memory

Outputs
-------
outputs/global_graph_memory/
    infragraph_global_graph.json   -- full node/edge list
    nodes.csv                      -- one row per node
    edges.csv                      -- one row per edge
    scenario_index.json            -- per-scenario metadata
    summary.json                   -- aggregate counts
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _infer_diagram_type(diagram_id: str) -> str:
    for keyword in ("branch", "wan", "datacenter", "app_db", "shared_services"):
        if keyword in diagram_id.lower():
            return keyword
    return "unknown"


# ── core builder ──────────────────────────────────────────────────────────────

def build_galaxy(dataset_root: Path, out_dir: Path) -> dict:
    global_nodes: list[dict] = []
    global_edges: list[dict] = []
    scenario_index: list[dict] = []

    node_type_counts: dict[str, int] = defaultdict(int)
    diagram_type_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    total_cross = 0

    for split in SPLITS:
        split_dir = dataset_root / "scenarios" / split
        if not split_dir.exists():
            print(f"  [skip] split not found: {split_dir}")
            continue

        for scenario_dir in sorted(split_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            scenario_id = scenario_dir.name
            eg_path = scenario_dir / "enterprise_graph.json"
            if not eg_path.exists():
                print(f"  [skip] no enterprise_graph.json: {scenario_dir}")
                continue

            eg = _load_json(eg_path)
            nodes = eg.get("nodes", [])
            edges = eg.get("edges", [])
            cross  = eg.get("cross_diagram_edges", [])

            local_to_global: dict[str, str] = {}
            sc_node_rows: list[dict] = []

            for n in nodes:
                nid = n.get("id") or n.get("node_id") or ""
                if not nid:
                    continue
                global_id = f"{scenario_id}::{nid}"
                local_to_global[nid] = global_id
                ntype = n.get("type", "unknown")
                diag_id = n.get("diagram_id", "")
                dtype = _infer_diagram_type(diag_id)
                row = {
                    "global_node_id":   global_id,
                    "scenario_id":      scenario_id,
                    "split":            split,
                    "node_id":          nid,
                    "type":             ntype,
                    "diagram_id":       diag_id,
                    "diagram_type":     dtype,
                    "canonical_id":     n.get("canonical_id", ""),
                    "is_shared_entity": n.get("is_shared_entity", False),
                    "ip_address":       n.get("ip_address", ""),
                    "zone":             n.get("zone", ""),
                }
                sc_node_rows.append(row)
                global_nodes.append(row)
                node_type_counts[ntype] += 1
                diagram_type_counts[dtype] += 1

            sc_edge_rows: list[dict] = []
            all_edges = list(edges) + list(cross)
            for e in all_edges:
                src = e.get("source", "")
                tgt = e.get("target", "")
                if not src or not tgt:
                    continue
                g_src = local_to_global.get(src, f"{scenario_id}::{src}")
                g_tgt = local_to_global.get(tgt, f"{scenario_id}::{tgt}")
                scope = e.get("edge_scope", "cross_diagram" if e in cross else "local")
                row = {
                    "source_global_id": g_src,
                    "target_global_id": g_tgt,
                    "scenario_id":      scenario_id,
                    "edge_scope":       scope,
                    "relationship":     e.get("relationship", ""),
                    "label":            e.get("label", ""),
                    "source_diagram":   e.get("source_diagram", ""),
                    "target_diagram":   e.get("target_diagram", ""),
                }
                sc_edge_rows.append(row)
                global_edges.append(row)

            n_cross_sc = len([r for r in sc_edge_rows if r["edge_scope"] == "cross_diagram"])
            total_cross += n_cross_sc
            split_counts[split] += 1

            scenario_index.append({
                "scenario_id":         scenario_id,
                "split":               split,
                "path":                str(eg_path),
                "node_count":          len(sc_node_rows),
                "edge_count":          len(sc_edge_rows),
                "cross_diagram_edges": n_cross_sc,
                "diagrams":            list({n.get("diagram_id", "") for n in nodes if n.get("diagram_id")}),
            })

            split_counts[split] += 0  # already counted above
            print(f"  [{split}] {scenario_id}: {len(sc_node_rows)} nodes, {len(sc_edge_rows)} edges")

    summary = {
        "total_scenarios":          len(scenario_index),
        "total_nodes":              len(global_nodes),
        "total_edges":              len(global_edges),
        "total_cross_diagram_edges": total_cross,
        "node_type_counts":         dict(node_type_counts),
        "diagram_type_counts":      dict(diagram_type_counts),
        "split_counts":             dict(split_counts),
    }

    global_graph = {
        "graph_type":  "global_infragraph_galaxy",
        "description": "Combined graph-memory index across all V3 InfraGraph scenarios.",
        "nodes":       global_nodes,
        "edges":       global_edges,
        "summary":     summary,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "infragraph_global_graph.json", global_graph)
    _save_json(out_dir / "scenario_index.json",          scenario_index)
    _save_json(out_dir / "summary.json",                 summary)
    _save_csv(out_dir / "nodes.csv",                     global_nodes)
    _save_csv(out_dir / "edges.csv",                     global_edges)

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Build Global InfraGraph Galaxy index")
    p.add_argument("--dataset-root", default="./datasets/infragraph_v3",
                   help="Root of V3 dataset (must contain scenarios/train|val|test/)")
    p.add_argument("--out", default="./runtime_state/global_graph_memory",
                   help="Output directory for galaxy files")
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir      = Path(args.out)

    if not dataset_root.exists():
        print(f"[ERROR] Dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Building Global InfraGraph Galaxy")
    print(f"  Dataset root : {dataset_root}")
    print(f"  Output dir   : {out_dir}")
    print()

    summary = build_galaxy(dataset_root, out_dir)

    print()
    print("Done.")
    print(f"  Scenarios : {summary['total_scenarios']}")
    print(f"  Nodes     : {summary['total_nodes']}")
    print(f"  Edges     : {summary['total_edges']}")
    print(f"  Cross-diag: {summary['total_cross_diagram_edges']}")
    print(f"  Output    : {out_dir}")


if __name__ == "__main__":
    main()
