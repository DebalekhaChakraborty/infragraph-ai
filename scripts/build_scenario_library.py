#!/usr/bin/env python3
"""
build_scenario_library.py — Build a clean, leakage-free scenario library from
datasets/infragraph_v3.

Design principles
-----------------
- events.json    : public — only what a simulation observer would see at runtime.
- labels.json    : private — ground-truth labels for ML training and evaluation.
- metadata.json  : bookkeeping fields (split, counts, provenance).
- graph_ref.json : repo-relative paths to graph files; no inline graph data.

No remediation text, root-cause answers, impact paths, or resolution plans ever
appear in events.json.  Consuming scripts must open labels.json explicitly.

Output layout
-------------
scenario_library/
  manifest.json
  topology_rca/{case_id}/{events,labels,metadata,graph_ref}.json
  enterprise_gnn_rca/{case_id}/{events,labels,metadata,graph_ref}.json

Usage
-----
python scripts/build_scenario_library.py
python scripts/build_scenario_library.py --dataset-root ./datasets/infragraph_v3 --out ./scenario_library
python scripts/build_scenario_library.py --dry-run        # validate only, no writes
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

VERSION = "1.0"

# Keys that must NEVER appear in events.json (top-level or inside events list)
FORBIDDEN_EVENT_KEYS: frozenset[str] = frozenset({
    "root_cause",
    "root_cause_diagram",
    "root_cause_pattern",
    "recommended_actions",
    "remediation_steps",
    "triage_steps",
    "rollback",
    "rollback_or_safety_notes",
    "rollback_plan",
    "servicenow",
    "servicenow_incident_summary",
    "impacted_nodes",
    "impacted_diagrams",
    "impact_paths",
    "validation_steps",
    "post_checks",
    "pre_checks",
})


# ── IO helpers ─────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: dict, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_events_doc(events_data: dict, case_id: str) -> None:
    """Raise ValueError if events_data contains any forbidden key."""
    bad = FORBIDDEN_EVENT_KEYS & set(events_data.keys())
    if bad:
        raise ValueError(
            f"[{case_id}] events.json top-level contains forbidden key(s): {sorted(bad)}"
        )
    for i, event in enumerate(events_data.get("events", [])):
        bad_event = FORBIDDEN_EVENT_KEYS & set(event.keys())
        if bad_event:
            raise ValueError(
                f"[{case_id}] events.json event[{i}] contains forbidden key(s): {sorted(bad_event)}"
            )


# ── Event builder ─────────────────────────────────────────────────────────────

def _build_events(alerts: list[dict], diagram_id: str | None = None) -> list[dict]:
    """Return public event records, optionally filtered to one diagram."""
    events = []
    for alert in alerts:
        if diagram_id is not None and alert.get("diagram_id") != diagram_id:
            continue
        events.append({
            "event_id": "",        # filled in below after filtering
            "time_offset_min": alert.get("time_offset_min", 0),
            "node": alert.get("node", ""),
            "diagram_id": alert.get("diagram_id", ""),
            "alert_type": alert.get("alert_type", ""),
            "severity": alert.get("severity", ""),
        })
    for i, event in enumerate(events, 1):
        event["event_id"] = f"EVT-{i:04d}"
    return events


# ── Per-scenario processor ────────────────────────────────────────────────────

def _process_scenario(
    scenario_dir: Path,
    split: str,
    out_dir: Path,
    manifest_topo: list[dict],
    manifest_ent: list[dict],
    dry_run: bool,
) -> tuple[int, int]:
    """
    Build all library cases for one scenario.
    Returns (enterprise_cases_written, topology_cases_written).
    """
    alerts_data = _load_json(scenario_dir / "alerts.json")
    enterprise_data = _load_json(scenario_dir / "enterprise_graph.json")

    scenario_id: str = alerts_data.get("scenario_id") or scenario_dir.name
    alerts: list[dict] = alerts_data.get("alerts", [])
    root_cause: str = alerts_data.get("root_cause", "")
    root_cause_diagram: str = alerts_data.get("root_cause_diagram", "")
    root_cause_pattern: str = alerts_data.get("root_cause_pattern", "")
    severity: str = alerts_data.get("severity", "")
    impacted_nodes: list = alerts_data.get("impacted_nodes", [])
    impacted_diagrams: list = alerts_data.get("impacted_diagrams", [])
    impact_paths: list = alerts_data.get("impact_paths", [])

    all_diagram_ids: list[str] = sorted(
        {a.get("diagram_id", "") for a in alerts if a.get("diagram_id")}
    )
    node_count = len(enterprise_data.get("nodes", []))
    edge_count = (
        len(enterprise_data.get("edges", []))
        + len(enterprise_data.get("cross_diagram_edges", []))
    )

    # Repo-relative path prefix used in graph_ref files
    rel_base = Path("datasets/infragraph_v3/scenarios") / split / scenario_id

    # ── 1. Enterprise GNN RCA case ────────────────────────────────────────────
    ent_case_id = f"ent_{scenario_id}"
    ent_dir = out_dir / "enterprise_gnn_rca" / ent_case_id

    ent_events_doc = {
        "case_id": ent_case_id,
        "mode": "enterprise_gnn_rca",
        "scenario_id": scenario_id,
        "events": _build_events(alerts),
    }
    _validate_events_doc(ent_events_doc, ent_case_id)

    _write_json(ent_dir / "events.json", ent_events_doc, dry_run)

    _write_json(ent_dir / "labels.json", {
        "case_id": ent_case_id,
        "root_cause_node": root_cause,
        "root_cause_diagram": root_cause_diagram,
        "root_cause_pattern": root_cause_pattern,
        "impacted_nodes": impacted_nodes,
        "impacted_diagrams": impacted_diagrams,
        "impact_paths": impact_paths,
        "severity": severity,
    }, dry_run)

    _write_json(ent_dir / "metadata.json", {
        "case_id": ent_case_id,
        "mode": "enterprise_gnn_rca",
        "split": split,
        "source_scenario_id": scenario_id,
        "diagram_ids": all_diagram_ids,
        "alert_count": len(alerts),
        "node_count": node_count,
        "edge_count": edge_count,
        "created_from": str(rel_base / "alerts.json"),
        "version": VERSION,
    }, dry_run)

    _write_json(ent_dir / "graph_ref.json", {
        "case_id": ent_case_id,
        "enterprise_graph_path": str(rel_base / "enterprise_graph.json"),
        "stitch_map_path": str(rel_base / "stitch_map.json"),
        "local_graphs_dir": str(rel_base / "local_graphs"),
    }, dry_run)

    manifest_ent.append({
        "case_id": ent_case_id,
        "mode": "enterprise_gnn_rca",
        "split": split,
        "scenario_id": scenario_id,
        "event_count": len(alerts),
        "root_cause_in_scope": True,
        "events_path": f"enterprise_gnn_rca/{ent_case_id}/events.json",
        "labels_path": f"enterprise_gnn_rca/{ent_case_id}/labels.json",
        "metadata_path": f"enterprise_gnn_rca/{ent_case_id}/metadata.json",
        "graph_ref_path": f"enterprise_gnn_rca/{ent_case_id}/graph_ref.json",
    })

    # ── 2. Topology RCA cases (one per in-scope diagram) ─────────────────────
    local_graphs_dir = scenario_dir / "local_graphs"
    diagrams_with_graph: set[str] = set()
    if local_graphs_dir.is_dir():
        diagrams_with_graph = {lg.stem for lg in local_graphs_dir.glob("*.json")}

    # Diagrams to generate cases for:
    # - always include root_cause_diagram (even if no alert in that diagram alone)
    # - include any diagram that has at least one alert AND a local graph
    diagrams_to_process: set[str] = set()
    if root_cause_diagram:
        diagrams_to_process.add(root_cause_diagram)
    for alert in alerts:
        diag = alert.get("diagram_id", "")
        if diag and diag in diagrams_with_graph:
            diagrams_to_process.add(diag)

    topo_count = 0
    for diagram_id in sorted(diagrams_to_process):
        diag_events = _build_events(alerts, diagram_id=diagram_id)
        # If no events AND this is not the root-cause diagram, skip
        if not diag_events and diagram_id != root_cause_diagram:
            continue

        topo_case_id = f"topo_{scenario_id}_{diagram_id}"
        topo_dir = out_dir / "topology_rca" / topo_case_id
        in_scope = diagram_id == root_cause_diagram

        lg_path = local_graphs_dir / f"{diagram_id}.json"
        lg_data = _load_json(lg_path) if lg_path.exists() else {}
        topo_node_count = len(lg_data.get("nodes", []))
        topo_edge_count = len(lg_data.get("edges", []))

        # Nodes impacted in this diagram only
        local_impacted = [
            a.get("node", "")
            for a in alerts
            if a.get("diagram_id") == diagram_id and a.get("node")
            and a.get("node") != root_cause
        ]

        topo_events_doc = {
            "case_id": topo_case_id,
            "mode": "topology_rca",
            "scenario_id": scenario_id,
            "diagram_id": diagram_id,
            "events": diag_events,
        }
        _validate_events_doc(topo_events_doc, topo_case_id)
        _write_json(topo_dir / "events.json", topo_events_doc, dry_run)

        topo_labels: dict = {
            "case_id": topo_case_id,
            "root_cause_in_scope": in_scope,
            "severity": severity,
            "impacted_nodes": local_impacted,
        }
        if in_scope:
            topo_labels["root_cause_node"] = root_cause
            topo_labels["root_cause_diagram"] = root_cause_diagram
            topo_labels["root_cause_pattern"] = root_cause_pattern
            topo_labels["impact_paths"] = impact_paths
        else:
            topo_labels["expected_behavior"] = "escalate_or_unknown"

        _write_json(topo_dir / "labels.json", topo_labels, dry_run)

        _write_json(topo_dir / "metadata.json", {
            "case_id": topo_case_id,
            "mode": "topology_rca",
            "split": split,
            "source_scenario_id": scenario_id,
            "diagram_ids": [diagram_id],
            "alert_count": len(diag_events),
            "node_count": topo_node_count,
            "edge_count": topo_edge_count,
            "created_from": str(rel_base / "alerts.json"),
            "version": VERSION,
        }, dry_run)

        _write_json(topo_dir / "graph_ref.json", {
            "case_id": topo_case_id,
            "diagram_id": diagram_id,
            "local_graph_path": str(rel_base / "local_graphs" / f"{diagram_id}.json"),
        }, dry_run)

        manifest_topo.append({
            "case_id": topo_case_id,
            "mode": "topology_rca",
            "split": split,
            "scenario_id": scenario_id,
            "diagram_id": diagram_id,
            "event_count": len(diag_events),
            "root_cause_in_scope": in_scope,
            "events_path": f"topology_rca/{topo_case_id}/events.json",
            "labels_path": f"topology_rca/{topo_case_id}/labels.json",
            "metadata_path": f"topology_rca/{topo_case_id}/metadata.json",
            "graph_ref_path": f"topology_rca/{topo_case_id}/graph_ref.json",
        })
        topo_count += 1

    return 1, topo_count


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build scenario_library/ from datasets/infragraph_v3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-root", default="./datasets/infragraph_v3",
        help="Path to infragraph_v3 dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--out", default="./scenario_library",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--splits", default="train,val,test",
        help="Comma-separated splits to process (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate without writing any files",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.out).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    dry_run: bool = args.dry_run

    if dry_run:
        print("DRY-RUN mode — no files will be written.")
    print(f"Dataset root : {dataset_root}")
    print(f"Output dir   : {out_dir}")
    print()

    manifest_topo: list[dict] = []
    manifest_ent: list[dict] = []
    split_counts: dict[str, dict[str, int]] = {}

    for split in splits:
        scenarios_dir = dataset_root / "scenarios" / split
        if not scenarios_dir.exists():
            print(f"  [skip] Split directory not found: {scenarios_dir}")
            continue

        ent_total = topo_total = 0
        for scenario_dir in sorted(scenarios_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            if not (scenario_dir / "alerts.json").exists():
                print(f"  [skip] No alerts.json in {scenario_dir.name}")
                continue

            ent_n, topo_n = _process_scenario(
                scenario_dir, split, out_dir,
                manifest_topo, manifest_ent, dry_run,
            )
            ent_total += ent_n
            topo_total += topo_n

        split_counts[split] = {"ent": ent_total, "topo": topo_total}

    # Write manifest
    manifest = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits_processed": splits,
        "topology_rca": manifest_topo,
        "enterprise_gnn_rca": manifest_ent,
    }
    _write_json(out_dir / "manifest.json", manifest, dry_run)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 54)
    print(" Scenario Library Build Summary")
    print("=" * 54)
    print(f"  {'split':<10}  {'enterprise_gnn_rca':>20}  {'topology_rca':>14}")
    print(f"  {'-'*10}  {'-'*20}  {'-'*14}")
    for split, c in split_counts.items():
        print(f"  {split:<10}  {c['ent']:>20}  {c['topo']:>14}")
    total_ent = sum(c["ent"] for c in split_counts.values())
    total_topo = sum(c["topo"] for c in split_counts.values())
    print(f"  {'TOTAL':<10}  {total_ent:>20}  {total_topo:>14}")
    print()
    if dry_run:
        print("  DRY-RUN complete — no files written.")
    else:
        print(f"  Output   : {out_dir}")
        print(f"  Manifest : {out_dir / 'manifest.json'}")
    print()
    print("  Validation: all events.json passed forbidden-key check.")


if __name__ == "__main__":
    main()
