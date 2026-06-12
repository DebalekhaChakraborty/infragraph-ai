"""
topology_dataset.py — Load scenario_library cases and build the feature DataFrame.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .features import compute_case_features, normalize_repo_path


def load_case(
    lib_root: Path,
    manifest_row: dict,
    repo_root: Path,
) -> tuple[list[dict], dict, dict, dict]:
    """
    Load events, labels, graph_ref, and local_graph for one manifest row.

    Returns:
        (events_list, labels_dict, graph_ref_dict, local_graph_dict)
    """
    def _r(relpath: str) -> dict:
        return json.loads((lib_root / relpath).read_text(encoding="utf-8"))

    events_doc = _r(manifest_row["events_path"])
    labels     = _r(manifest_row["labels_path"])
    graph_ref  = _r(manifest_row["graph_ref_path"])

    lg_path    = normalize_repo_path(repo_root, graph_ref["local_graph_path"])
    local_graph = json.loads(lg_path.read_text(encoding="utf-8"))

    return events_doc.get("events", []), labels, graph_ref, local_graph


def build_dataset(
    manifest_rows: list[dict],
    lib_root: Path,
    repo_root: Path,
    include_out_of_scope: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Build the flat node-level feature DataFrame from topology_rca cases.

    For in-scope cases (root_cause_in_scope=True), label_is_root=1 marks
    the ground-truth root node.  Out-of-scope cases are included only when
    include_out_of_scope=True; their label_is_root values are all 0.

    Returns:
        (DataFrame, case_index)
    """
    all_rows:   list[dict] = []
    case_index: list[dict] = []
    skipped = 0

    for row in manifest_rows:
        in_scope: bool = bool(row.get("root_cause_in_scope", False))
        if not in_scope and not include_out_of_scope:
            continue

        try:
            events, labels, graph_ref, local_graph = load_case(lib_root, row, repo_root)
        except Exception as exc:
            print(f"  [skip] {row['case_id']}: {exc}")
            skipped += 1
            continue

        root_cause = labels.get("root_cause_node") if in_scope else None
        case_id    = row["case_id"]
        split      = row["split"]
        scenario_id = row.get("scenario_id", "")
        diagram_id  = row.get("diagram_id", "")

        feature_rows = compute_case_features(
            case_id=case_id,
            split=split,
            scenario_id=scenario_id,
            diagram_id=diagram_id,
            events=events,
            local_graph=local_graph,
            root_cause_node=root_cause,
        )
        all_rows.extend(feature_rows)

        case_index.append({
            "case_id":            case_id,
            "split":              split,
            "scenario_id":        scenario_id,
            "diagram_id":         diagram_id,
            "root_cause_in_scope": in_scope,
            "root_cause_node":    root_cause,
            "node_count":         len(local_graph.get("nodes", [])),
            "alert_count":        len(events),
            "row_count":          len(feature_rows),
        })

    if skipped:
        print(f"  [warning] skipped {skipped} case(s) due to load errors.")

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    return df, case_index
