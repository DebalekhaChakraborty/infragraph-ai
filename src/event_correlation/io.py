"""
io.py — Case loading and cluster output writing helpers.

Handles file I/O for both topology_rca and enterprise_gnn_rca modes.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_case_for_correlation(
    lib_root: Path,
    repo_root: Path,
    case_id: str,
    mode: str | None = None,
) -> tuple[list[dict], dict | None, str, str, str]:
    """
    Load events and graph for a scenario-library case.

    Returns (events, graph_dict, mode, scenario_id, diagram_id).
      events      — list of event dicts from events.json
      graph_dict  — {"nodes": [...], "edges": [...]} or None if not found
      mode        — "topology_rca" or "enterprise_gnn_rca"
      scenario_id — scenario_id string from events.json (may be empty)
      diagram_id  — diagram_id string (topology only; empty for enterprise)

    Returns ([], None, mode, "", "") when the case directory or events file
    is not found.
    """
    if mode is None:
        if case_id.startswith("ent_"):
            mode = "enterprise_gnn_rca"
        else:
            mode = "topology_rca"

    subdir   = "enterprise_gnn_rca" if mode == "enterprise_gnn_rca" else "topology_rca"
    case_dir = lib_root / subdir / case_id

    if not case_dir.exists():
        return [], None, mode, "", ""

    events_path    = case_dir / "events.json"
    graph_ref_path = case_dir / "graph_ref.json"

    if not events_path.exists():
        return [], None, mode, "", ""

    events_doc  = json.loads(events_path.read_text(encoding="utf-8"))
    events      = events_doc.get("events", [])
    scenario_id = events_doc.get("scenario_id", "")
    diagram_id  = events_doc.get("diagram_id", "")

    graph: dict | None = None

    if graph_ref_path.exists():
        graph_ref = json.loads(graph_ref_path.read_text(encoding="utf-8"))

        if mode == "topology_rca":
            lg_path_raw = graph_ref.get("local_graph_path", "")
            if lg_path_raw:
                lg_path = _resolve_path(repo_root, lg_path_raw)
                if lg_path.exists():
                    graph = json.loads(lg_path.read_text(encoding="utf-8"))

        else:  # enterprise_gnn_rca
            eg_path_raw = graph_ref.get("enterprise_graph_path", "")
            if eg_path_raw:
                eg_path = _resolve_path(repo_root, eg_path_raw)
                if eg_path.exists():
                    graph = json.loads(eg_path.read_text(encoding="utf-8"))

    return events, graph, mode, scenario_id, diagram_id


def _resolve_path(repo_root: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    # Try relative to repo_root first (handles both / and \ separators)
    candidate = (repo_root / raw.replace("\\", "/")).resolve()
    if candidate.exists():
        return candidate
    # Fallback: raw path as-is (relative to cwd)
    return Path(raw).resolve()


def write_cluster_output(
    output: dict,
    case_id: str,
    scenario_id: str,
    mode: str,
    out_dir: Path,
) -> Path:
    """Write cluster output JSON.  Creates directories as needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = scenario_id if (mode == "enterprise_gnn_rca" and scenario_id) else case_id
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def load_cluster_file(cluster_path: Path) -> dict | None:
    """Load a cluster output JSON file.  Returns None on any error."""
    if not cluster_path.exists():
        return None
    try:
        return json.loads(cluster_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_cluster_for_case(
    cluster_data: dict,
    case_id: str,
    scenario_id: str = "",
) -> dict | None:
    """
    Return cluster_data if it matches the given case_id or scenario_id.

    The cluster file's top-level case_id / scenario_id must match.
    Returns None if there is no match.
    """
    if not isinstance(cluster_data, dict):
        return None
    if cluster_data.get("case_id") == case_id:
        return cluster_data
    if scenario_id and cluster_data.get("scenario_id") == scenario_id:
        return cluster_data
    return None
