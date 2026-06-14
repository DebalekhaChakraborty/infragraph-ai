"""Load and normalize InfraGraph global graph memory for the Graph Copilot."""
from __future__ import annotations

import csv
import json
from pathlib import Path

_GLOBAL_GRAPH_DIR = Path(__file__).resolve().parents[2] / "runtime_state" / "global_graph_memory"


def _safe_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_global_graph_context(
    global_graph_path: "Path | None" = None,
    scenario_graph: "dict | None" = None,
    scenario_alerts: "list | None" = None,
    stitch_map: "dict | None" = None,
    enterprise_rca: "dict | None" = None,
    incident: "dict | None" = None,
) -> dict:
    """Load and normalize graph data into lookup structures for the copilot.

    Merges global graph memory with the currently loaded scenario graph so the
    copilot can answer questions about any node/IP/edge across all diagrams.
    """
    gdir     = global_graph_path.parent if global_graph_path else _GLOBAL_GRAPH_DIR
    gg_path  = global_graph_path or (gdir / "infragraph_global_graph.json")

    global_graph    = _safe_json(gg_path) if gg_path.exists() else {}
    summary         = _safe_json(gdir / "summary.json")
    scenario_index  = _safe_json(gdir / "scenario_index.json")

    # Collect nodes/edges from global graph
    all_nodes: list[dict]  = list(global_graph.get("nodes") or [])
    all_edges: list[dict]  = list(global_graph.get("edges") or [])
    cross_edges: list[dict] = list(global_graph.get("cross_diagram_edges") or [])

    # Merge in scenario-level graph (avoids requiring a full global rebuild)
    if scenario_graph:
        _seen_ids = {n.get("id") for n in all_nodes if n.get("id")}
        for n in (scenario_graph.get("nodes") or []):
            if n.get("id") not in _seen_ids:
                all_nodes.append(n)
        for e in (scenario_graph.get("edges") or []):
            all_edges.append(e)
        for e in (scenario_graph.get("cross_diagram_edges") or []):
            cross_edges.append(e)

    # Fallback to CSV when JSON nodes list is empty (large global graph stored as CSV)
    if not all_nodes:
        nodes_csv = gdir / "nodes.csv"
        if nodes_csv.exists():
            try:
                with nodes_csv.open(encoding="utf-8", newline="") as f:
                    all_nodes = [dict(row) for row in csv.DictReader(f)]
            except Exception:
                pass

    if not all_edges:
        edges_csv = gdir / "edges.csv"
        if edges_csv.exists():
            try:
                with edges_csv.open(encoding="utf-8", newline="") as f:
                    all_edges = [dict(row) for row in csv.DictReader(f)]
            except Exception:
                pass

    # ── Build lookup indexes ─────────────────────────────────────────────────
    nodes_by_id:      dict[str, dict]        = {}
    nodes_by_ip:      dict[str, list[dict]]  = {}
    diagram_to_nodes: dict[str, list[str]]   = {}

    for n in all_nodes:
        nid = str(n.get("id") or n.get("node_id") or "")
        if not nid:
            continue
        nodes_by_id[nid] = n
        ip = str(n.get("ip_address") or "")
        if ip:
            nodes_by_ip.setdefault(ip, []).append(n)
        diag = str(n.get("diagram_id") or "")
        if diag:
            diagram_to_nodes.setdefault(diag, []).append(nid)

    edges_by_source: dict[str, list[dict]] = {}
    edges_by_target: dict[str, list[dict]] = {}
    for e in all_edges:
        src = str(e.get("source") or "")
        tgt = str(e.get("target") or "")
        if src:
            edges_by_source.setdefault(src, []).append(e)
        if tgt:
            edges_by_target.setdefault(tgt, []).append(e)

    cross_by_source: dict[str, list[dict]] = {}
    cross_by_target: dict[str, list[dict]] = {}
    for e in cross_edges:
        src = str(e.get("source") or e.get("source_node") or "")
        tgt = str(e.get("target") or e.get("target_node") or "")
        if src:
            cross_by_source.setdefault(src, []).append(e)
        if tgt:
            cross_by_target.setdefault(tgt, []).append(e)

    # ── Scenario index: scenario_id → [diagram_ids] ──────────────────────────
    scenario_to_diagrams: dict[str, list[str]] = {}
    diagram_to_scenario:  dict[str, str]       = {}
    if isinstance(scenario_index, dict):
        for sid, sinfo in scenario_index.items():
            diags = (sinfo.get("diagrams") or []) if isinstance(sinfo, dict) else []
            scenario_to_diagrams[sid] = diags
            for d in diags:
                diagram_to_scenario[d] = sid

    # ── Alert timeline ───────────────────────────────────────────────────────
    alert_timeline: list[dict] = []
    if incident:
        alert_timeline = list(incident.get("alert_timeline") or [])
    if not alert_timeline and scenario_alerts:
        alert_timeline = list(scenario_alerts) if isinstance(scenario_alerts, list) else []

    rca = enterprise_rca or {}

    return {
        "global_graph":         global_graph,
        "summary":              summary,
        "scenario_index":       scenario_index,
        "nodes_by_id":          nodes_by_id,
        "nodes_by_ip":          nodes_by_ip,
        "edges_by_source":      edges_by_source,
        "edges_by_target":      edges_by_target,
        "diagram_to_nodes":     diagram_to_nodes,
        "scenario_to_diagrams": scenario_to_diagrams,
        "diagram_to_scenario":  diagram_to_scenario,
        "cross_diagram_edges":  cross_edges,
        "cross_by_source":      cross_by_source,
        "cross_by_target":      cross_by_target,
        "alert_timeline":       alert_timeline,
        "gnn_rca":              rca,
        "impact_paths":         rca.get("impact_path") or [],
        # Totals for the status card
        "total_nodes":          len(nodes_by_id),
        "total_edges":          len(all_edges),
        "total_cross_edges":    len(cross_edges),
        "total_diagrams":       len(diagram_to_nodes),
        "total_scenarios":      len(scenario_to_diagrams),
    }
