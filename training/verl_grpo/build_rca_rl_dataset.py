"""
Build GRPO RL training dataset for Qwen3 remediation fine-tuning.

Reads V3 enterprise scenarios and per-diagram local sub-graphs,
constructs prompt/reference pairs, and writes a JSONL file suitable
for vERL/GRPO training.

Each record includes a ``scope`` field ("local" or "enterprise"),
``required_diagrams`` (for cross-diagram reward) and
``required_sections`` (JSON output keys that must be present).

Usage
-----
python training/verl_grpo/build_rca_rl_dataset.py \\
    --dataset-root ./datasets/infragraph_v3 \\
    --gnn-results  ./outputs/enterprise_gnn_rca \\
    --out          ./data/rl_training/infragraph_rca_remediation_grpo.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from ai_remediation.prompt_builder  import build_remediation_prompt
from ai_remediation.response_schema import make_remediation_input


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_alert_timeline_from_alerts(ad: dict) -> list[dict]:
    events = []
    for i, a in enumerate(ad.get("alerts", []), 1):
        events.append({
            "step":             i,
            "timestamp":        f"T+{(i-1)*4:02d}m",
            "node_id":          a.get("node_id", a.get("node", "")),
            "diagram_id":       a.get("diagram_id", a.get("source_diagram", "")),
            "device_type":      a.get("class", a.get("type", "server")),
            "alert_type":       a.get("alert_type", "Service alert"),
            "message":          a.get("message", a.get("alert_message", "")),
            "severity":         a.get("severity", "major"),
            "correlation_role": "",
        })
    return events


def _build_candidate_ranking_ent(gnn: "dict | None", ad: dict, root_cause: str) -> list[dict]:
    if gnn:
        ranking = []
        for c in gnn.get("top_candidates", [])[:6]:
            ranking.append({
                "node_id": c.get("node_id", ""),
                "score":   c.get("score", 0.0),
                "reason":  f"GNN rank {c.get('rank','?')} type={c.get('type','?')}",
            })
        if ranking:
            return ranking
    ranking = [{"node_id": root_cause, "score": 0.97, "reason": "ground-truth root cause"}]
    seen = {root_cause}
    for a in ad.get("alerts", [])[:5]:
        n = a.get("node_id", a.get("node", ""))
        if n and n not in seen:
            seen.add(n)
            ranking.append({
                "node_id": n,
                "score":   round(0.70 - len(ranking) * 0.07, 3),
                "reason":  "alert evidence",
            })
    return ranking[:6]


def _build_device_context_from_graph(eg: dict, max_nodes: int = 20) -> list[dict]:
    devices = []
    for n in eg.get("nodes", [])[:max_nodes]:
        devices.append({
            "node_id":    n.get("id", n.get("node_id", "")),
            "device_type": n.get("type", n.get("class_name", "")),
            "ip_address": n.get("ip_address", ""),
            "diagram_id": n.get("diagram_id", n.get("source_diagram", "")),
        })
    return devices


def _build_connector_context_from_graph(eg: dict) -> list[dict]:
    links = []
    for e in eg.get("edges", [])[:15]:
        links.append({
            "source":     e.get("source", ""),
            "target":     e.get("target", ""),
            "type":       e.get("label", e.get("relationship", "")),
            "diagram_id": e.get("diagram_id", ""),
        })
    for e in eg.get("cross_diagram_edges", [])[:5]:
        links.append({
            "source":     e.get("source", ""),
            "target":     e.get("target", ""),
            "type":       "cross_diagram",
            "diagram_id": "",
        })
    return links


def _find_gnn_result(gnn_root: Path, scenario_id: str) -> "dict | None":
    if not gnn_root.exists():
        return None
    for p in gnn_root.glob(f"*{scenario_id}*.json"):
        try:
            return _load_json(p)
        except Exception:
            pass
    return None


_REQUIRED_SECTIONS = [
    "executive_summary",
    "probable_root_cause",
    "scope",
    "evidence_from_graph",
    "triage_steps",
    "validation_steps",
    "remediation_steps",
    "rollback_or_safety_notes",
    "escalation_recommendation",
    "servicenow_incident_summary",
    "confidence_notes",
]


# ── Enterprise record ─────────────────────────────────────────────────────────

def _build_enterprise_reference(ad: dict, root_cause: str) -> dict:
    impact_paths = ad.get("impact_paths", [])
    impact_path  = impact_paths[0] if impact_paths else []
    all_nodes    = list({
        a.get("node_id", a.get("node", ""))
        for a in ad.get("alerts", [])
        if a.get("node_id", a.get("node", ""))
    })
    imp_diagrams = list(ad.get("impacted_diagrams", []))
    return {
        "root_cause":         root_cause,
        "required_nodes":     [root_cause] + [n for n in impact_path if n != root_cause],
        "required_diagrams":  imp_diagrams,
        "valid_node_set":     all_nodes,
        "required_sections":  _REQUIRED_SECTIONS,
        "scope":              "enterprise",
        "required_steps": [
            f"Investigate {root_cause} for connectivity or configuration fault.",
            "Validate upstream path from affected nodes to root cause.",
            "Restore service and confirm alert suppression.",
        ],
    }


def process_enterprise_scenario(
    scenario_dir: Path,
    gnn_root: Path,
) -> "dict | None":
    eg_path = scenario_dir / "enterprise_graph.json"
    ad_path = scenario_dir / "alerts.json"
    if not eg_path.exists() or not ad_path.exists():
        return None

    try:
        eg = _load_json(eg_path)
        ad = _load_json(ad_path)
    except Exception as exc:
        print(f"  skip {scenario_dir.name}: {exc}", file=sys.stderr)
        return None

    scenario_id   = ad.get("scenario_id", scenario_dir.name)
    root_cause    = ad.get("root_cause", "")
    rc_diagram    = ad.get("root_cause_diagram", "")
    imp_diagrams  = list(ad.get("impacted_diagrams", []))
    impact_paths  = ad.get("impact_paths", [])
    impact_path   = impact_paths[0] if impact_paths else []

    gnn           = _find_gnn_result(gnn_root, scenario_id)
    gnn_available = gnn is not None
    rca_source    = "Enterprise GNN RCA" if gnn_available else "Scenario-grounded RCA simulation"

    clusters = eg.get("diagram_clusters", {})
    n_diags  = len(clusters) if isinstance(clusters, dict) else len(clusters)
    graph_summary = (
        f"{len(eg.get('nodes',[]))} nodes, {len(eg.get('edges',[]))} edges, "
        f"{len(eg.get('cross_diagram_edges',[]))} cross-diagram edges across {n_diags} domains."
    )

    context = make_remediation_input(
        incident_id=f"ENT-{scenario_id}",
        scope="enterprise",
        selected_diagram_id=ad.get("selected_diagram_id", ""),
        scenario_id=scenario_id,
        alert_timeline=_build_alert_timeline_from_alerts(ad),
        graph_memory_summary=graph_summary,
        root_cause=root_cause,
        root_cause_diagram=rc_diagram,
        impacted_nodes=list({
            a.get("node_id", a.get("node", ""))
            for a in ad.get("alerts", [])
            if a.get("node_id", a.get("node", ""))
        }),
        impacted_diagrams=imp_diagrams,
        impact_path=impact_path,
        candidate_ranking=_build_candidate_ranking_ent(gnn, ad, root_cause),
        gnn_result_available=gnn_available,
        rca_source=rca_source,
        device_context=_build_device_context_from_graph(eg),
        connector_context=_build_connector_context_from_graph(eg),
    )

    messages    = build_remediation_prompt(context)
    prompt_text = "\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)

    return {
        "prompt":    prompt_text,
        "messages":  messages,
        "reference": _build_enterprise_reference(ad, root_cause),
        "metadata": {
            "scope":              "enterprise",
            "scenario_id":        scenario_id,
            "root_cause":         root_cause,
            "root_cause_diagram": rc_diagram,
            "impacted_diagrams":  imp_diagrams,
            "gnn_available":      gnn_available,
            "rca_source":         rca_source,
            "n_alerts":           len(ad.get("alerts", [])),
            "n_nodes":            len(eg.get("nodes", [])),
        },
    }


# ── Local record ──────────────────────────────────────────────────────────────

def _extract_diagram_subgraph(eg: dict, diagram_id: str) -> dict:
    """Extract nodes and edges belonging to a single diagram from enterprise_graph."""
    nodes = [
        n for n in eg.get("nodes", [])
        if n.get("diagram_id", n.get("source_diagram", "")) == diagram_id
    ]
    node_ids = {n.get("id", n.get("node_id", "")) for n in nodes}
    edges = [
        e for e in eg.get("edges", [])
        if e.get("source", "") in node_ids and e.get("target", "") in node_ids
    ]
    return {"nodes": nodes, "edges": edges}


def _build_local_candidate_ranking(ad: dict, diagram_id: str, root_cause: str) -> list[dict]:
    ranking = [{"node_id": root_cause, "score": 0.95, "reason": "ground-truth root cause"}]
    seen = {root_cause}
    for a in ad.get("alerts", []):
        if a.get("diagram_id", a.get("source_diagram", "")) != diagram_id:
            continue
        n = a.get("node_id", a.get("node", ""))
        if n and n not in seen:
            seen.add(n)
            ranking.append({
                "node_id": n,
                "score":   round(0.72 - len(ranking) * 0.08, 3),
                "reason":  "local alert evidence",
            })
    return ranking[:6]


def _build_local_reference(ad: dict, diagram_id: str, root_cause: str) -> dict:
    local_alerts = [
        a for a in ad.get("alerts", [])
        if a.get("diagram_id", a.get("source_diagram", "")) == diagram_id
    ]
    all_nodes = list({
        a.get("node_id", a.get("node", ""))
        for a in local_alerts
        if a.get("node_id", a.get("node", ""))
    })
    impact_paths = ad.get("impact_paths", [])
    impact_path  = [n for p in impact_paths for n in p if n in set(all_nodes + [root_cause])]
    return {
        "root_cause":        root_cause,
        "required_nodes":    [root_cause] + [n for n in impact_path if n != root_cause],
        "required_diagrams": [diagram_id],
        "valid_node_set":    all_nodes,
        "required_sections": _REQUIRED_SECTIONS,
        "scope":             "local",
        "required_steps": [
            f"Investigate {root_cause} on diagram {diagram_id}.",
            "Validate local connectivity and service health.",
            "Restore service and confirm alert suppression.",
        ],
    }


def process_local_scenarios(
    scenario_dir: Path,
) -> list[dict]:
    """Build one JSONL record per diagram from a scenario directory.

    Returns an empty list if required files are missing.
    """
    eg_path = scenario_dir / "enterprise_graph.json"
    ad_path = scenario_dir / "alerts.json"
    if not eg_path.exists() or not ad_path.exists():
        return []

    try:
        eg = _load_json(eg_path)
        ad = _load_json(ad_path)
    except Exception as exc:
        print(f"  skip (local) {scenario_dir.name}: {exc}", file=sys.stderr)
        return []

    scenario_id  = ad.get("scenario_id", scenario_dir.name)
    root_cause   = ad.get("root_cause", "")
    rc_diagram   = ad.get("root_cause_diagram", "")
    imp_diagrams = list(ad.get("impacted_diagrams", []))

    records = []
    # Only emit local records for diagrams mentioned in the scenario
    target_diagrams = list({rc_diagram} | set(imp_diagrams)) if imp_diagrams else [rc_diagram]
    target_diagrams = [d for d in target_diagrams if d]

    for diagram_id in target_diagrams:
        sub_graph = _extract_diagram_subgraph(eg, diagram_id)
        if not sub_graph["nodes"]:
            continue

        local_alerts = [
            a for a in ad.get("alerts", [])
            if a.get("diagram_id", a.get("source_diagram", "")) == diagram_id
        ]
        timeline = _build_alert_timeline_from_alerts({"alerts": local_alerts})

        # First observed node: earliest alert on this diagram
        first_obs = local_alerts[0].get("node_id", local_alerts[0].get("node", "")) if local_alerts else ""

        # For local records, root cause only applies if it's in this diagram
        local_root = root_cause if diagram_id == rc_diagram else (
            local_alerts[0].get("node_id", local_alerts[0].get("node", "")) if local_alerts else ""
        )
        if not local_root:
            continue

        imp_nodes = list({
            a.get("node_id", a.get("node", ""))
            for a in local_alerts
            if a.get("node_id", a.get("node", ""))
        })
        n_sub = len(sub_graph["nodes"])
        n_edge = len(sub_graph["edges"])
        graph_summary = f"{n_sub} nodes, {n_edge} edges in {diagram_id}."

        context = make_remediation_input(
            incident_id=f"LOC-{scenario_id}-{diagram_id}",
            scope="local",
            selected_diagram_id=diagram_id,
            diagram_type=diagram_id,
            scenario_id=scenario_id,
            alert_timeline=timeline,
            graph_memory_summary=graph_summary,
            root_cause=local_root,
            root_cause_diagram=diagram_id,
            first_observed_node=first_obs,
            impacted_nodes=imp_nodes,
            impacted_diagrams=[diagram_id],
            impact_path=[],
            candidate_ranking=_build_local_candidate_ranking(ad, diagram_id, local_root),
            gnn_result_available=False,
            rca_source="Topology BFS RCA",
            device_context=_build_device_context_from_graph(sub_graph, max_nodes=12),
            connector_context=_build_connector_context_from_graph(sub_graph),
        )

        messages    = build_remediation_prompt(context)
        prompt_text = "\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)

        records.append({
            "prompt":    prompt_text,
            "messages":  messages,
            "reference": _build_local_reference(ad, diagram_id, local_root),
            "metadata": {
                "scope":       "local",
                "scenario_id": scenario_id,
                "diagram_id":  diagram_id,
                "root_cause":  local_root,
                "rca_source":  "Topology BFS RCA",
                "n_alerts":    len(local_alerts),
                "n_nodes":     n_sub,
            },
        })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GRPO RL training dataset from V3 scenarios (local + enterprise)."
    )
    parser.add_argument("--dataset-root", required=True, help="Path to datasets/infragraph_v3")
    parser.add_argument("--gnn-results",  default="outputs/enterprise_gnn_rca",
                        help="Directory containing GNN result JSON files")
    parser.add_argument("--out", required=True,
                        help="Output JSONL path")
    parser.add_argument("--no-local",     action="store_true",
                        help="Skip local (per-diagram) records")
    parser.add_argument("--no-enterprise", action="store_true",
                        help="Skip enterprise (cross-diagram) records")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    gnn_root     = Path(args.gnn_results)
    out_path     = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenario_dirs = sorted([
        p.parent for p in dataset_root.rglob("enterprise_graph.json")
        if (p.parent / "alerts.json").exists()
    ])
    print(f"Found {len(scenario_dirs)} scenarios under {dataset_root}")

    ent_count   = 0
    local_count = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for sd in scenario_dirs:
            if not args.no_enterprise:
                rec = process_enterprise_scenario(sd, gnn_root)
                if rec:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    ent_count += 1

            if not args.no_local:
                for rec in process_local_scenarios(sd):
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    local_count += 1

            total = ent_count + local_count
            if total > 0 and total % 20 == 0:
                print(f"  {total} records written (ent={ent_count}, local={local_count})…")

    total = ent_count + local_count
    print(
        f"\nDataset complete: {total} records "
        f"(enterprise={ent_count}, local={local_count}) → {out_path}"
    )


if __name__ == "__main__":
    main()
