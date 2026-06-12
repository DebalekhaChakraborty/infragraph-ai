"""Build sample LoRA + GRPO/vERL alignment records from InfraGraph RCA artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_remediation.prompt_builder import build_remediation_prompt  # noqa: E402
from ai_remediation.response_schema import make_remediation_input  # noqa: E402


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _scenario_dirs(dataset_root: Path) -> list[Path]:
    return sorted({
        p.parent for p in dataset_root.rglob("enterprise_graph.json")
        if (p.parent / "alerts.json").exists()
    })


def _alerts(alerts: dict) -> list[dict]:
    return list(alerts.get("alerts") or alerts.get("alert_timeline") or [])


def _timeline(alerts: dict) -> list[dict]:
    rows = []
    for idx, alert in enumerate(_alerts(alerts), 1):
        rows.append({
            "timestamp": alert.get("timestamp") or alert.get("time_label") or f"T+{(idx - 1) * 4:02d}m",
            "node_id": alert.get("node_id") or alert.get("node") or "",
            "diagram_id": alert.get("diagram_id") or alert.get("source_diagram") or "",
            "severity": alert.get("severity") or "major",
            "alert_type": alert.get("alert_type") or "alert",
            "message": alert.get("message") or alert.get("alert_message") or "",
        })
    return rows


def _gnn_result(gnn_root: Path, scenario_id: str) -> dict:
    for path in gnn_root.glob(f"*{scenario_id}*.json"):
        data = _load_json(path)
        if data:
            return data
    return {}


def _candidate_ranking(gnn: dict, alerts: dict, root_cause: str) -> list[dict]:
    candidates = list(gnn.get("top_candidates") or gnn.get("ranking") or [])
    ranking = []
    for rank, candidate in enumerate(candidates[:6], 1):
        ranking.append({
            "node_id": candidate.get("node_id") or candidate.get("node") or candidate.get("id") or "",
            "score": candidate.get("score", candidate.get("rca_score", "")),
            "reason": candidate.get("reason") or f"GNN rank {rank}",
        })
    if ranking:
        return ranking
    seen = {root_cause}
    ranking.append({"node_id": root_cause, "score": 0.97, "reason": "reference root cause"})
    for alert in _alerts(alerts):
        node = alert.get("node_id") or alert.get("node") or ""
        if node and node not in seen:
            seen.add(node)
            ranking.append({"node_id": node, "score": round(0.8 - len(ranking) * 0.08, 3), "reason": "alert evidence"})
    return ranking[:6]


def _device_context(graph: dict) -> list[dict]:
    return [
        {
            "node_id": node.get("id") or node.get("node_id") or "",
            "device_type": node.get("type") or node.get("class_name") or "",
            "ip_address": node.get("ip_address") or "",
            "diagram_id": node.get("diagram_id") or node.get("source_diagram") or "",
        }
        for node in list(graph.get("nodes") or [])[:24]
        if isinstance(node, dict)
    ]


def _connector_context(graph: dict) -> list[dict]:
    edges = list(graph.get("edges") or [])[:20] + list(graph.get("cross_diagram_edges") or [])[:8]
    return [
        {
            "source": edge.get("source", ""),
            "target": edge.get("target", ""),
            "type": edge.get("relationship") or edge.get("label") or "connected_to",
            "diagram_id": edge.get("diagram_id") or edge.get("source_diagram") or "",
        }
        for edge in edges
        if isinstance(edge, dict)
    ]


def _chosen_response(record: dict) -> dict:
    scope = record["scope"]
    root = record["root_cause"]
    diagrams = record["impacted_diagrams"]
    evidence_ids = ["E001", "E002"]
    blast = "enterprise_wide" if len(set(diagrams)) >= 4 else ("cross_diagram" if len(set(diagrams)) > 1 else "single_diagram")
    return {
        "executive_summary": f"{scope.title()} incident is graph-grounded to root cause {root}; validate before remediation.",
        "probable_root_cause": root,
        "scope": scope,
        "risk_level": "critical" if blast == "enterprise_wide" else ("high" if blast == "cross_diagram" else "medium"),
        "automation_eligibility": "manual_only" if scope == "enterprise" else "human_approval_required",
        "blast_radius": blast,
        "evidence_ids_used": evidence_ids,
        "evidence_from_graph": record["graph_evidence"][:4],
        "pre_checks": [
            f"Confirm active alerts still reference {root}.",
            "Validate graph evidence and RCA ranking before changes.",
        ],
        "triage_steps": [
            f"Inspect {root} logs and health using only the loaded graph context.",
            "Compare current alerts with the impact path before remediation.",
        ],
        "validation_steps": [
            f"Validate reachability from impacted nodes to {root}.",
            "Confirm alert timeline freshness before any change.",
        ],
        "remediation_steps": [
            f"After validation, restore service on {root} using approved operational procedure.",
            "Verify downstream nodes on the impact path recover before closing the incident.",
        ],
        "post_checks": [
            "Confirm no new alerts fire for impacted nodes.",
            "Re-run graph path validation across impacted diagrams.",
        ],
        "do_not_execute_if": [
            f"Do not execute if {root} is not confirmed by graph evidence.",
            "Do not execute without rollback owner approval.",
        ],
        "rollback_or_safety_notes": [
            "Capture pre-change state and rollback owner before remediation.",
            "Restore prior config/service state if post-checks fail.",
        ],
        "escalation_recommendation": (
            "Escalate to enterprise network/SRE owners for cross-diagram impact."
            if scope == "enterprise" else
            "Escalate to network engineering if local validation does not confirm recovery."
        ),
        "servicenow_incident_summary": {
            "short_description": f"InfraGraph RCA root cause {root}",
            "description": f"Graph-grounded RCA found {root}; impacted diagrams: {', '.join(diagrams) or 'single diagram'}.",
            "affected_ci": root,
            "priority": "1-Critical" if scope == "enterprise" else "2-High",
            "assignment_group": "Network Engineering",
        },
        "audit_summary": f"Chosen response validates before remediation, cites {', '.join(evidence_ids)}, and includes rollback.",
        "confidence_notes": "Scenario-grounded alignment record; use graph evidence and RCA labels only.",
    }


def _rejected_response(record: dict, idx: int) -> str:
    bad = [
        '{"executive_summary": "Restart FAKE-RTR-99 immediately.", "probable_root_cause": "FAKE-RTR-99"}',
        json.dumps({"probable_root_cause": record["root_cause"], "remediation_steps": ["Restart now"], "validation_steps": []}),
        "not valid json: remediate first, rollback later",
        json.dumps({"probable_root_cause": "WRONG-NODE", "validation_steps": ["check"], "remediation_steps": ["change"], "rollback_or_safety_notes": []}),
    ]
    return bad[idx % len(bad)]


def _record_from_scenario(scenario_dir: Path, gnn_root: Path, idx: int) -> dict | None:
    graph = _load_json(scenario_dir / "enterprise_graph.json")
    alerts = _load_json(scenario_dir / "alerts.json")
    if not graph or not alerts:
        return None
    scenario_id = str(alerts.get("scenario_id") or graph.get("scenario_id") or scenario_dir.name)
    root = str(alerts.get("root_cause") or "")
    root_diagram = str(alerts.get("root_cause_diagram") or "")
    impacted_diagrams = list(alerts.get("impacted_diagrams") or [])
    impact_paths = list(alerts.get("impact_paths") or [])
    impact_path = impact_paths[0] if impact_paths else []
    timeline = _timeline(alerts)
    gnn = _gnn_result(gnn_root, scenario_id)
    ranking = _candidate_ranking(gnn, alerts, root)
    impacted_nodes = sorted({
        alert.get("node_id") or alert.get("node") or ""
        for alert in _alerts(alerts)
        if alert.get("node_id") or alert.get("node")
    })
    graph_evidence = [
        f"E001: Root cause {root} appears in diagram {root_diagram}.",
        f"E002: Impacted diagrams are {', '.join(impacted_diagrams) or root_diagram}.",
        f"E003: Impact path is {' -> '.join(map(str, impact_path)) or 'not provided'}.",
        f"E004: Candidate ranking top node is {ranking[0]['node_id'] if ranking else root}.",
    ]
    scope = "enterprise" if len(set(impacted_diagrams)) > 1 else "local"
    ctx = make_remediation_input(
        incident_id=f"RL-{scenario_id}",
        scope=scope,
        selected_diagram_id=root_diagram,
        scenario_id=scenario_id,
        alert_timeline=timeline,
        graph_memory_summary=f"{len(graph.get('nodes', []))} nodes and {len(graph.get('edges', []))} edges.",
        root_cause=root,
        root_cause_diagram=root_diagram,
        impacted_nodes=impacted_nodes,
        impacted_diagrams=impacted_diagrams,
        impact_path=impact_path,
        candidate_ranking=ranking,
        gnn_result_available=bool(gnn),
        rca_source="Enterprise GNN RCA" if gnn else "Scenario-grounded RCA simulation",
        device_context=_device_context(graph),
        connector_context=_connector_context(graph),
    )
    prompt = "\n".join(f"[{m['role']}]\n{m['content']}" for m in build_remediation_prompt(ctx))
    record = {
        "id": f"rca_rl_{idx:05d}",
        "scenario_id": scenario_id,
        "scope": scope,
        "root_cause": root,
        "root_cause_diagram": root_diagram,
        "impacted_nodes": impacted_nodes,
        "impacted_diagrams": impacted_diagrams,
        "impact_path": impact_path,
        "alert_timeline": timeline,
        "candidate_ranking": ranking,
        "graph_evidence": graph_evidence,
        "prompt": prompt,
        "chosen_response": json.dumps(_chosen_response({
            "scope": scope,
            "root_cause": root,
            "impacted_diagrams": impacted_diagrams,
            "graph_evidence": graph_evidence,
        }), ensure_ascii=False),
        "rejected_response": _rejected_response({"root_cause": root}, idx),
        "reward_tags": [
            "graph_grounded",
            "validation_before_remediation",
            "rollback_required",
            "enterprise_escalation_required" if scope == "enterprise" else "single_diagram_scope",
        ],
    }
    return record


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sample RCA remediation RL JSONL files.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--dataset-root", default="./datasets/infragraph_v3")
    parser.add_argument("--gnn-results", default="./demo_assets/enterprise_gnn_rca")
    parser.add_argument("--out-dir", default="./training/verl_grpo/data")
    parser.add_argument("--max-records", type=int, default=80)
    args = parser.parse_args()

    repo = Path(args.repo_root)
    dataset_root = (repo / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    gnn_root = (repo / args.gnn_results).resolve() if not Path(args.gnn_results).is_absolute() else Path(args.gnn_results)
    out_dir = (repo / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    rows = []
    for idx, scenario_dir in enumerate(_scenario_dirs(dataset_root), 1):
        record = _record_from_scenario(scenario_dir, gnn_root, idx)
        if record:
            rows.append(record)
        if len(rows) >= args.max_records:
            break

    if not rows:
        rows.append({
            "id": "rca_rl_sample_00001",
            "scenario_id": "sample_scenario",
            "scope": "enterprise",
            "root_cause": "APP-LB-01",
            "root_cause_diagram": "app_db_topology",
            "impacted_nodes": ["APP-01", "APP-02", "DB-01"],
            "impacted_diagrams": ["branch_topology", "wan_topology", "app_db_topology"],
            "impact_path": ["APP-LB-01", "APP-01", "DB-01"],
            "alert_timeline": [],
            "candidate_ranking": [{"node_id": "APP-LB-01", "score": 0.97, "reason": "sample"}],
            "graph_evidence": ["E001: APP-LB-01 is the reference root cause."],
            "prompt": "Return graph-grounded JSON remediation for APP-LB-01.",
            "chosen_response": json.dumps(_chosen_response({
                "scope": "enterprise",
                "root_cause": "APP-LB-01",
                "impacted_diagrams": ["branch_topology", "wan_topology", "app_db_topology"],
                "graph_evidence": ["E001: APP-LB-01 is the reference root cause."],
            })),
            "rejected_response": '{"probable_root_cause":"FAKE-DEVICE","remediation_steps":["restart now"]}',
            "reward_tags": ["sample", "graph_grounded"],
        })

    split = max(1, int(len(rows) * 0.8))
    train_rows = rows[:split]
    eval_rows = rows[split:] or rows[:1]
    train_path = out_dir / "rca_remediation_rl_train.jsonl"
    eval_path = out_dir / "rca_remediation_rl_eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    print(f"train records: {len(train_rows)} -> {train_path}")
    print(f"eval records: {len(eval_rows)} -> {eval_path}")


if __name__ == "__main__":
    main()
