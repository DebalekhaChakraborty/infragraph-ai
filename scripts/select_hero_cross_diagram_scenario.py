"""Select the strongest cross-diagram V3 hero scenario for the Streamlit app."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _scenario_score(scenario_dir: Path, gnn_dir: Path) -> tuple[int, dict]:
    enterprise_graph = _load_json(scenario_dir / "enterprise_graph.json")
    alerts = _load_json(scenario_dir / "alerts.json")
    if not enterprise_graph or not alerts:
        return -1, {}

    scenario_id = str(alerts.get("scenario_id") or enterprise_graph.get("scenario_id") or scenario_dir.name)
    impacted_diagrams = list(alerts.get("impacted_diagrams") or [])
    alert_events = list(alerts.get("alerts") or alerts.get("alert_timeline") or [])
    cross_edges = list(enterprise_graph.get("cross_diagram_edges") or [])
    clusters = enterprise_graph.get("diagram_clusters") or []
    if isinstance(clusters, dict):
        diagram_count = len(clusters)
    else:
        diagram_count = len(clusters)

    gnn_result_path = ""
    for candidate in gnn_dir.glob(f"*{scenario_id}*.json"):
        if candidate.name.endswith(".json"):
            gnn_result_path = str(candidate)
            break

    score = (
        min(diagram_count, 5) * 10
        + min(len(set(impacted_diagrams)), 5) * 14
        + min(len(cross_edges), 8) * 5
        + min(len(alert_events), 7) * 4
        + (20 if gnn_result_path else 0)
    )
    summary = {
        "scenario_id": scenario_id,
        "scenario_path": str(scenario_dir),
        "diagram_count": diagram_count,
        "impacted_diagram_count": len(set(impacted_diagrams)),
        "impacted_diagrams": impacted_diagrams,
        "alert_count": len(alert_events),
        "cross_diagram_edge_count": len(cross_edges),
        "root_cause": alerts.get("root_cause", ""),
        "root_cause_diagram": alerts.get("root_cause_diagram", ""),
        "gnn_result_available": bool(gnn_result_path),
        "gnn_result_path": gnn_result_path,
        "rca_label": "Enterprise GNN inference result" if gnn_result_path else "Scenario-grounded cross-diagram RCA simulation",
        "score": score,
    }
    return score, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best V3 cross-diagram hero scenario.")
    parser.add_argument("--dataset-root", default="./datasets/infragraph_v3")
    parser.add_argument("--gnn-results", default="./assets/preloaded/enterprise_gnn_rca")
    parser.add_argument("--out", default="./assets/preloaded/demo_hero/hero_scenario.json")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    gnn_dir = Path(args.gnn_results)
    out_path = Path(args.out)
    scenario_dirs = sorted({
        p.parent for p in dataset_root.rglob("enterprise_graph.json")
        if (p.parent / "alerts.json").exists()
    })

    best: dict = {}
    best_score = -1
    for scenario_dir in scenario_dirs:
        score, summary = _scenario_score(scenario_dir, gnn_dir)
        if score > best_score:
            best_score = score
            best = summary

    if not best:
        raise SystemExit(f"No usable scenarios found under {dataset_root}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Selected {best['scenario_id']} -> {out_path}")
    print(
        f"diagrams={best['diagram_count']} impacted={best['impacted_diagram_count']} "
        f"alerts={best['alert_count']} cross_edges={best['cross_diagram_edge_count']} "
        f"gnn={best['gnn_result_available']}"
    )


if __name__ == "__main__":
    main()
