#!/usr/bin/env python3
"""
predict_enterprise_gnn_rca.py — Run Enterprise GNN RCA prediction for one scenario.

Reads:
  scenario_library/enterprise_gnn_rca/<case_id>/events.json
  scenario_library/enterprise_gnn_rca/<case_id>/graph_ref.json
  model_artifacts/enterprise_gnn_rca/enterprise_gnn_rca.pt
  model_artifacts/enterprise_gnn_rca/enterprise_gnn_config.json

Writes:
  assets/preloaded/enterprise_gnn_rca/<scenario_id>.json

Output contains predicted root-cause ranking only.
No remediation content is produced here.

Usage:
  python scripts/predict_enterprise_gnn_rca.py --scenario-id enterprise_v3_0000
  python scripts/predict_enterprise_gnn_rca.py --case-id ent_enterprise_v3_0000
  python scripts/predict_enterprise_gnn_rca.py --scenario-id enterprise_v3_0000 --top-k 5 --no-eval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rca_ml.enterprise_gnn_model import (  # noqa: E402
    check_torch_geo_requirement,
    load_gnn,
)
from rca_ml.enterprise_gnn_dataset import (  # noqa: E402
    build_graph_dict,
    load_enterprise_case,
)
from rca_ml.enterprise_gnn_inference import predict_one  # noqa: E402


def _find_manifest_row(
    manifest_rows: list[dict],
    scenario_id: str | None,
    case_id: str | None,
) -> dict | None:
    for row in manifest_rows:
        if scenario_id and row.get("scenario_id") == scenario_id:
            return row
        if case_id and row.get("case_id") == case_id:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Enterprise GNN RCA prediction for one scenario."
    )
    parser.add_argument("--scenario-id",     default=None,
                        help="scenario_id from manifest (e.g. enterprise_v3_0000)")
    parser.add_argument("--case-id",         default=None,
                        help="case_id from manifest (e.g. ent_enterprise_v3_0000)")
    parser.add_argument("--manifest",        default="scenario_library/manifest.json")
    parser.add_argument("--scenario-library", default="scenario_library")
    parser.add_argument("--model",           default="model_artifacts/enterprise_gnn_rca/enterprise_gnn_rca.pt")
    parser.add_argument("--config",          default="model_artifacts/enterprise_gnn_rca/enterprise_gnn_config.json")
    parser.add_argument("--out",             default="assets/preloaded/enterprise_gnn_rca")
    parser.add_argument("--top-k",  type=int, default=3)
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip ground-truth comparison (don't read labels.json)")
    args = parser.parse_args()

    if not args.scenario_id and not args.case_id:
        parser.error("Provide --scenario-id or --case-id")

    check_torch_geo_requirement()

    repo_root  = REPO_ROOT
    lib_root   = (repo_root / args.scenario_library).resolve()
    model_path = (repo_root / args.model).resolve()
    config_path = Path(args.config) if Path(args.config).is_absolute() else (repo_root / args.config).resolve()
    out_dir    = (repo_root / args.out).resolve()

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("        Run scripts/train_enterprise_gnn_rca.py first.")
        sys.exit(1)

    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows     = manifest.get("enterprise_gnn_rca", [])

    if not rows:
        print("[ERROR] scenario_library manifest has no enterprise_gnn_rca cases.")
        print("        Re-run scripts/build_scenario_library.py with enterprise mode enabled.")
        sys.exit(1)

    row = _find_manifest_row(rows, args.scenario_id, args.case_id)
    if row is None:
        key = args.scenario_id or args.case_id
        print(f"[ERROR] Case not found in manifest: {key!r}")
        print(f"        Available scenario IDs: {[r['scenario_id'] for r in rows[:5]]} ...")
        sys.exit(1)

    # Load case
    events, labels, graph_ref, enterprise_graph, stitch_map = (
        load_enterprise_case(lib_root, row, repo_root)
    )

    root_cause_node = labels.get("root_cause_node", "") if not args.no_eval else None

    case_id     = row["case_id"]
    scenario_id = row.get("scenario_id", "")

    graph_dict = build_graph_dict(
        case_id=case_id,
        scenario_id=scenario_id,
        split=row.get("split", "infer"),
        enterprise_graph=enterprise_graph,
        events=events,
        root_cause_node=None,  # never leak ground truth into features
    )

    if graph_dict is None:
        print(f"[ERROR] Failed to build graph for case {case_id}. Check enterprise_graph.json.")
        sys.exit(1)

    # Load model
    model, config = load_gnn(model_path, config_path)

    # Predict
    labels_for_eval = labels if not args.no_eval else None
    result = predict_one(model, graph_dict, labels_dict=labels_for_eval, top_k=args.top_k)

    # Print summary
    print(f"Scenario         : {scenario_id}")
    print(f"Predicted root   : {result['predicted_root_cause']}")
    print(f"Diagram          : {result['root_cause_diagram']}")
    print(f"Confidence       : {result['confidence']}")
    print(f"Top-{args.top_k} candidates :")
    for c in result["top_candidates"]:
        gt_marker = " <-- expected" if c["node_id"] == result.get("expected_root_cause") else ""
        print(f"  #{c['rank']:>2}  {c['node_id']:<40}  score={c['score']:.4f}  "
              f"diag={c['diagram_id']}  type={c['node_type']}{gt_marker}")
    if "correct" in result and result["correct"] is not None:
        print(f"Correct          : {result['correct']}")
    if "expected_root_cause" in result:
        print(f"Expected root    : {result['expected_root_cause']}")

    # Write output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scenario_id}.json"
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nResult written   : {out_path}")


if __name__ == "__main__":
    main()
