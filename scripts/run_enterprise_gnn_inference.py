#!/usr/bin/env python3
"""
run_enterprise_gnn_inference.py

Run Enterprise GNN RCA inference for a specific V3 scenario using the
trained GraphSAGE EnterpriseRcaGNN checkpoint.

Usage
-----
python scripts/run_enterprise_gnn_inference.py \\
    --scenario-id  enterprise_v3_0077 \\
    --model-path   model_artifacts/enterprise_gnn_rca/enterprise_gnn_rca.pt \\
    --out          outputs/enterprise_gnn_rca

# Optional: restrict to a specific split
python scripts/run_enterprise_gnn_inference.py \\
    --scenario-id  enterprise_v3_0001 \\
    --split        train \\
    --out          outputs/enterprise_gnn_rca

Output
------
outputs/enterprise_gnn_rca/<scenario_id>_enterprise_gnn_rca_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

# Suppress FutureWarning noise before any torch imports so it doesn't
# obscure real errors in the UI / terminal output.
warnings.filterwarnings("ignore", category=FutureWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_src_dir   = str(_REPO_ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from rca_ml.enterprise_gnn_model import load_gnn, check_torch_geo_requirement
    from rca_ml.enterprise_gnn_inference import predict_one
    from rca_ml.enterprise_gnn_dataset import build_graph_dict
except ImportError as exc:
    print(f"[ERROR] Cannot import rca_ml: {exc}")
    print("        Ensure src/ is on PYTHONPATH and the rca_ml package is present.")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("[ERROR] PyTorch is not installed.  Install with:  pip install torch")
    sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Enterprise GNN RCA inference for a selected V3 scenario"
    )
    p.add_argument("--scenario-id",  required=True,
                   help="Scenario ID, e.g. enterprise_v3_0077")
    p.add_argument("--split",        default=None,
                   choices=["train", "val", "test"],
                   help="Dataset split (auto-detected from graph_index.json if omitted)")
    p.add_argument("--model-path",   default=None,
                   help="Path to enterprise_gnn_rca.pt  "
                        "(default: model_artifacts/enterprise_gnn_rca/enterprise_gnn_rca.pt)")
    p.add_argument("--graphs-path",  default=None,
                   help="Path to prebuilt graphs.pt  "
                        "(default: data/rca/enterprise_gnn/graphs.pt; "
                        "falls back to on-the-fly build from V3 dataset if absent)")
    p.add_argument("--index-path",   default=None,
                   help="Path to graph_index.json  "
                        "(default: data/rca/enterprise_gnn/graph_index.json)")
    p.add_argument("--dataset-root", default=None,
                   help="V3 dataset root for on-the-fly graph build  "
                        "(default: datasets/infragraph_v3)")
    p.add_argument("--out",          default="outputs/enterprise_gnn_rca",
                   help="Output directory for inference result JSON")
    return p.parse_args()


# ── V3 scenario loader ────────────────────────────────────────────────────────

def _load_scenario_v3(
    dataset_root: Path,
    scenario_id: str,
    split: str | None,
    index_record: dict | None,
) -> tuple[dict, list[dict], str | None, str]:
    """
    Load enterprise_graph and alert events from a V3 scenario directory.

    Returns (enterprise_graph, events, root_cause_node, split_found).
    root_cause_node is taken from index_record when available, else None.
    """
    splits_to_try = [split] if split else ["train", "val", "test"]
    for sp in splits_to_try:
        sc_dir = dataset_root / "scenarios" / sp / scenario_id
        if not sc_dir.exists():
            continue

        eg_path = sc_dir / "enterprise_graph.json"
        al_path = sc_dir / "alerts.json"

        if not eg_path.exists():
            raise FileNotFoundError(
                f"enterprise_graph.json not found in {sc_dir}"
            )
        enterprise_graph = json.loads(eg_path.read_text(encoding="utf-8"))

        events: list[dict] = []
        if al_path.exists():
            al_doc = json.loads(al_path.read_text(encoding="utf-8"))
            # V3 stores events under "alerts" key; same field shape as build_graph_dict expects
            events = al_doc.get("alerts", al_doc.get("events", []))

        root_cause_node: str | None = (
            index_record.get("root_cause_node") or None
            if index_record else None
        )
        return enterprise_graph, events, root_cause_node, sp

    searched = [
        f"{dataset_root}/scenarios/{sp}/{scenario_id}"
        for sp in splits_to_try
    ]
    raise FileNotFoundError(
        f"Scenario '{scenario_id}' not found.  Searched:\n"
        + "\n".join(f"  {p}" for p in searched)
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = _parse_args()
    out_dir = Path(args.out)
    sid     = args.scenario_id

    # ── Resolve paths ──────────────────────────────────────────────────────────
    model_path = (
        Path(args.model_path) if args.model_path
        else _REPO_ROOT / "model_artifacts" / "enterprise_gnn_rca" / "enterprise_gnn_rca.pt"
    )
    config_path = model_path.with_name("enterprise_gnn_config.json")
    graphs_path = (
        Path(args.graphs_path) if args.graphs_path
        else _REPO_ROOT / "data" / "rca" / "enterprise_gnn" / "graphs.pt"
    )
    index_path = (
        Path(args.index_path) if args.index_path
        else _REPO_ROOT / "data" / "rca" / "enterprise_gnn" / "graph_index.json"
    )
    dataset_root = (
        Path(args.dataset_root) if args.dataset_root
        else _REPO_ROOT / "datasets" / "infragraph_v3"
    )

    print("InfraGraph AI — Enterprise GNN Inference")
    print(f"  Scenario : {sid}")
    print(f"  Model    : {model_path}")
    print(f"  Out dir  : {out_dir}")

    # ── Validate model files ───────────────────────────────────────────────────
    if not model_path.exists():
        print(f"\n[ERROR] Model checkpoint not found: {model_path}")
        print("Train first with:")
        print("  python scripts/build_enterprise_gnn_dataset.py")
        print("  python scripts/train_enterprise_gnn_rca.py \\")
        print("      --graphs  data/rca/enterprise_gnn/graphs.pt \\")
        print("      --index   data/rca/enterprise_gnn/graph_index.json \\")
        print("      --out-dir model_artifacts/enterprise_gnn_rca")
        sys.exit(1)
    if not config_path.exists():
        print(f"\n[ERROR] Model config not found: {config_path}")
        sys.exit(1)

    # ── Load model ─────────────────────────────────────────────────────────────
    check_torch_geo_requirement()
    print("\nLoading model...")
    try:
        model, config = load_gnn(model_path, config_path)
    except Exception:
        print("[ERROR] Failed to load model checkpoint:")
        traceback.print_exc()
        sys.exit(1)
    print(f"  Architecture : {config.get('gnn_architecture', 'GraphSAGE')} "
          f"({config.get('num_layers', 3)} layers, "
          f"hidden={config.get('hidden_channels', 64)}, "
          f"in_channels={config.get('in_channels', 54)})")
    print(f"  Parameters   : {sum(p.numel() for p in model.parameters()):,}")

    # ── Load graph index (provides labels_dict for evaluation) ─────────────────
    index_record: dict | None = None
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for rec in index:
                if rec.get("scenario_id") == sid:
                    if args.split is None or rec.get("split") == args.split:
                        index_record = rec
                        break
        except Exception:
            print("[WARNING] Could not load graph_index.json:")
            traceback.print_exc()

        if index_record:
            print(f"  Index record : {index_record['case_id']} "
                  f"(split={index_record.get('split', '?')})")
        else:
            print(f"  [warning] '{sid}' not found in graph_index.json — "
                  f"evaluation fields will be omitted.")

    # ── Load / build graph_dict ────────────────────────────────────────────────
    graph_dict: dict | None = None

    # Attempt 1: prebuilt graphs.pt (fast path, may not exist)
    if graphs_path.exists():
        print(f"\nLoading prebuilt graphs from {graphs_path}...")
        try:
            graphs = torch.load(str(graphs_path), map_location="cpu")
            for g in graphs:
                if g.get("scenario_id") == sid:
                    if args.split is None or g.get("split") == args.split:
                        graph_dict = g
                        print(f"  Found in graphs.pt  "
                              f"(nodes={g.get('num_nodes')}, "
                              f"events={g.get('event_count')})")
                        break
            if graph_dict is None:
                print(f"  [warning] '{sid}' not in graphs.pt — "
                      f"falling back to on-the-fly build.")
        except Exception:
            print("[WARNING] Could not load graphs.pt — "
                  "falling back to on-the-fly build.")
            traceback.print_exc()

    # Attempt 2: on-the-fly build from V3 scenario directory
    if graph_dict is None:
        print(f"\nBuilding graph on-the-fly from {dataset_root}/scenarios/...")
        try:
            enterprise_graph, events, root_cause_node, split_found = (
                _load_scenario_v3(dataset_root, sid, args.split, index_record)
            )
        except FileNotFoundError as exc:
            print(f"\n[ERROR] {exc}")
            sys.exit(1)
        except Exception:
            print("[ERROR] Failed to load scenario files:")
            traceback.print_exc()
            sys.exit(1)

        print(f"  Split        : {split_found}")
        print(f"  Events       : {len(events)}")
        print(f"  Graph nodes  : {len(enterprise_graph.get('nodes', []))}")
        print(f"  Graph edges  : {len(enterprise_graph.get('edges', []))}")

        try:
            graph_dict = build_graph_dict(
                case_id=f"ent_{sid}",
                scenario_id=sid,
                split=split_found,
                enterprise_graph=enterprise_graph,
                events=events,
                root_cause_node=root_cause_node,
            )
        except Exception:
            print("[ERROR] build_graph_dict raised an exception:")
            traceback.print_exc()
            sys.exit(1)

        # If root_cause_node wasn't found in the graph nodes, retry without it
        if graph_dict is None and root_cause_node:
            print(f"  [warning] root_cause_node '{root_cause_node}' not in graph nodes — "
                  f"building without ground-truth label.")
            index_record = None  # also suppress evaluation fields
            try:
                graph_dict = build_graph_dict(
                    case_id=f"ent_{sid}",
                    scenario_id=sid,
                    split=split_found,
                    enterprise_graph=enterprise_graph,
                    events=events,
                    root_cause_node=None,
                )
            except Exception:
                print("[ERROR] build_graph_dict (unlabelled) raised an exception:")
                traceback.print_exc()
                sys.exit(1)

        if graph_dict is None:
            print(f"[ERROR] Could not build graph for scenario '{sid}'.")
            sys.exit(1)

    # ── Run inference ──────────────────────────────────────────────────────────
    top_k = config.get("top_k", 3)
    print(f"\nRunning GNN inference (top_k={top_k})...")

    # labels_dict drives the "evaluation" block in predict_one (ground truth nested only)
    labels_dict: dict | None = index_record

    try:
        result = predict_one(model, graph_dict, labels_dict=labels_dict, top_k=top_k)
    except Exception:
        print("[ERROR] GNN inference failed:")
        traceback.print_exc()
        sys.exit(1)

    # ── Add Streamlit-compatibility fields ─────────────────────────────────────
    result.update({
        "model_type":           config.get("model_type", "EnterpriseRcaGNN"),
        "backend":              "torch_geometric_graphsage",
        "inference_source":     "trained_enterprise_gnn",
        "gnn_result_available": True,
        "model_path":           str(model_path),
    })
    result.setdefault("scenario_id",          sid)
    result.setdefault("rca_source",           "Enterprise GNN RCA")
    result.setdefault("predicted_root_cause", "")
    result.setdefault("root_cause_diagram",   "")
    result.setdefault("confidence",           0.0)
    result.setdefault("top_candidates",       [])
    result.setdefault("impacted_diagrams",    [])
    result.setdefault("alert_count",          0)

    # ── Save output ────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{sid}_enterprise_gnn_rca_result.json"
    out_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("Done.")
    print(f"  Predicted root cause : {result['predicted_root_cause']}")
    print(f"  Root cause diagram   : {result['root_cause_diagram']}")
    print(f"  Confidence           : {result['confidence'] * 100:.1f}%")
    print(f"  GNN result available : {result['gnn_result_available']}")
    print(f"  RCA source           : {result['rca_source']}")
    if result.get("top_candidates"):
        print("  Top candidates:")
        for c in result["top_candidates"]:
            print(f"    #{c['rank']}  {c['node_id']}  ({c['diagram_id']})  "
                  f"score={c['score']:.4f}")
    if "evaluation" in result:
        ev     = result["evaluation"]
        marker = "CORRECT" if ev.get("correct_top1") else "WRONG"
        print(f"  Ground truth         : {ev.get('ground_truth_node', '-')}  "
              f"->  {marker}  (rank {ev.get('rank', '?')})")
    print(f"  Output file          : {out_file}")


if __name__ == "__main__":
    main()
