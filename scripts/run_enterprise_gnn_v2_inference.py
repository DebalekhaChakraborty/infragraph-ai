#!/usr/bin/env python3
"""
run_enterprise_gnn_v2_inference.py

Run Enterprise GNN RCA V2 (Temporal Relation-Aware) inference for a V3 scenario.

Does NOT overwrite V1 outputs. V1 artifacts remain valid and separate.

Usage
-----
python scripts/run_enterprise_gnn_v2_inference.py \\
    --scenario-id  enterprise_v3_0077

Output
------
outputs/enterprise_gnn_rca_v2/<scenario_id>_enterprise_gnn_v2_rca_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_src_dir   = str(_REPO_ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from rca_ml.enterprise_gnn_v2_model import (
        _select_device,
        load_gnn_v2,
        check_torch_geo_v2_requirement,
        predict_one_v2,
    )
    from rca_ml.enterprise_gnn_dataset import build_graph_dict
except ImportError as exc:
    print(f"[ERROR] Cannot import rca_ml V2: {exc}")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("[ERROR] PyTorch is not installed.")
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Enterprise GNN RCA V2 inference for a V3 scenario"
    )
    p.add_argument("--scenario-id",  required=True,
                   help="Scenario ID, e.g. enterprise_v3_0077")
    p.add_argument("--split",        default=None,
                   choices=["train", "val", "test"])
    p.add_argument("--model-path",   default=None,
                   help="Path to enterprise_gnn_v2_rca.pt")
    p.add_argument("--graphs-path",  default=None,
                   help="Path to graphs.pt (falls back to on-the-fly build)")
    p.add_argument("--index-path",   default=None,
                   help="Path to graph_index.json")
    p.add_argument("--dataset-root", default=None,
                   help="V3 dataset root (default: datasets/infragraph_v3)")
    p.add_argument("--out",          default="outputs/enterprise_gnn_rca_v2",
                   help="Output directory")
    p.add_argument("--device",       default="auto", choices=["auto", "cpu", "cuda"],
                   help="Compute device: auto (cuda if available), cpu, cuda")
    return p.parse_args()


def _load_scenario_v3(
    dataset_root: Path,
    scenario_id: str,
    split: str | None,
    index_record: dict | None,
) -> tuple[dict, list[dict], str | None, str]:
    splits_to_try = [split] if split else ["train", "val", "test"]
    for sp in splits_to_try:
        sc_dir = dataset_root / "scenarios" / sp / scenario_id
        if not sc_dir.exists():
            continue
        eg_path = sc_dir / "enterprise_graph.json"
        al_path = sc_dir / "alerts.json"
        if not eg_path.exists():
            raise FileNotFoundError(f"enterprise_graph.json not found in {sc_dir}")
        enterprise_graph = json.loads(eg_path.read_text(encoding="utf-8"))
        events: list[dict] = []
        if al_path.exists():
            al_doc = json.loads(al_path.read_text(encoding="utf-8"))
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


def main() -> None:
    args    = _parse_args()
    sid     = args.scenario_id
    out_dir = Path(args.out)

    model_path = (
        Path(args.model_path) if args.model_path
        else _REPO_ROOT / "model_artifacts" / "enterprise_gnn_rca_v2" / "enterprise_gnn_v2_rca.pt"
    )
    config_path  = model_path.with_name("enterprise_gnn_v2_config.json")
    graphs_path  = (
        Path(args.graphs_path) if args.graphs_path
        else _REPO_ROOT / "data" / "rca" / "enterprise_gnn" / "graphs.pt"
    )
    index_path   = (
        Path(args.index_path) if args.index_path
        else _REPO_ROOT / "data" / "rca" / "enterprise_gnn" / "graph_index.json"
    )
    dataset_root = (
        Path(args.dataset_root) if args.dataset_root
        else _REPO_ROOT / "datasets" / "infragraph_v3"
    )

    device = _select_device(args.device)

    print("InfraGraph AI — Enterprise GNN RCA V2 (Temporal Relation-Aware)")
    print(f"  Scenario : {sid}")
    print(f"  Model    : {model_path}")
    print(f"  Out dir  : {out_dir}")
    print(f"  Device   : {device}")
    if device.type == "cuda":
        print(f"  GPU name : {torch.cuda.get_device_name(0)}")

    if not model_path.exists():
        print(f"\n[ERROR] V2 model checkpoint not found: {model_path}")
        print("Train first with:")
        print("  python scripts/build_enterprise_gnn_dataset.py")
        print("  python scripts/train_enterprise_gnn_v2_rca.py")
        sys.exit(1)
    if not config_path.exists():
        print(f"\n[ERROR] V2 model config not found: {config_path}")
        sys.exit(1)

    check_torch_geo_v2_requirement()
    print("\nLoading V2 model...")
    try:
        model, config = load_gnn_v2(model_path, config_path, device=device)
    except Exception:
        print("[ERROR] Failed to load V2 model checkpoint:")
        traceback.print_exc()
        sys.exit(1)
    print(f"  Architecture : {config.get('gnn_architecture', 'RelationAwareTemporalGraphSAGE')} "
          f"({config.get('num_layers', 2)} layers, "
          f"hidden={config.get('hidden_channels', 64)}, "
          f"in_channels={config.get('in_channels', 54)})")
    print(f"  Uses edge_type : {config.get('uses_edge_type', True)}")
    print(f"  Uses temporal  : {config.get('uses_temporal_features', True)}")
    print(f"  Parameters   : {sum(p.numel() for p in model.parameters()):,}")

    # Load graph index
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
            traceback.print_exc()

    # Load / build graph_dict
    graph_dict: dict | None = None

    if graphs_path.exists():
        print(f"\nLoading prebuilt graphs from {graphs_path}...")
        try:
            graphs = torch.load(str(graphs_path), map_location="cpu")
            for g in graphs:
                if g.get("scenario_id") == sid:
                    if args.split is None or g.get("split") == args.split:
                        graph_dict = g
                        et_note = "edge_type present" if g.get("edge_type") is not None else "no edge_type (fallback)"
                        print(f"  Found in graphs.pt  "
                              f"(nodes={g.get('num_nodes')}, {et_note})")
                        break
            if graph_dict is None:
                print(f"  [warning] '{sid}' not in graphs.pt — falling back to on-the-fly build.")
        except Exception:
            print("[WARNING] Could not load graphs.pt — falling back to on-the-fly build.")
            traceback.print_exc()

    if graph_dict is None:
        print(f"\nBuilding graph on-the-fly from {dataset_root}/scenarios/...")
        try:
            enterprise_graph, events, root_cause_node, split_found = (
                _load_scenario_v3(dataset_root, sid, args.split, index_record)
            )
        except FileNotFoundError as exc:
            print(f"\n[ERROR] {exc}")
            sys.exit(1)

        print(f"  Split   : {split_found}")
        print(f"  Events  : {len(events)}")
        print(f"  Nodes   : {len(enterprise_graph.get('nodes', []))}")
        print(f"  Edges   : {len(enterprise_graph.get('edges', []))}")

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

        if graph_dict is None and root_cause_node:
            print(f"  [warning] root_cause_node '{root_cause_node}' not in graph — building without label.")
            index_record = None
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
                traceback.print_exc()
                sys.exit(1)

        if graph_dict is None:
            print(f"[ERROR] Could not build graph for scenario '{sid}'.")
            sys.exit(1)

    # Run V2 inference
    top_k = config.get("top_k", 3)
    print(f"\nRunning V2 GNN inference (top_k={top_k})...")
    try:
        result = predict_one_v2(model, graph_dict, labels_dict=index_record, top_k=top_k, device=device)
    except Exception:
        print("[ERROR] V2 GNN inference failed:")
        traceback.print_exc()
        sys.exit(1)

    # Add provenance fields
    result.update({
        "model_type":           config.get("model_type", "EnterpriseRcaTemporalRelGNN"),
        "backend":              "torch_geometric_relation_aware_graphsage",
        "inference_source":     "trained_enterprise_gnn_v2",
        "gnn_result_available": True,
        "model_path":           str(model_path),
    })
    result.setdefault("scenario_id",          sid)
    result.setdefault("rca_source",           "Enterprise GNN RCA V2 — Temporal Relation-Aware GraphSAGE")
    result.setdefault("predicted_root_cause", "")
    result.setdefault("root_cause_diagram",   "")
    result.setdefault("confidence",           0.0)
    result.setdefault("top_candidates",       [])
    result.setdefault("impacted_diagrams",    [])
    result.setdefault("alert_count",          0)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{sid}_enterprise_gnn_v2_rca_result.json"
    out_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print()
    print("Done.")
    print(f"  Model type           : {result.get('model_type')}")
    print(f"  Predicted root cause : {result['predicted_root_cause']}")
    print(f"  Root cause diagram   : {result['root_cause_diagram']}")
    print(f"  Confidence           : {result['confidence'] * 100:.1f}%")
    print(f"  RCA source           : {result['rca_source']}")
    _mn = result.get("model_notes", {})
    if _mn:
        print(f"  Uses edge_type       : {_mn.get('uses_edge_type')}")
        print(f"  Uses temporal feats  : {_mn.get('uses_temporal_features')}")
        print(f"  Local edges          : {_mn.get('local_edges', '—')}")
        print(f"  Cross-diagram edges  : {_mn.get('cross_diagram_edges', '—')}")
        print(f"  Vision edges         : {_mn.get('vision_edges', '—')}")
    if result.get("top_candidates"):
        print("  Top candidates:")
        for c in result["top_candidates"]:
            print(f"    #{c['rank']}  {c['node_id']}  ({c['diagram_id']})  score={c['score']:.4f}")
    if "evaluation" in result:
        ev     = result["evaluation"]
        marker = "CORRECT" if ev.get("correct_top1") else "WRONG"
        print(f"  Ground truth         : {ev.get('ground_truth_node', '-')}  ->  {marker}  (rank {ev.get('rank', '?')})")
    print(f"  Output file          : {out_file}")


if __name__ == "__main__":
    main()
