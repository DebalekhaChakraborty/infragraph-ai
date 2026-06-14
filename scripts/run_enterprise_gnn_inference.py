#!/usr/bin/env python3
"""
run_enterprise_gnn_inference.py

Run Enterprise GNN RCA inference for a specific V3 scenario using a
previously trained model checkpoint.  Does NOT retrain the model.

Usage
-----
python scripts/run_enterprise_gnn_inference.py \\
    --dataset-root ./datasets/infragraph_v3 \\
    --scenario-id  enterprise_v3_0072 \\
    --out          ./outputs/enterprise_gnn_rca

# --split is optional; if omitted the script searches train/val/test
python scripts/run_enterprise_gnn_inference.py \\
    --dataset-root ./datasets/infragraph_v3 \\
    --scenario-id  enterprise_v3_0001 \\
    --split        train \\
    --out          ./outputs/enterprise_gnn_rca

Output
------
outputs/enterprise_gnn_rca/<scenario_id>_enterprise_gnn_rca_result.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Shared GNN utilities ──────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_src_dir   = str(_REPO_ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from enterprise_gnn_rca import (   # type: ignore  # noqa: E402
    EnterpriseGCN, IN_FEAT,
    load_model, _load_scenario, find_scenario_dir,
    run_inference_for_scenario,
)

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
    p.add_argument("--dataset-root", default="./datasets/infragraph_v3",
                   help="Root of the V3 dataset (must contain scenarios/)")
    p.add_argument("--scenario-id",  required=True,
                   help="Scenario ID to run inference on, e.g. enterprise_v3_0072")
    p.add_argument("--split",        default=None,
                   choices=["train", "val", "test"],
                   help="Dataset split (optional — auto-detected if omitted)")
    p.add_argument("--model-path",   default=None,
                   help="Path to enterprise_gnn_model.pt "
                        "(default: <out>/enterprise_gnn_model.pt)")
    p.add_argument("--out",          default="./assets/preloaded/enterprise_gnn_rca",
                   help="Output directory for inference result JSON")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir      = Path(args.out)
    scenario_id  = args.scenario_id

    # Resolve model path — priority order:
    #   1. explicit --model-path CLI flag
    #   2. INFRAGRAPH_ENTERPRISE_GNN_MODEL_PATH env var
    #   3. model_artifacts/enterprise_gnn_rca/enterprise_gnn_rca.pt  (canonical trained artifact)
    #   4. <out>/enterprise_gnn_model.pt                              (outputs compat)
    #   5. assets/preloaded/enterprise_gnn_rca/enterprise_gnn_model.pt (preloaded compat)
    import os as _os_inf
    _candidates = []
    if args.model_path:
        _candidates = [Path(args.model_path)]
    else:
        _env_model = _os_inf.environ.get("INFRAGRAPH_ENTERPRISE_GNN_MODEL_PATH")
        if _env_model:
            _candidates.append(Path(_env_model))
        _candidates += [
            _REPO_ROOT / "model_artifacts" / "enterprise_gnn_rca" / "enterprise_gnn_rca.pt",
            out_dir / "enterprise_gnn_model.pt",
            _REPO_ROOT / "assets" / "preloaded" / "enterprise_gnn_rca" / "enterprise_gnn_model.pt",
        ]

    model_path: Path | None = None
    for _c in _candidates:
        if _c.exists():
            model_path = _c
            break

    if model_path is None:
        print(
            "\n[ERROR] No model checkpoint found. Searched:\n"
            + "\n".join(f"  {c}" for c in _candidates)
            + "\n\nTrain first with:\n"
            "  python scripts/build_enterprise_gnn_dataset.py\n"
            "  python scripts/train_enterprise_gnn_rca.py \\\n"
            "      --graphs      data/rca/enterprise_gnn/graphs.pt \\\n"
            "      --index       data/rca/enterprise_gnn/graph_index.json \\\n"
            "      --out-dir     model_artifacts/enterprise_gnn_rca \\\n"
            "      --report-dir  reports/enterprise_gnn_rca \\\n"
            "      --epochs      80\n\n"
            "  # Optional: create compat symlinks\n"
            "  python scripts/link_enterprise_gnn_model_compat.py"
        )
        sys.exit(1)

    print("InfraGraph AI — Enterprise GNN Inference")
    print(f"  Scenario    : {scenario_id}")
    print(f"  Dataset root: {dataset_root}")
    print(f"  Model       : {model_path}")
    print(f"  Output dir  : {out_dir}")

    # ── Find scenario directory ───────────────────────────────────────────────
    found = find_scenario_dir(dataset_root, scenario_id, split=args.split)
    if found is None:
        searched = [args.split] if args.split else ["train", "val", "test"]
        print(
            f"\n[ERROR] Scenario '{scenario_id}' not found in: "
            + ", ".join(f"{dataset_root}/scenarios/{s}" for s in searched)
        )
        sys.exit(1)

    scenario_dir, split_found = found
    print(f"  Split       : {split_found}")

    # ── Load scenario ────────────────────────────────────────────────────────
    print("\nLoading scenario...")
    sc = _load_scenario(scenario_dir)
    if sc is None:
        print(f"[ERROR] Could not load scenario from {scenario_dir}")
        sys.exit(1)
    print(f"  Nodes: {len(sc['node_ids'])}  Edges: {len(sc['graph_data'].get('edges', []))}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\nLoading model checkpoint...")
    device = torch.device("cpu")
    try:
        model = load_model(model_path, device)
    except Exception as exc:
        print(f"[ERROR] Could not load model: {exc}")
        sys.exit(1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── Run inference ────────────────────────────────────────────────────────
    print("\nRunning GNN inference...")
    result = run_inference_for_scenario(model, sc, out_dir, device, model_path=model_path)

    print("\nDone.")
    print(f"  Predicted root cause : {result['predicted_root_cause']}")
    print(f"  Ground truth         : {result['ground_truth_root_cause']}")
    correct = "CORRECT" if result["is_correct"] else "WRONG"
    print(f"  Result               : {correct}  (rank {result['ground_truth_rank']})")
    out_file = out_dir / f"{scenario_id}_enterprise_gnn_rca_result.json"
    print(f"  Output file          : {out_file}")


if __name__ == "__main__":
    main()
