"""
scripts/test_live_rfdetr_inference.py

Smoke test for live RF-DETR inference.

Usage:
    python scripts/test_live_rfdetr_inference.py \
        --image datasets/diagram_v3_enterprise/scenarios/train/enterprise_v3_0000/diagrams/branch_topology.png \
        --scenario-id enterprise_v3_0000 \
        --diagram-id branch_topology

Exit codes:
    0  inference succeeded
    2  checkpoint missing (expected on fresh machine without trained model)
    1  unexpected error (bug)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=str(
            REPO_ROOT
            / "datasets/diagram_v3_enterprise/scenarios/train"
            / "enterprise_v3_0000/diagrams/branch_topology.png"
        ),
    )
    parser.add_argument("--scenario-id", default="enterprise_v3_0000")
    parser.add_argument("--diagram-id",  default="branch_topology")
    parser.add_argument("--conf",        type=float, default=0.25)
    parser.add_argument("--dataset",     default="v3")
    parser.add_argument("--split",       default="train")
    args = parser.parse_args()

    try:
        from live_rfdetr_detector import find_best_rfdetr_checkpoint, run_live_rfdetr_detection
    except ImportError as exc:
        print(f"[ERROR] Could not import live_rfdetr_detector: {exc}")
        return 1

    # ── checkpoint check ──────────────────────────────────────────────────────
    ckpt = find_best_rfdetr_checkpoint(REPO_ROOT)
    if ckpt is None:
        print("[EXIT 2] No RF-DETR checkpoint found.")
        print("  Train a model first: python scripts/train_rfdetr_diagram_detector.py")
        print("  Expected checkpoint locations:")
        print("    outputs/rfdetr_v3/model/checkpoint_best_total.pth")
        print("    outputs/rfdetr_v3/model/checkpoint_best_ema.pth")
        return 2

    print(f"[OK] Checkpoint found: {ckpt}")

    # ── image check ───────────────────────────────────────────────────────────
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return 1

    print(f"[OK] Image: {image_path}")
    print(f"     Running RF-DETR inference (conf={args.conf})…")

    # ── run inference ─────────────────────────────────────────────────────────
    try:
        result = run_live_rfdetr_detection(
            repo_root   = REPO_ROOT,
            image_path  = image_path,
            dataset     = args.dataset,
            split       = args.split,
            scenario_id = args.scenario_id,
            diagram_id  = args.diagram_id,
            conf        = args.conf,
        )
    except Exception as exc:
        print(f"[ERROR] Unexpected exception: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    # ── report ────────────────────────────────────────────────────────────────
    summary = {
        "ok":               result.get("ok"),
        "detection_source": result.get("detection_source"),
        "n_detections":     result.get("n_detections"),
        "inference_time_s": result.get("inference_time_s"),
        "strategy":         result.get("strategy"),
        "model_path":       result.get("model_path"),
        "detected_image":   result.get("detected_image_path"),
        "error":            result.get("error"),
    }
    print(json.dumps(summary, indent=2))

    if result.get("ok"):
        print(f"\n[SUCCESS] {result['n_detections']} device(s) detected "
              f"in {result.get('inference_time_s', '?')}s")
        print(f"  Annotated image: {result['detected_image_path']}")
        return 0
    else:
        print(f"\n[FAILED] {result.get('error', 'unknown error')}")
        if result.get("traceback"):
            print(result["traceback"])
        return 1


if __name__ == "__main__":
    sys.exit(main())
