"""Run RF-DETR inference in an external detector Python runtime.

This script is intentionally separate from Streamlit. It may import `rfdetr`;
the Streamlit process must not.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEVICE_TYPES = {
    "router", "switch", "firewall", "server", "database",
    "load_balancer", "cloud_or_wan", "service",
}


def _write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _error_payload(message: str, args: argparse.Namespace) -> dict:
    return {
        "ok": False,
        "source": "live_rfdetr_subprocess",
        "error": message,
        "python_executable": sys.executable,
        "checkpoint_path": str(args.checkpoint),
        "image_path": str(args.image),
    }


def _bbox_dict(xyxy: list[Any]) -> dict:
    vals = [float(v) for v in xyxy[:4]]
    x1, y1, x2, y2 = vals
    return {
        "x1": round(min(x1, x2), 2),
        "y1": round(min(y1, y2), 2),
        "x2": round(max(x1, x2), 2),
        "y2": round(max(y1, y2), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RF-DETR inference and write structured JSON.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-image", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    out_json = Path(args.out_json)
    out_image = Path(args.out_image)

    if not image_path.exists():
        _write_result(out_json, _error_payload(f"Image not found: {image_path}", args))
        return 2
    if not checkpoint_path.exists():
        _write_result(out_json, _error_payload(f"RF-DETR checkpoint not found: {checkpoint_path}", args))
        return 3

    try:
        from PIL import Image
    except Exception as exc:
        _write_result(out_json, _error_payload(f"Pillow is not installed in detector runtime: {exc}", args))
        return 4

    try:
        import live_rfdetr_detector as live
        model = live.load_rfdetr_model(str(checkpoint_path))
        checkpoint_load_strategy = live.get_checkpoint_load_strategy(str(checkpoint_path))
        model_class = type(model).__name__
    except Exception as exc:
        message = str(exc)
        if "No module named" in message or "not installed" in message:
            message = f"RF-DETR package not installed in detector runtime: {message}"
        _write_result(out_json, _error_payload(message, args))
        return 5

    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        t0 = time.perf_counter()
        raw_detections, strategy = live._run_rfdetr_inference(model, image, args.confidence)  # type: ignore[attr-defined]
        runtime_ms = int((time.perf_counter() - t0) * 1000)

        detections = []
        for det in raw_detections:
            cls_id = int(det.get("class_id", 0))
            label = live._RFDETR_CLASS_NAMES.get(cls_id, f"cls{cls_id}")  # type: ignore[attr-defined]
            device_type = label if label in DEVICE_TYPES else "server"
            xyxy = det.get("xyxy") or []
            if len(xyxy) < 4:
                continue
            bbox = _bbox_dict(xyxy)
            bbox["x1"] = max(0, min(width - 1, bbox["x1"]))
            bbox["x2"] = max(0, min(width - 1, bbox["x2"]))
            bbox["y1"] = max(0, min(height - 1, bbox["y1"]))
            bbox["y2"] = max(0, min(height - 1, bbox["y2"]))
            detections.append({
                "label": label,
                "device_type": device_type,
                "confidence": float(det.get("confidence", 0.0)),
                "bbox": bbox,
            })

        annotated = live._draw_rfdetr_boxes(  # type: ignore[attr-defined]
            image.copy(),
            [
                {
                    "class_id": next((k for k, v in live._RFDETR_CLASS_NAMES.items() if v == d["label"]), 0),  # type: ignore[attr-defined]
                    "confidence": d["confidence"],
                    "xyxy": [d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]],
                }
                for d in detections
            ],
            width,
            height,
            runtime_ms / 1000.0,
        )
        out_image.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(out_image, "PNG")

        _write_result(out_json, {
            "ok": True,
            "source": "live_rfdetr_subprocess",
            "detector": "RF-DETR",
            "python_executable": sys.executable,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_load_strategy": checkpoint_load_strategy,
            "model_class": model_class,
            "image_path": str(image_path),
            "inference_runtime_ms": runtime_ms,
            "inference_strategy": strategy,
            "detections": detections,
            "annotated_image_path": str(out_image),
        })
        return 0
    except Exception as exc:
        payload = _error_payload(f"RF-DETR inference exception: {exc}", args)
        payload["traceback"] = traceback.format_exc(limit=8)
        _write_result(out_json, payload)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
