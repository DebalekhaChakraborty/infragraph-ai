"""
live_detector.py

YOLO inference helpers for the InfraGraph AI Streamlit demo.

Public API:
    find_best_yolo_checkpoint(repo_root) -> Path | None
    run_live_yolo_detection(...)         -> dict
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

# Checkpoint search order — first existing file wins
_CHECKPOINT_PRIORITY: list[str] = [
    "training_runs/infragraph_yolo_v3/weights/best.pt",
    "training_runs/infragraph_yolo_v2/weights/best.pt",
    "training_runs/infragraph_yolo_v1/weights/best.pt",
    "runs/detect/training_runs/infragraph_yolo_v2/weights/best.pt",
]


def find_best_yolo_checkpoint(repo_root: Path) -> "Path | None":
    """
    Return the first existing YOLO checkpoint from the priority list.
    Falls back to any best.pt found under training_runs/**/weights/.
    Returns None if nothing exists.
    """
    for rel in _CHECKPOINT_PRIORITY:
        p = repo_root / rel
        if p.exists():
            return p
    for p in sorted(repo_root.glob("training_runs/**/weights/best.pt")):
        return p
    return None


def run_live_yolo_detection(
    repo_root: Path,
    image_path: Path,
    dataset: str,
    split: str,
    diagram_id: str,
    conf: float = 0.25,
    imgsz: int = 960,
    device: str = "cpu",
    scenario_id: str | None = None,
) -> dict:
    """
    Run YOLO inference on image_path, save annotated PNG and detections JSON.

    Output layout:
        outputs/live_detection/<dataset>/<split>/<dir_key>/detected.png
        outputs/live_detection/<dataset>/<split>/<dir_key>/detections.json

    Where dir_key is "<scenario_id>__<diagram_id>" for V3 or "<diagram_id>" otherwise.

    Returns dict keys:
        success, detected_image_path, detections_json_path, detections,
        model_path, detection_source, n_detections
    On failure:
        success=False, error, traceback (no raise)
    """
    # ── dependency checks ─────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return {
            "success": False,
            "error": (
                "ultralytics not installed. "
                "Run: pip install ultralytics"
            ),
            "detection_source": None,
        }

    ckpt = find_best_yolo_checkpoint(repo_root)
    if ckpt is None:
        return {
            "success": False,
            "error": (
                "No YOLO checkpoint found under training_runs/**/weights/best.pt. "
                "Train a model first using scripts/train_*.py."
            ),
            "detection_source": None,
        }

    if not image_path.exists():
        return {
            "success": False,
            "error": f"Image not found: {image_path}",
            "detection_source": None,
        }

    # ── output paths ──────────────────────────────────────────────────────────
    dir_key = f"{scenario_id}__{diagram_id}" if scenario_id else diagram_id
    out_dir = repo_root / "outputs" / "live_detection" / dataset / split / dir_key
    out_dir.mkdir(parents=True, exist_ok=True)
    detected_path   = out_dir / "detected.png"
    detections_path = out_dir / "detections.json"

    # ── run inference ─────────────────────────────────────────────────────────
    try:
        model   = YOLO(str(ckpt))
        results = model.predict(
            source=str(image_path),
            conf=conf,
            imgsz=imgsz,
            device=device,
            save=False,
            verbose=False,
        )

        # parse boxes
        class_names: dict = getattr(model, "names", {}) or {}
        detections: list[dict] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                detections.append({
                    "class_id":   cls_id,
                    "class_name": class_names.get(cls_id, str(cls_id)),
                    "confidence": round(float(boxes.conf[i].item()), 4),
                    "xyxy":       [round(float(v), 1) for v in boxes.xyxy[i].tolist()],
                })

        # save annotated image
        if results:
            annotated_bgr = results[0].plot()   # numpy uint8 HxWx3 BGR
            try:
                from PIL import Image as _PIL
                import numpy as _np
                _PIL.fromarray(annotated_bgr[:, :, ::-1]).save(detected_path, "PNG")
            except Exception:
                # PIL unavailable — try cv2
                import cv2  # type: ignore
                cv2.imwrite(str(detected_path), annotated_bgr)

        # save detections manifest
        manifest = {
            "model_path":       str(ckpt),
            "image_path":       str(image_path),
            "dataset":          dataset,
            "split":            split,
            "diagram_id":       diagram_id,
            "scenario_id":      scenario_id,
            "conf":             conf,
            "imgsz":            imgsz,
            "device":           device,
            "detection_source": "Live YOLO detector",
            "n_detections":     len(detections),
            "detections":       detections,
        }
        detections_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "success":              True,
            "detected_image_path":  str(detected_path),
            "detections_json_path": str(detections_path),
            "detections":           detections,
            "model_path":           str(ckpt),
            "detection_source":     "Live YOLO detector",
            "n_detections":         len(detections),
            "image_path":           str(image_path),
            "diagram_id":           diagram_id,
        }

    except Exception as exc:
        return {
            "success":          False,
            "error":            str(exc),
            "traceback":        traceback.format_exc(),
            "detection_source": None,
            "image_path":       str(image_path),
            "diagram_id":       diagram_id,
        }
