"""
live_rfdetr_detector.py

RF-DETR inference helpers for the InfraGraph AI Streamlit app.

Public API:
    find_best_rfdetr_checkpoint(repo_root)  -> Path | None
    load_rfdetr_model(checkpoint_path_str)  -> model (cached per process)
    run_live_rfdetr_detection(...)          -> dict

Caching:
    Model loading is cached in a module-level dict (_MODEL_CACHE) keyed by
    checkpoint path.  In Streamlit, wrap load_rfdetr_model with
    @st.cache_resource so the model is loaded once per session:

        @st.cache_resource(show_spinner="Loading RF-DETR checkpoint…")
        def _rfdetr_model_cached(ckpt_str: str):
            from live_rfdetr_detector import load_rfdetr_model
            return load_rfdetr_model(ckpt_str)
"""
from __future__ import annotations

import importlib
import inspect
import json
import math
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

# ── checkpoint priority ────────────────────────────────────────────────────────
_CHECKPOINT_PRIORITY: list[str] = [
    "outputs/rfdetr_v3/model/checkpoint_best_total.pth",
    "outputs/rfdetr_v3/model/checkpoint_best_ema.pth",
    "outputs/rfdetr_v3/model/checkpoint_best_regular.pth",
    "outputs/rfdetr_v3/model/last.ckpt",
    "outputs/rfdetr_v3_smoke/model/checkpoint_best_total.pth",
    "outputs/rfdetr_v3_smoke/model/checkpoint_best_ema.pth",
]

# ── COCO-style class map (1-indexed, matching rfdetr dataset annotations) ──────
_RFDETR_CLASS_NAMES: dict[int, str] = {
    1: "router",
    2: "switch",
    3: "firewall",
    4: "server",
    5: "database",
    6: "load_balancer",
    7: "cloud_or_wan",
    8: "service",
}

# ── device colour palette (RGB) — same as annotation renderer ──────────────────
_CLS_COLORS: dict[str, tuple[int, int, int]] = {
    "router":        (255, 107,  53),
    "switch":        ( 78, 205, 196),
    "firewall":      (255,  78,  80),
    "server":        ( 69, 183, 209),
    "database":      (150, 206, 180),
    "load_balancer": (255, 234, 167),
    "cloud_or_wan":  (221, 160, 221),
    "service":       (152, 216, 200),
}
_DEFAULT_COLOR: tuple[int, int, int] = (168, 168, 168)

# ── module-level model cache ──────────────────────────────────────────────────
_MODEL_CACHE: dict[str, Any] = {}


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
def find_best_rfdetr_checkpoint(repo_root: Path) -> "Path | None":
    """Return the first existing RF-DETR checkpoint from the priority list."""
    for rel in _CHECKPOINT_PRIORITY:
        p = repo_root / rel
        if p.exists():
            return p
    # alternate path: any .pth under outputs/rfdetr_v3/model/
    for p in sorted((repo_root / "outputs" / "rfdetr_v3" / "model").glob("*.pth")):
        return p
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _import_rfdetr() -> "tuple[Any, Any, str | None]":
    """
    Import rfdetr and locate the best model class.
    Returns (module, model_cls, error_str).  error_str is None on success.
    """
    try:
        module = importlib.import_module("rfdetr")
    except ImportError as exc:
        return None, None, f"rfdetr not installed: {exc}"

    for attr in ["RFDETRBase", "RFDETRMedium", "RFDETRLarge"]:
        cls = getattr(module, attr, None)
        if cls is not None:
            return module, cls, None

    try:
        models_mod = importlib.import_module("rfdetr.models")
        for attr in ["RFDETRBase", "RFDETRMedium", "RFDETRLarge"]:
            cls = getattr(models_mod, attr, None)
            if cls is not None:
                return module, cls, None
    except ImportError:
        pass

    return module, None, "rfdetr imported but no model class (RFDETRBase/Medium/Large) found"


def load_rfdetr_model(checkpoint_path_str: str) -> Any:
    """
    Load RF-DETR model from checkpoint.  Results are cached per process.
    Raises RuntimeError on failure so callers can catch it explicitly.
    """
    if checkpoint_path_str in _MODEL_CACHE:
        return _MODEL_CACHE[checkpoint_path_str]

    _, model_cls, err = _import_rfdetr()
    if err:
        raise RuntimeError(err)

    ckpt = Path(checkpoint_path_str)
    if not ckpt.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt}")

    # Strategy 1: constructor with pretrain_weights kwarg
    try:
        sig = inspect.signature(model_cls.__init__)
        if "pretrain_weights" in sig.parameters:
            model = model_cls(pretrain_weights=str(ckpt))
            _MODEL_CACHE[checkpoint_path_str] = model
            return model
    except Exception:
        pass

    # Strategy 2: default constructor → load/load_checkpoint/load_weights method
    model = model_cls()
    for method in ("load", "load_checkpoint", "load_weights"):
        if hasattr(model, method):
            try:
                getattr(model, method)(str(ckpt))
                _MODEL_CACHE[checkpoint_path_str] = model
                return model
            except Exception:
                pass

    # Strategy 3: torch.load + load_state_dict
    try:
        import torch  # type: ignore
        ckpt_data = torch.load(str(ckpt), map_location="cpu")
        state = (
            ckpt_data.get("model")
            or ckpt_data.get("state_dict")
            or ckpt_data
        )
        model.load_state_dict(state, strict=False)
        _MODEL_CACHE[checkpoint_path_str] = model
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Could not load checkpoint {ckpt} with any strategy: {exc}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def _parse_detections(result: Any, conf: float) -> list[dict]:
    """
    Convert RF-DETR output (supervision Detections, dict, list, tensor) to
    a list of {class_id, confidence, xyxy}.  Best-effort; returns [] on unknown format.
    """
    if result is None:
        return []

    # supervision Detections (or anything with .xyxy, .class_id, .confidence)
    if hasattr(result, "xyxy"):
        import numpy as _np  # type: ignore
        detections: list[dict] = []
        boxes = result.xyxy
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()
        else:
            boxes = _np.asarray(boxes)
        n = len(boxes)
        confs   = _np.ones(n, dtype=float)
        cls_ids = _np.zeros(n, dtype=int)
        if hasattr(result, "confidence") and result.confidence is not None:
            confs = _np.asarray(result.confidence).flatten()
        if hasattr(result, "class_id") and result.class_id is not None:
            cls_ids = _np.asarray(result.class_id).flatten().astype(int)
        for i in range(n):
            score = float(confs[i]) if i < len(confs) else 1.0
            if score < conf:
                continue
            cls_id = int(cls_ids[i]) if i < len(cls_ids) else 0
            detections.append({
                "class_id":   cls_id,
                "confidence": round(score, 4),
                "xyxy":       [round(float(x), 1) for x in boxes[i]],
            })
        return detections

    # dict-like: {boxes, scores, labels}
    if isinstance(result, dict):
        boxes_raw  = result.get("boxes",  result.get("pred_boxes",  []))
        scores_raw = result.get("scores", result.get("pred_scores", []))
        labels_raw = result.get("labels", result.get("pred_labels", []))
        if hasattr(boxes_raw, "cpu"):
            boxes_raw = boxes_raw.cpu().tolist()
        if hasattr(scores_raw, "cpu"):
            scores_raw = scores_raw.cpu().tolist()
        if hasattr(labels_raw, "cpu"):
            labels_raw = labels_raw.cpu().tolist()
        detections = []
        for i, box in enumerate(boxes_raw):
            score = float(scores_raw[i]) if i < len(scores_raw) else 1.0
            if score < conf:
                continue
            cls_id = int(labels_raw[i]) if i < len(labels_raw) else 0
            detections.append({
                "class_id":   cls_id,
                "confidence": round(score, 4),
                "xyxy":       [round(float(x), 1) for x in box],
            })
        return detections

    # list of items (could be nested)
    if isinstance(result, (list, tuple)) and result:
        detections = []
        for item in result:
            detections.extend(_parse_detections(item, conf))
        return detections

    return []


def _run_rfdetr_inference(
    model: Any,
    pil_image: Any,
    conf: float,
) -> tuple[list[dict], str]:
    """
    Try several RF-DETR prediction API patterns.
    Returns (detections, strategy_name).  Raises RuntimeError if all fail.
    """
    errors: list[str] = []

    for strategy, call in [
        ("predict(image)", lambda: model.predict(pil_image)),
        ("predict(image, threshold)", lambda: model.predict(pil_image, threshold=conf)),
        ("predict(image, conf)", lambda: model.predict(pil_image, conf=conf)),
        ("infer(image)", lambda: model.infer(pil_image)),
        ("__call__(image)", lambda: model(pil_image)),
    ]:
        if not hasattr(model, strategy.split("(")[0]):
            continue
        try:
            result = call()
            parsed = _parse_detections(result, conf)
            return parsed, strategy
        except Exception as exc:
            errors.append(f"{strategy}: {exc}")

    raise RuntimeError("All inference strategies failed:\n" + "\n".join(errors))


# ── PIL drawing helper ─────────────────────────────────────────────────────────
def _draw_rfdetr_boxes(
    img: Any,   # PIL.Image
    detections: list[dict],
    img_w: int,
    img_h: int,
    inference_time_s: float = 0.0,
) -> Any:
    """Draw detection boxes + labels + footer on a PIL image. Returns modified image."""
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        return img

    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font    = ImageFont.truetype("arial.ttf", 14)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        try:
            font    = ImageFont.load_default(size=14)
            font_sm = ImageFont.load_default(size=11)
        except Exception:
            font    = ImageFont.load_default()
            font_sm = font

    for det in detections:
        try:
            xyxy    = det.get("xyxy", [])
            cls_id  = det.get("class_id", 0)
            score   = det.get("confidence", 0.0)
            if len(xyxy) < 4:
                continue
            # normalize bbox to image bounds
            x0 = max(0, min(img_w - 1, int(xyxy[0])))
            y0 = max(0, min(img_h - 1, int(xyxy[1])))
            x1 = max(0, min(img_w - 1, int(xyxy[2])))
            y1 = max(0, min(img_h - 1, int(xyxy[3])))
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            if (x1 - x0) < 2 or (y1 - y0) < 2:
                continue

            cls_name = _RFDETR_CLASS_NAMES.get(cls_id, f"cls{cls_id}")
            r, g, b  = _CLS_COLORS.get(cls_name, _DEFAULT_COLOR)

            draw.rectangle([x0, y0, x1, y1], outline=(r, g, b, 255), fill=(r, g, b, 35), width=2)

            label    = f"{cls_name} {score:.2f}"
            lh_est   = 16
            try:
                tb = draw.textbbox((x0, 0), label, font=font)
                lw, lh_est = tb[2] - tb[0], tb[3] - tb[1]
            except AttributeError:
                lw = len(label) * 8

            ly0 = max(0, y0 - lh_est - 4)
            ly1 = ly0 + lh_est + 4
            if ly1 > img_h - 1:
                ly0, ly1 = y0, min(img_h - 1, y0 + lh_est + 4)
            lx1 = min(img_w - 1, x0 + lw + 6)
            if lx1 > x0 and ly1 > ly0:
                draw.rectangle([x0, ly0, lx1, ly1], fill=(r, g, b, 210))
            draw.text((x0 + 3, ly0 + 2), label, fill=(255, 255, 255), font=font)
        except Exception:
            continue

    # footer
    footer_h = 26
    time_str = f"  ({inference_time_s:.2f}s)" if inference_time_s > 0 else ""
    draw.rectangle([0, img_h - footer_h, img_w, img_h], fill=(10, 20, 40, 220))
    draw.text(
        (10, img_h - footer_h + 5),
        f"Live RF-DETR detector — {len(detections)} device(s) detected{time_str}",
        fill=(120, 220, 120), font=font_sm,
    )
    return img


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC DETECTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def run_live_rfdetr_detection(
    repo_root: Path,
    image_path: Path,
    dataset: str,
    split: str,
    scenario_id: str,
    diagram_id: str,
    conf: float = 0.25,
    model: Any = None,           # pre-loaded (from cache); loaded here if None
) -> dict:
    """
    Run RF-DETR inference on image_path.

    Output layout:
        outputs/live_rfdetr/<scenario_id>__<diagram_id>/detected.png
        outputs/live_rfdetr/<scenario_id>__<diagram_id>/detections.json

    Returns:
        ok, detected_image_path, detections_json_path, detections,
        model_path, detection_source, n_detections, inference_time_s
    On failure:
        ok=False, error, traceback
    Never raises.
    """
    out_dir = repo_root / "outputs" / "live_rfdetr" / f"{scenario_id}__{diagram_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    detected_path   = out_dir / "detected.png"
    detections_path = out_dir / "detections.json"

    # ── resolve checkpoint ────────────────────────────────────────────────────
    ckpt = find_best_rfdetr_checkpoint(repo_root)
    if ckpt is None:
        return {
            "ok": False,
            "error": (
                "No RF-DETR checkpoint found. "
                "Train a model first using scripts/train_rfdetr_diagram_detector.py."
            ),
            "detection_source": None,
        }
    if not image_path.exists():
        return {
            "ok": False,
            "error": f"Image not found: {image_path}",
            "detection_source": None,
        }

    try:
        from PIL import Image as _PIL  # type: ignore
    except ImportError:
        return {
            "ok": False,
            "error": "Pillow not installed — required for image loading.",
            "detection_source": None,
        }

    try:
        # ── load model (use cache if provided) ────────────────────────────────
        if model is None:
            model = load_rfdetr_model(str(ckpt))

        pil_image = _PIL.open(image_path).convert("RGB")
        img_w, img_h = pil_image.size

        # ── run inference ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        raw_detections, strategy = _run_rfdetr_inference(model, pil_image, conf)
        inference_time_s = round(time.perf_counter() - t0, 3)

        # ── enrich detections with class name and source ──────────────────────
        detections: list[dict] = []
        for det in raw_detections:
            cls_id   = det.get("class_id", 0)
            cls_name = _RFDETR_CLASS_NAMES.get(cls_id, f"cls{cls_id}")
            xyxy     = det.get("xyxy", [])
            # clamp bbox
            if len(xyxy) >= 4:
                xyxy = [
                    max(0, min(img_w - 1, xyxy[0])),
                    max(0, min(img_h - 1, xyxy[1])),
                    max(0, min(img_w - 1, xyxy[2])),
                    max(0, min(img_h - 1, xyxy[3])),
                ]
            detections.append({
                "class_id":   cls_id,
                "class_name": cls_name,
                "confidence": det.get("confidence", 0.0),
                "xyxy":       xyxy,
                "source":     "rfdetr",
            })

        # ── draw annotated output ─────────────────────────────────────────────
        annotated = _draw_rfdetr_boxes(pil_image.copy(), detections, img_w, img_h, inference_time_s)
        annotated.save(detected_path, "PNG")

        # ── persist detections manifest ───────────────────────────────────────
        manifest = {
            "model_path":       str(ckpt),
            "image_path":       str(image_path),
            "dataset":          dataset,
            "split":            split,
            "scenario_id":      scenario_id,
            "diagram_id":       diagram_id,
            "conf":             conf,
            "inference_strategy": strategy,
            "inference_time_s": inference_time_s,
            "detection_source": "Live RF-DETR detector",
            "n_detections":     len(detections),
            "detections":       detections,
        }
        detections_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "ok":                   True,
            "detected_image_path":  str(detected_path),
            "detections_json_path": str(detections_path),
            "detections":           detections,
            "model_path":           str(ckpt),
            "detection_source":     "Live RF-DETR detector",
            "n_detections":         len(detections),
            "inference_time_s":     inference_time_s,
            "strategy":             strategy,
            "image_path":           str(image_path),
            "diagram_id":           diagram_id,
        }

    except Exception as exc:
        return {
            "ok":               False,
            "error":            str(exc),
            "traceback":        traceback.format_exc(),
            "detection_source": None,
            "image_path":       str(image_path),
            "diagram_id":       diagram_id,
        }
