"""evaluate_rfdetr_v3_detector.py

Evaluate RF-DETR inference against InfraGraph V3 COCO-format annotations.

Three evaluation modes in a single run:
  A. Strict class-aware: predicted label must match GT label AND IoU >= threshold.
  B. Class-agnostic localization: match predicted boxes to GT boxes by IoU only,
     ignoring class labels. Shows whether boxes are localization-correct.
  C. Class-ID shift diagnostic: evaluate with 0-indexed offsets (original, +1, -1)
     to detect a 0-vs-1 indexing mismatch between model output and COCO annotation IDs.

Also produces a confusion matrix (predicted_label vs GT_label for IoU-matched pairs)
and the first 20 sample matches.

All output JSON is strict-valid (NaN replaced with null).

This script calls run_rfdetr_inference.py as a subprocess — it never imports
rfdetr or torch directly, so it works even when the detector runtime is absent.

Usage:
    python scripts/evaluate_rfdetr_v3_detector.py \\
        --dataset-root datasets/infragraph_v3 \\
        --checkpoint model_artifacts/rfdetr_v3/checkpoint_best_total.pth \\
        --out-dir reports/rfdetr_v3_eval \\
        --max-images 30 \\
        --confidence 0.25 \\
        --iou-threshold 0.50 \\
        [--split val|test|train]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Class map helpers (mirrors live_rfdetr_detector._RFDETR_CLASS_NAMES)
# ---------------------------------------------------------------------------

_RFDETR_CLASS_NAMES: dict[int, str] = {
    1: "router", 2: "switch", 3: "firewall", 4: "server",
    5: "database", 6: "load_balancer", 7: "cloud_or_wan", 8: "service",
}
_CLASS_NAME_TO_ID: dict[str, int] = {v: k for k, v in _RFDETR_CLASS_NAMES.items()}


def _label_to_cls_id(label: str) -> int | None:
    """Reverse-map a string label back to its numeric class ID."""
    if label in _CLASS_NAME_TO_ID:
        return _CLASS_NAME_TO_ID[label]
    if label.startswith("cls"):
        try:
            return int(label[3:])
        except ValueError:
            pass
    return None


def _shift_label(label: str, shift: int) -> str:
    """Apply a class-ID offset and re-look up through _RFDETR_CLASS_NAMES."""
    cls_id = _label_to_cls_id(label)
    if cls_id is None:
        return label
    new_id = cls_id + shift
    return _RFDETR_CLASS_NAMES.get(new_id, f"cls{new_id}")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _coco_xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _det_dict_to_xyxy(det: dict) -> list[float]:
    b = det["bbox"]
    return [b["x1"], b["y1"], b["x2"], b["y2"]]


# ---------------------------------------------------------------------------
# AP computation (trapezoidal rule)
# ---------------------------------------------------------------------------

def _compute_ap(sorted_detections: list[dict], num_gt: int) -> float:
    if num_gt == 0:
        return float("nan")
    if not sorted_detections:
        return 0.0
    tp_cum = fp_cum = 0
    precisions, recalls = [], []
    for det in sorted_detections:
        if det["tp"]:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / num_gt)
    ap = 0.0
    prev_r, prev_p = 0.0, 1.0
    for p, r in zip(precisions, recalls):
        ap += (r - prev_r) * (p + prev_p) / 2.0
        prev_r, prev_p = r, p
    return ap


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _safe_f1(precision: float, recall: float) -> float:
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0.0:
        return float("nan")
    return 2.0 * precision * recall / (precision + recall)


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None for strict JSON validity."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Evaluation pass helpers
# ---------------------------------------------------------------------------

def _aggregate_class_metrics(
    per_class_tp: dict,
    per_class_fp: dict,
    per_class_fn: dict,
    per_class_det_records: dict,
    per_class_gt_count: dict,
    total_matched_iou: list[float],
) -> dict:
    all_classes = sorted(
        set(list(per_class_tp) + list(per_class_fp) +
            list(per_class_fn) + list(per_class_gt_count))
    )
    global_tp = sum(per_class_tp.values())
    global_fp = sum(per_class_fp.values())
    global_fn = sum(per_class_fn.values())
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else float("nan")
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else float("nan")
    f1 = _safe_f1(precision, recall)
    mean_iou = sum(total_matched_iou) / len(total_matched_iou) if total_matched_iou else float("nan")

    per_class_report: dict = {}
    ap_values: list[float] = []
    for cls in all_classes:
        tp = per_class_tp.get(cls, 0)
        fp = per_class_fp.get(cls, 0)
        fn = per_class_fn.get(cls, 0)
        gt_count = per_class_gt_count.get(cls, 0)
        cls_p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        cls_r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        cls_f1 = _safe_f1(cls_p, cls_r)
        det_records = sorted(
            per_class_det_records.get(cls, []),
            key=lambda d: d["confidence"], reverse=True,
        )
        ap = _compute_ap(det_records, gt_count)
        if not math.isnan(ap):
            ap_values.append(ap)
        per_class_report[cls] = {
            "tp": tp, "fp": fp, "fn": fn, "gt_count": gt_count,
            "precision": cls_p, "recall": cls_r, "f1": cls_f1, "ap_at_50": ap,
        }
    mean_ap = sum(ap_values) / len(ap_values) if ap_values else float("nan")

    return {
        "global_tp": global_tp,
        "global_fp": global_fp,
        "global_fn": global_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_matched": mean_iou,
        "mean_ap_at_50": mean_ap,
        "per_class_metrics": per_class_report,
    }


def _eval_strict_class_aware(
    per_image_data: list[dict],
    cat_id_to_name: dict,
    iou_threshold: float,
    label_override_fn=None,
) -> dict:
    """
    Pass A (and pass C with shifting): strict class-aware matching.
    label_override_fn(label: str) -> str  — applied to predicted labels when not None.
    """
    per_class_tp: dict[str, int] = defaultdict(int)
    per_class_fp: dict[str, int] = defaultdict(int)
    per_class_fn: dict[str, int] = defaultdict(int)
    per_class_det_records: dict[str, list] = defaultdict(list)
    per_class_gt_count: dict[str, int] = defaultdict(int)
    total_matched_iou: list[float] = []

    for item in per_image_data:
        gt_anns = item["gt_anns"]
        detections = item["detections"]

        gt_by_class: dict[str, list] = defaultdict(list)
        for ann in gt_anns:
            cls_name = cat_id_to_name.get(ann["category_id"], "unknown")
            per_class_gt_count[cls_name] += 1
            gt_by_class[cls_name].append(
                {"bbox_xyxy": _coco_xywh_to_xyxy(ann["bbox"]), "matched": False}
            )

        preds_by_label: dict[str, list] = defaultdict(list)
        for det in detections:
            label = det.get("label", "unknown")
            if label_override_fn is not None:
                label = label_override_fn(label)
            preds_by_label[label].append({**det, "_matched_label": label})

        for pred_label, preds in preds_by_label.items():
            preds_sorted = sorted(preds, key=lambda d: d.get("confidence", 0.0), reverse=True)
            gt_list = gt_by_class.get(pred_label, [])
            for pred in preds_sorted:
                pred_box = _det_dict_to_xyxy(pred)
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt_item in enumerate(gt_list):
                    if gt_item["matched"]:
                        continue
                    iou_val = _iou(pred_box, gt_item["bbox_xyxy"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = gt_idx
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    gt_list[best_gt_idx]["matched"] = True
                    per_class_tp[pred_label] += 1
                    total_matched_iou.append(best_iou)
                    per_class_det_records[pred_label].append(
                        {"confidence": pred.get("confidence", 0.0), "tp": True}
                    )
                else:
                    per_class_fp[pred_label] += 1
                    per_class_det_records[pred_label].append(
                        {"confidence": pred.get("confidence", 0.0), "tp": False}
                    )

        for cls_name, gt_list in gt_by_class.items():
            per_class_fn[cls_name] += sum(1 for g in gt_list if not g["matched"])

    return _aggregate_class_metrics(
        per_class_tp, per_class_fp, per_class_fn,
        per_class_det_records, per_class_gt_count, total_matched_iou,
    )


def _eval_localization_agnostic(
    per_image_data: list[dict],
    cat_id_to_name: dict,
    iou_threshold: float,
) -> tuple[dict, list[dict], dict]:
    """
    Pass B: class-agnostic localization.
    Match predicted boxes to GT boxes by IoU only, ignoring class labels.
    Returns (localization_metrics, first_20_samples, class_confusion_matrix).
    """
    tp_total = fp_total = fn_total = 0
    matched_ious: list[float] = []
    all_confusion_pairs: list[tuple[str, str]] = []
    first_20_samples: list[dict] = []

    for item in per_image_data:
        gt_anns = item["gt_anns"]
        detections = item["detections"]
        file_name = item["file_name"]

        all_gt = [
            {
                "bbox_xyxy": _coco_xywh_to_xyxy(ann["bbox"]),
                "matched": False,
                "gt_label": cat_id_to_name.get(ann["category_id"], "unknown"),
            }
            for ann in gt_anns
        ]

        preds_sorted = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)

        for pred in preds_sorted:
            pred_box = _det_dict_to_xyxy(pred)
            pred_label = pred.get("label", "unknown")
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt_item in enumerate(all_gt):
                if gt_item["matched"]:
                    continue
                iou_val = _iou(pred_box, gt_item["bbox_xyxy"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                all_gt[best_gt_idx]["matched"] = True
                tp_total += 1
                matched_ious.append(best_iou)
                gt_label = all_gt[best_gt_idx]["gt_label"]
                all_confusion_pairs.append((pred_label, gt_label))
                if len(first_20_samples) < 20:
                    first_20_samples.append({
                        "image": file_name,
                        "predicted_label": pred_label,
                        "gt_label": gt_label,
                        "confidence": round(pred.get("confidence", 0.0), 4),
                        "iou": round(best_iou, 4),
                        "pred_bbox": pred.get("bbox", {}),
                        "gt_bbox": all_gt[best_gt_idx]["bbox_xyxy"],
                    })
            else:
                fp_total += 1

        fn_total += sum(1 for g in all_gt if not g["matched"])

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else float("nan")
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else float("nan")
    f1 = _safe_f1(precision, recall)
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else float("nan")

    localization_metrics = {
        "localization_tp": tp_total,
        "localization_fp": fp_total,
        "localization_fn": fn_total,
        "localization_precision": precision,
        "localization_recall": recall,
        "localization_f1": f1,
        "localization_mean_iou": mean_iou,
    }

    matrix: dict[str, dict[str, int]] = {}
    for pred_label, gt_label in all_confusion_pairs:
        matrix.setdefault(pred_label, {})
        matrix[pred_label][gt_label] = matrix[pred_label].get(gt_label, 0) + 1

    return localization_metrics, first_20_samples, matrix


def _eval_class_shifts(
    per_image_data: list[dict],
    cat_id_to_name: dict,
    iou_threshold: float,
) -> dict:
    """
    Pass C: run strict class-aware eval with class-ID offsets 0, +1, -1.
    Identifies whether a 0-vs-1 indexing mismatch explains low strict metrics.
    """
    shift_results: dict[str, dict] = {}
    for shift, key in ((0, "original"), (+1, "plus1"), (-1, "minus1")):
        override = None if shift == 0 else (lambda lbl, s=shift: _shift_label(lbl, s))
        metrics = _eval_strict_class_aware(
            per_image_data, cat_id_to_name, iou_threshold, label_override_fn=override
        )
        shift_results[key] = {
            "global_tp": metrics["global_tp"],
            "global_fp": metrics["global_fp"],
            "global_fn": metrics["global_fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "mean_ap_at_50": metrics["mean_ap_at_50"],
        }

    def _sort_key(k: str) -> tuple[float, float]:
        m = shift_results[k]
        f1 = m.get("f1", float("nan"))
        recall = m.get("recall", float("nan"))
        return (
            0.0 if (f1 is None or isinstance(f1, float) and math.isnan(f1)) else float(f1),
            0.0 if (recall is None or isinstance(recall, float) and math.isnan(recall)) else float(recall),
        )

    best_shift = max(shift_results.keys(), key=_sort_key)

    if best_shift == "plus1":
        note = (
            "RF-DETR runtime appears to return 0-indexed class IDs while COCO annotations "
            "are 1-indexed. +1 class-ID shift improves strict metrics significantly. "
            "Verify by inspecting per-class detection outputs vs. GT labels before claiming accuracy."
        )
    elif best_shift == "minus1":
        note = (
            "A -1 class-ID shift improved metrics, suggesting the model output is 2-indexed "
            "or COCO categories start at 2 (unexpected). Verify annotations and class map."
        )
    else:
        note = (
            "Original class mapping performs best; no class-index offset detected. "
            "Low strict metrics may be due to localization error, not a class mapping mismatch. "
            "Inspect the class-agnostic localization metrics for box quality."
        )

    return {
        "shifts": shift_results,
        "best_shift": best_shift,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_unavailability_report(out_dir: Path, reason: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    status = "inference_unavailable"
    note = (
        "RF-DETR checkpoints and inference script are present, "
        "but detector metric evaluation was not completed in this environment."
    )
    report = {"status": status, "reason": reason, "note": note}
    (out_dir / "rfdetr_v3_eval_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_dir / "rfdetr_v3_eval_report.md").write_text(
        f"# RF-DETR V3 Detector Evaluation\n\n"
        f"**Status:** {status}\n\n**Reason:** {reason}\n\n**Note:** {note}\n",
        encoding="utf-8",
    )
    print(f"Unavailability report written to {out_dir / 'rfdetr_v3_eval_report.json'}")


def _write_full_report(out_dir: Path, results: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rfdetr_v3_eval_report.json"
    md_path = out_dir / "rfdetr_v3_eval_report.md"

    # Strict-valid JSON: NaN/Inf → null
    json_path.write_text(
        json.dumps(_sanitize_for_json(results), indent=2), encoding="utf-8"
    )

    status = results.get("status", "completed")
    summary_note = results.get("summary_note", "")
    num_processed = results.get("num_images_processed", 0) or 0
    first_5_errors = results.get("first_5_errors", [])

    HONEST_NOTE = (
        "Detector accuracy should only be claimed from strict class-aware metrics after "
        "class mapping is validated. Localization metrics indicate box quality independent "
        "of class labels."
    )

    lines = [
        "# RF-DETR V3 Detector Evaluation\n",
        f"**Status:** {status}\n",
        f"\n{summary_note}\n",
        "\n---\n",
        f"\n> **Honest note:** {HONEST_NOTE}\n",
        "\n---\n",
    ]

    if status != "completed" or num_processed == 0:
        lines += [
            "\n**RF-DETR evaluation did not complete because detector inference failed "
            "for all sampled images. Detector accuracy is not claimed from this report.**\n",
        ]
        if first_5_errors:
            lines.append("\n### First inference errors\n")
            lines.append("| Image | Error |\n|-------|-------|\n")
            for e in first_5_errors:
                lines.append(
                    f"| {e.get('file_name','')} | {(e.get('error','') or '')[:200]} |\n"
                )
        md_path.write_text("".join(lines), encoding="utf-8")
        print(f"Evaluation report written to {json_path}")
        return

    avg_rt = results.get("avg_inference_runtime_ms")

    # ── A. Strict class-aware ─────────────────────────────────────────────────
    strict = results.get("strict_metrics", {})
    per_class = strict.get("per_class_metrics", {})

    lines += [
        "\n## A. Strict Class-Aware Metrics\n",
        "| Metric | Value |\n|--------|-------|\n",
        f"| Global TP | {strict.get('global_tp', 0)} |\n",
        f"| Global FP | {strict.get('global_fp', 0)} |\n",
        f"| Global FN | {strict.get('global_fn', 0)} |\n",
        f"| Precision | {_fmt(strict.get('precision'))} |\n",
        f"| Recall | {_fmt(strict.get('recall'))} |\n",
        f"| F1 | {_fmt(strict.get('f1'))} |\n",
        f"| Mean AP@0.5 | {_fmt(strict.get('mean_ap_at_50'))} |\n",
        f"| Mean IoU (matched) | {_fmt(strict.get('mean_iou_matched'))} |\n",
        f"| Avg inference runtime | {_fmt(avg_rt)} ms |\n",
    ]
    if per_class:
        lines.append(
            "\n### Per-Class Strict Metrics\n"
            "| Class | TP | FP | FN | GT | Precision | Recall | F1 | AP@0.5 |\n"
            "|-------|----|----|----|----|-----------|--------|----|--------|\n"
        )
        for cls, cm in sorted(per_class.items()):
            lines.append(
                f"| {cls} | {cm.get('tp',0)} | {cm.get('fp',0)} | {cm.get('fn',0)} "
                f"| {cm.get('gt_count',0)} | {_fmt(cm.get('precision'))} "
                f"| {_fmt(cm.get('recall'))} | {_fmt(cm.get('f1'))} "
                f"| {_fmt(cm.get('ap_at_50'))} |\n"
            )

    # ── B. Class-agnostic localization ────────────────────────────────────────
    loc = results.get("localization_metrics", {})
    lines += [
        "\n## B. Class-Agnostic Localization Metrics (IoU Only)\n",
        "| Metric | Value |\n|--------|-------|\n",
        f"| Localization TP | {loc.get('localization_tp', 0)} |\n",
        f"| Localization FP | {loc.get('localization_fp', 0)} |\n",
        f"| Localization FN | {loc.get('localization_fn', 0)} |\n",
        f"| Localization Precision | {_fmt(loc.get('localization_precision'))} |\n",
        f"| Localization Recall | {_fmt(loc.get('localization_recall'))} |\n",
        f"| Localization F1 | {_fmt(loc.get('localization_f1'))} |\n",
        f"| Localization Mean IoU | {_fmt(loc.get('localization_mean_iou'))} |\n",
    ]

    # ── C. Class-ID shift diagnostic ──────────────────────────────────────────
    shift_diag = results.get("class_shift_diagnostic", {})
    shifts = shift_diag.get("shifts", {})
    best_shift = shift_diag.get("best_shift", "original")
    shift_note = shift_diag.get("note", "")

    lines += ["\n## C. Class-ID Shift Diagnostic\n"]
    if shifts:
        lines.append(
            "| Shift | TP | FP | FN | Precision | Recall | F1 | AP@0.5 |\n"
            "|-------|----|----|----|-----------|---------|----|--------|\n"
        )
        for key, label_str in (("original", "0 (original)"), ("plus1", "+1"), ("minus1", "-1")):
            m = shifts.get(key, {})
            marker = " **(best)**" if key == best_shift else ""
            lines.append(
                f"| {label_str}{marker} | {m.get('global_tp',0)} | {m.get('global_fp',0)} "
                f"| {m.get('global_fn',0)} | {_fmt(m.get('precision'))} "
                f"| {_fmt(m.get('recall'))} | {_fmt(m.get('f1'))} "
                f"| {_fmt(m.get('mean_ap_at_50'))} |\n"
            )
    if shift_note:
        lines.append(f"\n**Diagnostic note:** {shift_note}\n")

    # ── D. Confusion matrix ───────────────────────────────────────────────────
    conf_matrix = results.get("class_confusion_matrix", {})
    first_20 = results.get("first_20_matches", [])

    lines += ["\n## D. Confusion Matrix Summary (IoU >= 0.5, class-agnostic matches)\n"]
    if conf_matrix:
        lines.append("| Predicted | GT Label | Count |\n|-----------|----------|-------|\n")
        for pred_lbl in sorted(conf_matrix.keys()):
            for gt_lbl, count in sorted(conf_matrix[pred_lbl].items()):
                lines.append(f"| {pred_lbl} | {gt_lbl} | {count} |\n")
    else:
        lines.append("No class-agnostic matches found at this IoU threshold.\n")

    if first_20:
        lines.append(
            "\n### First 20 Matched Pairs\n"
            "| Image | Predicted | GT | Confidence | IoU |\n"
            "|-------|-----------|-----|------------|-----|\n"
        )
        for s in first_20:
            lines.append(
                f"| {s.get('image','')} | {s.get('predicted_label','')} "
                f"| {s.get('gt_label','')} | {s.get('confidence','')} "
                f"| {s.get('iou','')} |\n"
            )

    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"Evaluation report written to {json_path}")


# ---------------------------------------------------------------------------
# COCO annotation loader
# ---------------------------------------------------------------------------

def _load_coco(annotations_path: Path) -> tuple[dict, dict, dict]:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    image_id_to_filename: dict[int, str] = {
        img["id"]: img["file_name"] for img in data.get("images", [])
    }
    cat_id_to_name: dict[int, str] = {
        cat["id"]: cat["name"] for cat in data.get("categories", [])
    }
    image_id_to_anns: dict[int, list] = defaultdict(list)
    for ann in data.get("annotations", []):
        image_id_to_anns[ann["image_id"]].append(ann)
    return dict(image_id_to_anns), image_id_to_filename, cat_id_to_name


# ---------------------------------------------------------------------------
# Subprocess inference runner
# ---------------------------------------------------------------------------

def _run_inference_for_image(
    image_path: Path,
    checkpoint: Path,
    confidence: float,
    tmp_dir: Path,
    image_stem: str,
) -> tuple[dict | None, str | None]:
    inference_script = Path(__file__).parent / "run_rfdetr_inference.py"
    out_json = tmp_dir / f"{image_stem}_det.json"
    out_image = tmp_dir / f"{image_stem}_vis.jpg"

    cmd = [
        sys.executable, str(inference_script),
        "--image", str(image_path),
        "--checkpoint", str(checkpoint),
        "--out-json", str(out_json),
        "--out-image", str(out_image),
        "--confidence", str(confidence),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "Subprocess timed out after 120 s"
    except Exception as exc:
        return None, f"Subprocess launch error: {exc}"

    if proc.returncode != 0:
        return None, f"Non-zero exit code {proc.returncode}: {(proc.stderr or '')[:500]}"

    if not out_json.exists():
        return None, "Output JSON not created by inference script"

    try:
        result = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Failed to parse output JSON: {exc}"

    vis_path = out_image if out_image.exists() else None
    result["_vis_path"] = str(vis_path) if vis_path else None
    return result, None


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    dataset_root: Path,
    checkpoint: Path,
    out_dir: Path,
    split: str,
    max_images: int,
    confidence: float,
    iou_threshold: float,
) -> int:
    annotations_path = dataset_root / "rfdetr" / "annotations" / f"instances_{split}.json"
    if not annotations_path.exists():
        print(f"ERROR: Annotation file not found: {annotations_path}", file=sys.stderr)
        return 1

    if not checkpoint.exists():
        print(f"WARNING: Checkpoint not found: {checkpoint}. Writing unavailability report.")
        _write_unavailability_report(out_dir, reason=f"Checkpoint not found: {checkpoint}")
        return 0

    image_id_to_anns, image_id_to_filename, cat_id_to_name = _load_coco(annotations_path)
    all_image_ids = sorted(image_id_to_filename.keys())
    image_ids_to_eval = all_image_ids[:max_images]
    images_dir = dataset_root / "rfdetr" / "images" / split

    print(
        f"Evaluating {len(image_ids_to_eval)} images "
        f"(split={split}, confidence={confidence}, iou_threshold={iou_threshold})"
    )

    # ── Inference loop ─────────────────────────────────────────────────────────
    per_image_data: list[dict] = []
    inference_runtimes_ms: list[float] = []
    inference_errors: list[dict] = []
    skipped_images = 0
    processed_images = 0
    vis_paths: list[str] = []

    with tempfile.TemporaryDirectory() as _tmp_str:
        tmp_dir = Path(_tmp_str)

        for image_id in image_ids_to_eval:
            file_name = image_id_to_filename[image_id]
            image_path = images_dir / file_name

            if not image_path.exists():
                warnings.warn(f"Image file missing, skipping: {image_path}")
                skipped_images += 1
                continue

            gt_anns = image_id_to_anns.get(image_id, [])
            inference_result, error_msg = _run_inference_for_image(
                image_path=image_path,
                checkpoint=checkpoint,
                confidence=confidence,
                tmp_dir=tmp_dir,
                image_stem=image_path.stem,
            )

            if error_msg is not None:
                print(f"  Inference error for {file_name}: {error_msg}")
                inference_errors.append(
                    {"image_id": image_id, "file_name": file_name, "error": error_msg}
                )
                # Include with empty detections so GT contributes to FN counts
                per_image_data.append({
                    "image_id": image_id, "file_name": file_name,
                    "gt_anns": gt_anns, "detections": [],
                    "inference_runtime_ms": None, "inference_ok": False,
                })
                continue

            if not inference_result.get("ok", False):
                err = inference_result.get("error", "inference_not_ok")
                print(f"  Inference not ok for {file_name}: {err}")
                inference_errors.append(
                    {"image_id": image_id, "file_name": file_name, "error": err}
                )
                per_image_data.append({
                    "image_id": image_id, "file_name": file_name,
                    "gt_anns": gt_anns, "detections": [],
                    "inference_runtime_ms": None, "inference_ok": False,
                })
                continue

            runtime_ms = inference_result.get("inference_runtime_ms")
            if runtime_ms is not None:
                inference_runtimes_ms.append(float(runtime_ms))

            if inference_result.get("_vis_path"):
                vis_paths.append(inference_result["_vis_path"])

            processed_images += 1
            per_image_data.append({
                "image_id": image_id, "file_name": file_name,
                "gt_anns": gt_anns,
                "detections": inference_result.get("detections", []),
                "inference_runtime_ms": runtime_ms, "inference_ok": True,
            })

        # Copy sample visualizations
        sample_dir = out_dir / "sample_predictions"
        sample_count = 0
        for vp in vis_paths[:5]:
            vp_path = Path(vp)
            if vp_path.exists():
                sample_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(vp_path, sample_dir / vp_path.name)
                sample_count += 1

    # ── Early exit: all images failed inference ────────────────────────────────
    if processed_images == 0:
        _n_eval = len(image_ids_to_eval)
        fail_results = {
            "status": "evaluation_failed",
            "split": split,
            "num_images_in_split": len(all_image_ids),
            "num_images_evaluated": _n_eval,
            "num_images_processed": 0,
            "num_images_skipped": skipped_images,
            "num_inference_errors": len(inference_errors),
            "iou_threshold": iou_threshold,
            "confidence_threshold": confidence,
            "summary_note": (
                "RF-DETR evaluation did not complete because detector inference failed "
                f"for all {_n_eval} sampled images. "
                "Detector accuracy is not claimed from this report."
            ),
            "first_5_errors": [
                {"file_name": e.get("file_name", ""), "error": (e.get("error", "") or "")[:300]}
                for e in inference_errors[:5]
            ],
            "strict_metrics": {},
            "localization_metrics": {},
            "class_shift_diagnostic": {},
            "class_confusion_matrix": {},
            "first_20_matches": [],
        }
        _write_full_report(out_dir, fail_results)
        print(f"\nSummary: processed=0/{_n_eval}, evaluation_failed")
        return 0

    # ── Three evaluation passes ────────────────────────────────────────────────
    avg_rt = (
        sum(inference_runtimes_ms) / len(inference_runtimes_ms)
        if inference_runtimes_ms else float("nan")
    )

    print(f"  Pass A: strict class-aware ...")
    strict_metrics = _eval_strict_class_aware(per_image_data, cat_id_to_name, iou_threshold)

    print(f"  Pass B: class-agnostic localization ...")
    localization_metrics, first_20_matches, class_confusion_matrix = _eval_localization_agnostic(
        per_image_data, cat_id_to_name, iou_threshold
    )

    print(f"  Pass C: class-ID shift diagnostic ...")
    class_shift_diagnostic = _eval_class_shifts(per_image_data, cat_id_to_name, iou_threshold)

    num_evaluated = len(image_ids_to_eval)
    full_results = {
        "status": "completed",
        "split": split,
        "num_images_in_split": len(all_image_ids),
        "num_images_evaluated": num_evaluated,
        "num_images_processed": processed_images,
        "num_images_skipped": skipped_images,
        "num_inference_errors": len(inference_errors),
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence,
        "avg_inference_runtime_ms": avg_rt,
        "summary_note": (
            f"RF-DETR detector evaluated against V3 verified annotations on "
            f"{processed_images} diagrams. "
            "Metrics are prototype benchmark metrics, not production accuracy."
        ),
        "strict_metrics": strict_metrics,
        "localization_metrics": localization_metrics,
        "class_shift_diagnostic": class_shift_diagnostic,
        "class_confusion_matrix": class_confusion_matrix,
        "first_20_matches": first_20_matches,
        "inference_errors": inference_errors,
        "sample_predictions_dir": str(sample_dir) if sample_count > 0 else None,
    }

    _write_full_report(out_dir, full_results)

    def _f(v):
        return "N/A" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.4f}"

    print(
        f"\nSummary: processed={processed_images}/{num_evaluated}\n"
        f"  Strict:       precision={_f(strict_metrics.get('precision'))}, "
        f"recall={_f(strict_metrics.get('recall'))}, "
        f"f1={_f(strict_metrics.get('f1'))}, "
        f"mAP@0.5={_f(strict_metrics.get('mean_ap_at_50'))}\n"
        f"  Localization: precision={_f(localization_metrics.get('localization_precision'))}, "
        f"recall={_f(localization_metrics.get('localization_recall'))}, "
        f"f1={_f(localization_metrics.get('localization_f1'))}\n"
        f"  Best class-ID shift: {class_shift_diagnostic.get('best_shift', 'original')}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RF-DETR inference against InfraGraph V3 COCO annotations."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/infragraph_v3"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("model_artifacts/rfdetr_v3/checkpoint_best_total.pth"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/rfdetr_v3_eval"))
    parser.add_argument("--split", choices=["val", "test", "train"], default="val")
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sys.exit(evaluate(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        split=args.split,
        max_images=args.max_images,
        confidence=args.confidence,
        iou_threshold=args.iou_threshold,
    ))


if __name__ == "__main__":
    main()
