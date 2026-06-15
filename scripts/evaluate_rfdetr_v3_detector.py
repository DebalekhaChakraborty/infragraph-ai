"""evaluate_rfdetr_v3_detector.py

Evaluate RF-DETR inference against InfraGraph V3 COCO-format annotations.

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
import time
import warnings
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU of two axis-aligned bounding boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _coco_xywh_to_xyxy(bbox: list[float]) -> list[float]:
    """Convert COCO [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _det_dict_to_xyxy(det: dict) -> list[float]:
    """Convert detection bbox dict with x1/y1/x2/y2 keys to list."""
    b = det["bbox"]
    return [b["x1"], b["y1"], b["x2"], b["y2"]]


# ---------------------------------------------------------------------------
# AP computation (simple trapezoidal precision-recall curve)
# ---------------------------------------------------------------------------

def _compute_ap(
    sorted_detections: list[dict],  # sorted by confidence desc; each has "tp" bool
    num_gt: int,
) -> float:
    """Compute Average Precision at a single IoU threshold."""
    if num_gt == 0:
        return float("nan")
    if not sorted_detections:
        return 0.0

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for det in sorted_detections:
        if det["tp"]:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / num_gt
        precisions.append(precision)
        recalls.append(recall)

    # Trapezoidal rule over the precision-recall curve
    ap = 0.0
    prev_recall = 0.0
    prev_precision = 1.0
    for p, r in zip(precisions, recalls):
        ap += (r - prev_recall) * (p + prev_precision) / 2.0
        prev_recall = r
        prev_precision = p

    return ap


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

    report = {
        "status": status,
        "reason": reason,
        "note": note,
    }

    json_path = out_dir / "rfdetr_v3_eval_report.json"
    md_path = out_dir / "rfdetr_v3_eval_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path.write_text(
        f"# RF-DETR V3 Detector Evaluation\n\n"
        f"**Status:** {status}\n\n"
        f"**Reason:** {reason}\n\n"
        f"**Note:** {note}\n",
        encoding="utf-8",
    )
    print(f"Unavailability report written to {json_path}")


def _write_full_report(
    out_dir: Path,
    results: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "rfdetr_v3_eval_report.json"
    md_path = out_dir / "rfdetr_v3_eval_report.md"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    metrics = results.get("metrics", {})
    per_class = results.get("per_class_metrics", {})
    summary_note = results.get("summary_note", "")

    def fmt(v: object) -> str:
        if isinstance(v, float):
            if math.isnan(v):
                return "N/A"
            return f"{v:.4f}"
        return str(v)

    lines = [
        "# RF-DETR V3 Detector Evaluation\n",
        f"**Status:** completed\n",
        f"\n{summary_note}\n",
        "\n## Overall Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {k} | {fmt(v)} |")

    lines.append("\n## Per-Class Metrics\n")
    lines.append("| Class | TP | FP | FN | Precision | Recall | F1 | AP@0.5 |")
    lines.append("|-------|----|----|-----|-----------|--------|----|--------|")
    for cls, cm in per_class.items():
        lines.append(
            f"| {cls} "
            f"| {cm.get('tp', 0)} "
            f"| {cm.get('fp', 0)} "
            f"| {cm.get('fn', 0)} "
            f"| {fmt(cm.get('precision', float('nan')))} "
            f"| {fmt(cm.get('recall', float('nan')))} "
            f"| {fmt(cm.get('f1', float('nan')))} "
            f"| {fmt(cm.get('ap_at_50', float('nan')))} |"
        )

    md_content = "\n".join(lines) + "\n"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Evaluation report written to {json_path}")


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def _load_coco(annotations_path: Path) -> tuple[dict, dict, dict]:
    """Load COCO JSON and return (image_id->anns, image_id->filename, cat_id->name)."""
    data = json.loads(annotations_path.read_text(encoding="utf-8"))

    image_id_to_filename: dict[int, str] = {
        img["id"]: img["file_name"] for img in data.get("images", [])
    }
    cat_id_to_name: dict[int, str] = {
        cat["id"]: cat["name"] for cat in data.get("categories", [])
    }
    image_id_to_anns: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        image_id_to_anns[ann["image_id"]].append(ann)

    return dict(image_id_to_anns), image_id_to_filename, cat_id_to_name


def _run_inference_for_image(
    image_path: Path,
    checkpoint: Path,
    confidence: float,
    tmp_dir: Path,
    image_stem: str,
) -> tuple[dict | None, str | None]:
    """
    Call run_rfdetr_inference.py as a subprocess.
    Returns (parsed_json_or_None, error_message_or_None).
    """
    inference_script = (
        Path(__file__).parent / "run_rfdetr_inference.py"
    )
    out_json = tmp_dir / f"{image_stem}_det.json"
    out_image = tmp_dir / f"{image_stem}_vis.jpg"

    cmd = [
        sys.executable,
        str(inference_script),
        "--image", str(image_path),
        "--checkpoint", str(checkpoint),
        "--out-json", str(out_json),
        "--out-image", str(out_image),
        "--confidence", str(confidence),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None, "Subprocess timed out after 120 s"
    except Exception as exc:
        return None, f"Subprocess launch error: {exc}"

    if proc.returncode != 0:
        stderr_snippet = proc.stderr[:500] if proc.stderr else ""
        return None, f"Non-zero exit code {proc.returncode}: {stderr_snippet}"

    if not out_json.exists():
        return None, "Output JSON not created by inference script"

    try:
        result = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Failed to parse output JSON: {exc}"

    vis_path: Path | None = out_image if out_image.exists() else None
    result["_vis_path"] = str(vis_path) if vis_path else None

    return result, None


def evaluate(
    dataset_root: Path,
    checkpoint: Path,
    out_dir: Path,
    split: str,
    max_images: int,
    confidence: float,
    iou_threshold: float,
) -> int:
    """Run evaluation. Returns 0 on success, 1 on fatal error."""

    # Check annotation file
    annotations_path = (
        dataset_root / "rfdetr" / "annotations" / f"instances_{split}.json"
    )
    if not annotations_path.exists():
        print(
            f"ERROR: Annotation file not found: {annotations_path}",
            file=sys.stderr,
        )
        return 1

    # Check checkpoint
    if not checkpoint.exists():
        print(
            f"WARNING: Checkpoint not found: {checkpoint}. "
            "Writing unavailability report.",
        )
        _write_unavailability_report(
            out_dir,
            reason=f"Checkpoint not found: {checkpoint}",
        )
        return 0

    # Load COCO annotations
    image_id_to_anns, image_id_to_filename, cat_id_to_name = _load_coco(
        annotations_path
    )
    all_image_ids = sorted(image_id_to_filename.keys())

    images_dir = dataset_root / "rfdetr" / "images" / split
    image_ids_to_eval = all_image_ids[:max_images]

    print(
        f"Evaluating {len(image_ids_to_eval)} images "
        f"(split={split}, confidence={confidence}, iou_threshold={iou_threshold})"
    )

    # Per-class accumulators for TP/FP/FN
    per_class_tp: dict[str, int] = defaultdict(int)
    per_class_fp: dict[str, int] = defaultdict(int)
    per_class_fn: dict[str, int] = defaultdict(int)
    # For AP: class -> list of {confidence, tp}
    per_class_det_records: dict[str, list[dict]] = defaultdict(list)
    per_class_gt_count: dict[str, int] = defaultdict(int)

    total_matched_iou: list[float] = []
    inference_runtimes_ms: list[float] = []
    inference_errors: list[str] = []
    skipped_images: int = 0
    processed_images: int = 0
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

            # Count GT per class
            for ann in gt_anns:
                cls_name = cat_id_to_name.get(ann["category_id"], "unknown")
                per_class_gt_count[cls_name] += 1

            image_stem = image_path.stem
            inference_result, error_msg = _run_inference_for_image(
                image_path=image_path,
                checkpoint=checkpoint,
                confidence=confidence,
                tmp_dir=tmp_dir,
                image_stem=image_stem,
            )

            if error_msg is not None:
                print(f"  Inference error for {file_name}: {error_msg}")
                inference_errors.append(
                    {"image_id": image_id, "file_name": file_name, "error": error_msg}
                )
                # All GT become FN for this image
                for ann in gt_anns:
                    cls_name = cat_id_to_name.get(ann["category_id"], "unknown")
                    per_class_fn[cls_name] += 1
                continue

            if not inference_result.get("ok", False):
                err = inference_result.get("error", "inference_not_ok")
                print(f"  Inference not ok for {file_name}: {err}")
                inference_errors.append(
                    {"image_id": image_id, "file_name": file_name, "error": err}
                )
                for ann in gt_anns:
                    cls_name = cat_id_to_name.get(ann["category_id"], "unknown")
                    per_class_fn[cls_name] += 1
                continue

            if "inference_runtime_ms" in inference_result:
                inference_runtimes_ms.append(
                    float(inference_result["inference_runtime_ms"])
                )

            if inference_result.get("_vis_path"):
                vis_paths.append(inference_result["_vis_path"])

            detections = inference_result.get("detections", [])
            processed_images += 1

            # Group GT boxes by class
            gt_by_class: dict[str, list[dict]] = defaultdict(list)
            for ann in gt_anns:
                cls_name = cat_id_to_name.get(ann["category_id"], "unknown")
                gt_box_xyxy = _coco_xywh_to_xyxy(ann["bbox"])
                gt_by_class[cls_name].append(
                    {"bbox_xyxy": gt_box_xyxy, "matched": False}
                )

            # Greedy matching per image per class
            # Sort predictions by confidence descending
            preds_by_class: dict[str, list[dict]] = defaultdict(list)
            for det in detections:
                label = det.get("label", "unknown")
                preds_by_class[label].append(det)

            for pred_label, preds in preds_by_class.items():
                preds_sorted = sorted(
                    preds, key=lambda d: d.get("confidence", 0.0), reverse=True
                )
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
                        # True Positive
                        gt_list[best_gt_idx]["matched"] = True
                        per_class_tp[pred_label] += 1
                        total_matched_iou.append(best_iou)
                        per_class_det_records[pred_label].append(
                            {"confidence": pred.get("confidence", 0.0), "tp": True}
                        )
                    else:
                        # False Positive
                        per_class_fp[pred_label] += 1
                        per_class_det_records[pred_label].append(
                            {"confidence": pred.get("confidence", 0.0), "tp": False}
                        )

            # Unmatched GT = False Negatives
            for cls_name, gt_list in gt_by_class.items():
                fn_count = sum(1 for g in gt_list if not g["matched"])
                per_class_fn[cls_name] += fn_count

        # Copy sample visualizations
        sample_dir = out_dir / "sample_predictions"
        sample_count = 0
        for vp in vis_paths[:5]:
            vp_path = Path(vp)
            if vp_path.exists():
                sample_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(vp_path, sample_dir / vp_path.name)
                sample_count += 1

    # Aggregate metrics
    all_classes = sorted(
        set(list(per_class_tp.keys()) + list(per_class_fp.keys()) + list(per_class_fn.keys()) + list(per_class_gt_count.keys()))
    )

    global_tp = sum(per_class_tp.values())
    global_fp = sum(per_class_fp.values())
    global_fn = sum(per_class_fn.values())

    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else float("nan")
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if not (math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0)
        else float("nan")
    )
    mean_iou_matched = (
        sum(total_matched_iou) / len(total_matched_iou) if total_matched_iou else float("nan")
    )
    avg_inference_runtime_ms = (
        sum(inference_runtimes_ms) / len(inference_runtimes_ms)
        if inference_runtimes_ms
        else float("nan")
    )

    # Per-class metrics
    per_class_report: dict[str, dict] = {}
    ap_values: list[float] = []
    for cls in all_classes:
        tp = per_class_tp.get(cls, 0)
        fp = per_class_fp.get(cls, 0)
        fn = per_class_fn.get(cls, 0)
        gt_count = per_class_gt_count.get(cls, 0)

        cls_precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        cls_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        cls_f1 = (
            2 * cls_precision * cls_recall / (cls_precision + cls_recall)
            if not (math.isnan(cls_precision) or math.isnan(cls_recall) or (cls_precision + cls_recall) == 0)
            else float("nan")
        )

        det_records = sorted(
            per_class_det_records.get(cls, []),
            key=lambda d: d["confidence"],
            reverse=True,
        )
        ap = _compute_ap(det_records, gt_count)
        if not math.isnan(ap):
            ap_values.append(ap)

        per_class_report[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_count": gt_count,
            "precision": cls_precision,
            "recall": cls_recall,
            "f1": cls_f1,
            "ap_at_50": ap,
        }

    mean_ap = sum(ap_values) / len(ap_values) if ap_values else float("nan")

    num_evaluated = len(image_ids_to_eval)
    summary_note = (
        f"RF-DETR detector evaluated against V3 verified annotations on "
        f"{processed_images} diagrams. "
        f"Metrics are prototype benchmark metrics, not production accuracy."
    )

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
        "summary_note": summary_note,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou_matched": mean_iou_matched,
            "mean_ap_at_50": mean_ap,
            "avg_inference_runtime_ms": avg_inference_runtime_ms,
            "global_tp": global_tp,
            "global_fp": global_fp,
            "global_fn": global_fn,
        },
        "per_class_metrics": per_class_report,
        "inference_errors": inference_errors,
        "sample_predictions_dir": str(out_dir / "sample_predictions") if sample_count > 0 else None,
    }

    _write_full_report(out_dir, full_results)
    print(
        f"\nSummary: processed={processed_images}, "
        f"precision={precision:.4f}, recall={recall:.4f}, "
        f"f1={f1:.4f}, mAP@0.5={mean_ap:.4f}"
        if not (math.isnan(precision) or math.isnan(recall))
        else f"\nSummary: processed={processed_images}, insufficient data for metrics"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RF-DETR inference against InfraGraph V3 COCO annotations."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/infragraph_v3"),
        help="Root of the InfraGraph V3 dataset (contains rfdetr/ subdirectory).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model_artifacts/rfdetr_v3/checkpoint_best_total.pth"),
        help="Path to the RF-DETR checkpoint .pth file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/rfdetr_v3_eval"),
        help="Directory to write evaluation reports and sample predictions.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "train"],
        default="val",
        help="Dataset split to evaluate (default: val).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=30,
        help="Maximum number of images to evaluate (default: 30).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU threshold for TP/FP classification (default: 0.50).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    exit_code = evaluate(
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        split=args.split,
        max_images=args.max_images,
        confidence=args.confidence,
        iou_threshold=args.iou_threshold,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
