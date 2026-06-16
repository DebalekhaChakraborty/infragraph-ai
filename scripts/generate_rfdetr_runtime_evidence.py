"""generate_rfdetr_runtime_evidence.py

Run RF-DETR inference on a representative sample of V3 val images and produce
runtime/inference evidence reports.

This script does NOT claim detector accuracy or mAP.  Its sole purpose is to
show that the RF-DETR checkpoint loads and produces detections on real diagrams.
The production app uses RF-DETR-supported detection with verified annotation
fallback for reliable graph extraction.

Usage:
    python scripts/generate_rfdetr_runtime_evidence.py [--images-dir ...] [--checkpoint ...]

Outputs:
    reports/rfdetr_runtime_evidence/predictions/<stem>.json  — per-image JSON
    reports/rfdetr_runtime_evidence/annotated/<stem>_annotated.png — per-image overlay
    reports/rfdetr_runtime_evidence/rfdetr_runtime_evidence.json
    reports/rfdetr_runtime_evidence/rfdetr_runtime_evidence.md
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_SCRIPT = REPO_ROOT / "scripts" / "run_rfdetr_inference.py"

DEFAULT_IMAGES_DIR = (
    REPO_ROOT / "datasets" / "infragraph_v3" / "rfdetr" / "images" / "val"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "model_artifacts" / "rfdetr_v3" / "checkpoint_best_total.pth"
)
OUT_DIR = REPO_ROOT / "reports" / "rfdetr_runtime_evidence"
PRED_DIR = OUT_DIR / "predictions"
ANN_DIR = OUT_DIR / "annotated"

# Representative images: one per distinct topology type
SAMPLE_STEMS = [
    "enterprise_v3_0064__datacenter_topology",
    "enterprise_v3_0065__app_db_topology",
    "enterprise_v3_0066__shared_services_topology",
    "enterprise_v3_0067__branch_topology",
    "enterprise_v3_0068__wan_topology",
]

HONEST_NOTE = (
    "This is RF-DETR runtime/inference evidence. "
    "Detector accuracy/mAP is not claimed here. "
    "The production demo uses RF-DETR-supported detection with verified annotation "
    "fallback for reliability."
)


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def _run_one(
    image_path: Path,
    checkpoint: Path,
    out_json: Path,
    out_image: Path,
    confidence: float,
    timeout_s: int,
) -> tuple[bool, dict, float]:
    """
    Call run_rfdetr_inference.py as a subprocess.
    Returns (success, result_dict, elapsed_s).
    """
    cmd = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--image", str(image_path),
        "--checkpoint", str(checkpoint),
        "--out-json", str(out_json),
        "--out-image", str(out_image),
        "--confidence", str(confidence),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed_s = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return False, {"error": f"Subprocess timed out after {timeout_s}s"}, timeout_s
    except Exception as exc:
        return False, {"error": f"Subprocess launch error: {exc}"}, 0.0

    result: dict = {}
    if out_json.exists():
        try:
            result = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception as exc:
            result = {"error": f"Could not parse output JSON: {exc}"}
    else:
        stderr = (proc.stderr or "").strip()[:400]
        result = {"error": f"Exit code {proc.returncode}; no output JSON. stderr: {stderr}"}

    success = proc.returncode == 0 and result.get("ok", False)
    return success, result, elapsed_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RF-DETR runtime evidence on representative V3 val images. "
            "Does not claim accuracy or mAP."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory containing val images (default: datasets/infragraph_v3/rfdetr/images/val)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="RF-DETR checkpoint path (default: model_artifacts/rfdetr_v3/checkpoint_best_total.pth)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-image inference timeout in seconds (default: 180)",
    )
    args = parser.parse_args()

    images_dir: Path = args.images_dir
    checkpoint: Path = args.checkpoint

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    print(f"RF-DETR Runtime Evidence Generator")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Images dir : {images_dir}")
    print(f"  Confidence : {args.confidence}")
    print(f"  Timeout    : {args.timeout}s")
    print()

    if not INFERENCE_SCRIPT.exists():
        _write_unavailable(generated_at, reason=f"Inference script not found: {INFERENCE_SCRIPT}")
        return

    if not checkpoint.exists():
        _write_unavailable(generated_at, reason=f"Checkpoint not found: {checkpoint}")
        return

    if not images_dir.exists():
        _write_unavailable(generated_at, reason=f"Images directory not found: {images_dir}")
        return

    # Resolve image paths — prefer the specified stems, fall back to any 5 images
    candidate_paths: list[Path] = []
    for stem in SAMPLE_STEMS:
        for ext in (".png", ".jpg", ".jpeg"):
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                candidate_paths.append(p)
                break

    if not candidate_paths:
        # Fallback: pick first 5 images in the directory
        all_imgs = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        candidate_paths = all_imgs[:5]

    if not candidate_paths:
        _write_unavailable(generated_at, reason=f"No images found in {images_dir}")
        return

    # Run inference on each candidate
    per_image_results: list[dict] = []
    runtimes_ms: list[float] = []

    for img_path in candidate_paths:
        stem = img_path.stem
        out_json = PRED_DIR / f"{stem}.json"
        out_image = ANN_DIR / f"{stem}_annotated.png"

        print(f"  [{stem}]")
        success, result, elapsed_s = _run_one(
            image_path=img_path,
            checkpoint=checkpoint,
            out_json=out_json,
            out_image=out_image,
            confidence=args.confidence,
            timeout_s=args.timeout,
        )

        elapsed_ms = elapsed_s * 1000.0
        detections = result.get("detections", []) if success else []
        det_count = len(detections)

        if success:
            reported_ms = result.get("inference_runtime_ms")
            if reported_ms is not None:
                runtimes_ms.append(float(reported_ms))
            else:
                runtimes_ms.append(elapsed_ms)
            print(f"    OK — {det_count} detection(s) in {elapsed_ms:.0f} ms")
        else:
            print(f"    FAILED — {result.get('error', 'unknown error')[:120]}")

        per_image_results.append({
            "image": img_path.name,
            "image_path": str(img_path),
            "success": success,
            "elapsed_ms": round(elapsed_ms, 1),
            "inference_runtime_ms": result.get("inference_runtime_ms"),
            "detection_count": det_count,
            "detections_summary": [
                {"label": d.get("label"), "confidence": round(d.get("confidence", 0), 3)}
                for d in detections[:10]
            ],
            "checkpoint_load_strategy": result.get("checkpoint_load_strategy"),
            "model_class": result.get("model_class"),
            "inference_strategy": result.get("inference_strategy"),
            "annotated_image": str(out_image) if success and out_image.exists() else None,
            "prediction_json": str(out_json) if success and out_json.exists() else None,
            "error": result.get("error") if not success else None,
        })

    # Aggregate
    num_attempted = len(per_image_results)
    num_succeeded = sum(1 for r in per_image_results if r["success"])
    avg_runtime_ms = (
        sum(runtimes_ms) / len(runtimes_ms) if runtimes_ms else None
    )
    total_detections = sum(r["detection_count"] for r in per_image_results)

    evidence = {
        "generated_at": generated_at,
        "checkpoint_path": str(checkpoint),
        "images_dir": str(images_dir),
        "confidence_threshold": args.confidence,
        "num_images_attempted": num_attempted,
        "num_images_succeeded": num_succeeded,
        "avg_inference_runtime_ms": round(avg_runtime_ms, 1) if avg_runtime_ms is not None else None,
        "total_detections_across_images": total_detections,
        "honest_note": HONEST_NOTE,
        "per_image": per_image_results,
    }

    json_path = OUT_DIR / "rfdetr_runtime_evidence.json"
    json_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  JSON report -> {json_path}")

    _write_markdown(evidence, OUT_DIR)
    print(f"  MD report  -> {OUT_DIR / 'rfdetr_runtime_evidence.md'}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_unavailable(generated_at: str, reason: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence = {
        "generated_at": generated_at,
        "status": "unavailable",
        "reason": reason,
        "honest_note": HONEST_NOTE,
        "num_images_attempted": 0,
        "num_images_succeeded": 0,
    }
    (OUT_DIR / "rfdetr_runtime_evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "rfdetr_runtime_evidence.md").write_text(
        f"# RF-DETR Runtime Evidence\n\n"
        f"**Status:** unavailable\n\n"
        f"**Reason:** {reason}\n\n"
        f"**Note:** {HONEST_NOTE}\n",
        encoding="utf-8",
    )
    print(f"Unavailability report written to {OUT_DIR}")


def _write_markdown(evidence: dict, out_dir: Path) -> None:
    num_att = evidence["num_images_attempted"]
    num_ok = evidence["num_images_succeeded"]
    avg_ms = evidence.get("avg_inference_runtime_ms")
    avg_str = f"{avg_ms:.0f} ms" if avg_ms is not None else "N/A"
    ckpt = evidence.get("checkpoint_path", "N/A")
    total_det = evidence.get("total_detections_across_images", 0)

    lines = [
        "# RF-DETR Runtime Evidence\n",
        f"Generated: {evidence.get('generated_at', 'N/A')}\n",
        "\n---\n",
        "\n> **" + HONEST_NOTE + "**\n",
        "\n---\n",
        "\n## Summary\n",
        f"| Field | Value |\n|-------|-------|\n",
        f"| Images attempted | {num_att} |\n",
        f"| Successful inference runs | {num_ok} |\n",
        f"| Average inference runtime | {avg_str} |\n",
        f"| Total detections (all images) | {total_det} |\n",
        f"| Checkpoint | `{ckpt}` |\n",
        "\n## Per-Image Results\n",
        "| Image | Status | Runtime (ms) | Detections | Annotated output |\n",
        "|-------|--------|-------------|------------|------------------|\n",
    ]

    for r in evidence.get("per_image", []):
        status = "OK" if r["success"] else "FAILED"
        rt = f"{r['elapsed_ms']:.0f}" if r.get("elapsed_ms") is not None else "N/A"
        det = r.get("detection_count", 0) if r["success"] else "—"
        ann = r.get("annotated_image") or "—"
        ann_cell = f"`{Path(ann).name}`" if ann != "—" else "—"
        lines.append(f"| `{r['image']}` | {status} | {rt} | {det} | {ann_cell} |\n")

    # Per-image detection detail
    succeeded = [r for r in evidence.get("per_image", []) if r["success"]]
    if succeeded:
        lines += [
            "\n## Detection Summaries\n",
        ]
        for r in succeeded:
            lines.append(f"\n### {r['image']}\n")
            lines.append(f"- Inference runtime: {r.get('inference_runtime_ms', 'N/A')} ms\n")
            lines.append(f"- Model class: {r.get('model_class', 'N/A')}\n")
            lines.append(f"- Checkpoint strategy: {r.get('checkpoint_load_strategy', 'N/A')}\n")
            lines.append(f"- Detections ({r['detection_count']}):\n")
            for d in r.get("detections_summary", []):
                lines.append(f"  - `{d['label']}` confidence={d['confidence']:.3f}\n")

    else:
        lines += [
            "\n## Note\n",
            "No successful inference runs in this environment. "
            "Run on the AMD/ROCm machine with `rfdetr` installed.\n",
        ]

    (out_dir / "rfdetr_runtime_evidence.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
