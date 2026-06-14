"""Check the external RF-DETR detector runtime used by InfraGraph Streamlit."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from rfdetr_subprocess_bridge import (  # noqa: E402
    check_rfdetr_http_service,
    check_rfdetr_runtime,
    find_rfdetr_checkpoint_with_reason,
    is_valid_rfdetr_checkpoint_path,
    rfdetr_service_base_url,
    resolve_rfdetr_python,
    resolve_rfdetr_python_details,
    run_rfdetr_subprocess,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check InfraGraph external RF-DETR runtime.")
    parser.add_argument("--python", dest="python_executable", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--confidence", type=float, default=float(os.environ.get("INFRAGRAPH_RFDETR_CONFIDENCE", "0.25")))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("INFRAGRAPH_RFDETR_TIMEOUT", "180")))
    args = parser.parse_args()

    if args.python_executable:
        selected_python = args.python_executable
        resolution = {
            "python_resolution_mode": "cli",
            "requested_detector_python": args.python_executable,
            "resolved_detector_python": args.python_executable,
            "python_executable": args.python_executable,
            "resolved_from_env": False,
            "resolved_from_path": False,
            "streamlit_python": sys.executable,
            "fallback_reason": "",
        }
    else:
        resolution = resolve_rfdetr_python_details()
        selected_python = str(resolution.get("python_executable") or resolve_rfdetr_python())
    # ── Checkpoint resolution ─────────────────────────────────────────────────
    rejection_reason: str = ""
    checkpoint_status: str = ""
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
        if not checkpoint.exists():
            checkpoint_status = "NOT FOUND"
            rejection_reason = f"Path does not exist: {checkpoint}"
            checkpoint = None
        elif not is_valid_rfdetr_checkpoint_path(checkpoint):
            checkpoint_status = "REJECTED (not a valid RF-DETR checkpoint)"
            rejection_reason = (
                f"Path {checkpoint} does not meet RF-DETR checkpoint requirements. "
                "Allowed names: checkpoint_best_total.pth, checkpoint_best_regular.pth, "
                "checkpoint_best_ema.pth, last.ckpt. "
                "Path must contain 'rfdetr' and must not contain qwen/rng_state/optimizer/scheduler."
            )
            checkpoint = None
        else:
            checkpoint_status = "VALID RF-DETR"
    else:
        checkpoint, rejection_reason = find_rfdetr_checkpoint_with_reason(REPO_ROOT)
        checkpoint_status = "VALID RF-DETR" if checkpoint else "NONE"

    runtime = check_rfdetr_runtime(selected_python)

    print(f"Current Python executable: {sys.executable}")
    print(f"Python resolution mode: {resolution.get('python_resolution_mode', 'unknown')}")
    print(f"Requested detector python: {resolution.get('requested_detector_python') or 'python'}")
    print(f"Resolved detector python: {resolution.get('resolved_detector_python') or selected_python}")
    print(f"Resolved from env: {resolution.get('resolved_from_env', False)}")
    print(f"Resolved from PATH: {resolution.get('resolved_from_path', False)}")
    print(f"Streamlit/current Python: {resolution.get('streamlit_python') or sys.executable}")
    if resolution.get("fallback_reason"):
        print(f"Fallback reason: {resolution.get('fallback_reason')}")
    print(f"Selected RF-DETR Python executable: {selected_python}")
    print(f"RF-DETR import in selected runtime: {'PASS' if runtime.get('ok') else 'FAIL'}")
    if runtime.get("error"):
        print(f"Runtime error: {runtime.get('error')}")
    if runtime.get("stderr_preview"):
        print(f"Runtime stderr: {runtime.get('stderr_preview')}")
    print(f"Checkpoint discovered: {checkpoint if checkpoint else 'NONE'}")
    print(f"Checkpoint valid RF-DETR: {checkpoint_status}")
    if rejection_reason:
        print(f"Checkpoint rejection reason: {rejection_reason}")
    if not checkpoint:
        print(
            "FAIL: No valid RF-DETR checkpoint found. "
            "Set INFRAGRAPH_RFDETR_CHECKPOINT to checkpoint_best_total.pth "
            "under model_artifacts/rfdetr_v3/model or outputs/rfdetr_v3/model."
        )

    service_url = rfdetr_service_base_url()
    if service_url:
        service = check_rfdetr_http_service(service_url)
        print(f"RF-DETR HTTP service URL: {service_url}")
        print(f"RF-DETR HTTP service health: {'PASS' if service.get('ok') else 'FAIL'}")
        if service.get("error"):
            print(f"RF-DETR HTTP service error: {service.get('error')}")

    overall_ok = bool(runtime.get("ok")) and bool(checkpoint and checkpoint.exists())

    if args.image:
        image = Path(args.image)
        print(f"Optional inference image: {image}")
        if not image.exists():
            print("Inference test: FAIL (image not found)")
            overall_ok = False
        else:
            old_env = os.environ.get("INFRAGRAPH_RFDETR_PYTHON")
            os.environ["INFRAGRAPH_RFDETR_PYTHON"] = selected_python
            try:
                result = run_rfdetr_subprocess(image, checkpoint, args.confidence, args.timeout)
            finally:
                if old_env is None:
                    os.environ.pop("INFRAGRAPH_RFDETR_PYTHON", None)
                else:
                    os.environ["INFRAGRAPH_RFDETR_PYTHON"] = old_env
            print(f"Inference test: {'PASS' if result.get('ok') else 'FAIL'}")
            print(json.dumps({
                "ok": result.get("ok"),
                "error": result.get("error", ""),
                "detections": len(result.get("detections", [])),
                "annotated_image_path": result.get("annotated_image_path", ""),
            }, indent=2))
            overall_ok = overall_ok and bool(result.get("ok"))

    print(f"Overall: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
