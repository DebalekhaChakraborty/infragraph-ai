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
    find_best_rfdetr_checkpoint,
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
    checkpoint = Path(args.checkpoint) if args.checkpoint else find_best_rfdetr_checkpoint(REPO_ROOT)
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
