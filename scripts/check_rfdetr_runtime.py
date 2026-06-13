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
    check_rfdetr_runtime,
    find_best_rfdetr_checkpoint,
    resolve_rfdetr_python,
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

    selected_python = args.python_executable or resolve_rfdetr_python()
    checkpoint = Path(args.checkpoint) if args.checkpoint else find_best_rfdetr_checkpoint(REPO_ROOT)
    runtime = check_rfdetr_runtime(selected_python)

    print(f"Current Python executable: {sys.executable}")
    print(f"Selected RF-DETR Python executable: {selected_python}")
    print(f"RF-DETR import in selected runtime: {'PASS' if runtime.get('ok') else 'FAIL'}")
    if runtime.get("error"):
        print(f"Runtime error: {runtime.get('error')}")
    if runtime.get("stderr_preview"):
        print(f"Runtime stderr: {runtime.get('stderr_preview')}")
    print(f"Checkpoint discovered: {checkpoint if checkpoint else 'NONE'}")

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
