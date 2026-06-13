#!/usr/bin/env python
"""Small HTTP service for RF-DETR detector inference.

Run this from the detector/base environment:

    python scripts/serve_rfdetr_detector.py --host 0.0.0.0 --port 8010
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_rfdetr_inference.py"


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def _health_payload() -> dict[str, Any]:
    return {
        "ok": importlib.util.find_spec("rfdetr") is not None,
        "source": "live_rfdetr_http_service",
        "python_executable": sys.executable,
        "rfdetr_import_ok": importlib.util.find_spec("rfdetr") is not None,
        "runner_path": str(RUNNER),
    }


class RFDETRHandler(BaseHTTPRequestHandler):
    server_version = "InfraGraphRFDETR/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            payload = _health_payload()
            _json_response(self, 200 if payload["ok"] else 503, payload)
            return
        _json_response(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/detect":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return

        started = time.perf_counter()
        try:
            req = _read_json(self)
            image_path = Path(str(req.get("image_path", "")))
            checkpoint_path = Path(str(req.get("checkpoint_path", "")))
            confidence = float(req.get("confidence", 0.25))
            out_image = Path(str(req.get("out_image", ""))) if req.get("out_image") else None
            if not image_path.exists():
                _json_response(self, 400, {"ok": False, "source": "live_rfdetr_http_service", "error": f"image not found: {image_path}"})
                return
            if not checkpoint_path.exists():
                _json_response(self, 400, {"ok": False, "source": "live_rfdetr_http_service", "error": f"checkpoint not found: {checkpoint_path}"})
                return

            tmp_dir = Path(tempfile.mkdtemp(prefix="infragraph_rfdetr_service_"))
            out_json = tmp_dir / "result.json"
            if out_image is None:
                out_image = tmp_dir / "overlay.png"
            cmd = [
                sys.executable,
                str(RUNNER),
                "--image", str(image_path),
                "--checkpoint", str(checkpoint_path),
                "--out-json", str(out_json),
                "--out-image", str(out_image),
                "--confidence", str(confidence),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=int(req.get("timeout", 180)))
            if out_json.exists():
                payload = json.loads(out_json.read_text(encoding="utf-8"))
            else:
                payload = {"ok": False, "error": "detector runner did not write result JSON"}
            payload.setdefault("source", "live_rfdetr_http_service")
            payload["source"] = "live_rfdetr_http_service"
            payload["detector_runtime_mode"] = "live_rfdetr_http_service"
            payload["service_python_executable"] = sys.executable
            payload["python_executable"] = sys.executable
            payload["inference_runtime_ms"] = payload.get("inference_runtime_ms") or int((time.perf_counter() - started) * 1000)
            payload["returncode"] = proc.returncode
            payload["stdout_preview"] = (proc.stdout or "")[:1200]
            payload["stderr_preview"] = (proc.stderr or "")[:1200]
            if proc.returncode != 0 and payload.get("ok"):
                payload["ok"] = False
                payload["error"] = f"detector runner exited {proc.returncode}"
            _json_response(self, 200 if payload.get("ok") else 500, payload)
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "source": "live_rfdetr_http_service", "error": str(exc)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve RF-DETR detector inference over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RFDETRHandler)
    print(f"RF-DETR detector service listening on http://{args.host}:{args.port}")
    print(f"Python executable: {sys.executable}")
    print(f"RF-DETR import ok: {importlib.util.find_spec('rfdetr') is not None}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

