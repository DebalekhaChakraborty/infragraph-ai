"""Streamlit-side bridge for external RF-DETR subprocess inference."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


CHECKPOINT_CANDIDATES = [
    "model_artifacts/rfdetr/checkpoint_best_total.pth",
    "model_artifacts/rfdetr/checkpoint_best_regular.pth",
    "model_artifacts/rfdetr_v3/model/checkpoint_best_total.pth",
    "model_artifacts/rfdetr_v3/model/checkpoint_best_regular.pth",
    "model_artifacts/rfdetr_v3/model/checkpoint_best_ema.pth",
    "outputs/rfdetr/checkpoint_best_total.pth",
    "outputs/rfdetr/checkpoint_best_regular.pth",
    "outputs/rfdetr_v3/model/checkpoint_best_total.pth",
    "outputs/rfdetr_v3/model/checkpoint_best_regular.pth",
    "outputs/rfdetr_v3/model/checkpoint_best_ema.pth",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_rfdetr_python() -> str:
    env_value = os.environ.get("INFRAGRAPH_RFDETR_PYTHON", "").strip()
    if env_value:
        return env_value
    for candidate in [
        "/opt/conda/bin/python",
        "/usr/bin/python",
        "/workspace/shared/venvs/rfdetr/bin/python",
    ]:
        if Path(candidate).exists():
            return candidate
    return sys.executable


def find_best_rfdetr_checkpoint(repo_root: Path) -> Path | None:
    env_value = os.environ.get("INFRAGRAPH_RFDETR_CHECKPOINT", "").strip()
    if env_value:
        path = Path(env_value)
        if not path.is_absolute():
            path = repo_root / path
        return path if path.exists() else path

    for rel in CHECKPOINT_CANDIDATES:
        path = repo_root / rel
        if path.exists():
            return path
    for root in [
        repo_root / "model_artifacts",
        repo_root / "outputs",
    ]:
        if root.exists():
            matches = sorted(root.rglob("checkpoint_best_total.pth"))
            if matches:
                return matches[0]
            matches = sorted(root.rglob("checkpoint_best_regular.pth"))
            if matches:
                return matches[0]
            matches = sorted(root.rglob("*.pth"))
            if matches:
                return matches[0]
    return None


def check_rfdetr_runtime(python_executable: str) -> dict:
    try:
        proc = subprocess.run(
            [
                python_executable,
                "-c",
                "import sys, importlib.util; ok=importlib.util.find_spec('rfdetr') is not None; print(sys.executable); print('OK' if ok else 'MISSING')",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        return {
            "ok": proc.returncode == 0 and any(line == "OK" for line in lines),
            "python_executable": lines[0] if lines else python_executable,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[:1000],
            "stderr_preview": (proc.stderr or "")[:1000],
            "error": "" if proc.returncode == 0 else (proc.stderr or "runtime check failed")[:1000],
        }
    except Exception as exc:
        return {
            "ok": False,
            "python_executable": python_executable,
            "error": str(exc),
            "stdout": "",
            "stderr_preview": "",
        }


def run_rfdetr_subprocess(
    image_path: Path,
    checkpoint_path: Path | None,
    confidence: float,
    timeout: int,
) -> dict:
    repo_root = _repo_root()
    python_executable = resolve_rfdetr_python()
    if checkpoint_path is None:
        checkpoint_path = find_best_rfdetr_checkpoint(repo_root)

    if checkpoint_path is None:
        return {
            "ok": False,
            "source": "live_rfdetr_subprocess",
            "error": "RF-DETR checkpoint not found",
            "python_executable": python_executable,
            "checkpoint_path": "",
            "image_path": str(image_path),
        }

    tmp_dir = Path(tempfile.mkdtemp(prefix="infragraph_rfdetr_"))
    out_json = tmp_dir / "infragraph_rfdetr_result.json"
    out_image = tmp_dir / "infragraph_rfdetr_overlay.png"
    runner = repo_root / "scripts" / "run_rfdetr_inference.py"
    cmd = [
        python_executable,
        str(runner),
        "--image", str(image_path),
        "--checkpoint", str(checkpoint_path),
        "--out-json", str(out_json),
        "--out-image", str(out_image),
        "--confidence", str(confidence),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(repo_root))
        if out_json.exists():
            payload = json.loads(out_json.read_text(encoding="utf-8"))
        else:
            payload = {
                "ok": False,
                "source": "live_rfdetr_subprocess",
                "error": "RF-DETR subprocess did not write result JSON",
            }
        payload.setdefault("python_executable", python_executable)
        payload.setdefault("checkpoint_path", str(checkpoint_path))
        payload.setdefault("image_path", str(image_path))
        payload["returncode"] = proc.returncode
        payload["stdout_preview"] = (proc.stdout or "")[:1200]
        payload["stderr_preview"] = (proc.stderr or "")[:1200]
        if proc.returncode != 0 and payload.get("ok"):
            payload["ok"] = False
            payload["error"] = payload.get("error") or f"RF-DETR subprocess exited {proc.returncode}"
        return payload
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "source": "live_rfdetr_subprocess",
            "error": f"RF-DETR subprocess timeout after {timeout}s",
            "python_executable": python_executable,
            "checkpoint_path": str(checkpoint_path),
            "image_path": str(image_path),
            "stdout_preview": (exc.stdout or "")[:1200] if isinstance(exc.stdout, str) else "",
            "stderr_preview": (exc.stderr or "")[:1200] if isinstance(exc.stderr, str) else "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "source": "live_rfdetr_subprocess",
            "error": str(exc),
            "python_executable": python_executable,
            "checkpoint_path": str(checkpoint_path),
            "image_path": str(image_path),
        }
