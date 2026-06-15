"""Streamlit-side bridge for external RF-DETR inference.

Streamlit never imports RF-DETR here. It either calls a detector HTTP service
or launches a detector Python subprocess.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


CHECKPOINT_CANDIDATES = [
    # Canonical: flat layout (model_artifacts/rfdetr_v3/<file>)
    "model_artifacts/rfdetr_v3/checkpoint_best_total.pth",
    "model_artifacts/rfdetr_v3/checkpoint_best_ema.pth",
    "model_artifacts/rfdetr_v3/checkpoint_best_regular.pth",
    # Canonical: model/ subdirectory layout
    "model_artifacts/rfdetr_v3/model/checkpoint_best_total.pth",
    "model_artifacts/rfdetr_v3/model/checkpoint_best_ema.pth",
    "model_artifacts/rfdetr_v3/model/checkpoint_best_regular.pth",
    # Legacy rfdetr (v1/v2)
    "model_artifacts/rfdetr/checkpoint_best_total.pth",
    "model_artifacts/rfdetr/checkpoint_best_regular.pth",
    # outputs/ mirrors
    "outputs/rfdetr_v3/checkpoint_best_total.pth",
    "outputs/rfdetr_v3/checkpoint_best_ema.pth",
    "outputs/rfdetr_v3/checkpoint_best_regular.pth",
    "outputs/rfdetr_v3/model/checkpoint_best_total.pth",
    "outputs/rfdetr_v3/model/checkpoint_best_ema.pth",
    "outputs/rfdetr_v3/model/checkpoint_best_regular.pth",
    "outputs/rfdetr/checkpoint_best_total.pth",
    "outputs/rfdetr/checkpoint_best_regular.pth",
]

# Allowed checkpoint filenames — anything else is not an RF-DETR detector checkpoint
_RFDETR_ALLOWED_NAMES: frozenset[str] = frozenset({
    "checkpoint_best_total.pth",
    "checkpoint_best_regular.pth",
    "checkpoint_best_ema.pth",
    "last.ckpt",
})

# Substrings that mark a path as definitively NOT an RF-DETR checkpoint
_RFDETR_FORBIDDEN_PATH_SUBSTRINGS: tuple[str, ...] = (
    "qwen",
    "qwen_lora",
    "checkpoint-",
    "rng_state",
    "optimizer",
    "scheduler",
    "trainer_state",
    "adapter_model",
)


def is_valid_rfdetr_checkpoint_path(path: Path) -> bool:
    """Return True only if path looks like an RF-DETR detector checkpoint.

    Rejects Qwen LoRA artifacts, optimizer states, RNG states, and any file
    whose name is not on the known-good RF-DETR checkpoint allowlist.
    """
    p    = str(path).lower()
    name = path.name.lower()

    if name not in _RFDETR_ALLOWED_NAMES:
        return False

    if "rfdetr" not in p and "rf-detr" not in p:
        return False

    return not any(x in p for x in _RFDETR_FORBIDDEN_PATH_SUBSTRINGS)

COMMON_PYTHON_CANDIDATES = [
    "/opt/conda/bin/python",
    "/usr/bin/python",
    "/workspace/shared/venvs/rfdetr/bin/python",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _probe_import_cmd() -> str:
    return (
        "import sys, importlib.util; "
        "print(sys.executable); "
        "print(importlib.util.find_spec('rfdetr') is not None)"
    )


def check_rfdetr_runtime(python_executable: str) -> dict:
    try:
        proc = subprocess.run(
            [python_executable, "-c", _probe_import_cmd()],
            capture_output=True,
            text=True,
            timeout=20,
        )
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        import_ok = len(lines) >= 2 and lines[1].lower() == "true"
        return {
            "ok": proc.returncode == 0 and import_ok,
            "import_ok": proc.returncode == 0 and import_ok,
            "requested_python": python_executable,
            "python_executable": lines[0] if lines else python_executable,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[:1000],
            "stderr_preview": (proc.stderr or "")[:1000],
            "error": "" if proc.returncode == 0 else (proc.stderr or "runtime check failed")[:1000],
        }
    except Exception as exc:
        return {
            "ok": False,
            "import_ok": False,
            "requested_python": python_executable,
            "python_executable": python_executable,
            "error": str(exc),
            "stdout": "",
            "stderr_preview": "",
        }


def resolve_rfdetr_python_details() -> dict:
    env_value = os.environ.get("INFRAGRAPH_RFDETR_PYTHON", "").strip()
    use_path_python = os.environ.get("INFRAGRAPH_RFDETR_USE_PATH_PYTHON", "").strip() == "1"

    if env_value:
        return {
            "python_resolution_mode": "env",
            "python_executable": env_value,
            "requested_detector_python": env_value,
            "resolved_detector_python": env_value,
            "resolved_from_env": True,
            "resolved_from_path": False,
            "streamlit_python": sys.executable,
            "fallback_reason": "",
            "runtime": check_rfdetr_runtime(env_value),
        }

    if use_path_python:
        return {
            "python_resolution_mode": "path_python",
            "python_executable": "python",
            "requested_detector_python": "python",
            "resolved_detector_python": "python",
            "resolved_from_env": False,
            "resolved_from_path": True,
            "streamlit_python": sys.executable,
            "fallback_reason": "INFRAGRAPH_RFDETR_USE_PATH_PYTHON=1",
            "runtime": check_rfdetr_runtime("python"),
        }

    for candidate in COMMON_PYTHON_CANDIDATES:
        if Path(candidate).exists():
            return {
                "python_resolution_mode": "common_candidate",
                "python_executable": candidate,
                "requested_detector_python": "common detector candidates",
                "resolved_detector_python": candidate,
                "resolved_from_env": False,
                "resolved_from_path": False,
                "streamlit_python": sys.executable,
                "fallback_reason": "INFRAGRAPH_RFDETR_PYTHON unset and INFRAGRAPH_RFDETR_USE_PATH_PYTHON is not 1",
                "runtime": check_rfdetr_runtime(candidate),
            }

    return {
        "python_resolution_mode": "streamlit_python",
        "python_executable": sys.executable,
        "requested_detector_python": "streamlit python",
        "resolved_detector_python": sys.executable,
        "resolved_from_env": False,
        "resolved_from_path": False,
        "streamlit_python": sys.executable,
        "fallback_reason": "No configured or common detector Python found; using Streamlit Python",
        "runtime": check_rfdetr_runtime(sys.executable),
    }


def resolve_rfdetr_python() -> str:
    return str(resolve_rfdetr_python_details().get("python_executable") or sys.executable)


_NO_CHECKPOINT_MSG = (
    "No valid RF-DETR checkpoint found. "
    "Set INFRAGRAPH_RFDETR_CHECKPOINT to checkpoint_best_total.pth "
    "under model_artifacts/rfdetr_v3/model or outputs/rfdetr_v3/model."
)

# Known rfdetr-specific subdirectories for the fallback restricted search.
# Flat layout (rfdetr_v3/) is listed before model/ subdirectory layout.
_RFDETR_FALLBACK_SEARCH_DIRS: tuple[str, ...] = (
    "model_artifacts/rfdetr_v3",
    "model_artifacts/rfdetr_v3/model",
    "model_artifacts/rfdetr_v3_smoke",
    "model_artifacts/rfdetr_v3_smoke/model",
    "model_artifacts/rfdetr",
    "outputs/rfdetr_v3",
    "outputs/rfdetr_v3/model",
    "outputs/rfdetr_v3_smoke",
    "outputs/rfdetr_v3_smoke/model",
    "outputs/rfdetr",
)


def find_rfdetr_checkpoint_with_reason(repo_root: Path) -> "tuple[Path | None, str]":
    """Return (checkpoint_path, rejection_reason).

    rejection_reason is "" on success and a human-readable message on failure.
    Never returns a Qwen LoRA file, optimizer state, RNG file, or any path
    that fails is_valid_rfdetr_checkpoint_path().
    """
    env_value = os.environ.get("INFRAGRAPH_RFDETR_CHECKPOINT", "").strip()
    if env_value:
        path = Path(env_value)
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            return None, f"INFRAGRAPH_RFDETR_CHECKPOINT path does not exist: {path}"
        if not is_valid_rfdetr_checkpoint_path(path):
            return None, (
                f"INFRAGRAPH_RFDETR_CHECKPOINT rejected — not a valid RF-DETR checkpoint: {path}. "
                "Allowed names: checkpoint_best_total.pth, checkpoint_best_regular.pth, "
                "checkpoint_best_ema.pth, last.ckpt. "
                "Path must contain 'rfdetr' and must not contain qwen/rng_state/optimizer/scheduler."
            )
        return path, ""

    for rel in CHECKPOINT_CANDIDATES:
        path = repo_root / rel
        if path.exists() and is_valid_rfdetr_checkpoint_path(path):
            return path, ""

    # Restricted fallback: only search inside known rfdetr-specific directories
    for rel_dir in _RFDETR_FALLBACK_SEARCH_DIRS:
        search_dir = repo_root / rel_dir
        if not search_dir.exists():
            continue
        # Prefer in allowed-name priority order, not filesystem order
        for allowed_name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                             "checkpoint_best_regular.pth", "last.ckpt"):
            p = search_dir / allowed_name
            if p.exists() and is_valid_rfdetr_checkpoint_path(p):
                return p, ""

    return None, _NO_CHECKPOINT_MSG


def find_best_rfdetr_checkpoint(repo_root: Path) -> "Path | None":
    """Return the best RF-DETR checkpoint, or None if none found or all invalid."""
    path, _ = find_rfdetr_checkpoint_with_reason(repo_root)
    return path


def rfdetr_service_base_url() -> str:
    return os.environ.get("INFRAGRAPH_RFDETR_BASE_URL", "").strip().rstrip("/")


def check_rfdetr_http_service(base_url: str | None = None, timeout: int = 5) -> dict:
    url = (base_url or rfdetr_service_base_url()).rstrip("/")
    if not url:
        return {"ok": False, "source": "live_rfdetr_http_service", "error": "INFRAGRAPH_RFDETR_BASE_URL is not set"}
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        payload.setdefault("ok", True)
        payload.setdefault("source", "live_rfdetr_http_service")
        payload["service_url"] = url
        return payload
    except Exception as exc:
        return {
            "ok": False,
            "source": "live_rfdetr_http_service",
            "service_url": url,
            "error": str(exc),
        }


def run_rfdetr_http_service(
    image_path: Path,
    checkpoint_path: Path | None,
    confidence: float,
    timeout: int,
    base_url: str | None = None,
) -> dict:
    url = (base_url or rfdetr_service_base_url()).rstrip("/")
    if not url:
        return {"ok": False, "source": "live_rfdetr_http_service", "error": "INFRAGRAPH_RFDETR_BASE_URL is not set"}

    repo_root = _repo_root()
    if checkpoint_path is None:
        checkpoint_path = find_best_rfdetr_checkpoint(repo_root)
    if checkpoint_path is None:
        return {
            "ok": False,
            "source": "live_rfdetr_http_service",
            "service_url": url,
            "error": "RF-DETR checkpoint not found",
            "image_path": str(image_path),
            "checkpoint_path": "",
        }

    tmp_dir = Path(tempfile.mkdtemp(prefix="infragraph_rfdetr_http_"))
    out_image = tmp_dir / "infragraph_rfdetr_overlay.png"
    payload = {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "confidence": confidence,
        "out_image": str(out_image),
    }
    req = urllib.request.Request(
        f"{url}/detect",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        result.setdefault("source", "live_rfdetr_http_service")
        result.setdefault("service_url", url)
        result.setdefault("checkpoint_path", str(checkpoint_path))
        result.setdefault("image_path", str(image_path))
        result.setdefault("detector_runtime_mode", "live_rfdetr_http_service")
        return result
    except (TimeoutError, socket.timeout) as exc:
        return {
            "ok": False,
            "source": "live_rfdetr_http_service",
            "service_url": url,
            "detector_runtime_mode": "verified_annotation_fallback",
            "timed_out": True,
            "error": f"RF-DETR HTTP timeout after {timeout}s: {exc}",
            "fallback_reason": "Live RF-DETR timed out; using verified annotation fallback for demo continuity.",
            "checkpoint_path": str(checkpoint_path),
            "image_path": str(image_path),
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1200]
        return {
            "ok": False,
            "source": "live_rfdetr_http_service",
            "service_url": url,
            "error": f"HTTP {exc.code}: {body}",
            "checkpoint_path": str(checkpoint_path),
            "image_path": str(image_path),
        }
    except Exception as exc:
        timed_out = "timed out" in str(exc).lower() or "timeout" in str(exc).lower()
        return {
            "ok": False,
            "source": "live_rfdetr_http_service",
            "service_url": url,
            "detector_runtime_mode": "verified_annotation_fallback" if timed_out else "live_rfdetr_http_service",
            "timed_out": timed_out,
            "error": f"RF-DETR HTTP timeout after {timeout}s: {exc}" if timed_out else str(exc),
            "fallback_reason": (
                "Live RF-DETR timed out; using verified annotation fallback for demo continuity."
                if timed_out else str(exc)
            ),
            "checkpoint_path": str(checkpoint_path),
            "image_path": str(image_path),
        }


def run_rfdetr_subprocess(
    image_path: Path,
    checkpoint_path: Path | None,
    confidence: float,
    timeout: int,
) -> dict:
    repo_root = _repo_root()
    resolution = resolve_rfdetr_python_details()
    python_executable = str(resolution.get("python_executable") or sys.executable)
    if checkpoint_path is None:
        checkpoint_path = find_best_rfdetr_checkpoint(repo_root)

    if checkpoint_path is None:
        return {
            "ok": False,
            "source": "live_rfdetr_subprocess",
            "detector_runtime_mode": "live_rfdetr_subprocess",
            "error": "RF-DETR checkpoint not found",
            "python_executable": python_executable,
            "checkpoint_path": "",
            "image_path": str(image_path),
            "python_resolution": resolution,
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
        payload.setdefault("source", "live_rfdetr_subprocess")
        payload["detector_runtime_mode"] = "live_rfdetr_subprocess"
        payload["python_resolution"] = resolution
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
            "detector_runtime_mode": "live_rfdetr_subprocess",
            "error": f"RF-DETR subprocess timeout after {timeout}s",
            "python_executable": python_executable,
            "checkpoint_path": str(checkpoint_path),
            "image_path": str(image_path),
            "python_resolution": resolution,
            "stdout_preview": (exc.stdout or "")[:1200] if isinstance(exc.stdout, str) else "",
            "stderr_preview": (exc.stderr or "")[:1200] if isinstance(exc.stderr, str) else "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "source": "live_rfdetr_subprocess",
            "detector_runtime_mode": "live_rfdetr_subprocess",
            "error": str(exc),
            "python_executable": python_executable,
            "checkpoint_path": str(checkpoint_path),
            "image_path": str(image_path),
            "python_resolution": resolution,
        }


def run_rfdetr_detection(
    image_path: Path,
    checkpoint_path: Path | None,
    confidence: float,
    timeout: int,
) -> dict:
    base_url = rfdetr_service_base_url()
    if base_url:
        http_result = run_rfdetr_http_service(image_path, checkpoint_path, confidence, timeout, base_url=base_url)
        if http_result.get("ok"):
            return http_result
        if http_result.get("timed_out"):
            return http_result
        subprocess_result = run_rfdetr_subprocess(image_path, checkpoint_path, confidence, timeout)
        if subprocess_result.get("ok"):
            subprocess_result["fallback_reason"] = f"HTTP RF-DETR service unavailable: {http_result.get('error', 'unknown error')}"
            subprocess_result["http_attempt"] = http_result
            return subprocess_result
        subprocess_result["fallback_reason"] = (
            f"HTTP RF-DETR service unavailable: {http_result.get('error', 'unknown error')}; "
            f"subprocess unavailable: {subprocess_result.get('error', 'unknown error')}"
        )
        subprocess_result["http_attempt"] = http_result
        return subprocess_result
    return run_rfdetr_subprocess(image_path, checkpoint_path, confidence, timeout)
