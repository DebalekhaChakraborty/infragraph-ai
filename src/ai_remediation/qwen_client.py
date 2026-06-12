"""
vLLM / OpenAI-compatible client for Qwen3 remediation generation.

Calls the locally-served model via POST /v1/chat/completions.
No cloud API — all inference runs on the local vLLM server.

Public API
----------
generate_remediation_with_qwen(context, base_url, model, timeout) -> dict
check_vllm_available(base_url, timeout) -> bool
generate_resolution_plan(context, scope, prefer_qwen, base_url, model, timeout) -> dict
"""
from __future__ import annotations

import ast
import json
import os
import re

from .prompt_builder import build_remediation_prompt
from .response_schema import empty_remediation_output


DEFAULT_QWEN_BASE_URL = "http://localhost:8000/v1"
DEFAULT_QWEN_MODEL = "infragraph"
DEFAULT_QWEN_TIMEOUT = 60


def get_qwen_runtime_config(
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int | str | None = None,
) -> dict:
    """Resolve Qwen/vLLM configuration with InfraGraph env vars preferred."""
    resolved_base_url = (
        base_url
        or os.environ.get("INFRAGRAPH_QWEN_BASE_URL")
        or os.environ.get("QWEN_BASE_URL")
        or DEFAULT_QWEN_BASE_URL
    )
    resolved_model = (
        model
        or os.environ.get("INFRAGRAPH_QWEN_MODEL")
        or os.environ.get("QWEN_MODEL")
        or DEFAULT_QWEN_MODEL
    )
    timeout_raw = (
        timeout
        or os.environ.get("INFRAGRAPH_QWEN_TIMEOUT")
        or os.environ.get("QWEN_TIMEOUT")
        or DEFAULT_QWEN_TIMEOUT
    )
    try:
        resolved_timeout = int(timeout_raw)
    except Exception:
        resolved_timeout = DEFAULT_QWEN_TIMEOUT
    return {
        "base_url": str(resolved_base_url).rstrip("/"),
        "model": str(resolved_model),
        "timeout": resolved_timeout,
    }


def _strip_thinking(text: str) -> str:
    """Remove Qwen-style <think>...</think> blocks before JSON extraction."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _balanced_json_slice(text: str) -> str:
    """Return the first balanced JSON-object substring from text."""
    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_str = False
    esc = False

    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    # Incomplete JSON: return from first brace; caller will try repair.
    return text[start:]


def _repair_common_json_issues(s: str) -> str:
    """Best-effort repair for demo stability; does not fabricate content."""
    s = s.strip()

    # Remove trailing commas before object/array endings.
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # If the model stopped early, close basic brackets/braces.
    opens_curly = s.count("{")
    closes_curly = s.count("}")
    opens_square = s.count("[")
    closes_square = s.count("]")

    if closes_square < opens_square:
        s += "]" * (opens_square - closes_square)
    if closes_curly < opens_curly:
        s += "}" * (opens_curly - closes_curly)

    return s


def _extract_json_from_text(text: str) -> dict:
    """Extract JSON object from model output.

    Handles Qwen <think> traces, fenced JSON, bare JSON, Python-style dicts,
    and small truncation/trailing-comma issues.
    """
    raw = _strip_thinking(text)

    candidates = []
    candidates.append(raw)
    candidates.append(_balanced_json_slice(raw))
    candidates.append(_repair_common_json_issues(_balanced_json_slice(raw)))

    last_error: Exception | None = None
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            return json.loads(cand)
        except Exception as exc:
            last_error = exc
        try:
            obj = ast.literal_eval(cand)
            if isinstance(obj, dict):
                return obj
        except Exception as exc:
            last_error = exc

    raise json.JSONDecodeError(str(last_error or "No JSON object found"), raw, 0)


def generate_remediation_with_qwen(
    context: dict,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> dict:
    """Call the vLLM endpoint to generate a remediation plan.

    Parameters
    ----------
    context  : remediation input dict from make_remediation_input()
    base_url : base URL of the running vLLM server
    model    : model identifier to pass in the request
    timeout  : HTTP request timeout in seconds

    Returns
    -------
    {
        "source":   "qwen_vllm",
        "model":    model,
        "ok":       True | False,
        "response": parsed JSON dict | {},
        "error":    "" | error message,
        "raw":      raw text content from the model,
    }
    """
    import requests  # optional at module level; available via requirements.txt

    config = get_qwen_runtime_config(base_url=base_url, model=model, timeout=timeout)
    base_url = config["base_url"]
    model = config["model"]
    timeout = config["timeout"]
    messages = build_remediation_prompt(context)

    # Keep generation bounded because vLLM may be served with --max-model-len=2048
    # during the hackathon. A large RCA prompt + 1800 output tokens can trigger
    # HTTP 400 context-length errors.
    max_tokens = int(os.environ.get("INFRAGRAPH_QWEN_MAX_TOKENS", "900"))

    # Qwen3 can emit <think> traces unless explicitly discouraged.
    messages = [
        {
            "role": "system",
            "content": "You are InfraGraph AI. Return only valid JSON. Do not include markdown, explanation, or thinking text."
        }
    ] + messages

    if messages and isinstance(messages[-1], dict):
        messages[-1]["content"] = str(messages[-1].get("content", "")) + "\n/no_think"

    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": float(os.environ.get("INFRAGRAPH_QWEN_TEMPERATURE", "0.0")),
        "max_tokens":  max_tokens,
    }

    result: dict = {
        "source":   "qwen_vllm",
        "model":    model,
        "ok":       False,
        "response": empty_remediation_output(context.get("scope", "enterprise")),
        "error":    "",
        "raw":      "",
    }

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data    = resp.json()
        content = data["choices"][0]["message"]["content"]
        result["raw"]      = content
        result["response"] = _extract_json_from_text(content)
        result["ok"]       = True

    except requests.exceptions.ConnectionError:
        result["error"] = (
            f"Cannot connect to vLLM at {base_url}. "
            "Ensure the model server is running."
        )
    except requests.exceptions.Timeout:
        result["error"] = (
            f"vLLM request timed out after {timeout}s. "
            "Try increasing INFRAGRAPH_QWEN_TIMEOUT."
        )
    except requests.exceptions.HTTPError as exc:
        body = ""
        try:
            body = resp.text[:1000]
        except Exception:
            body = ""
        result["error"] = f"vLLM HTTP error: {exc}. Response body: {body}"
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        result["error"] = f"Could not parse model response as JSON: {exc}"
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"

    return result


def check_vllm_available(base_url: str, timeout: int = 4) -> bool:
    """Return True if the vLLM server responds at /models within timeout seconds."""
    base_url = get_qwen_runtime_config(base_url=base_url)["base_url"]
    try:
        import requests
        r = requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def generate_resolution_plan(
    context: dict,
    scope: str = "enterprise",
    prefer_qwen: bool = True,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> dict:
    """Unified entry point for remediation plan generation.

    Injects ``scope`` into the context dict, then either calls the vLLM
    Qwen endpoint (if ``prefer_qwen`` and server is reachable) or falls
    back to deterministic template mode.

    Returns a result dict with the same structure as
    generate_remediation_with_qwen():
    {
        "source":   "qwen_vllm" | "template",
        "model":    str,
        "ok":       bool,
        "response": dict,
        "error":    str,
        "raw":      str,
    }
    """
    from .template_mode import generate_template_remediation  # avoid circular

    ctx = dict(context)
    ctx["scope"] = scope
    config = get_qwen_runtime_config(base_url=base_url, model=model, timeout=timeout)
    base_url = config["base_url"]
    model = config["model"]
    timeout = config["timeout"]

    if prefer_qwen and check_vllm_available(base_url, timeout=min(timeout, 5)):
        return generate_remediation_with_qwen(ctx, base_url=base_url, model=model, timeout=timeout)

    template_result = generate_template_remediation(ctx)
    return {
        "source":   "template",
        "model":    "—",
        "ok":       bool(template_result),
        "response": template_result,
        "error":    "",
        "raw":      "",
    }
