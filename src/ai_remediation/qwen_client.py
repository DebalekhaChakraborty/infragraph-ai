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

import json
import re

from .prompt_builder import build_remediation_prompt
from .response_schema import empty_remediation_output


def _extract_json_from_text(text: str) -> dict:
    """Extract JSON object from model output.

    Handles ```json ... ``` fences and bare JSON.
    Raises json.JSONDecodeError on failure.
    """
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?([\s\S]*?)```\s*$", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def generate_remediation_with_qwen(
    context: dict,
    base_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-4B-Instruct",
    timeout: int = 60,
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

    messages = build_remediation_prompt(context)

    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": 0.2,
        "max_tokens":  1800,
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
        result["error"] = f"vLLM HTTP error: {exc}"
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        result["error"] = f"Could not parse model response as JSON: {exc}"
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"

    return result


def check_vllm_available(base_url: str, timeout: int = 4) -> bool:
    """Return True if the vLLM server responds at /models within timeout seconds."""
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
    base_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-4B-Instruct",
    timeout: int = 60,
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
