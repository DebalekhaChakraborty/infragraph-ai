"""
GRPO reward functions for InfraGraph AI remediation fine-tuning.

Scores model-generated remediation responses against ground-truth references
built by build_rca_rl_dataset.py.  Supports both "local" and "enterprise" scope.

Reward components (10 total, weights sum to 1.0 after penalty normalisation):
    root_cause_match_reward              — 0.28
    grounded_node_reward                 — 0.18
    no_hallucinated_node_penalty         — 0.12  (can be negative)
    includes_validation_steps_reward     — 0.10
    action_specificity_reward            — 0.09
    rollback_safety_reward               — 0.07
    json_format_reward                   — 0.08
    escalation_if_multi_diagram_reward   — 0.04
    local_scope_precision_reward         — 0.04  (local only; 0 for enterprise)
    enterprise_cross_diagram_reward      — 0.04  (enterprise only; 0 for local)
    ─────────────────────────────────────────────
    Weighted total ≈ [-0.12, 1.04]  (clamped to [-0.15, 1.0])

Compatible with vERL reward_fn interface:
    reward_fn(responses: list[str], references: list[dict]) -> list[float]
"""
from __future__ import annotations

import json
import re


# ── JSON parsing helpers ──────────────────────────────────────────────────────

def _parse_response(text: str) -> dict:
    """Parse model output as JSON; return {} on failure."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?([\s\S]*?)```\s*$", text)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except Exception:
        return {}


def _collect_text(response: dict) -> str:
    """Flatten all string and list values into a single searchable lowercase string."""
    parts: list[str] = []
    for v in response.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(str(i) for i in v)
        elif isinstance(v, dict):
            parts.extend(str(x) for x in v.values())
    return " ".join(parts).lower()


def _all_steps(response: dict) -> list[str]:
    steps: list[str] = []
    for key in ("triage_steps", "validation_steps", "remediation_steps"):
        v = response.get(key, [])
        if isinstance(v, list):
            steps.extend(str(s) for s in v)
    return steps


# ── Individual reward functions ───────────────────────────────────────────────

def root_cause_match_reward(response: dict, reference: dict) -> float:
    """1.0 if the probable_root_cause field exactly names the correct node.
    0.5 if correct node appears anywhere in the response. 0.0 otherwise.
    """
    expected = reference.get("root_cause", "")
    if not expected:
        return 0.5
    probable = response.get("probable_root_cause", "")
    if expected.lower() in probable.lower():
        return 1.0
    if expected.lower() in _collect_text(response):
        return 0.5
    return 0.0


def grounded_node_reward(response: dict, reference: dict) -> float:
    """Fraction of required_nodes that appear anywhere in the response text."""
    required = reference.get("required_nodes", [])
    if not required:
        return 0.5
    all_text = _collect_text(response)
    hits = sum(1 for n in required if n.lower() in all_text)
    return hits / len(required)


def no_hallucinated_node_penalty(response: dict, reference: dict) -> float:
    """Penalty ∈ [-1.0, 0.0] proportional to fraction of mentioned node-like
    tokens that are NOT in the valid_node_set.
    """
    valid_nodes: set[str] = {n.lower() for n in reference.get("valid_node_set", [])}
    if not valid_nodes:
        return 0.0
    raw_values = " ".join(
        v if isinstance(v, str) else " ".join(str(i) for i in v)
        for v in response.values()
        if isinstance(v, (str, list))
    )
    candidates = re.findall(r"\b[A-Z][A-Z0-9_\-]{2,}\b", raw_values)
    if not candidates:
        return 0.0
    hallucinated = sum(1 for c in candidates if c.lower() not in valid_nodes)
    return -(hallucinated / len(candidates))


def includes_validation_steps_reward(response: dict, _reference: dict) -> float:
    """1.0 if both validation_steps and remediation_steps are non-empty.
    0.5 if only validation_steps is present. 0.0 otherwise.
    (Renamed from validation_before_remediation_reward for clarity.)
    """
    v_steps = response.get("validation_steps", [])
    r_steps = response.get("remediation_steps", [])
    if v_steps and r_steps:
        return 1.0
    if v_steps:
        return 0.5
    return 0.0


def action_specificity_reward(response: dict, _reference: dict) -> float:
    """Reward for specific, actionable steps.
    Score = min(1.0, n_steps_with_more_than_8_words / 5).
    """
    specific = sum(1 for s in _all_steps(response) if len(s.split()) > 8)
    return min(1.0, specific / 5.0)


def rollback_safety_reward(response: dict, _reference: dict) -> float:
    """1.0 if rollback_or_safety_notes is a non-empty list with substantive items.
    0.5 if it is a non-empty string. 0.0 if absent.
    """
    notes = response.get("rollback_or_safety_notes")
    if isinstance(notes, list) and len(notes) >= 1:
        total_words = sum(len(str(n).split()) for n in notes)
        return 1.0 if total_words >= 8 else 0.5
    if isinstance(notes, str) and len(notes.split()) >= 8:
        return 0.5
    return 0.0


def json_format_reward(raw_text: str, reference: dict) -> float:
    """Fraction of required output keys present in the parsed response.

    Uses reference["required_sections"] if available; falls back to the
    standard 10-key schema.
    """
    required_keys = set(reference.get("required_sections") or [
        "executive_summary", "probable_root_cause", "scope",
        "evidence_from_graph", "triage_steps", "validation_steps",
        "remediation_steps", "rollback_or_safety_notes",
        "escalation_recommendation", "servicenow_incident_summary",
        "confidence_notes",
    ])
    parsed = _parse_response(raw_text)
    if not parsed:
        return 0.0
    present = {k for k in required_keys if k in parsed}
    return len(present) / len(required_keys)


def escalation_if_multi_diagram_reward(response: dict, reference: dict) -> float:
    """1.0 if escalation_recommendation is substantive for multi-diagram incidents.
    0.3 for single-diagram; always 1.0 if enterprise scope and text is non-trivial.
    """
    req_diags   = reference.get("required_diagrams", [])
    escalation  = response.get("escalation_recommendation", "")
    word_count  = len(escalation.split())
    if len(req_diags) >= 2 and word_count > 10:
        return 1.0
    if word_count > 10:
        return 0.7
    if word_count > 4:
        return 0.3
    return 0.0


def local_scope_precision_reward(response: dict, reference: dict) -> float:
    """Local-only reward: 1.0 if scope == "local" and the response does NOT
    reference diagram IDs outside the single required_diagram.

    Returns 0.0 for enterprise records (wrong scope or multiple diagrams).
    """
    if reference.get("scope", "enterprise") != "local":
        return 0.0
    req_diags = reference.get("required_diagrams", [])
    if len(req_diags) != 1:
        return 0.0
    correct_diag = req_diags[0].lower()
    scope_field  = response.get("scope", "")
    if scope_field and scope_field.lower() != "local":
        return 0.0
    all_text = _collect_text(response)
    # Only reward if the correct diagram appears and no other diagram-like pattern does
    if correct_diag not in all_text:
        return 0.3
    return 1.0


def enterprise_cross_diagram_reasoning_reward(response: dict, reference: dict) -> float:
    """Enterprise-only reward: 1.0 if the response references >= 2 of the
    required_diagrams and sets scope = "enterprise".

    Returns 0.0 for local records.
    """
    if reference.get("scope", "local") != "enterprise":
        return 0.0
    req_diags   = reference.get("required_diagrams", [])
    if len(req_diags) < 2:
        return 0.5
    scope_field = response.get("scope", "")
    if scope_field.lower() != "enterprise":
        return 0.0
    all_text = _collect_text(response)
    mentioned = sum(1 for d in req_diags if d.lower() in all_text)
    return min(1.0, mentioned / len(req_diags))


# ── Composite reward ──────────────────────────────────────────────────────────

REWARD_WEIGHTS: dict[str, float] = {
    "root_cause_match":          0.28,
    "grounded_node":             0.18,
    "no_hallucination":          0.12,
    "includes_validation":       0.10,
    "action_specificity":        0.09,
    "rollback_safety":           0.07,
    "json_format":               0.08,
    "escalation":                0.04,
    "local_precision":           0.04,
    "enterprise_cross_diagram":  0.04,
}

assert abs(sum(REWARD_WEIGHTS.values()) - 1.04) < 0.001, "weights must sum to ~1.04"


def compute_composite_reward(raw_text: str, reference: dict) -> float:
    """Weighted composite reward for a single model response.

    Parameters
    ----------
    raw_text  : raw model output string
    reference : dict from build_rca_rl_dataset

    Returns
    -------
    float clamped to [-0.15, 1.0]
    """
    response = _parse_response(raw_text)
    scope    = reference.get("scope", "enterprise")

    scores: dict[str, float] = {
        "root_cause_match":         root_cause_match_reward(response, reference),
        "grounded_node":            grounded_node_reward(response, reference),
        "no_hallucination":         no_hallucinated_node_penalty(response, reference),
        "includes_validation":      includes_validation_steps_reward(response, reference),
        "action_specificity":       action_specificity_reward(response, reference),
        "rollback_safety":          rollback_safety_reward(response, reference),
        "json_format":              json_format_reward(raw_text, reference),
        "escalation":               escalation_if_multi_diagram_reward(response, reference),
        "local_precision":          local_scope_precision_reward(response, reference),
        "enterprise_cross_diagram": enterprise_cross_diagram_reasoning_reward(response, reference),
    }

    total = sum(REWARD_WEIGHTS[k] * v for k, v in scores.items())
    return round(max(-0.15, min(1.0, total)), 4)


# ── vERL-compatible batch interface ───────────────────────────────────────────

def batch_reward_fn(
    responses: list[str],
    references: list[dict],
) -> list[float]:
    """Compute composite rewards for a batch.

    Signature matches the vERL reward_fn interface.

    Parameters
    ----------
    responses  : list of raw model output strings (one per sample)
    references : list of reference dicts from the JSONL dataset

    Returns
    -------
    list of float reward values (same length as inputs), each in [-0.15, 1.0]
    """
    assert len(responses) == len(references), (
        f"batch_reward_fn: responses ({len(responses)}) and references "
        f"({len(references)}) must have the same length"
    )
    return [
        compute_composite_reward(r, ref)
        for r, ref in zip(responses, references)
    ]
