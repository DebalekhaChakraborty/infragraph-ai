"""
verl_reward.py — vERL custom reward function for InfraGraph RCA remediation.

Entry point called by vERL during rollout scoring:

    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

ground_truth is the reference record from prepare_verl_dataset.py.
It may arrive as a dict (parquet struct column) or as a JSON string
(legacy / fallback path).  _parse_json() handles both transparently.

Expected fields:
{
    "root_cause":        str,
    "impacted_nodes":    [str, ...],
    "impacted_diagrams": [str, ...],
    "graph_evidence":    [str, ...],
    "scope":             "local" | "enterprise",
    "reward_tags":       [str, ...]
}

Reward components are deterministic (no model call).
Returns a float in [0.0, 1.0].  Never raises — invalid model output returns a low
but non-zero reward so GRPO still gets a gradient signal.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ── Allow running standalone (mirrors reward_functions imports) ───────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json(text: "str | dict") -> dict:
    """Parse model output as JSON; return {} on any failure."""
    if isinstance(text, dict):
        return text
    s = str(text).strip()
    m = re.match(r"^```(?:json)?\s*\n?([\s\S]*?)```\s*$", s)
    if m:
        s = m.group(1).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        s = s[start : end + 1]
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def _flat_text(candidate: "str | dict") -> str:
    data = _parse_json(candidate)
    parts = [str(candidate) if isinstance(candidate, str) else ""]
    for v in data.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(str(i) for i in v)
        elif isinstance(v, dict):
            parts.extend(str(i) for i in v.values())
    return " ".join(parts)


# ── Individual reward components ──────────────────────────────────────────────

def _json_format(candidate: "str | dict") -> float:
    data = _parse_json(candidate)
    required = {
        "executive_summary", "probable_root_cause", "scope", "risk_level",
        "automation_eligibility", "blast_radius", "validation_steps",
        "remediation_steps", "rollback_or_safety_notes", "servicenow_incident_summary",
    }
    if not data:
        return 0.0
    present = required & set(data)
    return len(present) / len(required)


def _root_cause_match(candidate: "str | dict", ref: dict) -> float:
    root = str(ref.get("root_cause", ""))
    if not root:
        return 0.5
    data  = _parse_json(candidate)
    text  = _flat_text(candidate).lower()
    probe = str(data.get("probable_root_cause", "")).lower()
    if root.lower() in probe:
        return 1.0
    if root.lower() in text:
        return 0.5
    return 0.0


def _grounded_node(candidate: "str | dict", ref: dict) -> float:
    nodes = {str(n) for n in ref.get("impacted_nodes", []) if str(n)}
    root  = str(ref.get("root_cause", ""))
    if root:
        nodes.add(root)
    if not nodes:
        return 0.5
    text = _flat_text(candidate)
    hits = sum(1 for n in nodes if n in text)
    return hits / len(nodes)


def _no_hallucinated_device(candidate: "str | dict", ref: dict) -> float:
    valid = {str(n).upper() for n in ref.get("impacted_nodes", []) if str(n)}
    root  = str(ref.get("root_cause", "")).upper()
    if root:
        valid.add(root)
    text       = _flat_text(candidate)
    mentioned  = set(re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+){1,4}\b", text))
    hallucinated = [n for n in mentioned if n not in valid]
    if not mentioned:
        return 0.6
    if hallucinated:
        return max(0.0, 1.0 - len(hallucinated) / len(mentioned))
    return 1.0


def _validation_before_remediation(candidate: "str | dict") -> float:
    data = _parse_json(candidate)
    validation  = data.get("validation_steps") or []
    remediation = data.get("remediation_steps") or []
    if not validation or not remediation:
        return 0.0
    raw = str(candidate).lower()
    v_pos = raw.find("validation")
    r_pos = raw.find("remediation")
    if v_pos != -1 and r_pos != -1 and v_pos < r_pos:
        return 1.0
    return 0.7


def _rollback_safety(candidate: "str | dict") -> float:
    data    = _parse_json(candidate)
    notes   = data.get("rollback_or_safety_notes") or []
    blockers = data.get("do_not_execute_if") or []
    total   = (
        len(notes    if isinstance(notes,    list) else [notes])
        + len(blockers if isinstance(blockers, list) else [blockers])
    )
    if total >= 3:
        return 1.0
    if total >= 1:
        return 0.5
    return 0.0


def _enterprise_escalation(candidate: "str | dict", ref: dict) -> float:
    diagrams = set(ref.get("impacted_diagrams", []) or [])
    text     = _flat_text(candidate).lower()
    if len(diagrams) <= 1:
        return 1.0
    if "escalat" in text and any(k in text for k in ("enterprise", "noc", "sre", "network")):
        return 1.0
    return 0.0


def _servicenow_summary(candidate: "str | dict") -> float:
    data = _parse_json(candidate)
    snow = data.get("servicenow_incident_summary")
    if not isinstance(snow, dict):
        return 0.0
    required = {"short_description", "description", "affected_ci", "priority", "assignment_group"}
    present  = {k for k in required if snow.get(k)}
    return len(present) / len(required)


# ── Weights ───────────────────────────────────────────────────────────────────

_WEIGHTS = {
    "json_format":                   0.16,
    "root_cause_match":              0.18,
    "grounded_node":                 0.14,
    "no_hallucinated_device":        0.14,
    "validation_before_remediation": 0.12,
    "rollback_safety":               0.12,
    "enterprise_escalation":         0.08,
    "servicenow_summary":            0.06,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9


# ── Public entry point (called by vERL) ───────────────────────────────────────

def compute_score(
    data_source: str,
    solution_str: "str | dict",
    ground_truth: "str | dict",
    extra_info: "dict | None" = None,
) -> float:
    """
    Score a model response for the vERL reward_fn interface.

    Parameters
    ----------
    data_source   : always "infragraph_rca_remediation" for this dataset
    solution_str  : raw model output string (or already-parsed dict)
    ground_truth  : JSON string (or dict) from prepare_verl_dataset.py
    extra_info    : optional dict; not used for scoring

    Returns
    -------
    float in [0.0, 1.0]
    """
    try:
        ref = _parse_json(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth or {})

        scores = {
            "json_format":                   _json_format(solution_str),
            "root_cause_match":              _root_cause_match(solution_str, ref),
            "grounded_node":                 _grounded_node(solution_str, ref),
            "no_hallucinated_device":        _no_hallucinated_device(solution_str, ref),
            "validation_before_remediation": _validation_before_remediation(solution_str),
            "rollback_safety":               _rollback_safety(solution_str),
            "enterprise_escalation":         _enterprise_escalation(solution_str, ref),
            "servicenow_summary":            _servicenow_summary(solution_str),
        }
        total = sum(_WEIGHTS[k] * v for k, v in scores.items())
        return round(max(0.0, min(1.0, total)), 4)

    except Exception:
        # Never crash the training loop — return a low but non-zero reward
        return 0.05


# ── Standalone smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    _GOOD = json.dumps({
        "executive_summary": "Root cause APP-LB-01 confirmed via graph evidence.",
        "probable_root_cause": "APP-LB-01",
        "scope": "enterprise",
        "risk_level": "high",
        "automation_eligibility": "manual_only",
        "blast_radius": "cross_diagram",
        "evidence_from_graph": ["E001: APP-LB-01 is the reference root cause."],
        "validation_steps": [
            "Validate APP-LB-01 reachability before any change.",
            "Confirm alert timestamps are fresh.",
        ],
        "remediation_steps": [
            "After validation, restore service on APP-LB-01.",
            "Verify downstream nodes APP-01 and DB-01 recover.",
        ],
        "rollback_or_safety_notes": [
            "Capture running config before change.",
            "Restore prior state if post-checks fail.",
        ],
        "do_not_execute_if": [
            "Do not execute if APP-LB-01 not confirmed by graph evidence.",
        ],
        "escalation_recommendation": "Escalate to enterprise NOC for cross-diagram impact.",
        "servicenow_incident_summary": {
            "short_description": "InfraGraph RCA root cause APP-LB-01",
            "description": "Graph-grounded RCA found APP-LB-01.",
            "affected_ci": "APP-LB-01",
            "priority": "1-Critical",
            "assignment_group": "Network Engineering",
        },
        "confidence_notes": "Scenario-grounded; graph evidence cited.",
    })

    _BAD = '{"probable_root_cause": "FAKE-RTR-99", "remediation_steps": ["restart everything"]}'

    _GT = json.dumps({
        "root_cause": "APP-LB-01",
        "impacted_nodes": ["APP-01", "APP-02", "DB-01"],
        "impacted_diagrams": ["branch_topology", "wan_topology", "app_db_topology"],
        "graph_evidence": ["E001: APP-LB-01 is the reference root cause."],
        "scope": "enterprise",
        "reward_tags": ["graph_grounded"],
    })

    good_score = compute_score("infragraph_rca_remediation", _GOOD, _GT)
    bad_score  = compute_score("infragraph_rca_remediation", _BAD,  _GT)
    crash_score = compute_score("infragraph_rca_remediation", "not json at all", "also not json")

    print(f"Good response score : {good_score}")
    print(f"Bad response score  : {bad_score}")
    print(f"Malformed input score: {crash_score}")
    assert good_score > bad_score,       "good should outscore bad"
    assert good_score >= 0.8,            "good response should score >= 0.8"
    assert bad_score  < 0.2,             "bad response should score < 0.2"
    assert 0.0 <= crash_score <= 1.0,    "score must be in [0, 1] even for malformed input"
    print("All assertions passed.")
