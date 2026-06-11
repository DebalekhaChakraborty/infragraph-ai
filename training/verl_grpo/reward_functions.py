"""Deterministic rewards for InfraGraph RCA remediation alignment records."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _parse(candidate: str | dict) -> tuple[dict, str]:
    if isinstance(candidate, dict):
        return candidate, json.dumps(candidate, sort_keys=True)
    text = str(candidate).strip()
    match = re.match(r"^```(?:json)?\s*\n?([\s\S]*?)```\s*$", text)
    if match:
        text = match.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}, text
    except Exception:
        return {}, text


def _flat_text(candidate: str | dict) -> str:
    data, raw = _parse(candidate)
    parts = [raw]
    for value in data.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif isinstance(value, dict):
            parts.extend(str(v) for v in value.values())
    return " ".join(parts)


def _score(value: float, reason: str) -> dict:
    return {"score": round(max(-1.0, min(1.0, value)), 4), "reason": reason}


def json_format_reward(candidate_response: str | dict, reference_record: dict) -> dict:
    data, _raw = _parse(candidate_response)
    required = {
        "executive_summary", "probable_root_cause", "scope", "risk_level",
        "automation_eligibility", "blast_radius", "validation_steps",
        "remediation_steps", "rollback_or_safety_notes", "servicenow_incident_summary",
    }
    if not data:
        return _score(0.0, "response is not valid JSON")
    present = required & set(data)
    return _score(len(present) / len(required), f"{len(present)}/{len(required)} required keys present")


def root_cause_match_reward(candidate_response: str | dict, reference_record: dict) -> dict:
    root = str(reference_record.get("root_cause", ""))
    if not root:
        return _score(0.5, "no reference root cause")
    data, _ = _parse(candidate_response)
    text = _flat_text(candidate_response).lower()
    probable = str(data.get("probable_root_cause", "")).lower()
    if root.lower() in probable:
        return _score(1.0, "probable_root_cause matches reference")
    if root.lower() in text:
        return _score(0.5, "root cause appears outside probable_root_cause")
    return _score(0.0, "root cause missing or incorrect")


def grounded_node_reward(candidate_response: str | dict, reference_record: dict) -> dict:
    nodes = {str(n) for n in reference_record.get("impacted_nodes", []) if str(n)}
    root = str(reference_record.get("root_cause", ""))
    if root:
        nodes.add(root)
    if not nodes:
        return _score(0.5, "no reference nodes")
    text = _flat_text(candidate_response)
    hits = [node for node in nodes if node in text]
    return _score(len(hits) / len(nodes), f"{len(hits)}/{len(nodes)} reference nodes cited")


def no_hallucinated_device_reward(candidate_response: str | dict, reference_record: dict) -> dict:
    valid = {str(n).upper() for n in reference_record.get("impacted_nodes", []) if str(n)}
    root = str(reference_record.get("root_cause", "")).upper()
    if root:
        valid.add(root)
    text = _flat_text(candidate_response)
    mentioned = set(re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+){1,4}\b", text))
    hallucinated = sorted(n for n in mentioned if n not in valid)
    if not mentioned:
        return _score(0.6, "no device-like tokens found")
    if hallucinated:
        return _score(max(0.0, 1.0 - len(hallucinated) / len(mentioned)), f"hallucinated devices: {', '.join(hallucinated[:5])}")
    return _score(1.0, "all mentioned devices are in reference evidence")


def validation_before_remediation_reward(candidate_response: str | dict, _reference_record: dict) -> dict:
    data, raw = _parse(candidate_response)
    validation = data.get("validation_steps") or []
    remediation = data.get("remediation_steps") or []
    if not validation or not remediation:
        return _score(0.0, "validation_steps or remediation_steps missing")
    raw_lower = raw.lower()
    v_pos = raw_lower.find("validation")
    r_pos = raw_lower.find("remediation")
    if v_pos != -1 and r_pos != -1 and v_pos < r_pos:
        return _score(1.0, "validation appears before remediation")
    return _score(0.7, "validation/remediation present but ordering is unclear")


def rollback_safety_reward(candidate_response: str | dict, _reference_record: dict) -> dict:
    data, _ = _parse(candidate_response)
    notes = data.get("rollback_or_safety_notes") or []
    blockers = data.get("do_not_execute_if") or []
    total = len(notes if isinstance(notes, list) else [notes]) + len(blockers if isinstance(blockers, list) else [blockers])
    if total >= 3:
        return _score(1.0, "rollback and do-not-execute safeguards present")
    if total >= 1:
        return _score(0.5, "some safety safeguards present")
    return _score(0.0, "rollback/safety missing")


def enterprise_escalation_reward(candidate_response: str | dict, reference_record: dict) -> dict:
    diagrams = set(reference_record.get("impacted_diagrams", []) or [])
    text = _flat_text(candidate_response).lower()
    if len(diagrams) <= 1:
        return _score(1.0, "single-diagram record does not require enterprise escalation")
    if "escalat" in text and ("enterprise" in text or "noc" in text or "sre" in text or "network" in text):
        return _score(1.0, "enterprise escalation present for cross-diagram incident")
    return _score(0.0, "cross-diagram incident missing enterprise escalation")


def servicenow_summary_reward(candidate_response: str | dict, _reference_record: dict) -> dict:
    data, _ = _parse(candidate_response)
    snow = data.get("servicenow_incident_summary")
    if not isinstance(snow, dict):
        return _score(0.0, "ServiceNow summary missing")
    required = {"short_description", "description", "affected_ci", "priority", "assignment_group"}
    present = {k for k in required if snow.get(k)}
    return _score(len(present) / len(required), f"{len(present)}/{len(required)} ServiceNow fields present")


REWARD_FNS = {
    "json_format": json_format_reward,
    "root_cause_match": root_cause_match_reward,
    "grounded_node": grounded_node_reward,
    "no_hallucinated_device": no_hallucinated_device_reward,
    "validation_before_remediation": validation_before_remediation_reward,
    "rollback_safety": rollback_safety_reward,
    "enterprise_escalation": enterprise_escalation_reward,
    "servicenow_summary": servicenow_summary_reward,
}


def overall_reward(candidate_response: str | dict, reference_record: dict) -> dict:
    weights = {
        "json_format": 0.16,
        "root_cause_match": 0.18,
        "grounded_node": 0.14,
        "no_hallucinated_device": 0.14,
        "validation_before_remediation": 0.12,
        "rollback_safety": 0.12,
        "enterprise_escalation": 0.08,
        "servicenow_summary": 0.06,
    }
    details = {name: fn(candidate_response, reference_record) for name, fn in REWARD_FNS.items()}
    total = sum(weights[name] * details[name]["score"] for name in weights)
    return {"score": round(total, 4), "reason": "weighted deterministic reward", "details": details}


def batch_reward_fn(responses: list[str], references: list[dict]) -> list[float]:
    return [overall_reward(response, ref)["score"] for response, ref in zip(responses, references)]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate InfraGraph remediation reward functions.")
    parser.add_argument("--data", default="training/verl_grpo/data/rca_remediation_rl_eval.jsonl")
    parser.add_argument("--out", default="training/verl_grpo/reward_eval_report.json")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    rows = _load_jsonl(data_path)
    reports = []
    for row in rows:
        chosen = overall_reward(row.get("chosen_response", ""), row)
        rejected = overall_reward(row.get("rejected_response", ""), row)
        reports.append({
            "id": row.get("id", ""),
            "scenario_id": row.get("scenario_id", ""),
            "chosen_score": chosen["score"],
            "rejected_score": rejected["score"],
            "margin": round(chosen["score"] - rejected["score"], 4),
            "chosen_details": chosen["details"],
            "rejected_details": rejected["details"],
        })
    summary = {
        "records": len(reports),
        "average_chosen_score": round(sum(r["chosen_score"] for r in reports) / max(len(reports), 1), 4),
        "average_rejected_score": round(sum(r["rejected_score"] for r in reports) / max(len(reports), 1), 4),
        "records_with_positive_margin": sum(1 for r in reports if r["margin"] > 0),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "records": reports}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
