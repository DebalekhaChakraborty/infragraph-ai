"""
policy_filter.py — Apply enterprise policy to the runbook chain.

apply_runbook_policy(runbooks, calibrated_confidence, risk_level) -> list[dict]
    Annotates each runbook with policy-derived execution mode and approval flag.
"""
from __future__ import annotations


def apply_runbook_policy(
    runbooks: list[dict],
    calibrated_confidence: float = 0.0,
    risk_level: str = "medium",
) -> list[dict]:
    """
    Annotate runbooks with policy decisions.

    Rules (applied in order, most restrictive wins):
      - confidence < 0.50         → manual_only for all runbooks
      - confidence < 0.75         → force approval_required
      - risk_level == "critical"  → manual_only, force approval
      - risk_level == "high"      → force approval_required
      - otherwise                 → preserve runbook defaults
    """
    rl  = (risk_level or "medium").lower()
    cal = float(calibrated_confidence or 0.0)

    result: list[dict] = []
    for rb in runbooks:
        rb_copy = dict(rb)

        approval_required   = rb_copy.get("approval_required", True)
        automation_eligible = rb_copy.get("automation_eligible", False)
        execution_mode      = "automated" if automation_eligible else "manual"
        policy_note         = ""

        if cal < 0.50:
            approval_required   = True
            automation_eligible = False
            execution_mode      = "manual_only"
            policy_note         = "Manual-only: calibrated confidence below 0.50."
        elif cal < 0.75:
            approval_required = True
            policy_note       = "Approval required: calibrated confidence below 0.75 gate."
        elif rl == "critical":
            approval_required   = True
            automation_eligible = False
            execution_mode      = "manual_only"
            policy_note         = "Manual-only: critical risk level — no automation allowed."
        elif rl == "high":
            approval_required = True
            policy_note       = "Approval required: high risk level."

        rb_copy["approval_required"]   = approval_required
        rb_copy["automation_eligible"] = automation_eligible
        rb_copy["execution_mode"]      = execution_mode
        rb_copy["policy_note"]         = policy_note
        result.append(rb_copy)

    return result
