"""
evidence_critic.py — Governance validator for RCA + Remediation.

validate_rca_and_remediation(agent_run, remediation_plan, graph_context) -> dict

Checks that RCA is evidence-grounded and remediation includes required
safety gates before approval. Purely rule-based — no LLM.
"""
from __future__ import annotations

_RISKY_KEYWORDS: frozenset[str] = frozenset({
    "shutdown", "restart", "failover", "disable", "terminate",
    "reboot", "isolate", "drop", "flush", "wipe",
})


def validate_rca_and_remediation(
    agent_run: dict,
    remediation_plan: dict | None = None,
    graph_context: dict | None = None,
) -> dict:
    """
    Validate RCA evidence and remediation plan for enterprise governance.

    Returns
    -------
    dict with:
      status                 : "passed" | "warning" | "failed"
      score                  : int  0–100
      findings               : list[str]
      blocking_issues        : list[str]
      approval_recommendation: "approve_for_demo" | "needs_human_review" | "block_execution"
    """
    findings:        list[str] = []
    blocking_issues: list[str] = []
    score = 100

    root_cause    = agent_run.get("root_cause", "")
    rca_source    = agent_run.get("rca_source", "")
    confidence    = float(agent_run.get("confidence") or 0.0)
    impacted      = agent_run.get("impacted_diagrams", [])
    steps         = agent_run.get("steps", [])
    approval_gate = agent_run.get("approval_gate", {})
    cal_conf      = agent_run.get("calibrated_confidence")

    # ── RCA checks ────────────────────────────────────────────────────────────

    # 1: Root cause presence
    if not root_cause:
        blocking_issues.append("Root cause is missing — RCA incomplete.")
        score -= 25
    else:
        findings.append(f"Root cause identified: '{root_cause}'.")

    # 2: Root cause exists in graph topology
    if root_cause and graph_context:
        node_ids = {n.get("id", "") for n in graph_context.get("nodes", [])}
        if root_cause in node_ids:
            findings.append("Root cause node confirmed in enterprise graph topology.")
        else:
            findings.append(
                f"'{root_cause}' not found in graph nodes "
                f"— may be a diagram ID or label alias."
            )
            score -= 5

    # 3: RCA source labelling
    if not rca_source:
        blocking_issues.append("RCA source field is missing.")
        score -= 10
    elif "gnn" in rca_source.lower():
        findings.append(f"RCA source: {rca_source} — Enterprise GNN backed.")
    elif "fallback" in rca_source.lower():
        findings.append(
            f"RCA source: {rca_source} — graph-grounded fallback (not Enterprise GNN)."
        )
    else:
        findings.append(f"RCA source: {rca_source}.")

    # 4: Calibrated confidence gate
    eff_conf = float(cal_conf) if cal_conf is not None else confidence
    if eff_conf >= 0.75:
        findings.append(
            f"Confidence gate: {eff_conf:.0%}"
            + (" (calibrated)" if cal_conf is not None else " (raw)") + " — passed."
        )
    else:
        findings.append(
            f"Confidence {eff_conf:.0%} is below 0.75 gate — human validation required."
        )
        score -= 10

    # 5: Evidence bullets from Step 5
    step5 = next((s for s in steps if s.get("step_id") == 5), {})
    ev_bullets = step5.get("evidence", [])
    if len(ev_bullets) >= 2:
        findings.append(f"Evidence validation: {len(ev_bullets)} item(s) from Step 5.")
    else:
        findings.append("Sparse evidence in Step 5 — fewer than 2 bullets.")
        score -= 10

    # ── Remediation checks ────────────────────────────────────────────────────

    if remediation_plan:
        resp = remediation_plan.get("response", {}) or {}
        if not isinstance(resp, dict):
            resp = {}

        # 6: Validation before remediation
        val_steps = resp.get("validation_steps", [])
        rem_steps = resp.get("remediation_steps", [])
        if val_steps:
            findings.append(f"{len(val_steps)} validation step(s) precede remediation.")
        else:
            findings.append("No validation steps before remediation steps.")
            score -= 10

        # 7: Rollback / safety notes
        rollback = resp.get("rollback_or_safety_notes", [])
        if rollback:
            findings.append(f"{len(rollback)} rollback / safety note(s) included.")
        else:
            blocking_issues.append(
                "Rollback or safety notes are missing — plan is incomplete."
            )
            score -= 15

        # 8: Evidence from graph
        ev_graph = resp.get("evidence_from_graph", [])
        if ev_graph:
            findings.append(f"{len(ev_graph)} graph evidence item(s) cited in plan.")
        else:
            findings.append(
                "No evidence_from_graph items — plan may not be graph-grounded."
            )
            score -= 5

        # 9: Runbook chain presence
        rb_chain = resp.get("runbook_chain", [])
        if rb_chain:
            rb_ids = [rb.get("runbook_id", "?") for rb in rb_chain]
            findings.append(f"Runbook chain present: {', '.join(rb_ids)}.")
        else:
            findings.append("Runbook chain absent — plan is not formally SOP-grounded.")
            score -= 5

        # 10: High-risk action approval gate
        rem_text = " ".join(str(s) for s in rem_steps).lower()
        risky    = [kw for kw in _RISKY_KEYWORDS if kw in rem_text]
        if risky:
            if not approval_gate.get("required"):
                blocking_issues.append(
                    f"High-risk actions ({', '.join(risky)}) without approval gate."
                )
                score -= 20
            else:
                findings.append(
                    f"High-risk actions ({', '.join(risky)}) — approval gate is set."
                )
    else:
        findings.append("No remediation plan generated yet — partial review only.")
        score -= 10

    # ── Approval gate check ───────────────────────────────────────────────────
    if approval_gate.get("required"):
        findings.append("Human approval gate is active.")
    else:
        findings.append("Approval gate not required — verify appropriateness.")
        score -= 5

    # Clamp
    score = max(0, min(100, score))

    # Status + recommendation
    if blocking_issues:
        status = "failed"
        rec    = "block_execution"
    elif score < 70:
        status = "warning"
        rec    = "needs_human_review"
    else:
        status = "passed"
        rec    = "approve_for_demo"

    return {
        "status":                  status,
        "score":                   score,
        "findings":                findings,
        "blocking_issues":         blocking_issues,
        "approval_recommendation": rec,
    }
