"""
Deterministic 9-step agentic incident orchestrator.

Root cause ALWAYS comes from graph/GNN RCA or deterministic fallback.
Qwen/vLLM is used ONLY for remediation generation (Step 6), after RCA evidence exists.
No LangChain / LangGraph / CrewAI / any external agent framework.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from agents.schemas import AgentRun, AgentStep
from agents import tools as _tools

try:
    from rca_ml.calibration import calibrate_confidence as _calibrate_confidence
    _CALIB_OK = True
except Exception:
    _calibrate_confidence = None  # type: ignore
    _CALIB_OK = False

try:
    from governance.evidence_critic import validate_rca_and_remediation as _validate_governance
    _GOVERNANCE_OK = True
except Exception:
    _validate_governance = None  # type: ignore
    _GOVERNANCE_OK = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _step(
    step_id: int,
    agent_name: str,
    objective: str,
    tool_name: str,
    status: str,
    started_at: str,
    summary: str,
    evidence: list | None = None,
    payload: dict | None = None,
) -> dict:
    return AgentStep(
        step_id=step_id,
        agent_name=agent_name,
        objective=objective,
        tool_name=tool_name,
        status=status,
        started_at=started_at,
        completed_at=_now_iso(),
        summary=summary,
        evidence=evidence or [],
        payload=payload or {},
    ).to_dict()


def run_agentic_incident_flow(
    repo_root: Path,
    selected_diagram_id: str | None = None,
    scenario_id: str | None = None,
    prefer_qwen: bool = True,
    mode: str = "demo",
) -> dict:
    """
    Execute the 10-step agentic incident flow and return a JSON-serializable AgentRun dict.

    Steps:
      1.  Alert Intake Agent
      2.  Topology Context Agent
      3.  Correlation Agent
      4.  RCA Agent
      5.  Evidence Validation Agent  (includes confidence calibration in payload)
      6.  Remediation Agent
      7.  ITSM Draft Agent
      8.  Human Approval Agent       (uses calibrated confidence for risk gate)
      9.  Governance Critic Agent
      10. Final Summary Agent
    """
    run_id     = f"run-{uuid.uuid4().hex[:12]}"
    started_at = _now_iso()
    steps: list[dict] = []

    # Working state
    ctx:               dict = {}
    local_incident:    dict = {}
    ent_incident:      dict = {}
    gnn_result:        dict = {}
    evidence_result:   dict = {}
    remediation_result: dict = {}
    ticket_draft:      dict = {}

    alert_source            = "unknown"
    topology_source         = "unknown"
    rca_source              = "unknown"
    remediation_source      = "unknown"
    root_cause              = ""
    root_cause_diagram      = ""
    confidence              = 0.0
    calibrated_confidence   = 0.0
    confidence_calibration: dict = {}
    governance_review:      dict = {}
    impacted_diagrams: list = []

    # ══════════════════════════════════════════════════════════════════
    # Step 1 — Alert Intake Agent
    # ══════════════════════════════════════════════════════════════════
    t0 = _now_iso()
    try:
        ctx             = _tools.load_selected_context(repo_root, selected_diagram_id, scenario_id)
        scenario_id     = ctx["scenario_id"]
        selected_diagram_id = ctx["diagram_id"]
        topology_source = ctx["topology_source"]

        local_result    = _tools.simulate_alert_intake(ctx["local_graph"], ctx["diagram_id"])
        local_incident  = local_result.get("incident", {})
        alert_source    = local_result.get("alert_source", "unknown")
        timeline        = local_result.get("alert_timeline", [])

        steps.append(_step(
            1, "Alert Intake Agent",
            "Simulate topology-aware alert stream from selected infrastructure diagram",
            "simulate_alert_intake",
            "success" if local_result.get("ok") else "warning",
            t0,
            f"Generated {len(timeline)} alert event(s). Source: {alert_source}.",
            evidence=[f"Alert source: {alert_source}", f"{len(timeline)} event(s)"],
            payload={"alert_count": len(timeline), "diagram_id": ctx["diagram_id"]},
        ))
    except Exception as exc:
        steps.append(_step(
            1, "Alert Intake Agent", "Simulate alert stream", "simulate_alert_intake",
            "error", t0, f"Alert intake failed: {exc}",
        ))

    # ══════════════════════════════════════════════════════════════════
    # Step 2 — Topology Context Agent
    # ══════════════════════════════════════════════════════════════════
    t0 = _now_iso()
    try:
        eg       = ctx.get("enterprise_graph", {})
        nodes    = eg.get("nodes", [])
        edges    = eg.get("edges", [])
        cross    = eg.get("cross_diagram_edges", [])
        clusters = eg.get("diagram_clusters", {})
        if isinstance(clusters, dict):
            n_dom = len(clusters)
        elif isinstance(clusters, list):
            n_dom = len(clusters)
        else:
            n_dom = 0
        if n_dom == 0 and nodes:
            n_dom = len(set(n.get("diagram_id", "") for n in nodes if n.get("diagram_id")))

        steps.append(_step(
            2, "Topology Context Agent",
            "Load enterprise graph: nodes, edges, diagram domains",
            "load_selected_context",
            "success" if nodes else "warning",
            t0,
            (
                f"{len(nodes)} nodes, {len(edges)} intra-diagram edges, "
                f"{len(cross)} cross-diagram edges across {n_dom} domain(s). "
                f"Source: {topology_source}."
            ),
            evidence=[
                f"Topology source: {topology_source}",
                f"Scenario: {scenario_id or '—'}",
                f"Diagram: {selected_diagram_id or '—'}",
                f"{len(nodes)} nodes | {len(edges)} edges | {len(cross)} cross-diagram edges",
            ],
            payload={
                "node_count": len(nodes), "edge_count": len(edges),
                "cross_edge_count": len(cross), "domain_count": n_dom,
            },
        ))
    except Exception as exc:
        steps.append(_step(
            2, "Topology Context Agent", "Load enterprise graph context", "load_selected_context",
            "error", t0, f"Context loading failed: {exc}",
        ))

    # ══════════════════════════════════════════════════════════════════
    # Step 3 — Correlation Agent
    # ══════════════════════════════════════════════════════════════════
    t0 = _now_iso()
    try:
        ent_result   = _tools.simulate_enterprise_alert_intake(
            ctx.get("enterprise_graph", {}),
            selected_diagram_id or "",
            scenario_id or "",
            alerts_data=ctx.get("alerts_data", {}),
            gnn_result=None,
        )
        ent_incident = ent_result.get("incident", {})
        imp_d        = ent_incident.get("impacted_diagrams", [])
        ent_timeline = ent_incident.get("alert_timeline", [])

        steps.append(_step(
            3, "Correlation Agent",
            "Map alerts to cross-diagram topology; build enterprise incident scope",
            "simulate_enterprise_alert_intake",
            "success" if ent_result.get("ok") else "warning",
            t0,
            f"Enterprise incident built. {len(imp_d)} impacted diagram(s), {len(ent_timeline)} correlated alert(s).",
            evidence=[
                f"Cross-diagram scope: {len(imp_d)} diagram(s)",
                f"Correlated alerts: {len(ent_timeline)}",
                f"Incident ID: {ent_incident.get('incident_id', '—')}",
            ],
            payload={"impacted_diagram_count": len(imp_d)},
        ))
    except Exception as exc:
        steps.append(_step(
            3, "Correlation Agent", "Map alerts to cross-diagram topology",
            "simulate_enterprise_alert_intake", "error", t0,
            f"Correlation failed: {exc}",
        ))

    # ══════════════════════════════════════════════════════════════════
    # Step 4 — RCA Agent
    # ══════════════════════════════════════════════════════════════════
    t0 = _now_iso()
    try:
        gnn_load = _tools.load_or_run_enterprise_gnn_rca(repo_root, scenario_id or "")

        if gnn_load.get("ok"):
            gnn_result         = gnn_load.get("gnn_result") or {}
            root_cause         = gnn_load["root_cause"] or ent_incident.get("root_cause", "")
            root_cause_diagram = gnn_load["root_cause_diagram"]
            confidence         = gnn_load["confidence"]
            impacted_diagrams  = gnn_load["impacted_diagrams"] or ent_incident.get("impacted_diagrams", [])
            rca_source         = gnn_load["rca_source"]
            steps.append(_step(
                4, "RCA Agent",
                "Load Enterprise GNN RCA result; identify root cause and confidence",
                "load_or_run_enterprise_gnn_rca",
                "success", t0,
                (
                    f"Enterprise GNN RCA loaded. Root cause: '{root_cause}' "
                    f"in '{root_cause_diagram}' (confidence {confidence:.0%}). "
                    f"Source: {rca_source}."
                ),
                evidence=[
                    f"RCA source: {rca_source}",
                    f"Root cause: {root_cause}",
                    f"Root cause diagram: {root_cause_diagram}",
                    f"Confidence: {confidence:.0%}",
                    f"Result path: {gnn_load.get('path', '—')}",
                ],
                payload={"root_cause": root_cause, "confidence": confidence, "rca_source": rca_source},
            ))
        else:
            # Scenario / incident graph fallback — clearly labelled
            root_cause         = ent_incident.get("root_cause") or local_incident.get("root_cause", "unknown")
            root_cause_diagram = ent_incident.get("root_cause_diagram", selected_diagram_id or "")
            confidence         = 0.5
            impacted_diagrams  = ent_incident.get("impacted_diagrams", [])
            rca_source         = "scenario_grounded_graph_fallback"
            warning            = gnn_load.get("warning", "GNN result unavailable.")
            steps.append(_step(
                4, "RCA Agent",
                "Load Enterprise GNN RCA (fallback: scenario-grounded graph RCA)",
                "load_or_run_enterprise_gnn_rca",
                "warning", t0,
                f"GNN RCA unavailable — using scenario-grounded fallback. Root cause: '{root_cause}'. {warning}",
                evidence=[
                    "RCA source: scenario-grounded graph fallback (GNN not available)",
                    f"Root cause (fallback): {root_cause}",
                    warning,
                ],
                payload={"root_cause": root_cause, "confidence": confidence, "rca_source": rca_source},
            ))
    except Exception as exc:
        root_cause = ent_incident.get("root_cause", "unknown")
        rca_source = "error_fallback"
        steps.append(_step(
            4, "RCA Agent", "Load Enterprise GNN RCA", "load_or_run_enterprise_gnn_rca",
            "error", t0, f"RCA step error: {exc}",
        ))

    # ══════════════════════════════════════════════════════════════════
    # Step 5 — Evidence Validation Agent
    # ══════════════════════════════════════════════════════════════════
    t0 = _now_iso()
    try:
        evidence_result = _tools.validate_rca_evidence(
            ctx.get("enterprise_graph", {}),
            ent_incident,
            gnn_result or None,
        )
        if not impacted_diagrams:
            impacted_diagrams = evidence_result.get("impacted_diagrams", [])

        ev_bullets = list(evidence_result.get("evidence_summary", []))

        # ── Confidence calibration (embedded in Step 5) ───────────────────────
        top_candidates = (
            (gnn_result or {}).get("top_candidates")
            or (gnn_result or {}).get("ranking")
            or []
        )
        if _CALIB_OK and _calibrate_confidence:
            try:
                confidence_calibration = _calibrate_confidence(
                    raw_confidence=confidence,
                    rca_source=rca_source,
                    impacted_diagrams=list(impacted_diagrams),
                    top_candidates=top_candidates,
                    evidence_summary=ev_bullets,
                )
                calibrated_confidence = confidence_calibration["calibrated_confidence"]
            except Exception:
                confidence_calibration = {}
                calibrated_confidence = confidence
        else:
            confidence_calibration = {}
            calibrated_confidence = confidence

        _gate_label = (
            "passed" if confidence_calibration.get("threshold_passed")
            else "needs validation"
        )
        ev_bullets = ev_bullets + [
            f"Raw confidence: {confidence:.0%}",
            f"Calibrated confidence: {calibrated_confidence:.0%}",
            f"Confidence gate: {_gate_label}",
        ]

        steps.append(_step(
            5, "Evidence Validation Agent",
            "Deterministically validate RCA with graph and alert evidence — no LLM",
            "validate_rca_evidence",
            "success", t0,
            (
                f"Validated {len(ev_bullets)} evidence point(s). "
                f"{evidence_result.get('alert_count', 0)} alert(s), "
                f"{len(impacted_diagrams)} impacted diagram(s). "
                f"Calibrated confidence: {calibrated_confidence:.0%}."
            ),
            evidence=ev_bullets,
            payload={
                "alert_count":            evidence_result.get("alert_count", 0),
                "impacted_count":         len(impacted_diagrams),
                "confidence_calibration": confidence_calibration,
            },
        ))
    except Exception as exc:
        calibrated_confidence = confidence
        steps.append(_step(
            5, "Evidence Validation Agent", "Validate RCA evidence", "validate_rca_evidence",
            "error", t0, f"Evidence validation error: {exc}",
        ))

    # ══════════════════════════════════════════════════════════════════
    # Step 6 — Remediation Agent (Qwen/vLLM or template fallback)
    # ══════════════════════════════════════════════════════════════════
    t0 = _now_iso()
    try:
        rem_ctx = _tools.build_remediation_context(
            run_id=run_id,
            selected_diagram_id=selected_diagram_id or "",
            scenario_id=scenario_id or "",
            enterprise_graph=ctx.get("enterprise_graph", {}),
            ent_incident=ent_incident,
            impacted_diagrams=impacted_diagrams,
            root_cause=root_cause,
            root_cause_diagram=root_cause_diagram,
            rca_source=rca_source,
            gnn_result=gnn_result or None,
        )
        remediation_result = _tools.generate_ai_remediation(rem_ctx, prefer_qwen=prefer_qwen)
        remediation_source = remediation_result.get("source", "unknown")
        used_qwen          = remediation_source == "qwen_vllm"

        steps.append(_step(
            6, "Remediation Agent",
            "Generate SOP-grounded remediation using Qwen/vLLM or template fallback",
            "generate_ai_remediation",
            "success" if (remediation_result.get("ok") or "template" in remediation_source) else "warning",
            t0,
            f"Remediation generated. Source: {remediation_source}. Qwen/vLLM: {used_qwen}.",
            evidence=[
                f"Remediation source: {remediation_source}",
                "Qwen/vLLM: yes" if used_qwen else "Qwen/vLLM: no — template fallback used",
            ] + ([f"Error: {remediation_result.get('error')}"] if remediation_result.get("error") else []),
            payload={"source": remediation_source, "qwen_used": used_qwen},
        ))
    except Exception as exc:
        remediation_result = {"source": "error", "ok": False, "error": str(exc), "response": {}}
        remediation_source = "error"
        steps.append(_step(
            6, "Remediation Agent", "Generate remediation", "generate_ai_remediation",
            "error", t0, f"Remediation error: {exc}",
        ))

    # ══════════════════════════════════════════════════════════════════
    # Step 7 — ITSM Draft Agent
    # ══════════════════════════════════════════════════════════════════
    t0 = _now_iso()
    try:
        rca_for_ticket = {
            "root_cause":        root_cause,
            "root_cause_diagram": root_cause_diagram,
            "confidence":        confidence,
            "impacted_diagrams": impacted_diagrams,
            "rca_source":        rca_source,
            "evidence_summary":  evidence_result.get("evidence_summary", []),
        }
        ticket_draft = _tools.draft_itsm_ticket(
            root_cause, rca_for_ticket, remediation_result, ent_incident,
        )
        steps.append(_step(
            7, "ITSM Draft Agent",
            "Generate demo ITSM ticket draft — no external ITSM API called",
            "draft_itsm_ticket",
            "success", t0,
            f"Demo ticket: {ticket_draft.get('ticket_id')} | {ticket_draft.get('priority')}. No external system contacted.",
            evidence=[
                f"Ticket ID: {ticket_draft.get('ticket_id')}",
                f"Priority: {ticket_draft.get('priority')}",
                f"Assignment: {ticket_draft.get('assignment_group')}",
                "DEMO ONLY — no real ITSM ticket created",
            ],
            payload={"ticket_id": ticket_draft.get("ticket_id"), "priority": ticket_draft.get("priority")},
        ))
    except Exception as exc:
        steps.append(_step(
            7, "ITSM Draft Agent", "Generate ITSM ticket draft", "draft_itsm_ticket",
            "error", t0, f"Ticket draft error: {exc}",
        ))

    # ══════════════════════════════════════════════════════════════════
    # Step 8 — Human Approval Agent
    # ══════════════════════════════════════════════════════════════════
    t0 = _now_iso()
    n_impacted          = len(impacted_diagrams)
    rem_text            = str(remediation_result.get("response", "")).lower()
    high_risk           = any(
        k in rem_text
        for k in ("shutdown", "restart", "failover", "disable", "terminate", "reboot", "isolate")
    )
    effective_confidence = calibrated_confidence if calibrated_confidence else confidence

    if high_risk or n_impacted >= 3 or effective_confidence >= 0.85:
        risk_level  = "high"
        risk_reason = (
            "High-risk remediation actions detected in plan"
            if high_risk
            else (
                f"{n_impacted} diagrams impacted with calibrated confidence "
                f"{effective_confidence:.0%}"
            )
        )
    elif n_impacted >= 2 or effective_confidence >= 0.6:
        risk_level  = "medium"
        risk_reason = f"{n_impacted} diagram(s) impacted — moderate blast radius"
    else:
        risk_level  = "low"
        risk_reason = "Localised impact, lower confidence threshold"

    approval_gate = {
        "required":                True,
        "status":                  "pending",
        "risk_level":              risk_level,
        "reason":                  risk_reason,
        "recommended_next_action": (
            "Review the agent trace and validate root cause, "
            "then approve or reject the remediation before any execution."
        ),
    }
    _calib_note = (
        f"Calibrated confidence: {effective_confidence:.0%}"
        if effective_confidence != confidence
        else f"Confidence: {effective_confidence:.0%} (no calibration applied)"
    )
    steps.append(_step(
        8, "Human Approval Agent",
        "Require human review and approval before any remediation is executed",
        "approval_gate",
        "success", t0,
        f"Approval gate set. Risk: {risk_level}. Status: pending. Human action required.",
        evidence=[
            f"Risk level: {risk_level}",
            f"Reason: {risk_reason}",
            f"Raw confidence: {confidence:.0%}",
            _calib_note,
            "No auto-execution — approval required",
        ],
        payload=approval_gate,
    ))

    # ══════════════════════════════════════════════════════════════════
    # Step 9 — Governance Critic Agent
    # ══════════════════════════════════════════════════════════════════
    t0 = _now_iso()
    if _GOVERNANCE_OK and _validate_governance:
        try:
            _agent_snapshot = {
                "root_cause":           root_cause,
                "root_cause_diagram":   root_cause_diagram,
                "rca_source":           rca_source,
                "confidence":           confidence,
                "calibrated_confidence": calibrated_confidence,
                "impacted_diagrams":    impacted_diagrams,
                "steps":                steps,
                "approval_gate":        approval_gate,
            }
            governance_review = _validate_governance(
                _agent_snapshot,
                remediation_plan=remediation_result,
                graph_context=ctx.get("enterprise_graph", {}),
            )
        except Exception as _gov_exc:
            governance_review = {
                "status": "skipped",
                "score": 0,
                "findings": [f"Governance validator error: {_gov_exc}"],
                "blocking_issues": [],
                "approval_recommendation": "needs_human_review",
            }
    else:
        governance_review = {
            "status": "skipped",
            "score": 0,
            "findings": ["Governance validator unavailable."],
            "blocking_issues": [],
            "approval_recommendation": "needs_human_review",
        }

    _gov_status      = governance_review.get("status", "skipped")
    _gov_score       = governance_review.get("score", 0)
    _gov_step_status = (
        "success" if _gov_status == "passed"
        else "warning" if _gov_status == "warning"
        else "error" if _gov_status == "failed"
        else "skipped"
    )
    _gov_findings  = governance_review.get("findings", [])[:5]
    _gov_blocking  = [f"⛔ {bi}" for bi in governance_review.get("blocking_issues", [])]
    steps.append(_step(
        9, "Governance Critic Agent",
        "Evidence-based governance and compliance validation of RCA and remediation plan",
        "validate_rca_and_remediation",
        _gov_step_status, t0,
        f"Governance review {_gov_status} with score {_gov_score}/100.",
        evidence=_gov_findings + _gov_blocking,
        payload=governance_review,
    ))

    # ══════════════════════════════════════════════════════════════════
    # Step 10 — Final Summary Agent
    # ══════════════════════════════════════════════════════════════════
    t0 = _now_iso()
    _calib_summary = (
        f"Calibrated confidence: {calibrated_confidence:.0%}. "
        if calibrated_confidence and calibrated_confidence != confidence else ""
    )
    final_summary = (
        f"InfraGraph AI detected a cross-diagram infrastructure incident. "
        f"Root cause: '{root_cause}' in '{root_cause_diagram}' "
        f"(confidence {confidence:.0%}, source: {rca_source}). "
        f"{_calib_summary}"
        f"{n_impacted} diagram(s) impacted. "
        f"Remediation: {remediation_source}. "
        f"Governance: {_gov_status} ({_gov_score}/100). "
        f"Ticket {ticket_draft.get('ticket_id', '—')} drafted ({ticket_draft.get('priority', '—')}). "
        f"Human approval: pending."
    )
    steps.append(_step(
        10, "Final Summary Agent",
        "Produce executive summary of end-to-end agentic incident flow",
        "final_summary",
        "success", t0,
        final_summary,
        evidence=[
            f"Root cause: {root_cause}",
            f"Raw confidence: {confidence:.0%}",
            f"Calibrated confidence: {calibrated_confidence:.0%}",
            f"RCA source: {rca_source}",
            f"Remediation source: {remediation_source}",
            f"Governance: {_gov_status} ({_gov_score}/100)",
            f"Ticket: {ticket_draft.get('ticket_id', '—')} ({ticket_draft.get('priority', '—')})",
            "Approval: pending",
        ],
        payload={"run_id": run_id},
    ))

    # ══════════════════════════════════════════════════════════════════
    # Assemble and persist AgentRun
    # ══════════════════════════════════════════════════════════════════
    _is_fallback = "fallback" in rca_source or rca_source in ("unknown", "error_fallback")
    _run_status  = "partial" if _is_fallback else "success"

    agent_run = AgentRun(
        run_id=run_id,
        scenario_id=scenario_id or "",
        selected_diagram_id=selected_diagram_id or "",
        mode=mode,
        started_at=started_at,
        completed_at=_now_iso(),
        status=_run_status,
        alert_source=alert_source,
        topology_source=topology_source,
        rca_source=rca_source,
        remediation_source=remediation_source,
        steps=steps,
        final_summary=final_summary,
        root_cause=root_cause,
        root_cause_diagram=root_cause_diagram,
        confidence=confidence,
        calibrated_confidence=calibrated_confidence if calibrated_confidence else None,
        confidence_calibration=confidence_calibration,
        impacted_diagrams=list(impacted_diagrams),
        approval_gate=approval_gate,
        ticket_draft=ticket_draft,
        governance_review=governance_review,
        runbook_chain=rem_ctx.get("runbook_chain", []),
    ).to_dict()

    try:
        _tools.persist_agent_run(repo_root, agent_run)
    except Exception:
        pass

    return agent_run
