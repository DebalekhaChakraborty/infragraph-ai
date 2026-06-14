"""
Template remediation mode — deterministic, not model-generated.

Used when vLLM / Qwen3 is unavailable.  Output is clearly labelled
"Template remediation mode" and must never be presented as AI output.

Dispatches on context["scope"]: "local" or "enterprise".
"""
from __future__ import annotations

from .response_schema import make_remediation_output

_SOURCE_LABEL = "template"


def _unique_strings(values: list) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values or []:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _evidence_ids(context: dict) -> list[str]:
    ids: list[str] = []
    for item in context.get("retrieved_graph_memory_evidence", []) or []:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata") or {}
        eid = meta.get("evidence_id")
        if eid:
            ids.append(str(eid))
    for item in context.get("causal_evidence", []) or []:
        eid = item.get("evidence_id")
        if eid:
            ids.append(str(eid))
    # Include RB-* evidence IDs from runbook chain
    for rb in context.get("runbook_chain", []) or []:
        for eid in rb.get("evidence_ids", []):
            if eid:
                ids.append(str(eid))
    return _unique_strings(ids)


def _build_automation_plan(context: dict) -> dict:
    """Build automation_plan from the first automation-eligible runbook in the chain."""
    runbook_chain = context.get("runbook_chain", []) or []
    # Find the first automation-eligible runbook
    for rb in runbook_chain:
        if rb.get("automation_eligible"):
            return {
                "can_execute":       True,
                "requires_approval": bool(rb.get("approval_required", False)),
                "connector":         str(rb.get("connector", "") or rb.get("tool_name", "")),
                "dry_run_supported": bool(rb.get("dry_run_supported", False)),
                "execution_note":    (
                    f"Runbook {rb.get('runbook_id', '')} is automation-eligible "
                    f"(execution_mode={rb.get('execution_mode', 'semi_automated')}). "
                    + ("Dry-run available." if rb.get("dry_run_supported") else "No dry-run.")
                ),
            }
    # Default: manual only
    approval_needed = any(rb.get("approval_required", True) for rb in runbook_chain)
    return {
        "can_execute":       False,
        "requires_approval": approval_needed or True,
        "connector":         "",
        "dry_run_supported": False,
        "execution_note":    "No automation-eligible runbook available — manual execution required.",
    }


def _kb_evidence_refs(context: dict, max_items: int = 5) -> list[str]:
    """Return formatted reference strings for retrieved KB evidence chunks.

    SOPs and runbooks are labelled "Retrieved SOP evidence: ...";
    known resolutions are labelled "Retrieved known resolution: ...".
    """
    out: list[str] = []
    for item in (context.get("retrieved_kb_evidence", []) or [])[:max_items]:
        if not isinstance(item, dict):
            continue
        eid      = item.get("evidence_id", "KB-unknown")
        meta     = item.get("metadata") or {}
        title    = meta.get("title", "")
        section  = meta.get("section", "")
        doc_type = meta.get("doc_type", "")
        sec_suffix = f" ({section})" if section else ""
        if doc_type == "known_resolution":
            out.append(f"Retrieved known resolution: {eid} — {title}{sec_suffix}")
        else:
            out.append(f"Retrieved SOP evidence: {eid} — {title}{sec_suffix}")
    return out


def _kb_section_steps(context: dict, section_keyword: str, max_items: int = 3) -> list[str]:
    """
    Return SOP-grounded action strings for chunks whose text contains a known section.

    Looks at retrieved_kb_evidence chunks. Matches are phrased as references,
    not invented commands.
    """
    out: list[str] = []
    seen: set[str] = set()
    for item in (context.get("retrieved_kb_evidence", []) or []):
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        if section_keyword.lower() not in text.lower():
            continue
        meta  = item.get("metadata") or {}
        kb_id = meta.get("kb_id", "")
        title = meta.get("title", "")
        eid   = item.get("evidence_id", "KB-unknown")
        key   = kb_id or eid
        if key in seen:
            continue
        seen.add(key)
        if "remediation" in section_keyword.lower():
            out.append(f"Follow {kb_id or 'SOP'} ({title}) remediation section [{eid}].")
        elif "validation" in section_keyword.lower():
            out.append(f"Validate according to {kb_id or 'SOP'} ({title}) validation section [{eid}].")
        elif "rollback" in section_keyword.lower() or "safety" in section_keyword.lower():
            out.append(f"Apply rollback/safety controls from {kb_id or 'SOP'} ({title}) [{eid}].")
        else:
            out.append(f"Refer to {kb_id or 'SOP'} ({title}) for {section_keyword} guidance [{eid}].")
        if len(out) >= max_items:
            break
    return out


def _causal_evidence_summaries(context: dict, max_items: int = 5) -> list[str]:
    out: list[str] = []
    for item in (context.get("causal_evidence", []) or [])[:max_items]:
        eid     = item.get("evidence_id", "CE")
        stage   = item.get("stage", "unknown")
        summary = item.get("summary", "")
        conf    = item.get("confidence", "")
        out.append(f"{eid} ({stage}, confidence={conf}): {summary}")
    return out


def _blast_radius(scope: str, impacted_diagrams: list[str], impacted_nodes: list[str]) -> str:
    diagrams = _unique_strings(impacted_diagrams)
    if scope == "enterprise":
        if len(diagrams) >= 4:
            return "enterprise_wide"
        if len(diagrams) >= 2:
            return "cross_diagram"
    if len(diagrams) == 1:
        return "single_diagram"
    if len(_unique_strings(impacted_nodes)) <= 1:
        return "single_node"
    return "single_diagram"


def _risk_level(scope: str, impacted_diagrams: list[str], alert_timeline: list[dict]) -> str:
    severities = {str(a.get("severity", "")).lower() for a in alert_timeline or [] if isinstance(a, dict)}
    n_diagrams = len(_unique_strings(impacted_diagrams))
    if "critical" in severities or n_diagrams >= 4:
        return "critical"
    if scope == "enterprise" or n_diagrams >= 2 or len(alert_timeline or []) >= 5:
        return "high"
    if len(alert_timeline or []) >= 3:
        return "medium"
    return "low"


def _automation_eligibility(scope: str, risk_level: str, impacted_diagrams: list[str]) -> str:
    if scope == "enterprise" or len(_unique_strings(impacted_diagrams)) >= 2 or risk_level == "critical":
        return "manual_only"
    if risk_level in {"medium", "high"}:
        return "human_approval_required"
    return "safe_to_automate"


def _do_not_execute(scope: str, root_cause: str) -> list[str]:
    base = [
        f"Do not execute changes if root cause {root_cause or 'node'} is not confirmed by graph evidence.",
        "Do not execute if a rollback owner and maintenance window are not approved.",
        "Do not execute if alert timestamps are stale or no longer match the active incident.",
    ]
    if scope == "enterprise":
        base.append("Do not execute cross-domain changes without NOC and owning team approval.")
    return base


# ── Domain inference (mirrors kb_retrieval.retriever._infer_domain) ──────────

_TEMPLATE_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "firewall": [
        "dc-fw", "fw-", "-fw", "firewall", "packet_drop", "packet drop", "acl",
    ],
    "load_balancer": [
        "app-lb", "-lb-", "-lb", "lb-", "load_balancer", "load balancer",
        "backend_pool", "backend pool", "unhealthy", "health_probe", "health probe",
    ],
    "database": [
        "db-master", "db_master", "db-", "-db", "database", "connection_pool",
        "connection pool", "replica", "replication",
    ],
    "wan": [
        "wan-pe", "wan_pe", "pe-0", "pe-1", "pe-", "-pe", "wan",
        "bgp", "ospf", "mpls", "circuit", "link_down", "link down", "sfp", "carrier",
    ],
}


def _infer_domain(root_cause: str) -> str:
    """Infer infra domain from root_cause node ID for domain-specific step selection."""
    rc = root_cause.lower()
    for domain, keywords in _TEMPLATE_DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in rc:
                return domain
    return ""


# ── Domain-specific remediation steps ────────────────────────────────────────

_REMEDIATION_BY_DOMAIN: dict[str, list[str]] = {
    "load_balancer": [
        "Drain the affected backend pool members gracefully before making any configuration change.",
        "Investigate backend health probe failures — check application health endpoint response codes.",
        "Verify backend server CPU, memory, and thread-pool saturation before re-adding to pool.",
        "After configuration change, confirm backend pool health-check status returns healthy.",
        "Verify no recent application deployment introduced a health-endpoint regression.",
        "Document the change and update the CMDB record for the affected load balancer.",
    ],
    "database": [
        "Check active connection count and connection-pool exhaustion on the database master node.",
        "Review the slow query log for queries holding locks or consuming excessive resources.",
        "Verify replication lag between master and replica — do not failover without DBA approval.",
        "Coordinate with the DBA team before applying schema changes or restarting the database.",
        "After restoration, confirm connection-pool recovery metrics return to baseline.",
        "Document the change and update the CMDB record for the affected database node.",
    ],
    "wan": [
        "Verify WAN circuit status with the carrier via the NOC portal or vendor dashboard.",
        "Inspect physical layer: check SFP module status and optic power levels if accessible.",
        "Confirm BGP session state on WAN edge nodes — look for session reset timestamps.",
        "If BGP is down, verify route advertisements are propagating after re-establishment.",
        "Confirm the backup WAN path is active and carrying traffic during the outage.",
        "Document the change and update the CMDB record for the affected WAN edge node.",
    ],
    "firewall": [
        "Review the firewall policy change log for any recently added or modified deny rules.",
        "Identify affected traffic flows from packet-drop counters on the suspect interface.",
        "Confirm the ACL or policy rule causing drops — do not remove rules without change approval.",
        "Apply firewall policy correction only within an approved maintenance window.",
        "After policy update, confirm packet-drop counters stop incrementing and traffic restores.",
        "Document the change and update the CMDB record for the affected firewall.",
    ],
}

# ── Domain-specific validation steps ─────────────────────────────────────────

_VALIDATION_BY_DOMAIN: dict[str, list[str]] = {
    "load_balancer": [
        "Confirm load balancer backend pool shows all members as healthy after remediation.",
        "Send a synthetic health-probe request and verify application response is within threshold.",
        "Verify application error rates in APM/logging have returned to baseline.",
        "Confirm connection-timeout alerts have cleared on the monitoring platform.",
    ],
    "database": [
        "Confirm active connection count is below the configured connection-pool limit.",
        "Verify no new slow-query alerts are firing after remediation.",
        "Confirm replication lag has returned to near-zero on all replica nodes.",
        "Check database latency metrics in the monitoring platform show return to baseline.",
    ],
    "wan": [
        "Confirm BGP session state shows Established on all WAN edge nodes.",
        "Verify end-to-end reachability across the WAN link from multiple test points.",
        "Confirm packet loss and error rate on the WAN interface have dropped to zero.",
        "Verify carrier circuit status shows up in the NOC portal.",
    ],
    "firewall": [
        "Confirm packet-drop counters on the affected firewall interface have stopped incrementing.",
        "Verify traffic flows that were blocked are now passing the firewall policy.",
        "Check the firewall deny log to confirm no residual ACL hits on legitimate traffic.",
        "Confirm the security team has reviewed and approved the policy change before closure.",
    ],
}


# ── Domain-specific triage templates ─────────────────────────────────────────

_TRIAGE_BY_DIAGRAM: dict[str, list[str]] = {
    "branch_topology": [
        "Verify WAN circuit health and packet loss on the branch uplink.",
        "Check branch router interface metrics: CRC errors, input/output drops.",
        "Confirm branch firewall policy has not changed in the past 24 hours.",
        "Ping the branch default gateway to confirm local Layer-3 reachability.",
    ],
    "wan_topology": [
        "Verify BGP session state on WAN edge routers.",
        "Check MPLS/carrier circuit health via NOC portal or vendor dashboard.",
        "Review WAN path utilisation: confirm no QoS policy is dropping traffic.",
        "Confirm backup path is available and not also degraded.",
    ],
    "app_db_topology": [
        "Check application server error logs for connection-refused or timeout patterns.",
        "Verify database connection pool utilisation and active session count.",
        "Confirm load balancer backend health checks are passing.",
        "Review recent application deployments or configuration changes.",
    ],
    "datacenter_topology": [
        "Inspect core switch interface error counters and spanning-tree topology.",
        "Verify inter-service communication across fabric via traceroute.",
        "Check data-centre firewall deny logs for new ACL hits.",
        "Confirm power and cooling alarms are not contributing to device instability.",
    ],
    "shared_services_topology": [
        "Check DNS resolution from multiple client hosts across diagrams.",
        "Verify IAM / authentication service health and certificate validity.",
        "Confirm NTP synchronisation across all dependent nodes.",
        "Review shared-service change log for recent updates.",
    ],
}

_DEFAULT_TRIAGE = [
    "Isolate the affected segment and confirm Layer-2/3 connectivity.",
    "Review device logs on the suspected root-cause node.",
    "Validate that upstream services are reachable from the affected segment.",
    "Check for recent configuration changes on the implicated devices.",
]

_VALIDATION_STEPS_LOCAL = [
    "Confirm alert frequency has decreased after initial triage actions.",
    "Verify that impact-path nodes in this diagram are returning to healthy state.",
    "Run ping / traceroute from the first-observed node to the suspected root cause.",
    "Confirm monitoring system shows green health on the root-cause node.",
]

_VALIDATION_STEPS_ENT = [
    "Confirm alert frequency has decreased after initial triage actions.",
    "Verify that impact-path nodes are returning to healthy state across all diagrams.",
    "Run connectivity tests across affected diagram boundaries.",
    "Confirm monitoring system shows green health on root-cause node.",
]

_ROLLBACK_LOCAL = [
    "Capture running config before applying any change to the affected device.",
    "Prepare a rollback config snippet before touching interface or routing settings.",
    "Use a maintenance window if the change affects production traffic on this diagram.",
    "Notify the local site team before modifying firewall or routing policies.",
]

_ROLLBACK_ENTERPRISE = [
    "Capture running configs on all affected devices before any change.",
    "Coordinate maintenance windows across affected sites — changes here have cross-domain blast radius.",
    "Confirm a rollback plan is in place for each affected diagram domain.",
    "Notify NOC before modifying WAN or shared-services components.",
    "Stage changes in the least-critical diagram first; validate before proceeding.",
]


# ── Local scope ───────────────────────────────────────────────────────────────

def _generate_local_template(context: dict) -> dict:
    root_cause        = context.get("root_cause", "")
    diagram_id        = context.get("selected_diagram_id", "")
    first_obs         = context.get("first_observed_node", "")
    imp_nodes         = context.get("impacted_nodes", [])
    impact_path       = context.get("impact_path", [])
    alert_tl          = context.get("alert_timeline", [])
    ranking           = context.get("candidate_ranking", [])
    rca_source        = context.get("rca_source", "")
    n_alerts          = len(alert_tl)
    risk              = _risk_level("local", [diagram_id] if diagram_id else [], alert_tl)
    blast             = _blast_radius("local", [diagram_id] if diagram_id else [], imp_nodes)
    automation        = _automation_eligibility("local", risk, [diagram_id] if diagram_id else [])
    evidence_ids      = _evidence_ids(context)

    root_str = root_cause or "undetermined root cause"
    path_str = " → ".join(str(n) for n in impact_path[:6]) if impact_path else "—"

    exec_sum = (
        f"Local incident on diagram '{diagram_id}'. "
        f"{n_alerts} alert event(s) detected; first observed on {first_obs or '—'}. "
        f"Primary suspected root cause: {root_str}. "
        "Targeted local triage recommended to validate and restore service."
    )

    probable = root_str
    if diagram_id:
        probable += f" (in {diagram_id})"

    top_candidate = ""
    if ranking:
        top_candidate = ranking[0].get("node_id", ranking[0].get("node", root_cause))
    evidence = [
        f"BFS impact path: {path_str}",
        f"First observed node: {first_obs or '—'}",
        f"Top RCA candidate: {top_candidate or root_str}",
        f"Alert events observed: {n_alerts}",
    ]
    if imp_nodes:
        evidence.append(f"Impacted nodes on this diagram: {', '.join(imp_nodes[:5])}")
    # KB evidence near top — before CE-* causal evidence
    evidence.extend(_kb_evidence_refs(context))
    evidence.extend(_causal_evidence_summaries(context))
    for reason in (context.get("correlation_reasons", []) or [])[:5]:
        evidence.append(f"Correlation reason: {reason}")

    triage = _TRIAGE_BY_DIAGRAM.get(diagram_id, _DEFAULT_TRIAGE)[:]
    if root_cause:
        triage.insert(0, f"Focus initial investigation on node '{root_cause}' in {diagram_id}.")

    domain = _infer_domain(root_cause)
    _dom_rem = _REMEDIATION_BY_DOMAIN.get(domain)
    _dom_val = _VALIDATION_BY_DOMAIN.get(domain)

    if _dom_rem:
        remediation = list(_dom_rem)
        remediation.insert(
            0, f"Restore '{root_cause}' in {diagram_id} following domain-specific steps below."
        )
    else:
        remediation = [
            f"Restore '{root_cause}' in {diagram_id} by restarting the affected service "
            "or applying corrective configuration.",
            "Verify interface up/down state and error counters on the root-cause node.",
            "Clear alert conditions on the monitoring platform for this diagram.",
            f"Confirm downstream nodes in the impact path have recovered: {path_str}.",
            "Document the applied change and update the CMDB record.",
        ]
    remediation.extend(_kb_section_steps(context, "Remediation Steps"))

    validation = list(_dom_val) if _dom_val else list(_VALIDATION_STEPS_LOCAL)
    if imp_nodes:
        validation.append(
            "Confirm health of impacted nodes: "
            + ", ".join(imp_nodes[:5])
            + ("…" if len(imp_nodes) > 5 else "") + "."
        )
    validation.extend(_kb_section_steps(context, "Validation Steps"))

    escalation = (
        f"Escalate to network engineering if '{root_cause}' cannot be restored within 30 minutes, "
        "or if the incident shows signs of cross-diagram propagation."
    )

    itsm = {
        "short_description": f"[TOPOLOGY] Network incident on {diagram_id} — root cause: {root_str}",
        "description": (
            f"Local incident on diagram '{diagram_id}'. "
            f"First observed: {first_obs or '—'}. "
            f"Root cause: {root_str}. "
            f"Impact path: {path_str}. "
            f"Impacted nodes: {', '.join(imp_nodes[:5])}{'…' if len(imp_nodes) > 5 else ''}. "
            f"RCA source: {rca_source}. Status: Under investigation."
        ),
        "affected_ci":      root_cause or diagram_id,
        "priority":         "2-High" if n_alerts >= 3 else "3-Medium",
        "assignment_group": "Network Operations — Topology Diagram",
    }

    cluster_id    = context.get("cluster_id", "")
    cluster_score = context.get("cluster_score")
    cluster_str   = f"{cluster_id} (score={cluster_score:.4f})" if cluster_id and cluster_score is not None else cluster_id or "—"
    n_kb = len(context.get("retrieved_kb_evidence", []) or [])
    kb_note = f"SOP/KB evidence retrieved: {n_kb} chunk(s). " if n_kb else ""
    confidence = (
        f"RCA source: {rca_source}. "
        "Topology BFS traversal used — candidate ranking reflects single-diagram graph topology. "
        f"Event correlation cluster: {cluster_str}. "
        f"{kb_note}"
        "Template mode: output is deterministic and not model-generated."
    )
    pre_checks = [
        f"Confirm current alerts still reference {root_cause or first_obs or diagram_id}.",
        f"Review graph path evidence before touching {root_cause or 'the suspected root-cause node'}.",
        "Run read-only reachability and health checks before applying any change.",
    ]
    audit = (
        f"Template remediation plan for {diagram_id}; "
        f"risk={risk}, blast_radius={blast}, automation={automation}. "
        f"Evidence IDs used: {', '.join(evidence_ids) if evidence_ids else 'none'}. "
        f"Event correlation cluster: {cluster_str}. "
        + (f"KB chunks retrieved: {n_kb}." if n_kb else "")
    )

    rollback = list(_ROLLBACK_LOCAL)
    rollback.extend(_kb_section_steps(context, "Rollback / Safety Notes"))

    runbook_chain   = context.get("runbook_chain") or []
    automation_plan = _build_automation_plan(context)

    output = make_remediation_output(
        executive_summary=exec_sum,
        probable_root_cause=probable,
        scope="local",
        risk_level=risk,
        automation_eligibility=automation,
        blast_radius=blast,
        evidence_ids_used=evidence_ids,
        evidence_from_graph=evidence,
        pre_checks=pre_checks,
        triage_steps=triage,
        validation_steps=validation,
        remediation_steps=remediation,
        post_checks=validation,
        do_not_execute_if=_do_not_execute("local", root_cause),
        rollback_or_safety_notes=rollback,
        escalation_recommendation=escalation,
        itsm_ticket_summary=itsm,
        audit_summary=audit,
        confidence_notes=confidence,
        runbook_chain=runbook_chain,
        automation_plan=automation_plan,
    )
    output["source"] = _SOURCE_LABEL
    return output


# ── Enterprise scope ──────────────────────────────────────────────────────────

def _generate_enterprise_template(context: dict) -> dict:
    root_cause    = context.get("root_cause", "")
    rc_diagram    = context.get("root_cause_diagram", "")
    scenario_id   = context.get("scenario_id", "—")
    imp_diagrams  = context.get("impacted_diagrams", [])
    imp_nodes     = context.get("impacted_nodes", [])
    impact_path   = context.get("impact_path", [])
    alert_tl      = context.get("alert_timeline", [])
    ranking       = context.get("candidate_ranking", [])
    rca_source    = context.get("rca_source", "")
    diagram_id    = context.get("selected_diagram_id", "")
    n_alerts      = len(alert_tl)
    n_diags       = len(imp_diagrams)
    risk          = _risk_level("enterprise", imp_diagrams, alert_tl)
    blast         = _blast_radius("enterprise", imp_diagrams, imp_nodes)
    automation    = _automation_eligibility("enterprise", risk, imp_diagrams)
    evidence_ids  = _evidence_ids(context)

    diag_list = ", ".join(imp_diagrams[:4]) if imp_diagrams else "multiple diagrams"
    root_str  = root_cause or "undetermined root cause"
    path_str  = " → ".join(str(n) for n in impact_path[:6]) if impact_path else "—"

    exec_sum = (
        f"Cross-diagram incident affecting {n_diags} topology domain(s) "
        f"({diag_list}). {n_alerts} alert event(s) detected. "
        f"Primary suspected root cause: {root_str} "
        f"in {rc_diagram or 'unknown diagram'}. "
        "Immediate triage recommended to validate and restore service."
    )

    probable = root_str
    if rc_diagram:
        probable += f" (in {rc_diagram})"

    top_candidate = ""
    if ranking:
        top_candidate = ranking[0].get("node_id", ranking[0].get("node", root_cause))
    evidence = [
        f"Alert propagation path: {path_str}",
        f"Top GNN candidate: {top_candidate or root_str}",
        f"Incident spans diagrams: {diag_list}",
        f"Alert events observed: {n_alerts}",
    ]
    if imp_nodes:
        evidence.append(f"Impacted nodes: {', '.join(imp_nodes[:5])}")
    # KB evidence near top — before CE-* causal evidence
    evidence.extend(_kb_evidence_refs(context))
    evidence.extend(_causal_evidence_summaries(context))
    for reason in (context.get("correlation_reasons", []) or [])[:5]:
        evidence.append(f"Correlation reason: {reason}")

    triage = _TRIAGE_BY_DIAGRAM.get(rc_diagram or diagram_id, _DEFAULT_TRIAGE)[:]
    if root_cause:
        triage.insert(0, f"Focus initial investigation on node '{root_cause}' in {rc_diagram or 'the root-cause diagram'}.")

    domain = _infer_domain(root_cause)
    _dom_rem = _REMEDIATION_BY_DOMAIN.get(domain)
    _dom_val = _VALIDATION_BY_DOMAIN.get(domain)

    if _dom_rem:
        remediation = list(_dom_rem)
        remediation.insert(
            0,
            f"Restore '{root_cause}' ({rc_diagram or 'root-cause diagram'}) "
            "following domain-specific steps below.",
        )
    else:
        remediation = [
            f"Restore '{root_cause}' ({rc_diagram or 'root-cause diagram'}) "
            "by restarting the affected service or applying corrective configuration.",
            "Verify device state and error counters on the root-cause node.",
            "Clear alert conditions on the monitoring platform and confirm no secondary triggers.",
            "Document the change applied and update the CMDB record for the affected device.",
        ]
    if n_diags >= 3:
        remediation.append(
            "Coordinate recovery across all affected diagrams in order: "
            + " → ".join(imp_diagrams[:n_diags]) + "."
        )
    remediation.extend(_kb_section_steps(context, "Remediation Steps"))

    validation = list(_dom_val) if _dom_val else list(_VALIDATION_STEPS_ENT)
    if imp_nodes:
        validation.append(
            "Confirm health of all impacted nodes: "
            + ", ".join(imp_nodes[:6])
            + ("…" if len(imp_nodes) > 6 else "") + "."
        )
    validation.extend(_kb_section_steps(context, "Validation Steps"))

    escalation = (
        "Escalate to network engineering team if root-cause node cannot be restored within "
        "30 minutes, or if the incident spans more than 3 topology domains. "
        "Notify on-call SRE if customer-facing services remain impacted."
    )
    if n_diags >= 3:
        escalation = (
            f"This incident spans {n_diags} topology domains — immediate escalation to "
            "senior network engineering is recommended. " + escalation
        )

    itsm = {
        "short_description": (
            f"[ENTERPRISE] Cross-diagram network incident — scenario {scenario_id}; "
            f"root cause: {root_str}"
        ),
        "description": (
            f"Cross-diagram incident on scenario {scenario_id}. "
            f"Root cause: {root_str} ({rc_diagram or 'unknown'}). "
            f"Impact: {n_diags} diagram(s), {n_alerts} alert event(s). "
            f"Impact path: {path_str}. "
            f"Impacted nodes: {', '.join(imp_nodes[:5])}{'…' if len(imp_nodes) > 5 else ''}. "
            f"RCA source: {rca_source}. Status: Under investigation."
        ),
        "affected_ci":      root_cause or diagram_id,
        "priority":         "1-Critical" if n_diags >= 3 else "2-High",
        "assignment_group": "Network Engineering — Enterprise Operations",
    }

    cluster_id    = context.get("cluster_id", "")
    cluster_score = context.get("cluster_score")
    cluster_str   = f"{cluster_id} (score={cluster_score:.4f})" if cluster_id and cluster_score is not None else cluster_id or "—"
    n_kb = len(context.get("retrieved_kb_evidence", []) or [])
    kb_note = f"SOP/KB evidence retrieved: {n_kb} chunk(s). " if n_kb else ""
    confidence = (
        f"RCA source: {rca_source}. "
        + ("GNN inference result was used — candidate ranking is model-derived. "
           if context.get("gnn_result_available") else
           "GNN inference result not available — candidate ranking derived from alert timeline heuristics. ")
        + f"Event correlation cluster: {cluster_str}. "
        + f"{kb_note}"
        + "Template mode: output is deterministic and not model-generated."
    )
    pre_checks = [
        f"Confirm root-cause candidate {root_cause or 'unknown'} is present in enterprise graph memory.",
        f"Validate impacted diagrams before any change: {diag_list}.",
        "Confirm RCA source and alert timeline freshness before remediation.",
        "Run read-only reachability checks across diagram boundaries first.",
    ]
    audit = (
        f"Template remediation plan for scenario {scenario_id}; "
        f"risk={risk}, blast_radius={blast}, automation={automation}, rca_source={rca_source}. "
        f"Evidence IDs used: {', '.join(evidence_ids) if evidence_ids else 'none'}. "
        f"Event correlation cluster: {cluster_str}. "
        + (f"KB chunks retrieved: {n_kb}." if n_kb else "")
    )

    rollback = list(_ROLLBACK_ENTERPRISE)
    rollback.extend(_kb_section_steps(context, "Rollback / Safety Notes"))

    runbook_chain   = context.get("runbook_chain") or []
    automation_plan = _build_automation_plan(context)

    output = make_remediation_output(
        executive_summary=exec_sum,
        probable_root_cause=probable,
        scope="enterprise",
        risk_level=risk,
        automation_eligibility=automation,
        blast_radius=blast,
        evidence_ids_used=evidence_ids,
        evidence_from_graph=evidence,
        pre_checks=pre_checks,
        triage_steps=triage,
        validation_steps=validation,
        remediation_steps=remediation,
        post_checks=validation,
        do_not_execute_if=_do_not_execute("enterprise", root_cause),
        rollback_or_safety_notes=rollback,
        escalation_recommendation=escalation,
        itsm_ticket_summary=itsm,
        audit_summary=audit,
        confidence_notes=confidence,
        runbook_chain=runbook_chain,
        automation_plan=automation_plan,
    )
    output["source"] = _SOURCE_LABEL
    return output


# ── Public API ────────────────────────────────────────────────────────────────

def generate_template_remediation(context: dict) -> dict:
    """Generate a deterministic remediation plan from incident context.

    Dispatches on ``context["scope"]``.  Output is clearly labelled
    "template" — do not present as model-generated.
    """
    scope = context.get("scope", "enterprise")
    if scope == "local":
        return _generate_local_template(context)
    return _generate_enterprise_template(context)
