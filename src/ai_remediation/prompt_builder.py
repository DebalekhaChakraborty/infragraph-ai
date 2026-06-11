"""
Scope-aware prompt builder for AI remediation.

Builds system + user message pairs for Qwen3 based on whether the incident
is local (single-diagram) or enterprise (cross-diagram).  Both paths return
the same JSON output schema so the renderer is unified.
"""
from __future__ import annotations

import json

from .response_schema import OUTPUT_SCHEMA_TEMPLATE


# ── System messages ────────────────────────────────────────────────────────────

_LOCAL_SYSTEM = (
    "You are a Topology RCA remediation agent for a single infrastructure diagram. "
    "You receive a single network-topology diagram, its alert timeline, the BFS "
    "impact path, and the RCA candidate ranking for that diagram only. "
    "Generate targeted triage, validation, and remediation steps that are "
    "grounded exclusively in the devices, IPs, and topology present in this "
    "single-diagram context. "
    "Every remediation, validation, rollback, pre-check, post-check, and escalation item must be grounded "
    "in known node IDs, diagram IDs, alert timeline entries, RCA result, or retrieved graph evidence. "
    "Do not invent devices, tools, commands, IPs, services, teams, or monitoring systems that are not referenced. "
    "Validation must come before remediation. Rollback/safety notes are mandatory. "
    "If evidence is insufficient, say what evidence is missing instead of inventing. "
    "Use evidence IDs when available. "
    "Do not escalate to enterprise or cross-diagram operations unless the "
    "evidence explicitly shows inter-domain blast radius. "
    "Respond with ONLY a single valid JSON object matching the schema — no markdown fences, "
    "no prose, and no additional keys."
)

_ENTERPRISE_SYSTEM = (
    "You are an enterprise AIOps remediation agent. "
    "You receive a stitched multi-diagram scenario graph, a cross-diagram alert "
    "timeline, GNN correlation scores, and a ranked list of root-cause candidates "
    "spanning multiple topology domains. "
    "Generate enterprise-wide remediation steps that coordinate across all affected "
    "domains and diagram clusters. "
    "Ground every recommendation in the provided node IDs, diagram identifiers, alert timeline, "
    "RCA result, GNN ranking, or retrieved graph evidence. "
    "Do not invent devices, tools, commands, IPs, services, teams, or monitoring systems. "
    "Validation must come before remediation. Rollback/safety notes are mandatory. "
    "If evidence is insufficient, say what evidence is missing instead of inventing. "
    "Use evidence IDs when available. "
    "Include cross-diagram escalation, blast-radius reasoning, explicit approvals, impacted diagram references, "
    "GNN ranking explanation when available, and rollback plans. "
    "Respond with ONLY a single valid JSON object matching the schema — no markdown fences, "
    "no prose, and no additional keys."
)


# ── Shared formatting helpers ──────────────────────────────────────────────────

def _fmt_alert_timeline(events: list) -> str:
    if not events:
        return "  (no alerts)"
    lines = []
    for ev in events:
        t   = ev.get("timestamp", ev.get("time_label", ""))
        nid = ev.get("node_id", ev.get("node", ev.get("source_node", "?")))
        sev = ev.get("severity", "")
        msg = ev.get("message", ev.get("alert_message", ""))
        cr  = ev.get("correlation_role", "")
        diag = ev.get("diagram_id", "")
        tag  = f" [{cr}]" if cr else ""
        dtag = f" ({diag})" if diag else ""
        lines.append(f"  [{t}] {nid}{dtag} ({sev}){tag}: {msg}")
    return "\n".join(lines)


def _fmt_list(items: list, max_items: int = 8) -> str:
    if not items:
        return "  (none)"
    truncated = items[:max_items]
    out = "\n".join(f"  - {x}" for x in truncated)
    if len(items) > max_items:
        out += f"\n  … ({len(items) - max_items} more)"
    return out


def _fmt_ranking(ranking: list) -> str:
    if not ranking:
        return "  (none)"
    lines = []
    for i, r in enumerate(ranking[:6], 1):
        nid   = r.get("node_id", r.get("node", "?"))
        score = r.get("score", r.get("rca_score", "?"))
        diag  = r.get("diagram_id", "")
        rsn   = r.get("reason", "")
        diag_s = f" [{diag}]" if diag else ""
        rsn_s  = f"  {rsn}" if rsn else ""
        lines.append(f"  {i}. {nid}{diag_s} — score {score}{rsn_s}")
    return "\n".join(lines)


def _fmt_device_context(devices: list) -> str:
    if not devices:
        return "  (none)"
    lines = []
    for d in devices[:8]:
        nid   = d.get("node_id", d.get("id", "?"))
        dtype = d.get("device_type", d.get("type", ""))
        ip    = d.get("ip_address", d.get("ip", ""))
        diag  = d.get("diagram_id", "")
        parts = [nid]
        if dtype:
            parts.append(dtype)
        if ip:
            parts.append(ip)
        if diag:
            parts.append(f"[{diag}]")
        lines.append(f"  - {' | '.join(parts)}")
    return "\n".join(lines)


def _fmt_connectors(connectors: list) -> str:
    if not connectors:
        return "  (none)"
    lines = []
    for c in connectors[:8]:
        src  = c.get("source", "?")
        tgt  = c.get("target", "?")
        ctyp = c.get("type", c.get("edge_type", c.get("label", "")))
        diag = c.get("diagram_id", "")
        line = f"  - {src} → {tgt}"
        if ctyp:
            line += f" ({ctyp})"
        if diag:
            line += f" [{diag}]"
        lines.append(line)
    return "\n".join(lines)


def _fmt_retrieved_evidence(evidence: list) -> str:
    if not evidence:
        return "  (none)"
    lines = []
    for i, item in enumerate(evidence[:8], 1):
        meta = item.get("metadata", {}) if isinstance(item, dict) else {}
        evidence_id = meta.get("evidence_id") or f"E{i}"
        source_type = meta.get("source_type", "unknown")
        scenario_id = meta.get("scenario_id", "")
        diagram_id = meta.get("diagram_id", "")
        node_id = meta.get("node_id", "")
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        lines.append(
            f"  - {evidence_id} | {source_type} | scenario={scenario_id} | "
            f"diagram={diagram_id} | node={node_id}: {text}"
        )
    return "\n".join(lines)


# ── Local user message ─────────────────────────────────────────────────────────

def _build_local_user_message(ctx: dict) -> str:
    schema_json = json.dumps(OUTPUT_SCHEMA_TEMPLATE, indent=2)
    return (
        "== TOPOLOGY RCA REMEDIATION REQUEST ==\n\n"
        f"Diagram        : {ctx.get('selected_diagram_id', '—')}\n"
        f"Incident ID    : {ctx.get('incident_id', '—')}\n"
        "Scope          : local (single-diagram)\n\n"
        "--- Alert Timeline ---\n"
        f"{_fmt_alert_timeline(ctx.get('alert_timeline', []))}\n\n"
        "--- RCA Result ---\n"
        f"First observed : {ctx.get('first_observed_node', '—')}\n"
        f"Root cause     : {ctx.get('root_cause', '—')}\n"
        f"RCA source     : {ctx.get('rca_source', '—')}\n\n"
        "Impact path:\n"
        f"{_fmt_list(ctx.get('impact_path', []))}\n\n"
        "Impacted nodes:\n"
        f"{_fmt_list(ctx.get('impacted_nodes', []))}\n\n"
        "--- Candidate Ranking ---\n"
        f"{_fmt_ranking(ctx.get('candidate_ranking', []))}\n\n"
        "--- Device Context ---\n"
        f"{_fmt_device_context(ctx.get('device_context', []))}\n\n"
        "--- Connector Context ---\n"
        f"{_fmt_connectors(ctx.get('connector_context', []))}\n\n"
        "--- Retrieved Graph Memory Evidence ---\n"
        f"{_fmt_retrieved_evidence(ctx.get('retrieved_graph_memory_evidence', []))}\n\n"
        "--- Graph Summary ---\n"
        f"{ctx.get('graph_memory_summary', '—')}\n\n"
        "Grounding rules:\n"
        "- Use only listed node IDs, diagram IDs, alerts, RCA results, and retrieved evidence IDs.\n"
        "- Do not invent tools, commands, IPs, services, or devices.\n"
        "- Validation must come before remediation.\n"
        "- Rollback/safety notes are mandatory.\n"
        "- If evidence is insufficient, say what is missing instead of inventing.\n"
        "- Include evidence IDs such as E1 or E2 whenever retrieved evidence supports a step.\n\n"
        "--- Output Schema (return ONLY this JSON shape) ---\n"
        f"{schema_json}\n\n"
        'Generate the remediation plan. Set "scope" to "local" in your response.'
    )


# ── Enterprise user message ────────────────────────────────────────────────────

def _build_enterprise_user_message(ctx: dict) -> str:
    schema_json = json.dumps(OUTPUT_SCHEMA_TEMPLATE, indent=2)
    gnn_note = (
        "GNN correlation result is available — use node rankings from it."
        if ctx.get("gnn_result_available")
        else "GNN result not available — use alert-timeline and BFS rankings only."
    )
    return (
        "== ENTERPRISE RCA REMEDIATION REQUEST ==\n\n"
        f"Incident ID       : {ctx.get('incident_id', '—')}\n"
        f"Primary diagram   : {ctx.get('selected_diagram_id', '—')}\n"
        f"Root-cause diagram: {ctx.get('root_cause_diagram', ctx.get('selected_diagram_id', '—'))}\n"
        "Scope             : enterprise (cross-diagram)\n"
        f"GNN               : {gnn_note}\n\n"
        "--- Cross-Diagram Alert Timeline ---\n"
        f"{_fmt_alert_timeline(ctx.get('alert_timeline', []))}\n\n"
        "--- RCA Result ---\n"
        f"Root cause : {ctx.get('root_cause', '—')}\n"
        f"RCA source : {ctx.get('rca_source', '—')}\n\n"
        "Impact path:\n"
        f"{_fmt_list(ctx.get('impact_path', []))}\n\n"
        "Impacted nodes:\n"
        f"{_fmt_list(ctx.get('impacted_nodes', []))}\n\n"
        "Impacted diagrams:\n"
        f"{_fmt_list(ctx.get('impacted_diagrams', []))}\n\n"
        "--- Candidate Ranking ---\n"
        f"{_fmt_ranking(ctx.get('candidate_ranking', []))}\n\n"
        "--- Device Context ---\n"
        f"{_fmt_device_context(ctx.get('device_context', []))}\n\n"
        "--- Connector Context ---\n"
        f"{_fmt_connectors(ctx.get('connector_context', []))}\n\n"
        "--- Retrieved Graph Memory Evidence ---\n"
        f"{_fmt_retrieved_evidence(ctx.get('retrieved_graph_memory_evidence', []))}\n\n"
        "--- Graph Memory Summary ---\n"
        f"{ctx.get('graph_memory_summary', '—')}\n\n"
        "Enterprise grounding rules:\n"
        "- Use only listed node IDs, diagram IDs, alerts, RCA results, GNN ranking, and retrieved evidence IDs.\n"
        "- Do not invent tools, commands, IPs, services, or devices.\n"
        "- Validation must come before remediation.\n"
        "- Rollback/safety notes are mandatory.\n"
        "- If evidence is insufficient, say what is missing instead of inventing.\n"
        "- Explain cross-diagram escalation and blast radius when more than one diagram is impacted.\n"
        "- Explain the GNN ranking when GNN is available; otherwise name the scenario-grounded RCA source.\n"
        "- Include evidence IDs such as E1 or E2 whenever retrieved evidence supports a step.\n\n"
        "--- Output Schema (return ONLY this JSON shape) ---\n"
        f"{schema_json}\n\n"
        'Generate the enterprise remediation plan. Set "scope" to "enterprise" in your response.'
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def build_remediation_prompt(context: dict) -> list:
    """
    Return a Qwen3 chat-format messages list for the given context dict.

    Dispatches on ``context["scope"]`` — "local" vs "enterprise".
    Returns [{"role": "system", ...}, {"role": "user", ...}].
    """
    scope = context.get("scope", "enterprise")
    if scope == "local":
        system_msg = _LOCAL_SYSTEM
        user_msg   = _build_local_user_message(context)
    else:
        system_msg = _ENTERPRISE_SYSTEM
        user_msg   = _build_enterprise_user_message(context)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
