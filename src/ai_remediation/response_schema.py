"""
AI remediation input/output schema definitions.

Unified schema for both local (single-diagram) and enterprise (cross-diagram)
remediation plans.  The same JSON structure is used for Qwen output and
template mode, and rendered by a single Streamlit renderer.

Output schema
─────────────
{
  "executive_summary":           str,
  "probable_root_cause":         str,
  "scope":                       "local" | "enterprise",
  "risk_level":                  str,
  "automation_eligibility":      str,
  "blast_radius":                str,
  "evidence_ids_used":           [str, ...],
  "evidence_from_graph":         [str, ...],
  "pre_checks":                  [str, ...],
  "triage_steps":                [str, ...],
  "validation_steps":            [str, ...],
  "remediation_steps":           [str, ...],
  "post_checks":                 [str, ...],
  "do_not_execute_if":           [str, ...],
  "rollback_or_safety_notes":    [str, ...],
  "escalation_recommendation":   str,
  "servicenow_incident_summary": {
      "short_description": str,
      "description":       str,
      "affected_ci":       str,
      "priority":          str,
      "assignment_group":  str
  },
  "audit_summary": str,
  "confidence_notes": str
}
"""
from __future__ import annotations


# ── Input ─────────────────────────────────────────────────────────────────────

def make_remediation_input(
    *,
    incident_id: str = "",
    scope: str = "enterprise",
    selected_diagram_id: str = "",
    diagram_type: str = "",
    scenario_id: str = "",
    alert_timeline: list[dict] | None = None,
    graph_memory_summary: str = "",
    root_cause: str = "",
    root_cause_diagram: str = "",
    first_observed_node: str = "",
    impacted_nodes: list[str] | None = None,
    impacted_diagrams: list[str] | None = None,
    impact_path: list[str] | None = None,
    candidate_ranking: list[dict] | None = None,
    gnn_result_available: bool = False,
    rca_source: str = "",
    device_context: list[dict] | None = None,
    connector_context: list[dict] | None = None,
    interface_context: list[dict] | None = None,
) -> dict:
    """Return a normalised remediation input dict for either scope."""
    return {
        "incident_id":           incident_id,
        "scope":                 scope,
        "selected_diagram_id":   selected_diagram_id,
        "diagram_type":          diagram_type or selected_diagram_id,
        "scenario_id":           scenario_id,
        "alert_timeline":        alert_timeline or [],
        "graph_memory_summary":  graph_memory_summary,
        "root_cause":            root_cause,
        "root_cause_diagram":    root_cause_diagram,
        "first_observed_node":   first_observed_node,
        "impacted_nodes":        impacted_nodes or [],
        "impacted_diagrams":     impacted_diagrams or [],
        "impact_path":           impact_path or [],
        "candidate_ranking":     candidate_ranking or [],
        "gnn_result_available":  gnn_result_available,
        "rca_source":            rca_source,
        "device_context":        device_context or [],
        "connector_context":     connector_context or [],
        "interface_context":     interface_context or [],
    }


# ── Output ────────────────────────────────────────────────────────────────────

def empty_remediation_output(scope: str = "enterprise") -> dict:
    """Return an empty remediation output with all expected keys."""
    return {
        "executive_summary":           "",
        "probable_root_cause":         "",
        "scope":                       scope,
        "risk_level":                  "",
        "automation_eligibility":      "",
        "blast_radius":                "",
        "evidence_ids_used":           [],
        "evidence_from_graph":         [],
        "pre_checks":                  [],
        "triage_steps":                [],
        "validation_steps":            [],
        "remediation_steps":           [],
        "post_checks":                 [],
        "do_not_execute_if":           [],
        "rollback_or_safety_notes":    [],
        "escalation_recommendation":   "",
        "servicenow_incident_summary": {
            "short_description": "",
            "description":       "",
            "affected_ci":       "",
            "priority":          "",
            "assignment_group":  "",
        },
        "audit_summary":               "",
        "confidence_notes":            "",
    }


def make_remediation_output(
    *,
    executive_summary: str = "",
    probable_root_cause: str = "",
    scope: str = "enterprise",
    risk_level: str = "",
    automation_eligibility: str = "",
    blast_radius: str = "",
    evidence_ids_used: list[str] | None = None,
    evidence_from_graph: list[str] | None = None,
    pre_checks: list[str] | None = None,
    triage_steps: list[str] | None = None,
    validation_steps: list[str] | None = None,
    remediation_steps: list[str] | None = None,
    post_checks: list[str] | None = None,
    do_not_execute_if: list[str] | None = None,
    rollback_or_safety_notes: list[str] | None = None,
    escalation_recommendation: str = "",
    servicenow_incident_summary: "dict | None" = None,
    audit_summary: str = "",
    confidence_notes: str = "",
) -> dict:
    """Return a normalised remediation output dict."""
    return {
        "executive_summary":           executive_summary,
        "probable_root_cause":         probable_root_cause,
        "scope":                       scope,
        "risk_level":                  risk_level,
        "automation_eligibility":      automation_eligibility,
        "blast_radius":                blast_radius,
        "evidence_ids_used":           evidence_ids_used or [],
        "evidence_from_graph":         evidence_from_graph or [],
        "pre_checks":                  pre_checks or [],
        "triage_steps":                triage_steps or [],
        "validation_steps":            validation_steps or [],
        "remediation_steps":           remediation_steps or [],
        "post_checks":                 post_checks or validation_steps or [],
        "do_not_execute_if":           do_not_execute_if or [],
        "rollback_or_safety_notes":    rollback_or_safety_notes or [],
        "escalation_recommendation":   escalation_recommendation,
        "servicenow_incident_summary": servicenow_incident_summary or {
            "short_description": "",
            "description":       "",
            "affected_ci":       "",
            "priority":          "",
            "assignment_group":  "",
        },
        "audit_summary":               audit_summary,
        "confidence_notes":            confidence_notes,
    }


# Output schema shown to the model in the prompt
OUTPUT_SCHEMA_TEMPLATE: dict = {
    "executive_summary":        "<1-2 sentence summary of the incident and recommended action>",
    "probable_root_cause":      "<node-id or service identified as root cause>",
    "scope":                    "<local or enterprise>",
    "risk_level":               "<low | medium | high | critical>",
    "automation_eligibility":   "<safe_to_automate | human_approval_required | manual_only>",
    "blast_radius":             "<single_node | single_diagram | cross_diagram | enterprise_wide>",
    "evidence_ids_used":        ["<E1>", "<E2>"],
    "evidence_from_graph":      ["<evidence item 1>", "<evidence item 2>"],
    "pre_checks":               ["<read-only check 1>", "<read-only check 2>"],
    "triage_steps":             ["<step 1>", "<step 2>"],
    "validation_steps":         ["<step 1>", "<step 2>"],
    "remediation_steps":        ["<step 1>", "<step 2>"],
    "post_checks":              ["<post-change validation 1>", "<post-change validation 2>"],
    "do_not_execute_if":        ["<condition that blocks execution>"],
    "rollback_or_safety_notes": ["<safety note 1>", "<safety note 2>"],
    "escalation_recommendation": "<when and to whom to escalate>",
    "servicenow_incident_summary": {
        "short_description": "<one-line incident title>",
        "description":       "<multi-sentence incident detail>",
        "affected_ci":       "<primary affected node or service>",
        "priority":          "<1-Critical / 2-High / 3-Medium>",
        "assignment_group":  "<team responsible for resolution>",
    },
    "audit_summary": "<operator-ready audit note with evidence IDs and approvals>",
    "confidence_notes": "<strength of evidence and known uncertainties>",
}
