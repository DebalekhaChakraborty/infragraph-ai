"""
Common incident data-factory functions for the InfraGraph AI incident simulation layer.
"""
from __future__ import annotations


def make_alert_event(
    *,
    step: int,
    time_label: str,
    node: str,
    diagram_id: str,
    device_type: str,
    alert_type: str,
    message: str,
    severity: str,
    signal_strength: str = "medium",
    is_first_observed: bool = False,
    is_root_signal: bool = False,
    correlation_role: str = "",
) -> dict:
    """Create an AlertTimelineEvent dict."""
    return {
        "step": step,
        "time_label": time_label,
        "node": node,
        "diagram_id": diagram_id,
        "device_type": device_type,
        "alert_type": alert_type,
        "message": message,
        "severity": severity,
        "signal_strength": signal_strength,
        "is_first_observed": is_first_observed,
        "is_root_signal": is_root_signal,
        "correlation_role": correlation_role,
    }


def make_incident(
    *,
    incident_id: str,
    incident_title: str,
    severity: str,
    scope: str,
    selected_diagram_id: str,
    scenario_id: str,
    suspected_domain: str,
    alert_summary: str,
    alert_timeline: list[dict],
    first_observed_node: str,
    root_cause: str,
    root_cause_diagram: str,
    impacted_nodes: list[str],
    impacted_diagrams: list[str],
    impact_path: list[str],
    reasoning_steps: list[str],
    recommended_actions: list[str],
    rca_source: str,
    candidate_ranking: list[dict],
    propagation_steps: list[dict] | None = None,
) -> dict:
    """Create an IncidentScenario dict."""
    return {
        "incident_id": incident_id,
        "incident_title": incident_title,
        "severity": severity,
        "scope": scope,
        "selected_diagram_id": selected_diagram_id,
        "scenario_id": scenario_id,
        "suspected_domain": suspected_domain,
        "alert_summary": alert_summary,
        "alert_timeline": alert_timeline,
        "first_observed_node": first_observed_node,
        "root_cause": root_cause,
        "root_cause_diagram": root_cause_diagram,
        "impacted_nodes": impacted_nodes,
        "impacted_diagrams": impacted_diagrams,
        "impact_path": impact_path,
        "reasoning_steps": reasoning_steps,
        "recommended_actions": recommended_actions,
        "rca_source": rca_source,
        "candidate_ranking": candidate_ranking,
        "propagation_steps": propagation_steps or [],
    }
