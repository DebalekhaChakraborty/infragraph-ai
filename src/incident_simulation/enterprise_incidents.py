"""
Enterprise incident simulation — builds a cross-diagram alert story for the
selected scenario graph.  Respects priority chain: alerts_data ground truth
→ GNN result → topology-derived simulation.
"""
from __future__ import annotations

import hashlib

from .schemas import make_alert_event, make_incident

# ── Topology ordering ─────────────────────────────────────────────────────────
# Symptom diagrams come first; root-cause diagrams (shared services, WAN) last.
_SYMPTOM_FIRST_ORDER = [
    "branch_topology",
    "wan_topology",
    "app_db_topology",
    "datacenter_topology",
    "shared_services_topology",
]

_DIAGRAM_DOMAIN = {
    "branch_topology":          "Branch Edge",
    "wan_topology":             "WAN Transport",
    "app_db_topology":          "Application / Database Tier",
    "datacenter_topology":      "Data Center Fabric",
    "shared_services_topology": "Shared Infrastructure",
}

# Per-diagram alert messages for when no specific node alerts are available.
# Each tuple: (message, alert_type, default_severity)
_DIAGRAM_ALERT_MSGS: dict[str, list[tuple[str, str, str]]] = {
    "branch_topology": [
        ("Branch endpoints reporting application access timeouts", "Application timeout", "critical"),
        ("Branch router showing elevated WAN packet loss", "WAN degradation", "major"),
    ],
    "wan_topology": [
        ("WAN edge router BGP session instability detected", "BGP instability", "critical"),
        ("MPLS circuit health degraded on primary path", "Circuit degradation", "major"),
    ],
    "app_db_topology": [
        ("Application tier reporting elevated transaction error rate", "Transaction errors", "critical"),
        ("Database connection pool saturation and query timeouts", "DB degradation", "major"),
    ],
    "datacenter_topology": [
        ("Data center fabric showing inter-service communication failures", "Fabric failure", "critical"),
        ("Core switch interface errors and packet drops detected", "Switch errors", "major"),
    ],
    "shared_services_topology": [
        ("DNS resolution failures cascading across dependent services", "DNS failure", "critical"),
        ("IAM / shared service health check failures", "Shared service down", "major"),
    ],
}
_DEFAULT_DIAGRAM_MSGS: list[tuple[str, str, str]] = [
    ("Service connectivity degradation reported", "Connectivity alert", "critical"),
    ("Upstream dependency health check failures", "Health check failure", "major"),
]

# Node-type alert messages for enriching real alert entries
_NODE_TYPE_MSGS: dict[str, str] = {
    "router":        "BGP/WAN session instability — upstream path degradation detected",
    "cloud_or_wan":  "Carrier path packet loss above threshold — upstream BGP withdrawn",
    "firewall":      "Packet drop rate elevated — possible policy or routing change",
    "database":      "Database query latency spike — connection pool saturation",
    "load_balancer": "Backend pool health check failures — 5xx error rate spike",
    "server":        "Service health check failing — connection refused from upstream",
    "switch":        "Interface errors detected — downstream forwarding affected",
    "dns_server":    "DNS query failure rate elevated — consumers unable to resolve",
    "ntp_server":    "NTP sync failures — clock drift detected on dependent nodes",
    "iam_server":    "Identity service auth failures — downstream access blocked",
}


def _incident_id(scenario_id: str, root: str) -> str:
    raw = f"enterprise::{scenario_id}::{root}"
    return "ENT-" + hashlib.sha1(raw.encode()).hexdigest()[:8].upper()


def _enrich_message(ntype: str, diagram_id: str) -> str:
    domain = _DIAGRAM_DOMAIN.get(diagram_id, "")
    prefix = f"[{domain}] " if domain else ""
    return prefix + _NODE_TYPE_MSGS.get(ntype, "Service connectivity degradation detected")


def _enterprise_title(alerts_data: dict, imp_diagrams: list[str]) -> str:
    if alerts_data.get("incident_title"):
        return alerts_data["incident_title"]
    n = len(imp_diagrams)
    return f"Cross-diagram enterprise incident — {n} diagram{'s' if n != 1 else ''} affected"


def _enterprise_severity(alert_list: list[dict]) -> str:
    return "Critical" if any(a.get("severity") == "critical" for a in alert_list) else "High"


def _enterprise_actions(rc_diagram: str, root_cause: str) -> list[str]:
    return [
        f"Investigate {root_cause} in {rc_diagram or 'the root cause diagram'} "
        "for connectivity or configuration changes",
        "Review cross-diagram stitch links for recent topology changes",
        "Check shared service dependencies (DNS, IAM, NTP) that span multiple diagrams",
        "Validate WAN and backbone path health metrics",
        "Correlate syslog and interface metrics across all impacted diagrams",
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def build_enterprise_incident(
    enterprise_graph: dict,
    alerts_data: dict,
    selected_local_diagram: str | None,
    gnn_result: dict | None = None,
) -> dict:
    """Build a cross-diagram alert story for the selected scenario.

    Priority chain:
    1. ``alerts_data`` ground truth if real alerts present.
    2. ``gnn_result`` predicted_root_cause / top_candidates if available.
    3. Diagram-level topology simulation otherwise.

    Returns an IncidentScenario dict with a cross-diagram alert_timeline,
    root_cause, impact_path, reasoning_steps, and candidate_ranking.
    """
    scenario_id  = alerts_data.get("scenario_id", "")
    alert_list   = alerts_data.get("alerts", [])
    root_cause   = alerts_data.get("root_cause", "")
    rc_diagram   = alerts_data.get("root_cause_diagram", "")
    imp_nodes    = list(alerts_data.get("impacted_nodes", []))
    imp_diagrams = list(alerts_data.get("impacted_diagrams", []))
    impact_paths = alerts_data.get("impact_paths", [])
    impact_path  = impact_paths[0] if impact_paths else []

    rca_source = "Scenario-grounded RCA simulation"

    # Override with GNN result when available
    if gnn_result:
        gnn_root = (gnn_result.get("predicted_root_cause")
                    or gnn_result.get("root_cause", ""))
        if gnn_root:
            root_cause = gnn_root
            rca_source = "Enterprise GNN RCA"
        if gnn_result.get("root_cause_diagram"):
            rc_diagram = gnn_result["root_cause_diagram"]
        if gnn_result.get("impacted_diagrams"):
            imp_diagrams = list(gnn_result["impacted_diagrams"])

    # Map alert nodes to their diagrams from alerts_data
    alert_node_diagram: dict[str, str] = {}
    for a in alert_list:
        n = a.get("node", "")
        d = a.get("diagram_id", "") or a.get("source_diagram", "")
        if n and d:
            alert_node_diagram[n] = d

    # Collect all relevant diagrams
    alert_diagrams = sorted(set(alert_node_diagram.values()))
    if not alert_diagrams:
        clusters = enterprise_graph.get("diagram_clusters", {})
        if isinstance(clusters, dict):
            alert_diagrams = sorted(clusters.keys())
        else:
            alert_diagrams = sorted(
                {c.get("diagram_id", "") for c in clusters if isinstance(c, dict)}
            )
    if not imp_diagrams:
        imp_diagrams = alert_diagrams or [selected_local_diagram or ""]

    # Order: symptom-first, then root diagram last
    ordered_diagrams: list[str] = []
    for d in _SYMPTOM_FIRST_ORDER:
        if d in imp_diagrams:
            ordered_diagrams.append(d)
    for d in imp_diagrams:
        if d not in ordered_diagrams:
            ordered_diagrams.append(d)
    if rc_diagram and rc_diagram not in ordered_diagrams:
        ordered_diagrams.append(rc_diagram)

    # Build cross-diagram timeline
    alert_events: list[dict] = []
    step = 1
    time_min = 0
    used_nodes: set[str] = set()

    if alert_list:
        # Group real alerts by diagram and walk in dependency order
        by_diagram: dict[str, list[dict]] = {}
        for a in alert_list:
            d = (a.get("diagram_id", "")
                 or alert_node_diagram.get(a.get("node", ""), "unknown"))
            by_diagram.setdefault(d, []).append(a)

        for diag in ordered_diagrams:
            for a in by_diagram.get(diag, [])[:2]:
                node_id = a.get("node", "")
                if node_id in used_nodes:
                    continue
                used_nodes.add(node_id)
                ntype   = a.get("class", a.get("device_type", "server")).lower()
                sev     = a.get("severity", "major")
                atype   = a.get("alert_type", "Service alert")
                msg     = _enrich_message(ntype, diag)
                is_root = (node_id == root_cause)
                is_fst  = len(alert_events) == 0
                alert_events.append(make_alert_event(
                    step=step,
                    time_label=f"T+{time_min:02d}m",
                    node=node_id,
                    diagram_id=diag,
                    device_type=ntype,
                    alert_type=atype,
                    message=msg,
                    severity="critical" if is_root else sev,
                    signal_strength="high" if is_root or sev == "critical" else "medium",
                    is_first_observed=is_fst,
                    is_root_signal=is_root,
                ))
                step += 1
                time_min += 5
    else:
        # No real alert records — generate from diagram-level templates
        for diag in ordered_diagrams:
            msgs = _DIAGRAM_ALERT_MSGS.get(diag, _DEFAULT_DIAGRAM_MSGS)
            msg_text, atype, sev = msgs[0]
            is_fst = len(alert_events) == 0
            alert_events.append(make_alert_event(
                step=step,
                time_label=f"T+{time_min:02d}m",
                node=f"{diag}",
                diagram_id=diag,
                device_type=_DIAGRAM_DOMAIN.get(diag, "network device"),
                alert_type=atype,
                message=msg_text,
                severity=sev,
                signal_strength="high" if sev == "critical" else "medium",
                is_first_observed=is_fst,
                is_root_signal=(diag == rc_diagram),
            ))
            step += 1
            time_min += 5

    # Ensure root cause node appears with is_root_signal in timeline
    if root_cause and not any(e.get("is_root_signal") for e in alert_events):
        alert_events.append(make_alert_event(
            step=step,
            time_label=f"T+{time_min:02d}m",
            node=root_cause,
            diagram_id=rc_diagram or (selected_local_diagram or "unknown"),
            device_type="router",
            alert_type="Root signal confirmed",
            message=(
                f"Cross-diagram correlation identifies {root_cause} "
                "as common upstream dependency"
            ),
            severity="critical",
            signal_strength="high",
            is_first_observed=False,
            is_root_signal=True,
        ))

    # First observed
    fst_event = next((e for e in alert_events if e.get("is_first_observed")), None)
    first_obs_node = (fst_event or (alert_events[0] if alert_events else {})).get("node", "")
    first_obs_diag = (fst_event or (alert_events[0] if alert_events else {})).get("diagram_id", "")

    # Candidate ranking
    ranking: list[dict] = []
    if gnn_result:
        for c in gnn_result.get("top_candidates", [])[:6]:
            ranking.append({
                "node": c.get("node_id", ""),
                "score": c.get("score", 0.0),
                "reason": f"GNN rank {c.get('rank', '?')}, type={c.get('type', '?')}",
            })
    if not ranking and root_cause:
        ranking = [{
            "node": root_cause,
            "score": 0.97,
            "reason": "scenario ground truth root cause",
        }]
        for a in alert_list[:5]:
            n = a.get("node", "")
            if n and n != root_cause:
                ranking.append({
                    "node": n,
                    "score": round(0.74 - len(ranking) * 0.05, 3),
                    "reason": "alert propagation evidence",
                })

    # Reasoning steps
    n_diags    = len({e.get("diagram_id", "") for e in alert_events})
    diag_names = ", ".join(ordered_diagrams[:4]) + ("…" if len(ordered_diagrams) > 4 else "")
    ip_str     = " → ".join(str(n) for n in impact_path[:5])
    ip_str    += "…" if len(impact_path) > 5 else ""
    reasoning_steps = [
        f"1. First symptom observed at {first_obs_node} "
        f"({first_obs_diag}) — "
        f"{alert_events[0].get('message', '') if alert_events else ''}.",
        f"2. Alert propagation detected across {n_diags} diagrams: {diag_names}.",
        f"3. Cross-diagram correlation identifies common upstream dependency: "
        f"{root_cause} in {rc_diagram or 'unknown'}.",
        (f"4. Impact path: {ip_str}." if ip_str
         else "4. Impact path derived from scenario topology."),
        f"5. RCA source: {rca_source}.",
    ]

    return make_incident(
        incident_id=_incident_id(scenario_id, root_cause),
        incident_title=_enterprise_title(alerts_data, imp_diagrams),
        severity=_enterprise_severity(alert_list),
        scope="enterprise",
        selected_diagram_id=selected_local_diagram or "",
        scenario_id=scenario_id,
        suspected_domain=_DIAGRAM_DOMAIN.get(rc_diagram, "Enterprise Network"),
        alert_summary=(
            alerts_data.get("incident_summary", "")
            or (
                f"Cross-diagram incident across {len(imp_diagrams)} diagrams — "
                f"root cause in {rc_diagram or 'unknown'}."
            )
        ),
        alert_timeline=alert_events,
        first_observed_node=first_obs_node,
        root_cause=root_cause,
        root_cause_diagram=rc_diagram,
        impacted_nodes=imp_nodes,
        impacted_diagrams=imp_diagrams,
        impact_path=impact_path,
        reasoning_steps=reasoning_steps,
        recommended_actions=_enterprise_actions(rc_diagram, root_cause),
        rca_source=rca_source,
        candidate_ranking=ranking[:6],
    )
