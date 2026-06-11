"""
Enterprise incident simulation — builds cross-diagram alert stories for the
selected scenario graph.

Two public entry points:
  build_enterprise_incident          — original function, unchanged behaviour.
  build_cross_diagram_hero_incident  — always produces 3+ diagram coverage,
                                       guaranteed propagation_steps, per-event
                                       correlation_role.
"""
from __future__ import annotations

import hashlib

from .schemas import make_alert_event, make_incident

# ── Topology ordering ─────────────────────────────────────────────────────────
_SYMPTOM_FIRST_ORDER = [
    "branch_topology",
    "wan_topology",
    "app_db_topology",
    "datacenter_topology",
    "shared_services_topology",
]

_DIAGRAM_DOMAIN: dict[str, str] = {
    "branch_topology":           "Branch Edge",
    "wan_topology":              "WAN Transport",
    "app_db_topology":           "Application / Database Tier",
    "datacenter_topology":       "Data Center Fabric",
    "shared_services_topology":  "Shared Infrastructure",
}

# Chain preference starting from each diagram type.
# Symptom diagrams first → bridge → downstream → root.
_CHAIN_FROM_START: dict[str, list[str]] = {
    "branch_topology": [
        "branch_topology", "wan_topology", "datacenter_topology",
        "app_db_topology", "shared_services_topology",
    ],
    "app_db_topology": [
        "app_db_topology", "datacenter_topology", "shared_services_topology",
        "wan_topology", "branch_topology",
    ],
    "wan_topology": [
        "wan_topology", "branch_topology", "datacenter_topology",
        "app_db_topology", "shared_services_topology",
    ],
    "datacenter_topology": [
        "datacenter_topology", "app_db_topology", "shared_services_topology",
        "wan_topology", "branch_topology",
    ],
    "shared_services_topology": [
        "shared_services_topology", "datacenter_topology", "app_db_topology",
        "wan_topology", "branch_topology",
    ],
}
_DEFAULT_CHAIN = [
    "branch_topology", "wan_topology", "datacenter_topology",
    "app_db_topology", "shared_services_topology",
]

# Node-type groupings used for selecting representative nodes.
_SYMPTOM_NODE_TYPES = [
    "endpoint", "workstation", "web_server", "server", "client", "vm", "application",
]
_BRIDGE_NODE_TYPES = [
    "router", "firewall", "cloud_or_wan", "switch", "load_balancer",
]
_ROOT_NODE_TYPES = [
    "dns_server", "ntp_server", "iam_server", "cloud_or_wan",
    "database", "router", "firewall",
]

# Per-diagram alert templates for simulation.
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

# Node-type alert messages for enriching entries.
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
    "endpoint":      "Application connection timeout — service unreachable from client",
    "workstation":   "Application access failed — connection refused or DNS failure",
    "web_server":    "HTTP/HTTPS request failures — upstream service unreachable",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _incident_id(scenario_id: str, root: str) -> str:
    raw = f"enterprise::{scenario_id}::{root}"
    return "ENT-" + hashlib.sha1(raw.encode()).hexdigest()[:8].upper()


def _enrich_message(ntype: str, diagram_id: str) -> str:
    domain  = _DIAGRAM_DOMAIN.get(diagram_id, "")
    prefix  = f"[{domain}] " if domain else ""
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


def _nodes_by_diagram(enterprise_graph: dict) -> dict[str, list[dict]]:
    """Group enterprise graph nodes by their diagram cluster."""
    by_diag: dict[str, list[dict]] = {}
    clusters = enterprise_graph.get("diagram_clusters", {})
    nodes    = enterprise_graph.get("nodes", [])

    nid_to_diag: dict[str, str] = {}
    if isinstance(clusters, dict):
        for did, cluster_data in clusters.items():
            nids = cluster_data if isinstance(cluster_data, list) else cluster_data.get("node_ids", [])
            for nid in nids:
                nid_to_diag[nid] = did
    elif isinstance(clusters, list):
        for c in clusters:
            did = c.get("diagram_id", "")
            for nid in c.get("node_ids", []):
                nid_to_diag[nid] = did

    for n in nodes:
        nid  = n.get("id", "")
        diag = (nid_to_diag.get(nid)
                or n.get("diagram_id", "")
                or n.get("source_diagram", ""))
        if diag:
            by_diag.setdefault(diag, []).append(n)

    return by_diag


def _pick_node_of_type(nodes: list[dict], preferred_types: list[str]) -> dict | None:
    """Return the first node whose type is in preferred_types (in order)."""
    for ptype in preferred_types:
        for n in nodes:
            if (n.get("type") or n.get("class_name") or "").lower() == ptype:
                return n
    return nodes[0] if nodes else None


def _build_hero_chain(
    available: set[str],
    selected_local: str | None,
    rc_diagram: str,
    alert_diagrams: set[str],
) -> list[str]:
    """Return an ordered list of 3–5 diagrams for the cross-diagram story.

    Priority:
    1. Start from selected_local diagram (or branch as generic default).
    2. Prefer diagrams that already have real alert evidence.
    3. Fill remaining slots with other available diagrams in dependency order.
    4. Move rc_diagram to the last position.
    """
    start    = selected_local or "branch_topology"
    preferred = _CHAIN_FROM_START.get(start, _DEFAULT_CHAIN)

    chain: list[str] = []

    # Diagrams that have real alerts come first (within preferred order)
    for d in preferred:
        if d in alert_diagrams and d in available and d not in chain:
            chain.append(d)
    # Fill with remaining available diagrams
    for d in preferred:
        if d not in chain and d in available:
            chain.append(d)
        if len(chain) >= 5:
            break

    # Always include start even if no nodes mapped (will fall back to diagram name)
    if start not in chain:
        chain.insert(0, start)

    # Move rc_diagram to last position
    if rc_diagram and rc_diagram in chain and chain[-1] != rc_diagram:
        chain.remove(rc_diagram)
        chain.append(rc_diagram)

    return chain[:5]


def _build_synthetic_timeline(
    by_diagram: dict[str, list[dict]],
    chain: list[str],
    root_cause: str,
    rc_diagram: str,
    alert_list: list[dict],
) -> list[dict]:
    """Build 5–7 synthetic cross-diagram alert events from the graph structure.

    Generates two events for the first (symptom) diagram and one for each
    subsequent diagram, plus an extra pre-root event on the root diagram.
    """
    # Real alert nodes by diagram (for node selection preference)
    real_nodes_by_diag: dict[str, list[str]] = {}
    for a in alert_list:
        d = a.get("diagram_id", "") or a.get("source_diagram", "")
        n = a.get("node", "")
        if d and n:
            real_nodes_by_diag.setdefault(d, []).append(n)

    events: list[dict] = []
    step    = 1
    time_min = 0
    n_chain = len(chain)

    for i, diag in enumerate(chain):
        nodes_in_diag = by_diagram.get(diag, [])
        real_nodes    = real_nodes_by_diag.get(diag, [])
        domain        = _DIAGRAM_DOMAIN.get(diag, "")
        domain_pfx    = f"[{domain}] " if domain else ""
        is_root_diag  = (diag == rc_diagram)

        # Determine events to emit for this diagram position
        # First diagram: 2 events (first_observed + local_symptom)
        # Root diagram: 2 events (pre-root + root_signal)
        # Others: 1 event
        if i == 0:
            slots = [("first_observed", _SYMPTOM_NODE_TYPES),
                     ("local_symptom",  _BRIDGE_NODE_TYPES)]
        elif is_root_diag:
            slots = [("downstream_impact", _ROOT_NODE_TYPES),
                     ("root_signal",        _ROOT_NODE_TYPES)]
        elif i == 1:
            slots = [("bridge", _BRIDGE_NODE_TYPES)]
        else:
            slots = [("downstream_impact", _SYMPTOM_NODE_TYPES)]

        for slot_idx, (corr_role, pref_types) in enumerate(slots):
            # Pick a node
            node_obj = None
            if real_nodes and slot_idx == 0:
                real_id  = real_nodes[0]
                node_obj = next((n for n in nodes_in_diag if n.get("id") == real_id), None)
            if not node_obj:
                node_obj = _pick_node_of_type(nodes_in_diag, pref_types)
            if not node_obj and nodes_in_diag:
                node_obj = nodes_in_diag[slot_idx % len(nodes_in_diag)]

            if node_obj:
                node_id = node_obj.get("id", diag)
                ntype   = (node_obj.get("type") or node_obj.get("class_name") or "server").lower()
            else:
                node_id = diag
                ntype   = "network device"

            # Determine if this is actually the root cause node
            is_this_root   = bool(root_cause) and (node_id == root_cause)
            is_root_signal = is_this_root or (corr_role == "root_signal" and is_root_diag)
            is_first       = (len(events) == 0)

            # Build message + alert type
            if corr_role == "first_observed":
                atype = "Service timeout"
                msg   = f"{domain_pfx}User-facing service failure — connection refused or timeout from upstream"
                sev   = "critical"
            elif corr_role == "local_symptom":
                atype = "Network path degradation"
                msg   = f"{domain_pfx}{_NODE_TYPE_MSGS.get(ntype, 'Local dependency degradation detected')}"
                sev   = "major"
            elif corr_role == "bridge":
                atype = "Cross-diagram path failure"
                msg   = (f"{domain_pfx}Incident propagates across diagram boundary — "
                         f"{ntype} node on inter-domain link")
                sev   = "critical"
            elif is_root_signal:
                atype = "Root signal confirmed"
                msg   = (f"{domain_pfx}Cross-diagram correlation confirms this as "
                         "the common upstream dependency for all impacted domains.")
                sev   = "critical"
            else:
                atype = "Downstream impact"
                msg   = f"{domain_pfx}{_NODE_TYPE_MSGS.get(ntype, 'Downstream dependency degradation')}"
                sev   = "major"

            events.append(make_alert_event(
                step=step,
                time_label=f"T+{time_min:02d}m",
                node=node_id,
                diagram_id=diag,
                device_type=ntype,
                alert_type=atype,
                message=msg,
                severity="critical" if is_root_signal or sev == "critical" else sev,
                signal_strength="high" if sev == "critical" or is_root_signal else "medium",
                is_first_observed=is_first,
                is_root_signal=is_root_signal,
                correlation_role="root_signal" if is_root_signal else corr_role,
            ))
            step     += 1
            time_min += 3 if slot_idx == 0 else 2

        time_min += 1  # gap between diagrams

    return events


def _build_timeline_from_alerts(
    alert_list: list[dict],
    chain: list[str],
    root_cause: str,
    rc_diagram: str,
) -> list[dict]:
    """Build an enriched timeline from real alert records (3+ diagram case).

    Adds correlation_role to each event.  At most 2 events per diagram,
    max 7 total.
    """
    alert_node_diagram: dict[str, str] = {}
    for a in alert_list:
        n = a.get("node", "")
        d = a.get("diagram_id", "") or a.get("source_diagram", "")
        if n and d:
            alert_node_diagram[n] = d

    by_diagram: dict[str, list[dict]] = {}
    for a in alert_list:
        d = (a.get("diagram_id", "")
             or alert_node_diagram.get(a.get("node", ""), "unknown"))
        by_diagram.setdefault(d, []).append(a)

    events: list[dict]    = []
    step     = 1
    time_min = 0
    used: set[str] = set()
    n_chain  = len(chain)

    for ci, diag in enumerate(chain):
        for a in by_diagram.get(diag, [])[:2]:
            if len(events) >= 7:
                break
            node_id = a.get("node", "")
            if node_id in used:
                continue
            used.add(node_id)
            ntype      = a.get("class", a.get("device_type", "server")).lower()
            sev        = a.get("severity", "major")
            atype      = a.get("alert_type", "Service alert")
            msg        = _enrich_message(ntype, diag)
            is_root    = (node_id == root_cause)
            is_first   = (len(events) == 0)

            # Assign correlation_role by chain position
            if is_root or (diag == rc_diagram and ci == n_chain - 1 and len(by_diagram.get(diag, [])) < 2):
                corr_role = "root_signal"
            elif ci == 0 and len(events) == 0:
                corr_role = "first_observed"
            elif ci == 0:
                corr_role = "local_symptom"
            elif ci == 1:
                corr_role = "bridge"
            else:
                corr_role = "downstream_impact"

            events.append(make_alert_event(
                step=step,
                time_label=f"T+{time_min:02d}m",
                node=node_id,
                diagram_id=diag,
                device_type=ntype,
                alert_type=atype,
                message=msg,
                severity="critical" if is_root else sev,
                signal_strength="high" if is_root or sev == "critical" else "medium",
                is_first_observed=is_first,
                is_root_signal=is_root,
                correlation_role="root_signal" if is_root else corr_role,
            ))
            step     += 1
            time_min += 3

    return events


def _build_propagation_steps(
    alert_events: list[dict],
    chain: list[str],
    root_cause: str,
    rc_diagram: str,
) -> list[dict]:
    """Derive 5 propagation steps from the alert timeline."""

    def _find(role: str) -> dict | None:
        return next((e for e in alert_events if e.get("correlation_role") == role), None)

    first_ev      = next((e for e in alert_events if e.get("is_first_observed")), None)
    local_ev      = _find("local_symptom")
    bridge_ev     = _find("bridge")
    downstream_ev = _find("downstream_impact")
    root_ev       = next((e for e in alert_events if e.get("is_root_signal")), None)

    slots = [
        (first_ev,      "First observed symptom",
         "User-facing failure detected at the ingress point of the incident."),
        (local_ev,      "Local dependency expansion",
         "Alert propagates through local network layer toward upstream."),
        (bridge_ev,     "Cross-diagram bridge",
         "Incident crosses diagram boundary — inter-domain link implicated."),
        (downstream_ev, "Downstream enterprise impact",
         "Downstream topology domain begins reporting cascading failures."),
        (root_ev,       "Root cause ranked",
         "Cross-diagram correlation identifies the common upstream dependency."),
    ]

    steps: list[dict] = []
    step_num = 1
    for ev, title, desc in slots:
        if ev:
            steps.append({
                "step":        step_num,
                "title":       title,
                "node":        ev.get("node", ""),
                "diagram_id":  ev.get("diagram_id", ""),
                "description": desc,
            })
            step_num += 1

    return steps


def _build_ranking(
    gnn_result: dict | None,
    alert_list: list[dict],
    root_cause: str,
    by_diagram: dict[str, list[dict]],
    rc_diagram: str,
) -> list[dict]:
    if gnn_result:
        ranking = []
        for c in gnn_result.get("top_candidates", [])[:6]:
            ranking.append({
                "node":   c.get("node_id", ""),
                "score":  c.get("score", 0.0),
                "reason": f"GNN rank {c.get('rank', '?')}, type={c.get('type', '?')}",
            })
        if ranking:
            return ranking

    if not root_cause:
        return []

    ranking = [{
        "node":   root_cause,
        "score":  0.97,
        "reason": "scenario ground truth root cause — highest cross-diagram reach",
    }]
    seen = {root_cause}
    for a in alert_list[:8]:
        n = a.get("node", "")
        if n and n not in seen:
            seen.add(n)
            ranking.append({
                "node":   n,
                "score":  round(0.74 - len(ranking) * 0.06, 3),
                "reason": "alert propagation evidence",
            })
    return ranking[:6]


# ── Public API ────────────────────────────────────────────────────────────────

def build_cross_diagram_hero_incident(
    enterprise_graph: dict,
    alerts_data: dict,
    selected_local_diagram: str | None,
    gnn_result: dict | None = None,
) -> dict:
    """Build a guaranteed cross-diagram hero incident for GNN RCA storytelling.

    Behaviour:
    - If ``alerts_data`` already covers ≥ 3 distinct diagrams, enriches those
      alerts with correlation_role and propagation_steps.
    - Otherwise synthesises a cross-diagram story from the enterprise graph
      topology, targeting 3–5 affected diagrams and 5–7 alert events.
    - Never modifies ground-truth files.
    - GNN result, when present, overrides root_cause / rc_diagram and sets
      rca_source = "Enterprise GNN RCA".

    Returns an IncidentScenario dict with:
      alert_timeline (5–7 events), propagation_steps (≤ 5 steps),
      impacted_diagrams (≥ 3 when possible), and all standard fields.
    """
    scenario_id  = alerts_data.get("scenario_id", "")
    alert_list   = alerts_data.get("alerts", [])
    root_cause   = alerts_data.get("root_cause", "")
    rc_diagram   = alerts_data.get("root_cause_diagram", "")
    imp_diagrams = list(alerts_data.get("impacted_diagrams", []))
    impact_paths = alerts_data.get("impact_paths", [])
    impact_path  = impact_paths[0] if impact_paths else []

    rca_source = "Scenario-grounded RCA simulation"

    # Override with GNN result
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

    # Discover diagram clusters from graph
    by_diagram     = _nodes_by_diagram(enterprise_graph)
    available_diags = set(by_diagram.keys())

    # Supplement from node diagram_id fields if clusters not mapped
    if not available_diags:
        for n in enterprise_graph.get("nodes", []):
            d = n.get("diagram_id", "") or n.get("source_diagram", "")
            if d:
                available_diags.add(d)

    # Count diagrams in real alerts
    alert_diagrams: set[str] = set()
    for a in alert_list:
        d = a.get("diagram_id", "") or a.get("source_diagram", "")
        if d:
            alert_diagrams.add(d)

    hero_needs_synthesis = len(alert_diagrams) < 3

    # Build ordered chain of 3–5 affected diagrams
    chain = _build_hero_chain(available_diags, selected_local_diagram, rc_diagram, alert_diagrams)

    # If chain is tiny (graph has few clusters), pad with known diagram names
    if len(chain) < 3:
        for d in _DEFAULT_CHAIN:
            if d not in chain:
                chain.append(d)
            if len(chain) >= 3:
                break

    # Build timeline
    if not hero_needs_synthesis:
        alert_events = _build_timeline_from_alerts(alert_list, chain, root_cause, rc_diagram)
    else:
        alert_events = _build_synthetic_timeline(
            by_diagram, chain, root_cause, rc_diagram, alert_list,
        )

    # Ensure root cause appears as root_signal
    if root_cause and not any(e.get("is_root_signal") for e in alert_events):
        rc_nodes  = by_diagram.get(rc_diagram, []) if rc_diagram else []
        rc_node   = next((n for n in rc_nodes if n.get("id") == root_cause), None)
        if not rc_node and rc_nodes:
            rc_node = _pick_node_of_type(rc_nodes, _ROOT_NODE_TYPES)
        rc_node_id = rc_node.get("id", root_cause) if rc_node else root_cause
        rc_ntype   = (rc_node.get("type") or "router").lower() if rc_node else "router"
        domain_pfx = f"[{_DIAGRAM_DOMAIN.get(rc_diagram, '')}] " if rc_diagram else ""
        alert_events.append(make_alert_event(
            step=len(alert_events) + 1,
            time_label=f"T+{len(alert_events) * 3:02d}m",
            node=rc_node_id,
            diagram_id=rc_diagram or (chain[-1] if chain else "unknown"),
            device_type=rc_ntype,
            alert_type="Root signal confirmed",
            message=(
                f"{domain_pfx}Cross-diagram correlation confirms {rc_node_id} "
                "as the common upstream dependency across all impacted domains."
            ),
            severity="critical",
            signal_strength="high",
            is_first_observed=False,
            is_root_signal=True,
            correlation_role="root_signal",
        ))

    # Derive key metadata from final timeline
    fst_ev          = next((e for e in alert_events if e.get("is_first_observed")),
                           alert_events[0] if alert_events else {})
    first_obs_node  = fst_ev.get("node", "")
    first_obs_diag  = fst_ev.get("diagram_id", "")

    all_diag_ids = list(dict.fromkeys(e.get("diagram_id", "") for e in alert_events if e.get("diagram_id")))
    if rc_diagram and rc_diagram not in all_diag_ids:
        all_diag_ids.append(rc_diagram)

    # Propagation steps
    prop_steps = _build_propagation_steps(alert_events, chain, root_cause, rc_diagram)

    # Impact path (from alerts_data or derived from propagation steps)
    if not impact_path and prop_steps:
        impact_path = [s["node"] for s in prop_steps if s.get("node")]

    # Candidate ranking
    ranking = _build_ranking(gnn_result, alert_list, root_cause, by_diagram, rc_diagram)

    # Reasoning steps
    n_diags    = len(all_diag_ids)
    diag_names = ", ".join(all_diag_ids[:4]) + ("…" if len(all_diag_ids) > 4 else "")
    ip_str     = " → ".join(str(n) for n in impact_path[:5])
    ip_str    += "…" if len(impact_path) > 5 else ""
    reasoning_steps = [
        f"1. First symptom at {first_obs_node} ({first_obs_diag}) — "
        + (fst_ev.get("message", "") if fst_ev else ""),
        f"2. Alert propagation across {n_diags} diagrams: {diag_names}.",
        f"3. Cross-diagram correlation identifies: {root_cause or '(root pending)'} "
        f"in {rc_diagram or 'unknown'}.",
        (f"4. Impact path: {ip_str}." if ip_str
         else "4. Impact path derived from scenario topology."),
        f"5. RCA source: {rca_source}.",
    ]

    incident = make_incident(
        incident_id=_incident_id(scenario_id, root_cause),
        incident_title=f"Cross-diagram enterprise incident — {n_diags} diagrams affected",
        severity="Critical" if n_diags >= 3 else "High",
        scope="enterprise",
        selected_diagram_id=selected_local_diagram or "",
        scenario_id=scenario_id,
        suspected_domain=_DIAGRAM_DOMAIN.get(rc_diagram, "Enterprise Network"),
        alert_summary=(
            alerts_data.get("incident_summary", "")
            or f"Enterprise-wide fault propagating across {n_diags} topology domains."
        ),
        alert_timeline=alert_events,
        first_observed_node=first_obs_node,
        root_cause=root_cause,
        root_cause_diagram=rc_diagram,
        impacted_nodes=list({e["node"] for e in alert_events if e.get("node")}),
        impacted_diagrams=all_diag_ids,
        impact_path=impact_path,
        reasoning_steps=reasoning_steps,
        recommended_actions=_enterprise_actions(rc_diagram, root_cause),
        rca_source=rca_source,
        candidate_ranking=ranking,
        propagation_steps=prop_steps,
    )
    # Add first_observed_diagram (not in schema, tacked on for UI convenience)
    incident["first_observed_diagram"] = first_obs_diag
    return incident


def build_enterprise_incident(
    enterprise_graph: dict,
    alerts_data: dict,
    selected_local_diagram: str | None,
    gnn_result: dict | None = None,
) -> dict:
    """Build a cross-diagram alert story for the selected scenario.

    Original function — behaviour unchanged.  Prefer
    build_cross_diagram_hero_incident for GNN RCA storytelling.
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

    alert_node_diagram: dict[str, str] = {}
    for a in alert_list:
        n = a.get("node", "")
        d = a.get("diagram_id", "") or a.get("source_diagram", "")
        if n and d:
            alert_node_diagram[n] = d

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

    ordered_diagrams: list[str] = []
    for d in _SYMPTOM_FIRST_ORDER:
        if d in imp_diagrams:
            ordered_diagrams.append(d)
    for d in imp_diagrams:
        if d not in ordered_diagrams:
            ordered_diagrams.append(d)
    if rc_diagram and rc_diagram not in ordered_diagrams:
        ordered_diagrams.append(rc_diagram)

    alert_events: list[dict] = []
    step     = 1
    time_min = 0
    used_nodes: set[str] = set()

    if alert_list:
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
                step     += 1
                time_min += 5
    else:
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
            step     += 1
            time_min += 5

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

    fst_event      = next((e for e in alert_events if e.get("is_first_observed")), None)
    first_obs_node = (fst_event or (alert_events[0] if alert_events else {})).get("node", "")

    ranking: list[dict] = []
    if gnn_result:
        for c in gnn_result.get("top_candidates", [])[:6]:
            ranking.append({
                "node":   c.get("node_id", ""),
                "score":  c.get("score", 0.0),
                "reason": f"GNN rank {c.get('rank', '?')}, type={c.get('type', '?')}",
            })
    if not ranking and root_cause:
        ranking = [{"node": root_cause, "score": 0.97, "reason": "scenario ground truth root cause"}]
        for a in alert_list[:5]:
            n = a.get("node", "")
            if n and n != root_cause:
                ranking.append({
                    "node":   n,
                    "score":  round(0.74 - len(ranking) * 0.05, 3),
                    "reason": "alert propagation evidence",
                })

    n_diags    = len({e.get("diagram_id", "") for e in alert_events})
    diag_names = ", ".join(ordered_diagrams[:4]) + ("…" if len(ordered_diagrams) > 4 else "")
    ip_str     = " → ".join(str(n) for n in impact_path[:5])
    ip_str    += "…" if len(impact_path) > 5 else ""
    reasoning_steps = [
        f"1. First symptom observed at {first_obs_node} "
        f"({(fst_event or {}).get('diagram_id', '')}): "
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
