"""
Local incident simulation — builds a deterministic, topology-aware alert story
for one selected diagram.  No model training, no random output.
"""
from __future__ import annotations

import hashlib

from .schemas import make_alert_event, make_incident

# ── Topology-specific templates ───────────────────────────────────────────────

_TOPOLOGY_TEMPLATES: dict[str, dict] = {
    "branch_topology": {
        "incident_title":   "Branch application access degradation",
        "severity":         "High",
        "suspected_domain": "WAN / Branch Edge",
        "alert_summary":    "Branch endpoints report application access failures and elevated packet loss.",
        "root_types":       ["cloud_or_wan", "router", "firewall", "switch"],
        "first_obs_types":  ["endpoint", "workstation", "pc", "server"],
        "recommended_actions": [
            "Check WAN / MPLS handoff status and verify circuit health",
            "Validate router uplink interface metrics (packet loss, latency, CRC errors)",
            "Review firewall tunnel and NAT policy logs for recent changes",
            "Notify network operations if upstream packet loss continues",
        ],
    },
    "app_db_topology": {
        "incident_title":   "Application transaction failure",
        "severity":         "Critical",
        "suspected_domain": "Application / Database Tier",
        "alert_summary":    "Application layer reports transaction errors and elevated response latency.",
        "root_types":       ["database", "load_balancer", "switch"],
        "first_obs_types":  ["server", "web_server", "client", "endpoint"],
        "recommended_actions": [
            "Check database connection pool utilization and query latency",
            "Verify load balancer backend health probe responses",
            "Review application error logs for connection exceptions",
            "Restart affected service instances if connection pool is exhausted",
        ],
    },
    "datacenter_topology": {
        "incident_title":   "Data center service degradation",
        "severity":         "High",
        "suspected_domain": "Data Center Fabric",
        "alert_summary":    "Inter-service communication failures detected across the data center fabric.",
        "root_types":       ["core_switch", "firewall", "load_balancer", "router", "switch"],
        "first_obs_types":  ["server", "database", "vm", "application"],
        "recommended_actions": [
            "Check core switch interface error counters and spanning tree state",
            "Review firewall policy change audit logs for recent modifications",
            "Verify load balancer health probe responses from backend servers",
            "Inspect physical link status on affected uplinks and trunk ports",
        ],
    },
    "shared_services_topology": {
        "incident_title":   "Shared service dependency failure",
        "severity":         "High",
        "suspected_domain": "Shared Infrastructure",
        "alert_summary":    "Shared infrastructure service failure is cascading to dependent workloads.",
        "root_types":       ["dns_server", "ntp_server", "iam_server", "monitoring", "switch"],
        "first_obs_types":  ["endpoint", "server", "workstation", "client"],
        "recommended_actions": [
            "Verify DNS server availability and zone file consistency",
            "Check IAM / identity service health and TLS certificate validity",
            "Validate NTP server reachability and stratum level",
            "Review monitoring system logs for cascading service failures",
        ],
    },
    "wan_topology": {
        "incident_title":   "WAN backbone path degradation",
        "severity":         "High",
        "suspected_domain": "WAN Transport",
        "alert_summary":    "Branch sites reporting high latency and packet loss toward the datacenter.",
        "root_types":       ["cloud_or_wan", "router", "switch"],
        "first_obs_types":  ["switch", "router", "endpoint", "server"],
        "recommended_actions": [
            "Check MPLS circuit utilization and carrier status page",
            "Verify BGP session state and routing table completeness",
            "Test failover path reachability from affected branch sites",
            "Contact carrier if physical layer errors are detected",
        ],
    },
}

_DEFAULT_TEMPLATE: dict = {
    "incident_title":   "Network topology incident",
    "severity":         "High",
    "suspected_domain": "Network",
    "alert_summary":    "Network topology incident detected.",
    "root_types":       ["router", "switch", "firewall"],
    "first_obs_types":  ["endpoint", "server", "workstation"],
    "recommended_actions": [
        "Investigate the root cause node connectivity",
        "Review recent configuration changes on the root node",
        "Check interface error counters and logs",
    ],
}

# ── Per-device-type alert message templates ───────────────────────────────────
# Each entry: (human-readable message, alert_type label, default severity)

_ALERT_MESSAGES: dict[str, list[tuple[str, str, str]]] = {
    "router": [
        ("BGP session instability detected — route withdrawals observed", "BGP flap", "critical"),
        ("WAN packet loss elevated above 5% threshold", "WAN degradation", "major"),
        ("Interface utilisation approaching saturation", "Interface alert", "warning"),
    ],
    "switch": [
        ("Downstream interface unreachable — link state down", "Interface down", "critical"),
        ("VLAN adjacency lost on access port", "VLAN unreachable", "major"),
        ("Spanning tree topology change detected", "STP event", "warning"),
    ],
    "firewall": [
        ("Packet drop rate elevated on WAN-facing interface", "Packet drops", "critical"),
        ("Policy deny spike — possible connectivity or routing change", "Policy alert", "major"),
        ("VPN tunnel negotiation failures observed", "VPN instability", "warning"),
    ],
    "server": [
        ("Service health check failing — no response from endpoint", "Service down", "critical"),
        ("CPU utilisation exceeding 95% threshold", "CPU alert", "major"),
        ("Memory pressure rising — GC pause durations increasing", "Memory alert", "warning"),
    ],
    "database": [
        ("Query response time exceeding SLA threshold", "DB latency", "critical"),
        ("Connection pool saturation — new connections rejected", "Connection timeout", "major"),
        ("Replication lag growing on replica nodes", "Replication lag", "warning"),
    ],
    "load_balancer": [
        ("Backend pool health check failures — pool degraded", "Pool unhealthy", "critical"),
        ("5xx error rate spike from origin servers", "Error rate spike", "major"),
        ("Request queue depth increasing beyond threshold", "Queue buildup", "warning"),
    ],
    "cloud_or_wan": [
        ("Upstream BGP session withdrawn by carrier", "BGP session down", "critical"),
        ("ISP circuit packet loss above 3% threshold", "WAN packet loss", "major"),
        ("Latency spike on external peering path", "Latency spike", "warning"),
    ],
    "endpoint": [
        ("Application connection timeout reported by client", "Connection timeout", "critical"),
        ("DNS resolution failures from endpoint", "DNS failure", "major"),
        ("Service unreachable from client perspective", "Service unreachable", "warning"),
    ],
    "workstation": [
        ("Application access failed — connection refused", "Access failure", "critical"),
        ("Network connectivity timeout from workstation", "Connection timeout", "major"),
        ("DNS or proxy failure reported from workstation", "DNS failure", "warning"),
    ],
    "dns_server": [
        ("DNS query failure rate elevated — consumers affected", "DNS failure", "critical"),
        ("Zone transfer failing for authoritative domain", "Zone transfer failure", "major"),
        ("Recursive resolver timeout detected", "Resolver timeout", "warning"),
    ],
    "ntp_server": [
        ("NTP sync failures cascading across dependent consumers", "NTP failure", "critical"),
        ("Stratum level degraded beyond acceptable range", "Stratum degraded", "major"),
        ("Clock drift detected on dependent nodes", "Clock drift", "warning"),
    ],
    "iam_server": [
        ("Identity service auth failures — logins rejected", "Auth failure", "critical"),
        ("Token validation latency exceeding threshold", "Token latency", "major"),
        ("Certificate expiry warning on IAM service", "Cert warning", "warning"),
    ],
    "monitoring": [
        ("Monitoring heartbeats missing from downstream nodes", "Heartbeat missing", "critical"),
        ("Metric collection pipeline stalled", "Collection failure", "major"),
        ("Alert delivery latency elevated", "Alert delay", "warning"),
    ],
}

_DEFAULT_ALERT: list[tuple[str, str, str]] = [
    ("Service connectivity degradation detected", "Connectivity alert", "critical"),
    ("Upstream dependency health check failure", "Health check failure", "major"),
    ("Performance metrics exceeding normal thresholds", "Performance alert", "warning"),
]


# ── Graph helpers ─────────────────────────────────────────────────────────────

def _bfs_path(src: str, dst: str, adj: dict[str, list[str]]) -> list[str]:
    """BFS shortest path from src to dst using the given adjacency list."""
    if src == dst:
        return [src]
    prev: dict[str, str | None] = {src: None}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        for nxt in adj.get(cur, []):
            if nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
                if nxt == dst:
                    queue.clear()
                    break
    if dst not in prev:
        return []
    path: list[str] = []
    cur_: str | None = dst
    while cur_ is not None:
        path.append(cur_)
        cur_ = prev.get(cur_)
    return list(reversed(path))


def _pick_alert(node_type: str, index: int) -> tuple[str, str, str]:
    entries = _ALERT_MESSAGES.get(node_type, _DEFAULT_ALERT)
    return entries[index % len(entries)]


def _incident_id(diagram_id: str, root: str, first_obs: str) -> str:
    raw = f"{diagram_id}::{root}::{first_obs}"
    return "INC-" + hashlib.sha1(raw.encode()).hexdigest()[:8].upper()


def _root_reason(ntype: str, diagram_id: str) -> str:
    reasons: dict[str, str] = {
        "cloud_or_wan":  "WAN handoff is the common upstream dependency for all impacted nodes",
        "router":        "Router aggregates all downstream traffic — on the path of every affected flow",
        "firewall":      "Firewall sits on all flows between impacted nodes and the upstream network",
        "database":      "Database is the shared backend dependency for all application tiers",
        "load_balancer": "Load balancer distributes traffic; its failure impacts every backend",
        "dns_server":    "DNS is a shared upstream dependency — failure cascades to all consumers",
        "ntp_server":    "NTP provides shared timing — failure causes drift across all dependents",
        "switch":        "Core switch is on the forwarding path for all affected inter-node traffic",
        "iam_server":    "IAM service is shared auth dependency — its failure blocks all access",
    }
    return reasons.get(ntype, f"Common upstream dependency for {diagram_id.replace('_', ' ')} nodes")


# ── Public API ────────────────────────────────────────────────────────────────

def build_local_incident(
    local_graph: dict,
    diagram_id: str,
    diagram_type: str | None = None,
) -> dict:
    """Build a deterministic local incident story for one diagram.

    Returns an IncidentScenario dict containing:
    - alert_timeline: ordered list of AlertTimelineEvent dicts
    - root_cause, first_observed_node, impact_path, candidate_ranking
    - reasoning_steps, recommended_actions

    Parameters
    ----------
    local_graph:  Graph dict with ``nodes`` and ``edges`` lists.
    diagram_id:   Identifier for this diagram (e.g. ``branch_topology``).
    diagram_type: Optional override for topology type lookup key.
    """
    nodes_raw = local_graph.get("nodes", [])
    edges_raw = local_graph.get("edges", [])
    topo_key  = diagram_type or diagram_id
    tmpl      = _TOPOLOGY_TEMPLATES.get(topo_key, _DEFAULT_TEMPLATE)

    nodes: dict[str, dict] = {n["id"]: n for n in nodes_raw if n.get("id")}
    node_ids = list(nodes.keys())

    if not node_ids:
        return make_incident(
            incident_id="INC-EMPTY",
            incident_title=tmpl["incident_title"],
            severity=tmpl["severity"],
            scope="local",
            selected_diagram_id=diagram_id,
            scenario_id="",
            suspected_domain=tmpl["suspected_domain"],
            alert_summary=tmpl["alert_summary"],
            alert_timeline=[],
            first_observed_node="",
            root_cause="",
            root_cause_diagram=diagram_id,
            impacted_nodes=[],
            impacted_diagrams=[diagram_id],
            impact_path=[],
            reasoning_steps=["No nodes found in local graph."],
            recommended_actions=tmpl["recommended_actions"],
            rca_source="Scenario-guided graph RCA",
            candidate_ranking=[],
        )

    # Build adjacency lists
    adj_out: dict[str, list[str]] = {n: [] for n in node_ids}
    adj_in:  dict[str, list[str]] = {n: [] for n in node_ids}
    adj_un:  dict[str, list[str]] = {n: [] for n in node_ids}
    for e in edges_raw:
        s, t = e.get("source", ""), e.get("target", "")
        if s in adj_out and t in adj_in:
            adj_out[s].append(t)
            adj_in[t].append(s)
            if t not in adj_un[s]:
                adj_un[s].append(t)
            if s not in adj_un[t]:
                adj_un[t].append(s)

    def _first_matching(type_priority: list[str]) -> str | None:
        for ptype in type_priority:
            for n_id, n_obj in nodes.items():
                ntype = (n_obj.get("type") or n_obj.get("class_name") or "").lower()
                if ntype == ptype:
                    return n_id
        return None

    # Select root and first-observed nodes
    root = _first_matching(tmpl["root_types"])
    if not root:
        root = max(node_ids, key=lambda n: (len(adj_out[n]), -len(adj_in.get(n, []))))

    first_obs = _first_matching(tmpl["first_obs_types"])
    if not first_obs or first_obs == root:
        leaves = [n for n in node_ids if n != root
                  and not adj_out.get(n) and adj_in.get(n)]
        first_obs = leaves[0] if leaves else next(
            (n for n in node_ids if n != root), root
        )

    # Build chronological path: first_obs → root (alerts propagate toward root)
    path_fo_to_root = _bfs_path(first_obs, root, adj_un)
    if not path_fo_to_root:
        # Try reversed directed path
        directed = _bfs_path(root, first_obs, adj_out)
        path_fo_to_root = list(reversed(directed)) if directed else []
    if not path_fo_to_root:
        path_fo_to_root = [first_obs, root] if first_obs != root else [root]

    # Impact path for display: root → first_obs
    impact_path = list(reversed(path_fo_to_root))

    # Impacted nodes: BFS downstream from root
    seen: set[str] = {root}
    queue = list(adj_out.get(root, []))
    impacted: list[str] = []
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        impacted.append(cur)
        queue.extend(adj_out.get(cur, []))
    if not impacted:
        impacted = [n for n in node_ids if n != root][:4]

    # Build alert timeline: walk path first_obs → root (max 5 events)
    timeline_path = path_fo_to_root
    if len(timeline_path) > 5:
        step_size = max(1, (len(timeline_path) - 1) // 4)
        sampled = [timeline_path[i * step_size] for i in range(4)]
        sampled.append(timeline_path[-1])
        timeline_path = sampled

    alert_events: list[dict] = []
    time_step = 0
    for i, node_id in enumerate(timeline_path):
        node_obj = nodes.get(node_id, {})
        ntype = (node_obj.get("type") or node_obj.get("class_name") or "server").lower()
        msg, atype, sev = _pick_alert(ntype, i)
        is_first = node_id == first_obs
        is_root  = node_id == root
        if is_root:
            sev = "critical"
        alert_events.append(make_alert_event(
            step=i + 1,
            time_label=f"T+{time_step:02d}m",
            node=node_id,
            diagram_id=diagram_id,
            device_type=ntype,
            alert_type=atype,
            message=msg,
            severity=sev,
            signal_strength="high" if sev == "critical" else "medium",
            is_first_observed=is_first,
            is_root_signal=is_root,
        ))
        time_step += 5

    # Candidate ranking
    outdeg   = {n: len(adj_out[n]) for n in node_ids}
    indeg    = {n: len(adj_in[n])  for n in node_ids}
    path_set = set(path_fo_to_root)
    imp_set  = set(impacted)
    ranking: list[dict] = []
    for n_id in node_ids:
        score = 0.40 + outdeg[n_id] * 0.11 + (0.14 if indeg[n_id] == 0 else 0.0)
        if n_id == root:
            score += 0.20
        ntype = (nodes[n_id].get("type") or nodes[n_id].get("class_name") or "").lower()
        if n_id == root:
            reason = _root_reason(ntype, diagram_id)
        elif n_id == first_obs:
            reason = "First observed symptom — downstream impact evidence points here"
        elif n_id in path_set:
            reason = f"On the propagation path between {first_obs} and {root}"
        elif n_id in imp_set:
            reason = "Downstream node reachable from root — impacted rather than causal"
        else:
            reason = "Peripheral node — no direct evidence of involvement"
        ranking.append({
            "node": n_id,
            "type": ntype,
            "score": round(min(score, 0.99), 3),
            "reason": reason,
        })
    ranking.sort(key=lambda r: r["score"], reverse=True)

    # Reasoning steps
    path_str      = " → ".join(impact_path[:6]) + ("…" if len(impact_path) > 6 else "")
    imp_preview   = ", ".join(impacted[:3]) + ("…" if len(impacted) > 3 else "")
    top_score_str = str(ranking[0]["score"]) if ranking else "N/A"
    reasoning_steps = [
        f"1. First symptom observed at {first_obs} — "
        f"{(nodes.get(first_obs,{}).get('type') or nodes.get(first_obs,{}).get('class_name','?')).lower()} "
        f"reports connectivity degradation.",
        f"2. Alert propagation traced upstream: {path_str}.",
        f"3. {root} has highest topological reach — zero or minimal in-degree, maximum out-degree.",
        f"4. All impacted nodes ({imp_preview}) are downstream-reachable from {root}.",
        f"5. Candidate ranking confirms {root} as primary root cause (score: {top_score_str}).",
    ]

    return make_incident(
        incident_id=_incident_id(diagram_id, root, first_obs),
        incident_title=tmpl["incident_title"],
        severity=tmpl["severity"],
        scope="local",
        selected_diagram_id=diagram_id,
        scenario_id="",
        suspected_domain=tmpl["suspected_domain"],
        alert_summary=tmpl["alert_summary"],
        alert_timeline=alert_events,
        first_observed_node=first_obs,
        root_cause=root,
        root_cause_diagram=diagram_id,
        impacted_nodes=impacted,
        impacted_diagrams=[diagram_id],
        impact_path=impact_path,
        reasoning_steps=reasoning_steps,
        recommended_actions=tmpl["recommended_actions"],
        rca_source="Scenario-guided graph RCA",
        candidate_ranking=ranking[:6],
    )
