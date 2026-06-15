"""
kb_schema.py — In-code runbook/SOP catalog for InfraGraph AI demo.

Covers: firewall, router/WAN, load balancer, database, server, storage,
        DNS/shared services, cross-diagram enterprise incidents.
No external files required — the catalog is defined here.
"""
from __future__ import annotations

RUNBOOK_CATALOG: list[dict] = [
    # ── Security / Firewall ───────────────────────────────────────────────────
    {
        "runbook_id":            "SEC-FW-001",
        "title":                 "Firewall Policy Violation — Emergency Isolation",
        "domain":                "firewall",
        "applicable_node_types": ["firewall"],
        "applicable_alert_types": ["auth_errors", "packet_drop", "connection_timeout"],
        "risk_level":            "high",
        "approval_required":     True,
        "automation_eligible":   False,
        "dry_run_supported":     False,
        "tool_name":             "",
        "evidence_ids":          ["RB-SEC-FW-001-000", "RB-SEC-FW-001-001"],
        "steps": [
            "Verify firewall rule-set for unauthorized policy change (read-only).",
            "Check access control logs for auth_errors in the last 60 min.",
            "Isolate affected VLAN segment — requires CAB approval.",
            "Re-apply baseline policy from approved config backup.",
            "Post-change: confirm packet forwarding restored on all uplinks.",
        ],
    },
    {
        "runbook_id":            "SEC-FW-002",
        "title":                 "Firewall ACL Review and Rollback",
        "domain":                "firewall",
        "applicable_node_types": ["firewall"],
        "applicable_alert_types": ["auth_errors", "link_errors", "packet_drop"],
        "risk_level":            "medium",
        "approval_required":     True,
        "automation_eligible":   False,
        "dry_run_supported":     True,
        "tool_name":             "",
        "evidence_ids":          ["RB-SEC-FW-002-000"],
        "steps": [
            "Pull current ACL config (dry-run: show running-config).",
            "Diff against last-known-good config in version control.",
            "Rollback to approved ACL snapshot — dry-run first.",
            "Validate connectivity from branch to datacenter after rollback.",
        ],
    },
    # ── Network / Router / WAN ────────────────────────────────────────────────
    {
        "runbook_id":            "NET-RTR-001",
        "title":                 "BGP Route Flap — WAN Stabilisation",
        "domain":                "router",
        "applicable_node_types": ["router", "wan"],
        "applicable_alert_types": ["route_flap", "packet_drop", "link_errors"],
        "risk_level":            "high",
        "approval_required":     True,
        "automation_eligible":   False,
        "dry_run_supported":     False,
        "tool_name":             "",
        "evidence_ids":          ["RB-NET-RTR-001-000", "RB-NET-RTR-001-001"],
        "steps": [
            "Check BGP peer state on affected router: show bgp summary.",
            "Verify physical/optical layer for link errors and CRC counts.",
            "Increase BGP hold-timer to suppress transient flaps (CAB approval).",
            "Redistribute traffic to backup WAN path.",
            "Restore primary path after BGP session stable > 5 min.",
        ],
    },
    {
        "runbook_id":            "NET-WAN-001",
        "title":                 "WAN Link Degradation — Failover",
        "domain":                "wan",
        "applicable_node_types": ["wan", "router"],
        "applicable_alert_types": ["latency", "packet_drop", "link_errors"],
        "risk_level":            "medium",
        "approval_required":     True,
        "automation_eligible":   True,
        "dry_run_supported":     True,
        "tool_name":             "wan_controller_api",
        "evidence_ids":          ["RB-NET-WAN-001-000"],
        "steps": [
            "Run latency/jitter test to WAN provider PoP.",
            "Check WAN circuit utilisation (dry-run threshold check).",
            "Trigger SD-WAN failover to secondary link if > 30% packet loss.",
            "Open ISP ticket if physical layer degradation confirmed.",
        ],
    },
    # ── Application / Load Balancer ───────────────────────────────────────────
    {
        "runbook_id":            "APP-LB-001",
        "title":                 "Load Balancer Backend Pool Unhealthy",
        "domain":                "load_balancer",
        "applicable_node_types": ["load_balancer"],
        "applicable_alert_types": ["backend_pool_unhealthy", "user_timeout", "latency"],
        "risk_level":            "high",
        "approval_required":     True,
        "automation_eligible":   True,
        "dry_run_supported":     True,
        "tool_name":             "lb_api",
        "evidence_ids":          ["RB-APP-LB-001-000", "RB-APP-LB-001-001"],
        "steps": [
            "List unhealthy backend members from LB health monitor.",
            "Probe backend servers for CPU/memory saturation.",
            "Mark failed backends as MAINT (dry-run first).",
            "Redirect live traffic to healthy pool members.",
            "Re-add replacement backends after health checks pass (> 2 min stable).",
        ],
    },
    # ── Data / Database ───────────────────────────────────────────────────────
    {
        "runbook_id":            "DB-REPL-001",
        "title":                 "Database Replication Lag — Recovery",
        "domain":                "database",
        "applicable_node_types": ["database"],
        "applicable_alert_types": ["latency", "connection_timeout", "dependency_error"],
        "risk_level":            "high",
        "approval_required":     True,
        "automation_eligible":   False,
        "dry_run_supported":     False,
        "tool_name":             "",
        "evidence_ids":          ["RB-DB-REPL-001-000"],
        "steps": [
            "Check replication lag on replica nodes: SHOW SLAVE STATUS.",
            "Identify long-running queries blocking replication threads.",
            "Kill blocking queries after DBA approval.",
            "Monitor lag — target < 5 sec before read routing resumes.",
            "Restore read-replica routing only after sustained lag < 2 sec.",
        ],
    },
    # ── Infrastructure / Server ───────────────────────────────────────────────
    {
        "runbook_id":            "INF-SRV-001",
        "title":                 "Server I/O Saturation — Resource Recovery",
        "domain":                "server",
        "applicable_node_types": ["server", "compute"],
        "applicable_alert_types": ["cpu", "latency", "user_timeout"],
        "risk_level":            "medium",
        "approval_required":     True,
        "automation_eligible":   True,
        "dry_run_supported":     True,
        "tool_name":             "compute_api",
        "evidence_ids":          ["RB-INF-SRV-001-000"],
        "steps": [
            "Check CPU/memory/disk utilisation with read-only monitoring.",
            "Identify top resource-consuming processes.",
            "Rebalance workload to healthy compute nodes (dry-run).",
            "Scale up resources if auto-scaling is in approved policy.",
        ],
    },
    # ── Infrastructure / Storage ──────────────────────────────────────────────
    {
        "runbook_id":            "INF-STG-001",
        "title":                 "Storage Mount Failure — NFS/SAN Recovery",
        "domain":                "storage",
        "applicable_node_types": ["storage"],
        "applicable_alert_types": ["latency", "connection_timeout", "dependency_error"],
        "risk_level":            "high",
        "approval_required":     True,
        "automation_eligible":   False,
        "dry_run_supported":     False,
        "tool_name":             "",
        "evidence_ids":          ["RB-INF-STG-001-000"],
        "steps": [
            "Check NFS/SAN mount status on affected servers.",
            "Verify storage controller reachability and array health.",
            "Remount volumes after confirming data integrity.",
            "Validate filesystem consistency before re-enabling workloads.",
        ],
    },
    # ── Shared Services / DNS ─────────────────────────────────────────────────
    {
        "runbook_id":            "SVC-DNS-001",
        "title":                 "DNS / Shared Services Resolution Failure",
        "domain":                "shared_services",
        "applicable_node_types": ["service", "gateway", "dns"],
        "applicable_alert_types": ["connection_timeout", "dependency_error", "user_timeout"],
        "risk_level":            "high",
        "approval_required":     True,
        "automation_eligible":   False,
        "dry_run_supported":     True,
        "tool_name":             "",
        "evidence_ids":          ["RB-SVC-DNS-001-000"],
        "steps": [
            "Check DNS resolver availability from multiple VLANs (read-only).",
            "Verify DNS zone files for missing or incorrect records.",
            "Restart DNS resolver service after configuration validation.",
            "Test resolution of all critical service FQDNs.",
        ],
    },
    # ── Enterprise / Cross-Diagram ────────────────────────────────────────────
    {
        "runbook_id":            "ENT-XDIAG-001",
        "title":                 "Cross-Diagram Enterprise Incident — Coordinated Recovery",
        "domain":                "enterprise",
        "applicable_node_types": [
            "firewall", "router", "switch", "load_balancer", "database",
            "server", "storage", "wan", "gateway", "service",
        ],
        "applicable_alert_types": [
            "packet_drop", "connection_timeout", "latency",
            "route_flap", "backend_pool_unhealthy", "dependency_error",
            "auth_errors", "link_errors",
        ],
        "risk_level":            "critical",
        "approval_required":     True,
        "automation_eligible":   False,
        "dry_run_supported":     False,
        "tool_name":             "",
        "evidence_ids":          ["RB-ENT-XDIAG-001-000", "RB-ENT-XDIAG-001-001"],
        "steps": [
            "Convene cross-team war room: Network, App, DB, Security.",
            "Establish blast radius — identify all impacted topology domains.",
            "Isolate root-cause node from cross-diagram traffic (CAB approval).",
            "Apply domain-specific runbooks per affected diagram cluster.",
            "Restore services in dependency order: network → compute → app → data.",
            "Validate cross-diagram connectivity before closing the incident.",
        ],
    },
]


def get_runbook_by_id(runbook_id: str) -> dict | None:
    for rb in RUNBOOK_CATALOG:
        if rb["runbook_id"] == runbook_id:
            return rb
    return None
