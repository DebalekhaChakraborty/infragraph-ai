---
kb_id: RUNBOOK-ENT-XDIAG-001
runbook_id: ENT-XDIAG-001
title: "Enterprise Cross-Diagram Incident Validation Runbook"
doc_type: runbook
version: "1.0"
source: approved_kb_repo
domain: enterprise
owner_group: "Network Engineering — Enterprise Operations"
approval_required: true
automation_eligible: false
execution_mode: manual
applies_to_node_types:
  - any
applies_to_diagrams:
  - datacenter_topology
  - app_db_topology
  - branch_topology
  - wan_topology
  - shared_services_topology
applies_to_alert_types:
  - any
rca_patterns:
  - cross_diagram_cascade
  - enterprise_wide_incident
last_reviewed: "2026-03-20"
evidence_tags:
  - cross_diagram
  - enterprise
  - validation
  - blast_radius
---

## Purpose

Define the validation procedure for enterprise-wide incidents that span two or more topology domains. This runbook is executed in parallel with or immediately after SOP-specific remediation steps to ensure cross-domain blast radius is fully assessed and all affected diagrams are confirmed healthy before incident closure.

## Trigger Symptoms

- GNN RCA result identifies a root cause node with causal evidence spanning multiple topology diagrams.
- Alert correlation clusters contain alerts from ≥ 2 distinct topology domains within a single correlation window.
- Blast radius classification is `cross_diagram` or `enterprise_wide`.
- Incident auto-escalation triggered because more than 2 topology teams are listed as stakeholders.

## Applicable RCA Patterns

- **cross_diagram_cascade**: An upstream failure (firewall, WAN router, database) propagates alerts into multiple dependent diagrams within the correlation time window.
- **enterprise_wide_incident**: Shared services (DNS, IAM, NTP) become unavailable, causing simultaneous alerts across all topology domains.

## Pre-Checks

1. List all topology diagrams with active alerts for this incident — this is the confirmed blast radius scope.
2. Confirm the GNN RCA root cause candidate and verify it appears in the causal evidence for the majority of affected diagrams.
3. Identify the owner/team for each affected topology diagram.
4. Establish an incident bridge call or communication channel if 3 or more teams are involved.
5. Confirm no concurrent incidents are active on the same devices (avoid conflating two separate incidents).

## Triage Steps

1. For each affected diagram, pull the current active alert count and timestamp of the most recent alert.
2. Determine the alert propagation order from the correlation cluster evidence — identify the diagram that received the first alert (root source diagram).
3. Confirm the inter-diagram connectivity path from root source to each affected diagram (e.g., datacenter_topology → app_db_topology → branch_topology).
4. For each diagram: confirm whether the local diagram-level SOP has been initiated or completed.
5. Identify any diagram that is NOT yet in remediation — escalate ownership and initiate that diagram's SOP.
6. Check shared services (DNS, NTP, IAM) — if shared services are affected, their SOP takes priority over diagram-specific SOPs.

## Remediation Steps

1. Execute the SOP for the root-cause diagram first (e.g., SOP-DC-FW-001 for datacenter_topology root cause).
2. Validate the root-cause diagram is recovering before proceeding to downstream diagram remediation.
3. For each downstream affected diagram: follow the domain-specific SOP in blast-radius order (most impacted first).
4. Do not apply changes to multiple diagrams simultaneously unless each change is on a separate, isolated device.
5. After each diagram-level SOP is applied: run the diagram-level validation steps from that SOP before proceeding to the next diagram.

## Validation Steps

1. **Root source diagram**: Confirm all active alerts have cleared (zero active alerts, monitoring shows green).
2. **Downstream diagrams**: For each affected diagram, confirm alert count is zero and services are returning to baseline metrics.
3. Run end-to-end connectivity tests across diagram boundaries:
   - Branch → datacenter: ping/traceroute to datacenter services
   - App-tier → database: synthetic connection and query health check
   - All sites → shared services: DNS resolution, NTP synchronisation check
4. Confirm the event correlation cluster associated with this incident shows no new alerts in the most recent 15-minute window.
5. Obtain explicit confirmation from the owner of each affected diagram that their domain is healthy.
6. Verify that GNN model's top-1 candidate (root cause node) shows healthy status in monitoring.

## Rollback / Safety Notes

- Maintain a single authoritative incident record (ITSM ticket) for the entire cross-diagram incident — do not split into per-diagram tickets during active remediation.
- All diagram-level changes must be recorded in the central ticket with timestamps.
- If remediation of a downstream diagram worsens the root-source diagram, stop immediately and re-evaluate the blast-radius sequencing.
- Shared services changes (DNS, NTP, IAM) require approval from the Shared Services team lead regardless of which SOP initiated the change request.
- Retain the correlation cluster JSON export and GNN RCA output as evidence in the central ticket.

## Do Not Execute If

- Root cause has not been confirmed by GNN RCA or causal evidence — do not begin cross-diagram remediation based on a low-confidence candidate alone.
- Fewer than two diagrams have active alerts — use the diagram-specific SOP directly rather than this runbook.
- A disaster recovery or business continuity plan has already been activated — this runbook is for normal incident response, not DR failover.

## ITSM Routing

- **Assignment Group**: Network Engineering — Enterprise Operations (lead)
- **Co-Assignees**: Owner team for each affected topology diagram
- **Category**: Network / Enterprise Incident
- **Priority**: 1-Critical for blast_radius = enterprise_wide; 2-High for blast_radius = cross_diagram
- **Escalation**: Network Engineering Manager if more than 3 topology domains are impacted or if resolution exceeds 60 minutes

## Evidence Tags

`cross_diagram`, `enterprise_wide`, `blast_radius`, `cascade`, `correlation_cluster`, `gnn_rca`, `shared_services`
