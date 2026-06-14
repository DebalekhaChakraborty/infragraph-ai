---
kb_id: SOP-DC-FW-001
title: "Datacenter Firewall Packet Drop and Link Error Response"
doc_type: sop
version: "1.4"
owner_group: "Network Engineering — Datacenter Operations"
applies_to_node_types:
  - firewall
  - datacenter_firewall
applies_to_diagrams:
  - datacenter_topology
  - app_db_topology
  - branch_topology
  - shared_services_topology
applies_to_alert_types:
  - packet_drop
  - link_errors
  - latency
  - connection_timeout
  - high_error_rate
rca_patterns:
  - datacenter_firewall_policy_block
  - cross_diagram_packet_loss
  - firewall_session_table_exhaustion
last_reviewed: "2026-03-15"
evidence_tags:
  - DC-FW
  - packet_drop
  - cross_diagram
  - firewall
---

## Purpose

Provide structured triage and remediation steps for incidents where the datacenter firewall (DC-FW-01 or equivalent) is the suspected root cause of packet drops, link errors, or connection timeouts propagating across one or more topology domains.

## Trigger Symptoms

- Alerts of type `packet_drop`, `link_errors`, or `high_error_rate` originating on or upstream of the datacenter firewall.
- Cross-diagram alert correlation: datacenter_topology alerts reflected in app_db_topology and/or branch_topology within the same correlation window.
- Spike in denied connections in firewall deny logs without a corresponding approved change ticket.
- RCA candidate ranking places the datacenter firewall node at rank 1 with GNN confidence above 0.50.
- Session table utilisation on the firewall exceeds 80% concurrent with alert onset.

## Applicable RCA Patterns

- **datacenter_firewall_policy_block**: Stateful firewall inadvertently blocking established connections after a policy reload or ACL change.
- **cross_diagram_packet_loss**: Packet drops on the datacenter firewall propagating latency and timeout alerts into downstream diagrams (app_db, branch, shared_services).
- **firewall_session_table_exhaustion**: Session table capacity exceeded, causing new connection establishment failures without explicit deny-rule hits.

## Pre-Checks

1. Confirm alert timestamps in the active incident match the firewall deny-log timestamps. Do not act on stale data.
2. Verify the firewall node is the active member (not standby) of the HA pair before making configuration changes.
3. Confirm no approved change-management window is currently in progress on the datacenter firewall or adjacent devices.
4. Check whether the same alert pattern occurred within the past 30 days; if so, retrieve the prior incident's record before proceeding.
5. Run read-only health checks: interface error counters, session table utilisation, CPU and memory utilisation.

## Triage Steps

1. Pull the current firewall deny log filtered to the incident time window. Identify the most frequent destination ports and source IP ranges in denied entries.
2. Check interface counters on the DC-FW-01 LAN and WAN interfaces for incremented `input_errors`, `output_drops`, `CRC_errors`.
3. Verify firewall stateful session table utilisation — if above 80%, the firewall may be dropping new connections without an explicit deny rule hit.
4. Confirm HA pair state: both members should show Active/Standby (not Active/Active split-brain).
5. Review routing table downstream of the firewall — confirm no route withdrawal has occurred concurrent with the incident.
6. Cross-correlate the alert timeline: if the first alert in the timeline originated on the firewall node, this is consistent with the firewall as root cause.
7. Inspect audit log for the past 24 hours for policy reloads, ACL changes, or software updates.

## Remediation Steps

1. **If a rogue ACL entry is identified**: Prepare a targeted ACL change, obtain change-management approval, and apply during the next available maintenance window.
2. **If session table exhaustion is confirmed**: Apply a session table cleanup command (vendor-specific) to free stale half-open sessions. Do not reboot the firewall without NOC approval.
3. **If HA pair is in split-brain state**: Follow the enterprise cross-diagram validation runbook before making any further changes.
4. **If a policy reload caused the disruption**: Revert to the prior approved policy configuration snapshot if available.
5. After any change: flush the ARP cache on downstream distribution switches and verify forwarding table entries.
6. Confirm downstream services (app_db_topology, branch_topology) recover within 5–10 minutes of the firewall fix.

## Validation Steps

1. Confirm `packet_drop` and `link_errors` alerts on the firewall node have cleared.
2. Verify app_db_topology: database connection pools should return to normal utilisation within 5 minutes.
3. Verify branch_topology: branch uplink connectivity to datacenter services should be restored.
4. Run traceroute from at least one host in each affected diagram to the datacenter services VLAN.
5. Confirm monitoring shows green health on the datacenter firewall node.
6. Verify no new firewall deny-log entries matching the incident pattern appear for 10 minutes after the fix.

## Rollback / Safety Notes

- Capture a running configuration backup of the firewall before applying any change.
- All changes must be applied under an approved change-management ticket.
- Apply one change at a time; validate; then proceed to the next change.
- If the incident worsens after a change, immediately revert to the previous configuration snapshot.
- Notify NOC and all dependent diagram owners (app_db, branch, shared_services) before making changes.
- Retain all deny-log excerpts and interface counter outputs as evidence in the ITSM record.

## Do Not Execute If

- A maintenance window has not been approved and customer-facing services are live.
- The HA standby member is in a degraded state — changing an HA pair with a failed standby creates a single point of failure.
- The firewall root-cause candidate has a GNN confidence score below 0.35 without strong causal evidence corroboration.
- A parallel change-management window is already in progress on any device in the same traffic path.
- The alert timestamps are more than 2 hours old — validate that the incident is still active before proceeding.

## ITSM Routing

- **Assignment Group**: Network Engineering — Datacenter Operations
- **Category**: Network / Firewall
- **Priority**: 1-Critical if cross-diagram blast radius ≥ 3 topology domains; 2-High otherwise
- **Escalation**: Senior Network Engineer on-call if root cause cannot be confirmed within 30 minutes

## Evidence Tags

`DC-FW`, `datacenter_topology`, `packet_drop`, `link_errors`, `cross_diagram`, `firewall_policy`, `session_table`, `HA_pair`
