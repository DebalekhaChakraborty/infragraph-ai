---
kb_id: SOP-WAN-001
title: "WAN Provider Edge Link Down and High Error Rate Response"
doc_type: sop
version: "1.1"
owner_group: "Network Engineering — WAN Operations"
applies_to_node_types:
  - wan_router
  - provider_edge
  - pe_router
applies_to_diagrams:
  - wan_topology
  - branch_topology
  - datacenter_topology
  - shared_services_topology
applies_to_alert_types:
  - link_down
  - high_error_rate
  - bgp_session_down
  - packet_loss
  - latency
rca_patterns:
  - wan_pe_link_failure
  - bgp_session_drop
  - wan_circuit_error_rate
  - pe_router_hardware_fault
last_reviewed: "2026-01-20"
evidence_tags:
  - WAN-PE
  - link_down
  - high_error_rate
  - wan_topology
  - bgp
---

## Purpose

Provide structured triage and remediation steps for incidents where the WAN Provider Edge router (WAN-PE-01 or equivalent) is the suspected root cause of link-down events or persistent high error rates causing service degradation across branch sites and interconnected topology domains.

## Trigger Symptoms

- `link_down` alert on the WAN-PE-01 interface (physical or logical) with concurrent BGP session drops.
- `high_error_rate` alert on a WAN-PE-01 interface exceeding the configured threshold (typically > 0.1% error rate sustained for 5 minutes).
- Branch sites lose connectivity to datacenter services concurrently with PE-router alerts.
- BGP route withdrawal notifications from the PE-router affecting downstream prefix reachability.
- Physical layer alarms: LOS (Loss of Signal), LOF (Loss of Frame), or AIS on WAN circuit interfaces.

## Applicable RCA Patterns

- **wan_pe_link_failure**: Physical WAN circuit failure or carrier-side fault causing the PE-router interface to go down.
- **bgp_session_drop**: BGP session drops on the PE-router due to link failure, hold-timer expiry, or route reflector issues, causing route withdrawal and traffic blackholing.
- **wan_circuit_error_rate**: Physical or framing errors on the WAN circuit — often carrier-side — causing packet loss and CRC errors without a full link-down event.
- **pe_router_hardware_fault**: Hardware fault on the PE-router (line card, SFP, or memory) causing interface flapping or persistent errors.

## Pre-Checks

1. Confirm WAN-PE-01 is the active PE-router for the affected branch/site — verify the redundant PE path state.
2. Check whether the carrier NOC has an active incident or planned maintenance for this circuit.
3. Verify whether physical alarms (LOS, LOF, AIS) are present — these indicate a physical/carrier issue, not a configuration issue.
4. Confirm no configuration change was applied to WAN-PE-01 within 30 minutes of the incident.
5. Check backup/secondary WAN path state — if the backup path is also degraded, priority escalation is required.

## Triage Steps

1. Check interface status on WAN-PE-01: `show interface` or equivalent — confirm link state, error counters, and utilisation.
2. Verify BGP session state on WAN-PE-01: confirm which BGP neighbors are down and which prefixes have been withdrawn.
3. Open a carrier NOC case immediately if physical layer alarms (LOS, LOF, AIS) are present — these cannot be resolved without carrier intervention.
4. Check the PE-router hardware fault log: SFP optical power levels, line card error counters, memory utilisation.
5. Confirm whether the backup WAN path is carrying traffic — if yes, the primary circuit is down and the backup is providing partial service.
6. Verify that branch sites are routing over the backup path by checking branch router routing tables.
7. Cross-correlate with alert timeline: if WAN-PE-01 alerts preceded branch-site and datacenter alerts, the PE-router is the upstream source of the cascade.

## Remediation Steps

1. **If physical carrier fault (LOS/LOF/AIS)**: Escalate to the carrier NOC immediately. Provide circuit ID, affected interface, and timestamp. Do not attempt hardware changes without carrier guidance.
2. **If BGP session is down but the interface is up**: Investigate BGP hold-timer settings and peer configuration. Manually clear the BGP session if the peer is confirmed reachable: `clear bgp neighbor <peer-ip>` (requires approval).
3. **If SFP or optical power is degraded**: Replace the SFP (from spares) under a maintenance window. Confirm optical power levels after replacement.
4. **If configuration drift is suspected**: Compare the running configuration against the last approved baseline using the CMDB configuration audit tool.
5. **If the primary circuit cannot be restored within 30 minutes**: Formally activate the backup WAN path as the primary and notify branch sites of the temporary routing change.
6. After any BGP session restoration: verify that all withdrawn prefixes have been re-advertised and that branch sites have restored routing.

## Validation Steps

1. Confirm the WAN-PE-01 interface returns to `up/up` state with error counters not incrementing.
2. Verify all BGP sessions on WAN-PE-01 return to `Established` state.
3. Confirm all branch sites regain datacenter reachability (ping/traceroute from branch to datacenter services).
4. Verify `link_down` and `high_error_rate` alerts clear on the monitoring platform.
5. Check that no routes remain withdrawn in the BGP route table due to the incident.
6. Confirm carrier NOC has closed their incident ticket if physical fault was involved.

## Rollback / Safety Notes

- Do not apply PE-router configuration changes without carrier coordination if the fault is physical.
- Any BGP session clear must be approved by a senior network engineer — clearing BGP on a PE-router can briefly blackhole traffic for all downstream sites.
- Document carrier case numbers, circuit IDs, and all engineer actions in the ServiceNow record.
- If the backup path is activated as primary, create a follow-up task to restore the original primary path and revert the routing change during a planned maintenance window.
- Retain interface counters and BGP state outputs captured during triage as evidence.

## Do Not Execute If

- The carrier NOC confirms a planned maintenance is in progress — wait for carrier completion before initiating remediation.
- Both primary and backup WAN paths are down — escalate immediately to senior network engineering and management; manual intervention beyond this SOP is required.
- The incident has not been confirmed active within the last 30 minutes (alerts may have auto-cleared).

## ServiceNow Routing

- **Assignment Group**: Network Engineering — WAN Operations
- **Category**: Network / WAN / Carrier
- **Priority**: 1-Critical if all branch connectivity is lost; 2-High if partial degradation with backup path active
- **Escalation**: WAN Operations manager and carrier account team if outage exceeds 60 minutes

## Evidence Tags

`WAN-PE`, `wan_topology`, `link_down`, `high_error_rate`, `bgp_session`, `carrier_fault`, `branch_topology`
