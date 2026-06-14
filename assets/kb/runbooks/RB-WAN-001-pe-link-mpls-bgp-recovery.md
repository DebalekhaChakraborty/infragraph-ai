---
kb_id: RB-WAN-001
runbook_id: WAN-001
title: "WAN PE Link and MPLS/BGP Circuit Recovery"
doc_type: runbook
version: "1.0"
source: approved_kb_repo
domain: wan
owner_group: "Network Engineering — WAN Operations"
approval_required: true
automation_eligible: false
execution_mode: manual
applies_to_node_types:
  - wan_router
  - pe_router
  - mpls_node
applies_to_diagrams:
  - wan_topology
  - datacenter_topology
applies_to_alert_types:
  - link_down
  - bgp_session_down
  - mpls_circuit_failure
  - high_latency
  - packet_loss
rca_patterns:
  - wan_link_failure
  - bgp_session_loss
  - mpls_circuit_outage
last_reviewed: "2026-03-20"
evidence_tags:
  - wan
  - pe_router
  - bgp
  - mpls
  - circuit
  - link_down
---

## Purpose

Recover a WAN Provider Edge (PE) link or MPLS/BGP circuit that is down or degraded. Applies when GNN RCA identifies a WAN router or PE node (WAN-PE-*, PE-*) as the root cause of a `link_down`, `bgp_session_down`, or `mpls_circuit_failure` alert.

## Trigger Conditions

- RCA root cause node matches pattern `WAN-PE-*`, `PE-*`, or `WAN-*`
- Active alerts include `link_down`, `bgp_session_down`, or `mpls_circuit_failure`
- BGP peer session is in `Idle`, `Active`, or `Connect` state (not `Established`)
- MPLS circuit shows label-switched path (LSP) failure or high packet loss (> 5%)

## Pre-Checks

1. Confirm RCA root cause node is WAN/PE tier — check causal evidence for `link_down` or `bgp_session_down` as the initiating alert.
2. Identify the physical circuit and carrier: `show interfaces {interface_id}` on the PE router — note `Line protocol is down` vs. `administratively down`.
3. Verify carrier contact availability: check NOC carrier contact list for the affected circuit's provider.
4. Confirm the maintenance window status — WAN changes typically require a maintenance window.
5. Check for recently expired SFP modules or physical link events: `show log | include {interface_id}`.
6. Confirm redundant path availability before making any change: verify alternate PE or backup circuit is up.

## Triage Steps

1. Determine if the circuit failure is physical (Layer 1) or logical (Layer 3 BGP/MPLS):
   - Layer 1: `show interfaces {interface_id}` — check `Input errors`, `CRC`, `Line protocol is down`
   - Layer 3 BGP: `show bgp summary` — check peer state and uptime
   - MPLS: `show mpls forwarding-table` and `show mpls ldp neighbors`
2. Check carrier portal or NOC ticket system for provider-side incident on the circuit.
3. If physical link is up but BGP is down: attempt BGP session reset (soft reset preferred): `clear ip bgp {peer_ip} soft`.
4. If physical link is down: escalate to carrier immediately — open a carrier ticket with the circuit ID.
5. Check redundant circuits: `show ip route` — confirm traffic has failed over to backup paths if available.
6. Measure packet loss and latency on alternate paths: `ping {gateway_ip} repeat 100 source {loopback_ip}`.

## Remediation Steps

1. **BGP session soft reset (if BGP is down but link is up)**: `clear ip bgp {peer_ip} soft` — this is non-disruptive. Wait 60 seconds for session to re-establish.
2. **BGP hard reset (if soft reset fails)**: `clear ip bgp {peer_ip}` — this will briefly drop all prefixes learned from this peer. Requires NOC approval.
3. **Interface bounce (if Layer 1 / Layer 2 issue)**: `interface {interface_id} / shutdown / no shutdown` — disruptive, requires maintenance window and NOC approval. Warn downstream teams before execution.
4. **Carrier escalation (if physical circuit is down)**: Open carrier ticket immediately with: circuit ID, failure start time, impacted sites. Request circuit restoration ETA.
5. **Failover to backup circuit (if primary is unrecoverable)**: Modify route maps or adjust BGP local-preference to route traffic via backup: `route-map BACKUP-PATH permit 10 / set local-preference 200`. Requires NOC approval.
6. **Update CMDB**: Log the change in ITSM with affected CI = WAN-PE-* node, circuit ID, and carrier ticket number.

## Automation Hooks

- **Tool**: Network automation platform (Ansible/Nornir via internal_network_api)
- **Connector**: internal_network_api
- **Dry-run**: `show bgp summary` and `show interfaces` — read-only verification supported
- **Automation gate**: `approval_required=true` for all write operations (clear bgp, interface bounce, failover)
- **Rollback hook**: Restore original BGP local-preference settings via playbook: `network_admin rollback-bgp-policy --pe={pe_node_id}`

## Validation Steps

1. Confirm BGP session is `Established` with all configured peers: `show bgp summary`.
2. Confirm prefix count from affected peer has returned to baseline.
3. Verify MPLS LSP is active: `show mpls forwarding-table` and `ping mpls {lsp_id}`.
4. Confirm packet loss to remote sites is < 0.1%: `ping {remote_site_ip} repeat 500`.
5. Confirm `link_down` and `bgp_session_down` alerts have cleared on the monitoring platform.
6. Verify impacted downstream diagrams (app_db_topology, branch_topology) are also showing healthy.

## Rollback / Safety Notes

- **Interface bounce is disruptive** — notify affected downstream teams (app, database, branch) before execution.
- Capture running configuration before any change: `show running-config | include {interface_id}`.
- Do not modify BGP policies without the Network Engineering Manager's approval if > 1 site is impacted.
- Carrier failover (diverting traffic to alternate circuit) must be coordinated with the carrier and NOC — do not make unilateral routing changes.
- Maintain a carrier escalation ticket number in the ITSM ticket for traceability.

## Do Not Execute If

- Root cause has not been confirmed as WAN/PE tier in causal evidence.
- No maintenance window has been approved for interface bounce or BGP policy changes.
- An alternate/backup circuit is not available — do not remove the primary path without confirming a fallback exists.
- A carrier maintenance window is already in progress on the circuit.

## ITSM Routing

- **Assignment Group**: Network Engineering — WAN Operations
- **Category**: Network / WAN Circuit
- **Priority**: 1-Critical for complete circuit failure; 2-High for BGP degradation with active failover
- **Escalation**: Network Engineering Manager if resolution exceeds 30 minutes or if more than 2 PE nodes are affected

## Evidence Tags

`wan`, `pe_router`, `bgp`, `mpls`, `circuit`, `link_down`, `carrier_escalation`, `manual_only`, `noc_required`
