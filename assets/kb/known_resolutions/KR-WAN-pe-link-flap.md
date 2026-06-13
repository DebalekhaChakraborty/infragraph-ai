---
kb_id: KR-WAN-001
title: "Known Resolution: WAN Provider Edge Router Link Flap and Error Rate"
doc_type: known_resolution
version: "1.0"
owner_group: "Network Engineering — WAN Operations"
applies_to_node_types:
  - wan_router
  - provider_edge
  - pe_router
applies_to_diagrams:
  - wan_topology
  - branch_topology
  - datacenter_topology
applies_to_alert_types:
  - link_down
  - high_error_rate
  - bgp_session_down
rca_patterns:
  - wan_pe_link_failure
  - wan_circuit_error_rate
last_reviewed: "2026-01-15"
evidence_tags:
  - WAN-PE
  - link_flap
  - known_resolution
  - high_error_rate
  - bgp
---

## Purpose

Record confirmed resolution patterns for WAN provider edge (WAN-PE-01 or equivalent) link flap and persistent high error rate incidents.

## Incident Summary

WAN-PE-01 link flap events cause BGP session drops, triggering route withdrawals and connectivity loss to branch sites and downstream topology domains. High error rate alerts (without full link-down) indicate physical layer degradation that can progress to a full link-down if not addressed.

## Root Cause Confirmed

Prior confirmed root causes for WAN-PE-01 link flap and error rate incidents include:

1. **Faulty SFP transceiver**: An aging or mismatched SFP on the WAN-PE-01 WAN interface caused intermittent optical signal degradation, manifesting as CRC errors and periodic link-down events.
2. **Carrier-side physical fault**: A splice or connector fault on the provider's fibre caused signal attenuation beyond the SFP receive power budget, triggering LOS alarms.
3. **BGP hold-timer misconfiguration after router upgrade**: A router software upgrade reset the BGP hold-timer to the vendor default (90 seconds), which was incompatible with the carrier's BGP peer configuration, causing periodic BGP session drops without physical link failure.
4. **Interface error dampening suppression**: Excessive link flaps triggered the router's error dampening feature, suppressing the interface and preventing BGP from re-establishing even after the physical issue was resolved.

## Resolution Steps Applied

1. **For faulty SFP**: Replace the SFP with an identical approved spare. Confirm optical RX power is within the vendor's acceptable range after replacement. Monitor the interface for 30 minutes before closing.
2. **For carrier-side physical fault**: Open a carrier NOC P1 case with circuit ID, affected interface identifier, and alarm timestamps. Follow up every 30 minutes until carrier confirms physical repair. Activate backup WAN path as primary during carrier repair.
3. **For BGP hold-timer misconfiguration**: Restore the BGP hold-timer to the pre-upgrade agreed value under a change ticket. Clear the BGP session to the affected peer after the change.
4. **For error dampening suppression**: After the physical fault is resolved, apply the `no dampening` command (or equivalent) on the affected interface. Verify the BGP session re-establishes and routes are re-advertised.

## Lessons Learned

- SFP optical power levels should be monitored with threshold alerts at -3 dBm above the minimum receive power to provide early warning of SFP degradation.
- Router software upgrades must include a post-upgrade validation step for BGP timer configuration and carrier peer settings.
- WAN-PE routers should have approved spare SFP transceivers staged on-site to reduce MTTR.
- BGP error dampening parameters should be reviewed after any link flap incident to prevent dampening from masking a resolved physical issue.

## Evidence Tags

`WAN-PE`, `wan_topology`, `link_flap`, `high_error_rate`, `bgp_session`, `sfp_fault`, `carrier_noc`, `known_resolution`
