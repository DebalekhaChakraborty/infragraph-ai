---
kb_id: RB-ROLLBACK-001
runbook_id: ROLLBACK-001
title: "Cross-Domain Rollback and Safety Validation Runbook"
doc_type: runbook
version: "1.0"
source: approved_kb_repo
domain: cross_domain
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
  - post_remediation_validation
  - rollback_required
  - change_induced_regression
last_reviewed: "2026-03-20"
evidence_tags:
  - rollback
  - safety
  - validation
  - cross_domain
  - post_change
---

## Purpose

Validate that a remediation action has succeeded and determine whether a rollback is required. Applies after any domain-specific runbook (APP-LB-001, DB-001, WAN-001, DC-FW-001) has been executed. Also applies as the first step when a post-change regression is detected by new alerts or monitoring signals.

## Trigger Conditions

- Any domain-specific runbook has been executed (APP-LB-001, DB-001, WAN-001, DC-FW-001, ENT-XDIAG-001).
- New alerts appear within 30 minutes of completing a remediation step.
- Application error rates, latency, or packet drop metrics have not returned to baseline within 15 minutes of applying the fix.
- A GNN RCA re-run on the updated state identifies a new root cause candidate.

## Pre-Checks

1. Document the exact set of changes made (device, command, timestamp) from the domain-specific runbook execution log.
2. Confirm which diagram domains were modified — list all affected CI nodes and their change timestamps.
3. Identify the pre-change baseline metrics: error rate, latency P95, packet loss, CPU utilisation.
4. Confirm all monitoring dashboards are active and current (not showing stale data).
5. Confirm a configuration archive or snapshot was taken before the change (required by all domain runbooks).

## Triage Steps (Post-Change Regression)

1. Compare current alert state with pre-change alert state: are the same alerts present, or are there new alerts?
2. Check if new alerts are on nodes that were NOT previously impacted — this suggests the change introduced a new fault.
3. Run read-only verification commands for each modified domain:
   - **App LB**: `GET /api/lb/{lb_node_id}/pool/health` — check pool member status
   - **Database**: `SELECT count(*) FROM pg_stat_activity WHERE state = 'active';` — check connection count
   - **WAN**: `show bgp summary` — check peer state
   - **Firewall**: `show conn count; show access-list {acl_name} counters` — check session and ACL state
4. Determine if the regression is in the same domain as the original incident or in a different domain (cross-domain regression indicates a deeper systemic issue).
5. Escalate to the appropriate domain team if the regression is in a domain different from the original fix.

## Rollback Decision Matrix

| Condition | Action |
|-----------|--------|
| Original alerts cleared, no new alerts, metrics at baseline | Validation passed — close incident |
| Original alerts cleared, but latency/error rate still elevated | Continue monitoring for 15 min; escalate if not improving |
| New alerts on previously-healthy nodes after change | Immediate rollback of the most recent change |
| Original alerts persist unchanged after fix | Fix was ineffective — re-evaluate RCA; do not double-apply the fix |
| Alert storm — > 5 new alerts in 10 minutes | Immediate rollback AND activate incident bridge call |

## Rollback Steps

1. **Identify rollback target**: Determine the specific configuration snapshot or archive to restore.
   - App LB: `lb_admin_cli rollback-pool --pool={pool_id} --snapshot={snapshot_id}`
   - Database: coordinate with DBA for connection pool or policy rollback
   - WAN: `configure replace nvram:archive-{version}` on affected PE router
   - Firewall: `configure replace nvram:archive-{version}` — requires dual Security Ops approval
2. **Notify affected teams before rollback**: Inform all teams whose diagrams may be impacted by the rollback traffic change.
3. **Apply rollback in reverse order of changes**: If multiple changes were made, roll them back in reverse chronological order (last change first).
4. **Confirm rollback took effect**: Run the same read-only verification commands as in Triage Steps — confirm the state matches the pre-change baseline snapshot.
5. **Re-evaluate RCA**: After rollback, re-run the RCA analysis or request a GNN RCA re-run to re-assess the root cause with the reverted state.
6. **Update ITSM**: Log the rollback in the original ITSM ticket with timestamps, reason, and the new post-rollback state.

## Validation Steps

1. Confirm all active alerts from the original incident have cleared.
2. Confirm no new alerts have been generated within 15 minutes of completing the rollback.
3. Confirm all monitoring metrics (error rate, latency, packet loss, CPU) are at or below pre-incident baseline.
4. Get explicit sign-off from each diagram domain owner that their domain is healthy.
5. Confirm the GNN RCA or correlation cluster shows no active anomalies for this scenario.
6. If rollback was performed: confirm the rolled-back configuration is stable and not generating new issues.

## Safety Notes

- **All rollbacks must be logged in the central ITSM ticket ticket** with timestamps and approver names.
- No change should be rolled back without first notifying the team that originally applied the change.
- If the rollback itself causes an alert storm or degradation, stop immediately and escalate to the Network Engineering Manager and NOC.
- Rollback of a firewall ACL change requires dual Security Ops approval — same as the original change.
- Rollback of a WAN failover requires NOC + carrier coordination — do not revert WAN routing without confirming the primary circuit is stable.
- Maintain a full audit trail: all commands executed, all approvals received, all timestamps — these are evidence for the post-incident review.

## Do Not Execute If

- No domain-specific remediation has been applied yet — this runbook is for post-change validation and rollback only.
- The original incident has been caused by a known carrier outage or vendor issue — rollback of internal changes will not restore service until the external dependency is resolved.
- A security incident investigation (SIRT) is in progress — preserve the changed state as forensic evidence and coordinate rollback timing with the SIRT team.

## ITSM Routing

- **Assignment Group**: Network Engineering — Enterprise Operations (lead), plus domain-specific teams for affected CIs
- **Category**: Network / Post-Change Validation
- **Priority**: Inherit from the parent incident ticket
- **Escalation**: Network Engineering Manager if rollback does not restore service within 30 minutes

## Evidence Tags

`rollback`, `safety`, `validation`, `cross_domain`, `post_change`, `incident_closure`, `audit_trail`
