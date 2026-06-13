---
kb_id: RB-DC-FW-001
runbook_id: DC-FW-001
title: "Datacenter Firewall Packet Drop and ACL Recovery"
doc_type: runbook
version: "1.0"
source: approved_kb_repo
domain: firewall
owner_group: "Network Engineering — Security Operations"
approval_required: true
automation_eligible: false
execution_mode: manual
applies_to_node_types:
  - firewall
  - security_device
applies_to_diagrams:
  - datacenter_topology
applies_to_alert_types:
  - packet_drop
  - acl_deny
  - firewall_cpu_high
  - session_table_full
  - policy_violation
rca_patterns:
  - firewall_acl_block
  - session_table_exhaustion
  - firewall_cpu_overload
last_reviewed: "2026-03-20"
evidence_tags:
  - firewall
  - packet_drop
  - acl
  - datacenter
  - security
  - dc_fw
---

## Purpose

Recover a datacenter firewall experiencing packet drops due to ACL mismatches, session table exhaustion, or CPU overload. Applies when GNN RCA identifies a firewall node (DC-FW-*, FW-*) as the root cause of a `packet_drop` or `acl_deny` alert propagating to downstream datacenter or application services.

## Trigger Conditions

- RCA root cause node matches pattern `DC-FW-*` or `FW-*`
- Active alerts include `packet_drop`, `acl_deny`, or `session_table_full`
- Firewall CPU exceeds 90% for > 5 minutes
- Session table utilisation exceeds 80% of maximum capacity
- Packet drop rate exceeds 0.5% of forwarded traffic

## Pre-Checks

1. Confirm the RCA root cause node is DC-FW-* in causal evidence — verify the initiating alert is `packet_drop` or `acl_deny` (not a downstream effect).
2. Run read-only firewall status check: `show system resources` — document CPU, memory, and session table utilisation.
3. Identify which ACL or policy rule is generating the denies: `show log | grep DENY | tail -100`.
4. Confirm the Security Operations on-call engineer is notified — firewall changes require Security Ops approval.
5. Verify that the packet drops are NOT caused by a known DDoS or active attack — check threat intelligence feed and IPS logs before proceeding.
6. Confirm change management approval exists for any ACL modification.

## Triage Steps

1. Determine the drop category:
   - **ACL deny**: `show access-list {acl_name} counters` — identify which rule is matching and incrementing.
   - **Session table full**: `show conn count` or `show resource usage` — check if session limit is reached.
   - **CPU overload**: `show processes cpu sorted` — identify which process is consuming CPU (e.g., inspection engine, logging daemon).
2. For ACL denies: extract the source/destination IP pairs being denied — determine if they are legitimate traffic or an attack.
3. For session table exhaustion: identify which source IPs have the most half-open connections (potential SYN flood): `show conn | sort | head -50`.
4. For CPU overload: determine if a deep-packet inspection (DPI) policy is triggering on a high-rate traffic pattern.
5. Correlate the denial timestamps with the alert timeline from the GNN RCA causal evidence to confirm the root cause.

## Remediation Steps

1. **ACL rule correction (if legitimate traffic is being blocked)**: With Security Ops approval, add a permit rule above the blocking deny: `ip access-list extended {acl_name} / permit {protocol} {src_ip} {dst_ip}`. Changes must be reviewed by a second Security Ops engineer before application.
2. **Session table cleanup (if session table is full due to half-open connections)**: Reduce TCP half-open timeout: `timeout half-closed 0:00:30` — requires Security Ops approval. Alternatively, apply a temporary rate-limit on the offending source: `rate-limit input access-group {deny_acl} bps {rate}`.
3. **CPU relief (if CPU overload)**: Identify and temporarily disable or tune the DPI inspection policy causing overload — do not remove security policies without explicit Security Ops sign-off. Consider offloading logging to a remote syslog if logging is the CPU consumer.
4. **Emergency ACL revert (if a recent policy change caused the issue)**: Identify the last change: `show archive log config all | tail -20`. Revert with change management approval: `configure replace nvram:archive-{config_version}`.
5. **Failover to backup firewall (if primary is unrecoverable)**: Activate the HA standby unit with NOC and Security Ops approval: `failover active` on the standby unit. Confirm state synchronisation before failover.
6. **Update CMDB**: Log the change in ServiceNow with affected CI = DC-FW-* node, rule name, change type = Emergency Security Change.

## Automation Hooks

- **Tool**: Security automation platform (internal_security_api)
- **Connector**: internal_security_api
- **Dry-run**: `show access-list {acl_name} counters` and `show conn count` — read-only checks supported
- **Automation gate**: `approval_required=true` for ALL write operations — dual-approval required (Security Ops + NOC)
- **Rollback hook**: `configure replace nvram:archive-{version}` — restores previous ACL configuration

## Validation Steps

1. Confirm packet drop rate has returned to < 0.1%: `show interface {interface_id} | include drops`.
2. Verify the specific ACL rule hit counts for the deny rule are no longer incrementing.
3. Confirm session table utilisation is below 50%.
4. Confirm firewall CPU has returned below 70%.
5. Confirm `packet_drop` and `acl_deny` alerts have cleared on the monitoring platform.
6. Run end-to-end connectivity test from the affected application tier to the datacenter services: verify no new drops appear in logs after the fix.

## Rollback / Safety Notes

- **All firewall ACL changes require dual approval** (Security Ops engineer + Network NOC). Do not apply changes with a single approver.
- Capture a timestamped configuration archive before any change: `archive config`.
- Do not permit traffic that was previously denied without a confirmed business justification — improper ACL widening is a security risk.
- If the firewall is under active DDoS attack, escalate to the Security Incident Response team immediately — do not attempt standard runbook steps during a security incident.
- Revert must be verified within 15 minutes of any emergency ACL change — if traffic validation fails, roll back immediately.

## Do Not Execute If

- Root cause has not been confirmed as DC-FW-* in causal evidence with confidence >= 0.8.
- A concurrent Security Incident Response (SIRT) investigation is active for the same firewall.
- The packet drops are caused by an active DDoS or security attack — hand off to SIRT team.
- A scheduled firewall maintenance window is already in progress.
- Change management approval has not been obtained.

## ServiceNow Routing

- **Assignment Group**: Network Engineering — Security Operations
- **Category**: Security / Firewall
- **Priority**: 1-Critical for > 5% packet drop rate; 2-High for ACL deny with limited impact
- **Escalation**: Security Operations Manager if packet drops are caused by an active attack, or if HA failover is required

## Evidence Tags

`firewall`, `packet_drop`, `acl`, `datacenter`, `security`, `dc_fw`, `dual_approval`, `manual_only`, `sirt_escalation`
