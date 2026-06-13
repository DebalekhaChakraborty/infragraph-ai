---
kb_id: RB-APP-LB-001
runbook_id: APP-LB-001
title: "Application Load Balancer Backend Pool Recovery"
doc_type: runbook
version: "1.0"
source: approved_kb_repo
domain: load_balancer
owner_group: "Application Engineering — Platform Reliability"
approval_required: false
automation_eligible: true
execution_mode: semi_automated
tool_name: lb_admin_cli
connector: internal_lb_api
action: drain_and_recover_backend_pool
dry_run_supported: true
applies_to_node_types:
  - load_balancer
  - application_server
applies_to_diagrams:
  - app_db_topology
applies_to_alert_types:
  - backend_unhealthy
  - health_probe_fail
  - connection_timeout
  - cpu_spike
  - latency_spike
rca_patterns:
  - load_balancer_backend_failure
  - connection_pool_exhaustion
last_reviewed: "2026-03-20"
evidence_tags:
  - load_balancer
  - backend_pool
  - health_probe
  - app_lb
---

## Purpose

Recover a load balancer backend pool that is reporting unhealthy members or failing health probes. Applies when GNN RCA identifies an application-tier load balancer (APP-LB-*) node as the root cause of a cpu_spike, latency_spike, or connection_timeout alert sequence.

## Trigger Conditions

- RCA root cause node matches pattern `APP-LB-*`
- Active alerts include `backend_unhealthy`, `health_probe_fail`, or `connection_timeout`
- Backend pool member count in healthy state has dropped below threshold (< 50% of configured members)
- Load balancer CPU exceeds 85% for > 3 minutes

## Pre-Checks

1. Confirm the RCA root cause node is confirmed as APP-LB-* in causal evidence with confidence >= 0.8.
2. Run read-only health-check API call: `GET /api/lb/{lb_node_id}/pool/health` (dry_run=true supported).
3. Verify current backend pool membership count and healthy/unhealthy split.
4. Confirm no active maintenance window for the app-db topology.
5. Check APM for baseline error rate — document pre-change error percentage.
6. Verify the load balancer config version in CMDB matches the running config.

## Triage Steps

1. Identify which backend pool members are failing health probes (check LB admin console or `GET /api/lb/{lb_node_id}/pool/members`).
2. For each unhealthy member: SSH to the backend server and check: CPU usage, memory, thread pool queue depth, health endpoint `/health` response code.
3. Determine if the health probe failure is due to application logic (500 errors) or resource exhaustion (timeout, OOM).
4. Review recent deployments: `git log --since=2h --oneline` on the application repo or check CI/CD deployment log.
5. If a recent deployment is the suspect: identify rollback candidate version.
6. Check load balancer access logs for any upstream 502/503 spike correlated with the alert timestamp.

## Remediation Steps

1. **Drain affected members**: Mark unhealthy backend pool members as `draining` (not immediately removed) to allow in-flight requests to complete. Use `lb_admin_cli drain --pool={pool_id} --member={member_id} --dry-run` first to validate.
2. **Remove confirmed-unhealthy members**: After drain completes (wait for active connection count to reach 0), remove the member from the pool: `lb_admin_cli remove-member --pool={pool_id} --member={member_id}`.
3. **Application restart (if resource exhaustion)**: On the affected backend server, restart the application service: `systemctl restart {app_service_name}`. Wait 30 seconds for JVM warm-up or startup probe to pass.
4. **Health endpoint validation**: Confirm the application health endpoint returns HTTP 200: `curl -s -o /dev/null -w "%{http_code}" http://{member_ip}:{port}/health`.
5. **Re-add to pool**: Once the member health endpoint returns 200 consistently for 60 seconds, re-add to the load balancer pool: `lb_admin_cli add-member --pool={pool_id} --member={member_id} --weight=50`.
6. **Ramp weight**: After 2 minutes with no errors, restore the member weight to 100: `lb_admin_cli set-weight --pool={pool_id} --member={member_id} --weight=100`.
7. **Update CMDB**: Log the change in ServiceNow with affected CI = APP-LB-* node, change type = Emergency.

## Automation Hooks

- **Tool**: `lb_admin_cli`
- **Connector**: `internal_lb_api`
- **Dry-run**: `lb_admin_cli drain --pool={pool_id} --dry-run` — lists members that would be drained without making changes
- **Automation gate**: `approval_required=false` for pool drain/re-add; `approval_required=true` if member count drops below 1 (would cause total outage)
- **Rollback hook**: `lb_admin_cli rollback-pool --pool={pool_id} --snapshot={snapshot_id}` — restores previous pool membership state

## Validation Steps

1. Confirm all backend pool members show status `healthy` in the LB admin console.
2. Send synthetic health probe: `curl -s -o /dev/null -w "%{http_code}" http://{lb_vip}:{port}/health` — expect HTTP 200.
3. Verify application error rate in APM has returned to baseline (< 1% 5xx rate).
4. Confirm `backend_unhealthy` and `health_probe_fail` alerts have cleared on the monitoring platform.
5. Verify load balancer CPU has returned below 60%.
6. Run end-to-end synthetic transaction: `curl -s -o /dev/null -w "%{http_code}" http://{lb_vip}/api/v1/ping` — expect 200.

## Rollback / Safety Notes

- Capture the pool membership snapshot before any drain/remove: `lb_admin_cli snapshot-pool --pool={pool_id}`.
- Do not remove more than 50% of pool members in a single operation without NOC approval.
- If total pool membership drops to 0 (all members unhealthy), immediately escalate to Platform SRE on-call — do not attempt further automation.
- If a deployment rollback is initiated, coordinate with the owning application team before pushing the rollback to production.

## Do Not Execute If

- Root cause node is NOT confirmed as APP-LB-* by causal evidence with confidence >= 0.8.
- An active DR/failover is in progress for the app_db_topology domain.
- All backend pool members are currently healthy (investigate alternate root cause).
- A concurrent change is in progress on the load balancer by another operator.

## ServiceNow Routing

- **Assignment Group**: Application Engineering — Platform Reliability
- **Category**: Application / Load Balancer
- **Priority**: 2-High for partial backend pool failure; 1-Critical if 100% of members unhealthy
- **Automation tag**: `auto_eligible=true` for pool drain/re-add with > 1 healthy member remaining

## Evidence Tags

`load_balancer`, `backend_pool`, `health_probe`, `app_lb`, `cpu_spike`, `latency`, `connection_timeout`, `semi_automated`
