---
kb_id: KR-APP-LB-001
title: "Known Resolution: Application Load Balancer Backend Health Degradation"
doc_type: known_resolution
version: "1.0"
owner_group: "Platform Engineering — Application Delivery"
applies_to_node_types:
  - load_balancer
  - application_load_balancer
applies_to_diagrams:
  - app_db_topology
  - datacenter_topology
applies_to_alert_types:
  - cpu_spike
  - connection_timeout
  - backend_pool_unhealthy
rca_patterns:
  - backend_pool_member_failure
  - load_balancer_cpu_spike
last_reviewed: "2026-02-10"
evidence_tags:
  - APP-LB
  - backend_pool
  - known_resolution
  - connection_timeout
---

## Purpose

Record a confirmed resolution pattern for application load balancer backend health degradation incidents, for use as reference in future incident response.

## Incident Summary

A recurring pattern has been observed where the APP-LB-01 load balancer reports `backend_pool_unhealthy` and `cpu_spike` alerts concurrently, with downstream `connection_timeout` alerts appearing on application server nodes within 3–5 minutes.

## Root Cause Confirmed

The confirmed root cause in prior occurrences was one of the following:

1. **Asymmetric connection drain**: A backend member was removed from the pool without a connection drain period, causing in-flight requests to fail and the load balancer to log them as `connection_timeout`.
2. **Health check misconfiguration after a deployment**: A new application version changed the health check endpoint path but the load balancer health check configuration was not updated, causing all members to fail health checks immediately after deployment.
3. **Database-tier cascade**: DB-MASTER latency caused application backend members to exceed their own timeout thresholds, appearing to the load balancer as unhealthy members.

## Resolution Steps Applied

1. **For asymmetric connection drain**: Re-add the removed member with a connection drain timeout of 30 seconds. Verify the pool reaches minimum healthy member count. Enable connection drain on all future pool member removals by policy.
2. **For health check misconfiguration after deployment**: Update the load balancer health check URL path to match the new application version's health endpoint. Verify member health within 2 minutes of the update.
3. **For database-tier cascade**: Escalate to DB-MASTER remediation per SOP-DB-001. The load balancer resolves automatically once the database tier recovers — no load balancer-specific change is needed in this case.

## Lessons Learned

- Load balancer health check configuration must be included in application deployment change tickets as a required checklist item.
- A connection drain policy of minimum 30 seconds should be enforced on all pool members.
- When `cpu_spike` and `backend_pool_unhealthy` alerts appear together, database-tier involvement should always be checked before assuming the load balancer is the root cause.

## Evidence Tags

`APP-LB`, `app_db_topology`, `backend_pool`, `connection_timeout`, `deployment_health_check`, `connection_drain`, `known_resolution`
