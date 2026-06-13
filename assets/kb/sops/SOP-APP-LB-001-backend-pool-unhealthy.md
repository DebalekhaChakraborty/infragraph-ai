---
kb_id: SOP-APP-LB-001
title: "Application Load Balancer Backend Pool Unhealthy Response"
doc_type: sop
version: "1.2"
owner_group: "Platform Engineering — Application Delivery"
applies_to_node_types:
  - load_balancer
  - application_load_balancer
applies_to_diagrams:
  - app_db_topology
  - datacenter_topology
  - shared_services_topology
applies_to_alert_types:
  - cpu_spike
  - connection_timeout
  - backend_pool_unhealthy
  - health_check_fail
  - high_latency
rca_patterns:
  - backend_pool_member_failure
  - load_balancer_cpu_spike
  - upstream_connection_exhaustion
last_reviewed: "2026-02-28"
evidence_tags:
  - APP-LB
  - backend_pool
  - cpu_spike
  - connection_timeout
  - app_db_topology
---

## Purpose

Provide structured triage and remediation steps for incidents where an application load balancer (APP-LB-01 or equivalent) is the suspected root cause of backend pool health failures, CPU spikes, or connection timeouts affecting application tiers.

## Trigger Symptoms

- `cpu_spike` alert on the load balancer node with concurrent `backend_pool_unhealthy` or `health_check_fail` alerts on backend members.
- `connection_timeout` alerts on application server nodes referencing the load balancer as the upstream.
- Backend pool member count drops below the configured minimum healthy threshold.
- Load balancer health check failure rate exceeds 20% for any backend pool.
- High active-connection queue depth on the load balancer concurrent with elevated CPU.

## Applicable RCA Patterns

- **backend_pool_member_failure**: One or more backend members fail health checks, causing the load balancer to exhaust remaining capacity while redistributing connections.
- **load_balancer_cpu_spike**: CPU saturation on the load balancer itself, leading to delayed health check processing and connection queuing.
- **upstream_connection_exhaustion**: Upstream clients exhaust the connection table on the load balancer due to connection keep-alive misconfiguration or flood.

## Pre-Checks

1. Identify which backend pool members have failed health checks — confirm their current operational state from the application server side.
2. Verify whether the load balancer CPU spike preceded or followed the backend health failures (timeline causality).
3. Confirm no application deployment or configuration change was initiated within 30 minutes of the incident onset.
4. Check whether the load balancer session persistence (sticky session) configuration is routing disproportionate traffic to a single backend member.
5. Verify SSL/TLS certificate validity on the load balancer if HTTPS health checks are configured.

## Triage Steps

1. Pull the load balancer access and error logs for the incident time window. Identify the ratio of 5xx responses to total requests.
2. Check CPU, memory, and active-connection metrics on APP-LB-01 over the incident window.
3. For each backend pool member reported unhealthy: SSH to the member and verify the application process is running and listening on the health-check port.
4. Verify that health check parameters (interval, timeout, threshold) are appropriate — overly aggressive health checks can cause false failures under load.
5. Check database connectivity from the backend members — if DB-MASTER or the database tier is degraded, backend health checks may fail at the application layer.
6. Review connection drain timers if a backend member was recently removed from the pool.

## Remediation Steps

1. **If a backend member is down (process not running)**: Restart the application service on that member, verify it passes the health check, then re-add it to the pool.
2. **If load balancer CPU is saturated**: Reduce active connections by temporarily reducing the connection rate limit, then investigate the traffic source.
3. **If session persistence is causing imbalance**: Temporarily disable session persistence, redistribute connections, then re-enable with corrected configuration.
4. **If health check parameters are too aggressive**: Increase health check interval and failure threshold under a change-management ticket.
5. **If the database tier is degraded**: Escalate to the database team per SOP-DB-001 — the load balancer cannot recover without the backend being healthy.
6. Verify the pool reaches the minimum healthy member count before closing the incident.

## Validation Steps

1. Confirm the backend pool member count returns to the configured minimum healthy threshold.
2. Verify CPU on APP-LB-01 returns to baseline (typically below 70% sustained).
3. Confirm `connection_timeout` and `backend_pool_unhealthy` alerts clear on the monitoring platform.
4. Run an end-to-end synthetic transaction through the load balancer to a backend member and confirm a successful response.
5. Check application error rates (5xx) in the 10 minutes following remediation — they should return to baseline.

## Rollback / Safety Notes

- Do not remove a backend member from the pool unless it is confirmed unhealthy — premature removal reduces redundancy.
- If adjusting health check parameters, revert immediately if the unhealthy member rate increases.
- Document all changes applied to the load balancer configuration in the ServiceNow record.
- Coordinate with the application team before restarting backend services — some applications have warm-up periods that affect response time after restart.
- Notify database team if the triage reveals database-tier involvement — escalate concurrently per SOP-DB-001.

## Do Not Execute If

- The load balancer is the sole ingress point and a restart would cause a total outage — validate a maintenance window first.
- Only one healthy backend member remains — removing or restarting it could cause a full service outage.
- A database-tier incident is in progress — resolve the database issue first before attempting load balancer remediation.
- The alert timestamps are stale (more than 1 hour old without active alert recurrence).

## ServiceNow Routing

- **Assignment Group**: Platform Engineering — Application Delivery
- **Category**: Application / Load Balancer
- **Priority**: 2-High if backend pool is partially healthy; 1-Critical if all members are unhealthy
- **Escalation**: Application Engineering on-call if backend member count reaches zero

## Evidence Tags

`APP-LB`, `app_db_topology`, `backend_pool`, `cpu_spike`, `connection_timeout`, `health_check`, `load_balancer`
