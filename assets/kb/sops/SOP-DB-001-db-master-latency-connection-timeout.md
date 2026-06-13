---
kb_id: SOP-DB-001
title: "Database Master Node Latency and Connection Timeout Response"
doc_type: sop
version: "1.3"
owner_group: "Database Engineering — Data Platform"
applies_to_node_types:
  - database
  - database_primary
  - database_master
applies_to_diagrams:
  - app_db_topology
  - datacenter_topology
  - shared_services_topology
applies_to_alert_types:
  - cpu_spike
  - latency
  - connection_timeout
  - replication_lag
  - lock_wait_timeout
rca_patterns:
  - db_master_cpu_saturation
  - connection_pool_exhaustion
  - long_running_query
  - replication_failover_lag
last_reviewed: "2026-03-01"
evidence_tags:
  - DB-MASTER
  - database
  - cpu_spike
  - latency
  - connection_timeout
  - app_db_topology
---

## Purpose

Provide structured triage and remediation steps for incidents where the database master node (DB-MASTER or equivalent) is the suspected root cause of latency spikes, connection timeouts, or CPU saturation affecting application and downstream services.

## Trigger Symptoms

- `cpu_spike` alert on DB-MASTER concurrent with `latency` or `connection_timeout` alerts on application nodes.
- Database connection pool utilisation exceeds 90% on application servers.
- `lock_wait_timeout` or `replication_lag` alerts on the database tier.
- Application-tier health checks begin failing with "connection refused" or "too many connections" errors.
- Active database session count is at or above the configured `max_connections` limit.

## Applicable RCA Patterns

- **db_master_cpu_saturation**: Long-running or poorly indexed queries consuming CPU, degrading query response time for all sessions.
- **connection_pool_exhaustion**: Application connection pools reach maximum capacity, queuing or failing new requests.
- **long_running_query**: A single high-cost query holding locks, blocking other sessions and cascading into timeouts.
- **replication_failover_lag**: High replication lag causing read-replica routing failures, redirecting all reads to the master and causing overload.

## Pre-Checks

1. Confirm DB-MASTER is the active primary and not a replica — check replication topology before making any changes.
2. Identify the current active session count versus the configured `max_connections` limit.
3. Verify whether a batch job, ETL process, or deployment activity was initiated within 30 minutes of the incident.
4. Check replication lag on all read replicas — elevated lag may indicate the master is already under stress.
5. Confirm that the application connection pool configuration (min, max, timeout settings) has not changed recently.

## Triage Steps

1. Pull the top-N queries by CPU time and execution duration from the database slow query log for the incident window.
2. Check active locks: query the information schema (or equivalent) for lock wait graphs and identify blocking session IDs.
3. Review the active connection count and connection state breakdown (active, idle, idle_in_transaction).
4. Check for elevated disk I/O or memory pressure on the database host — I/O saturation can cause query latency independent of query complexity.
5. Verify network latency between the application tier (APP-LB-01, app servers) and DB-MASTER is within normal bounds.
6. Review the database error log for OOM events, crash-recovery activity, or file system alerts.
7. Confirm the database buffer pool hit rate — a significant drop may indicate buffer pool eviction under load.

## Remediation Steps

1. **If a long-running blocking query is identified**: Obtain DBA approval and terminate the blocking session using the appropriate kill command. Document the session ID and query in the incident record.
2. **If connection pool is exhausted**: Temporarily reduce the application connection pool maximum in the application configuration to shed load, then investigate the connection leak or usage spike.
3. **If CPU is saturated by a batch job**: Coordinate with the batch job owner to pause or rate-limit the job; schedule it to a low-traffic window.
4. **If replication lag is causing read-redirect to master**: Configure application read routing to fail over to a less-lagged replica or temporarily disable read splitting.
5. **If disk I/O is saturated**: Identify the tablespace with highest I/O and coordinate storage expansion or query-level I/O reduction with the DBA team.
6. After any session kill or configuration change: monitor the active session count and CPU for 5 minutes before declaring the incident resolved.

## Validation Steps

1. Confirm DB-MASTER CPU returns to below 70% sustained load.
2. Verify active session count is below 80% of `max_connections`.
3. Confirm `latency` and `connection_timeout` alerts clear on the monitoring platform.
4. Run a synthetic database health check query (e.g., `SELECT 1`) from the application tier and confirm response time is within SLA bounds.
5. Verify application server connection pool utilisation returns to normal (typically below 60% pool capacity).
6. Confirm replication lag on read replicas returns to within acceptable bounds (typically under 5 seconds).

## Rollback / Safety Notes

- Never kill a session without DBA approval and without first capturing the session's query and wait state.
- Do not reduce `max_connections` at the database level without coordinating with all application teams that share the database.
- Document every session killed and configuration change applied in the ServiceNow record.
- If a connection pool parameter is changed in the application, ensure the original value is recorded for rollback.
- Notify the application team before terminating sessions — some application frameworks do not handle unexpected session termination gracefully.

## Do Not Execute If

- DB-MASTER is currently undergoing a planned failover or maintenance event.
- The replication topology is unknown — confirm primary/replica roles before any write-affecting operation.
- A current deployment or schema migration is in progress — killing sessions may corrupt in-flight DDL operations.
- The incident has been open for more than 2 hours without active alerts — validate the incident is still ongoing.

## ServiceNow Routing

- **Assignment Group**: Database Engineering — Data Platform
- **Category**: Database / Performance
- **Priority**: 1-Critical if application tier is fully unable to connect; 2-High for partial degradation
- **Escalation**: Senior DBA on-call if session kill does not resolve CPU spike within 15 minutes

## Evidence Tags

`DB-MASTER`, `app_db_topology`, `cpu_spike`, `latency`, `connection_timeout`, `connection_pool`, `lock_wait`, `replication_lag`
