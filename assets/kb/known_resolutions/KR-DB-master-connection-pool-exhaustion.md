---
kb_id: KR-DB-001
title: "Known Resolution: Database Master Connection Pool Exhaustion"
doc_type: known_resolution
version: "1.1"
owner_group: "Database Engineering — Data Platform"
applies_to_node_types:
  - database
  - database_primary
  - database_master
applies_to_diagrams:
  - app_db_topology
  - datacenter_topology
applies_to_alert_types:
  - cpu_spike
  - connection_timeout
  - latency
rca_patterns:
  - connection_pool_exhaustion
  - db_master_cpu_saturation
last_reviewed: "2026-02-20"
evidence_tags:
  - DB-MASTER
  - connection_pool
  - known_resolution
  - cpu_spike
  - latency
---

## Purpose

Record a confirmed resolution pattern for database master connection pool exhaustion incidents leading to CPU saturation and application-layer timeout cascades.

## Incident Summary

Connection pool exhaustion on DB-MASTER is a recurring contributor to cross-tier incidents. When the database's `max_connections` limit is reached, new connection requests from application servers are refused, triggering `connection_timeout` alerts on the application tier and appearing as load balancer backend health failures upstream.

## Root Cause Confirmed

Prior confirmed root causes for DB-MASTER connection pool exhaustion include:

1. **Long-running transaction holding connections**: A batch report job or migration script holds database connections for extended periods, depleting the available connection pool for normal application traffic.
2. **Application connection leak**: An application service is not properly releasing database connections after use, causing the connection count to grow monotonically until `max_connections` is reached.
3. **Replica promotion with connection redirect**: A read-replica promotion event caused all read traffic to temporarily be redirected to DB-MASTER, doubling the connection load.

## Resolution Steps Applied

1. **For long-running transaction**: Kill the identified blocking session(s) with DBA approval. Verify that no DDL operation is in progress before killing. Add a statement timeout to the batch job configuration to prevent recurrence.
2. **For application connection leak**: Identify the application service with the highest idle connection count (from `pg_stat_activity` or equivalent). Restart that application service to force connection pool reset. File a bug for connection leak remediation.
3. **For replica promotion redirect**: After the replica promotion stabilises, update application connection strings to route reads to the new replica. Restart application services to pick up the new configuration.

## Lessons Learned

- A connection pool monitoring alert at 80% of `max_connections` should be configured to provide early warning before exhaustion.
- Application deployments should include a connection pool validation step in the deployment runbook.
- Read-replica promotions must include an application-layer connection routing update procedure in the runbook.
- Statement timeouts on batch jobs prevent connection pool depletion by long-running ad-hoc queries.

## Evidence Tags

`DB-MASTER`, `app_db_topology`, `connection_pool`, `max_connections`, `connection_timeout`, `latency`, `known_resolution`
