---
kb_id: RB-DB-001
runbook_id: DB-001
title: "Database Connection Pool and Query Latency Recovery"
doc_type: runbook
version: "1.0"
source: approved_kb_repo
domain: database
owner_group: "Database Engineering — Platform Reliability"
approval_required: true
automation_eligible: false
execution_mode: manual
applies_to_node_types:
  - database
  - database_replica
applies_to_diagrams:
  - app_db_topology
applies_to_alert_types:
  - connection_pool_exhausted
  - query_latency_high
  - replica_lag
  - db_cpu_high
  - latency_spike
rca_patterns:
  - database_connection_pool_exhaustion
  - replica_lag_cascade
last_reviewed: "2026-03-20"
evidence_tags:
  - database
  - connection_pool
  - query_latency
  - db_master
  - replica
---

## Purpose

Recover a database node experiencing connection pool exhaustion or high query latency. Applies when GNN RCA identifies a database node (DB-MASTER, DB-REPLICA-*) as the root cause or a downstream victim of a propagating fault from an upstream load balancer or application tier.

## Trigger Conditions

- RCA root cause node or impacted node matches pattern `DB-MASTER` or `DB-REPLICA-*`
- Active alerts include `connection_pool_exhausted`, `query_latency_high`, or `replica_lag`
- Active connection count exceeds 80% of `max_connections`
- Query latency P95 exceeds 500ms for > 5 minutes

## Pre-Checks

1. Confirm impacted database node in causal evidence — note if DB is root cause vs. downstream victim.
2. Run read-only connection count query: `SELECT count(*) FROM pg_stat_activity WHERE state != 'idle';` (or equivalent for MySQL: `SHOW STATUS LIKE 'Threads_connected';`).
3. Identify long-running queries: `SELECT pid, query, now() - query_start AS duration FROM pg_stat_activity WHERE state = 'active' ORDER BY duration DESC LIMIT 20;`
4. Confirm replica lag: `SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag_seconds;` on replica.
5. Check database error logs for OOM, deadlock, or lock contention events in the last 15 minutes.
6. Verify database backup status — do not make changes if a backup is in progress.

## Triage Steps

1. Determine if the root cause is: (a) connection pool exhaustion from upstream application, (b) long-running/blocking query, (c) replica lag causing read-traffic failover to master, or (d) disk I/O saturation.
2. If upstream application (APP-LB, APP-*) is the confirmed root cause: focus on reducing upstream connection rate before touching the database.
3. Identify blocking queries and their pids: `SELECT pid, query, wait_event, wait_event_type FROM pg_stat_activity WHERE wait_event IS NOT NULL;`
4. Check if any table-level or row-level locks are blocking progress: `SELECT * FROM pg_locks WHERE granted = false;`
5. Check disk I/O utilisation: `iostat -xz 5 3` on the database host.
6. Review pg_stat_statements for the top-10 slowest queries in the incident window.

## Remediation Steps

1. **Terminate idle connections**: If connection pool is exhausted by idle connections, terminate idle sessions: `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < now() - INTERVAL '10 minutes';` — requires DBA approval before execution.
2. **Kill blocking query**: If a single long-running query is causing lock contention, terminate it: `SELECT pg_terminate_backend({blocking_pid});` — document the query and the justification.
3. **Reduce upstream connection pressure**: Coordinate with the application team to reduce application connection pool size (`max_pool_size` in the connection pool config) — restart the connection pool proxy if applicable (e.g., PgBouncer: `systemctl restart pgbouncer`).
4. **Promote replica (failover — if master is unrecoverable)**: Only with explicit DBA and NOC approval. Promote: `pg_ctl promote -D {data_dir}` — this is irreversible without a rebuild of the old master.
5. **Clear replication lag**: If replica lag is the trigger, confirm replication slot is not stuck: `SELECT slot_name, active, restart_lsn FROM pg_replication_slots;` — drop stuck slots with DBA approval: `SELECT pg_drop_replication_slot('{slot_name}');`
6. **Update CMDB**: Log the change in ServiceNow with affected CI = DB-MASTER or DB-REPLICA, change type = Emergency.

## Automation Hooks

- **Tool**: `db_admin_cli` (internal wrapper for psql/mysql admin commands)
- **Connector**: internal_db_api
- **Dry-run**: `db_admin_cli show-blocking-queries --host={db_host}` — read-only, lists blocking queries
- **Automation gate**: `approval_required=true` for all write operations (terminate, promote, drop slot)
- **Rollback hook**: Not applicable for termination. Failover rollback requires DBA-led rebuild — document in ServiceNow.

## Validation Steps

1. Confirm active connection count has returned below 60% of `max_connections`.
2. Verify query latency P95 has returned below 100ms.
3. Confirm no active blocking queries in `pg_stat_activity`.
4. Confirm replica lag is < 30 seconds.
5. Verify `connection_pool_exhausted` and `query_latency_high` alerts have cleared.
6. Run synthetic read/write health check: `db_admin_cli health-check --host={db_host} --read --write`.

## Rollback / Safety Notes

- **Never terminate replication without DBA approval** — this can cause data loss or split-brain.
- Capture a list of all terminated connection pids and their queries before terminating.
- Coordinate with application teams before changing connection pool settings — mis-configuration can worsen the outage.
- Do not promote a replica while the master is still accepting writes without explicit NOC sign-off.
- If the incident is caused by an upstream application, fixing the DB without fixing the upstream cause will result in recurrence.

## Do Not Execute If

- Active backup or PITR restore is in progress on the target database.
- A concurrent DBA change is in progress on the same database.
- Root cause has not been confirmed as database-tier — if upstream (load balancer, application) is the root cause, address that domain first.
- Replica promotion is not warranted (master is recoverable within 15 minutes).

## ServiceNow Routing

- **Assignment Group**: Database Engineering — Platform Reliability
- **Category**: Database / Connection Pool
- **Priority**: 2-High for connection pool exhaustion; 1-Critical for master failover
- **Escalation**: Database Engineering Manager for any failover or replication slot drop

## Evidence Tags

`database`, `connection_pool`, `query_latency`, `replica_lag`, `db_master`, `manual_only`, `dba_required`
