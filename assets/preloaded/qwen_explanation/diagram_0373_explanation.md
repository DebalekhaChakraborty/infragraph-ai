# InfraGraph AI RCA Explanation -- diagram_0373

> Generated: 2026-06-09 16:10 UTC  
> Provider: mock (deterministic template)

---

## Executive Summary

A network incident triggered 3 alerts across 2 nodes, impacting 10 downstream services. The heuristic RCA incorrectly identified **SW-CORE** as root cause. The GNN-based RCA correctly identified **FW-01** (firewall), demonstrating the value of learned propagation-direction signals over rule-based scoring.

---

## What Happened

**3 alerts** fired across **2 alerting nodes** in a 17-node, 17-edge topology:

- **FW-01** [CRITICAL] Packet drops elevated (t+0 min)
- **SW-CORE** [MAJOR] Policy deny spike (t+2 min)
- **SW-CORE** [MAJOR] App unreachable (t+4 min)

---

## Root Cause Conclusion

**Root cause: FW-01 (firewall)**  
Confidence: HIGH (GNN score: 30.733, margin over 2nd-ranked: 8.12)

The firewall **FW-01** initiated the incident. Its position in the topology as an upstream chokepoint means its failure cascades to all downstream services through the core switching and application layers.

---

## Heuristic vs GNN Comparison

| | Heuristic (Stage 2) | GNN (Stage 3) |
|--|--|--|
| Predicted root cause | SW-CORE | FW-01 |
| Ground truth | FW-01 | FW-01 |
| Correct? | **No** | **Yes** |
| Top candidates | SW-CORE (switch, score=20.86), FW-01 (firewall, score=20.62) | FW-01 (firewall, score=30.73), FW-02 (firewall, score=22.61), SW-APP (switch, score=13.85) |

**Why the heuristic was wrong**: The heuristic scorer ranked nodes by severity, timing, and downstream reach. **SW-CORE** received a higher composite score because it had more correlated alert events and higher downstream reach in the topology. The heuristic cannot distinguish a downstream aggregation node from the true upstream origin.

**Why the GNN is more reliable**: The GNN learned propagation direction from graph structure and temporal alert features across 400 training scenarios. **FW-01**'s features — firewall priority flag, earliest alert time, and critical severity — combined with its upstream topology position produce a score of 30.733, clearly separating it from downstream nodes. Test set: top-1=100%, MRR=1.000.

---

## Impacted Nodes / Services

**10 nodes impacted**: APP-01, APP-02, APP-03, APP-04, CLOUD-01, DB-01, DB-02, LB-01, MGMT-01, SW-APP

**Shortest propagation path (predicted root cause)**: `FW-01 -> SW-CORE`  
**Shortest propagation path (ground truth)**: `FW-01 -> SW-CORE`

---

## Recommended Next Actions (L1/L2)

1. **Immediate**: SSH to **FW-01** and check interface counters, packet drops, and syslog
2. **Verify**: Identify the failure mode — ACL misconfiguration, upstream link fault, or hardware error
3. **Escalate if**: Packet drop rate exceeds 5% or FW-01 CPU/memory is critically high
4. **Failover**: If FW-01 is faulty and a redundant path exists, activate it now
5. **Validate**: After remediation, confirm downstream services (APP-01, APP-02, APP-03, ...) are reachable
6. **Post-incident**: Update CMDB topology and retrain GNN if root cause pattern is novel

---

## ServiceNow Incident Summary

**Short description**: Network fault on FW-01 causing 10-service outage  
**Affected CI**: FW-01 (firewall)  
**Priority**: P1 -- 10 downstream nodes impacted  
**Assignment group**: Network Operations  
**Symptom**: Alerts on FW-01, SW-CORE; 10 downstream services unreachable  
**Root cause (automated)**: FW-01 identified by GNN RCA (confidence: HIGH)  
**Services impacted**: APP-01, APP-02, APP-03, APP-04, CLOUD-01  
**Suggested action**: Inspect FW-01 interfaces, ACLs, and upstream connectivity

---

## Confidence and Limitations

- **GNN confidence**: HIGH -- score margin 30.733 vs 22.61 (2nd candidate)
- **Model**: trained on 400 synthetic infragraph_v2 scenarios (test top-1=100%, MRR=1.000)
- **Limitations**: Synthetic training data; real-world noise, partial observability, or novel topologies may reduce accuracy
- **Human review recommended** before executing remediation on production firewall