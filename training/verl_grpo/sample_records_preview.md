# Sample RCA Remediation Alignment Records

## Example 1: Cross-Diagram Load Balancer Incident

Input evidence:
- Root cause: `APP-LB-01`
- Impacted diagrams: `branch_topology`, `wan_topology`, `app_db_topology`
- Evidence IDs: `E001` root-cause diagram, `E002` impacted diagrams, `E003` impact path

Chosen response:
- Valid JSON.
- Validates reachability and alert freshness before remediation.
- Remediates `APP-LB-01` only after validation.
- Includes rollback and enterprise escalation.
- Cites evidence IDs.

Rejected response:
- Names `FAKE-RTR-99`.
- Starts with restart action.
- Has no rollback notes.

Reward explanation:
- Chosen response scores high on root-cause match, grounding, validation-before-remediation, rollback safety, and enterprise escalation.
- Rejected response loses points for hallucinated device, missing rollback, and wrong root cause.

## Example 2: Shared Services Incident

Input evidence:
- Root cause: `DNS-01`
- Impacted diagrams: `branch_topology`, `app_db_topology`, `shared_services_topology`
- Candidate ranking: `DNS-01` first

Chosen response:
- Confirms DNS alert timeline and graph dependencies.
- Requires read-only checks before any service restart.
- Escalates to shared-services owner and NOC for cross-diagram blast radius.

Rejected response:
- Returns invalid JSON and recommends remediation before validation.

Reward explanation:
- Chosen response preserves operator safety.
- Rejected response fails JSON format and validation ordering.

## Example 3: Single-Diagram Firewall Incident

Input evidence:
- Root cause: `BR-FW-01`
- Impacted diagram: `branch_topology`

Chosen response:
- Keeps scope single-diagram.
- Validates firewall deny logs and branch reachability.
- Requires config backup and rollback owner before changes.

Rejected response:
- Escalates to unrelated enterprise teams and cites devices not present in graph evidence.

Reward explanation:
- Chosen response is graph-grounded and scoped.
- Rejected response is penalized for hallucinated nodes and unnecessary escalation.
