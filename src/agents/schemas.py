"""
Typed schemas for the Agentic Ops Orchestrator.
Uses stdlib dataclasses — no external dependencies.
All types are JSON-serializable via .to_dict() / dataclasses.asdict().
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentToolResult:
    tool_name: str
    status: str          # success | warning | error | skipped
    summary: str
    evidence: list = field(default_factory=list)
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentStep:
    step_id: int
    agent_name: str
    objective: str
    tool_name: str
    status: str          # success | warning | error | skipped
    started_at: str
    completed_at: str
    summary: str
    evidence: list = field(default_factory=list)
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ApprovalGate:
    required: bool
    status: str          # pending | approved | rejected
    risk_level: str      # low | medium | high
    reason: str
    recommended_next_action: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TicketDraft:
    ticket_id: str
    short_description: str
    description: str
    priority: str
    category: str
    assignment_group: str
    affected_ci: str
    impacted_diagrams: list = field(default_factory=list)
    evidence_summary: str = ""
    remediation_summary: str = ""
    approval_status: str = "pending"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentRun:
    run_id: str
    scenario_id: str
    selected_diagram_id: str
    mode: str
    started_at: str
    completed_at: str
    status: str
    alert_source: str
    topology_source: str
    rca_source: str
    remediation_source: str
    steps: list = field(default_factory=list)
    final_summary: str = ""
    root_cause: str = ""
    root_cause_diagram: str = ""
    confidence: float = 0.0
    impacted_diagrams: list = field(default_factory=list)
    approval_gate: dict = field(default_factory=dict)
    ticket_draft: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
