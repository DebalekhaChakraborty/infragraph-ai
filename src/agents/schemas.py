"""
Typed schemas for the Agentic Ops Orchestrator.
Uses Pydantic BaseModel for validation and serialisation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentToolResult(BaseModel):
    tool_name: str
    status: str                          # success | warning | error | skipped
    summary: str
    evidence: list[Any] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()


class AgentStep(BaseModel):
    step_id: int
    agent_name: str
    objective: str
    tool_name: str
    status: str                          # success | warning | error | skipped
    started_at: str
    completed_at: str
    summary: str
    evidence: list[Any] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()


class ApprovalGate(BaseModel):
    required: bool
    status: str                          # pending | approved | rejected
    risk_level: str                      # low | medium | high
    reason: str
    recommended_next_action: str

    def to_dict(self) -> dict:
        return self.model_dump()


class TicketDraft(BaseModel):
    ticket_id: str
    short_description: str
    description: str
    priority: str
    category: str
    assignment_group: str
    affected_ci: str
    impacted_diagrams: list[str] = Field(default_factory=list)
    evidence_summary: str = ""
    remediation_summary: str = ""
    approval_status: str = "pending"

    def to_dict(self) -> dict:
        return self.model_dump()


class AgentRun(BaseModel):
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
    steps: list[Any] = Field(default_factory=list)
    final_summary: str = ""
    root_cause: str = ""
    root_cause_diagram: str = ""
    confidence: float = 0.0
    impacted_diagrams: list[str] = Field(default_factory=list)
    approval_gate: dict[str, Any] = Field(default_factory=dict)
    ticket_draft: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()
