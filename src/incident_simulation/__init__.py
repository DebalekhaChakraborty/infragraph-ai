"""
InfraGraph AI — incident simulation layer.

Provides deterministic, topology-aware incident builders for both
local (single-diagram) and enterprise (cross-diagram) RCA narratives.
"""
from .local_incidents import build_local_incident
from .enterprise_incidents import (
    build_enterprise_incident,
    build_cross_diagram_hero_incident,
)
from .schemas import make_alert_event, make_incident

__all__ = [
    "build_local_incident",
    "build_enterprise_incident",
    "build_cross_diagram_hero_incident",
    "make_alert_event",
    "make_incident",
]
