"""
_patterns.py — Alert type classification and propagation patterns.

Internal module: imported by both correlator.py and evidence.py to avoid
circular imports.  No remediation content, root-cause labels, or evaluation data.
"""
from __future__ import annotations

_AT_BUCKETS: dict[str, list[str]] = {
    "cpu":                    ["cpu_spike", "cpu_high", "high_cpu"],
    "latency":                ["latency", "api_latency", "high_latency", "slow_response", "response_time"],
    "packet_drop":            ["packet_drop", "packet_loss"],
    "link_errors":            ["link_errors", "link_down", "interface_errors", "port_down",
                               "interface_flap", "port_flap"],
    "connection_timeout":     ["connection_timeout", "connection_refused", "connection_drop",
                               "connection_reset"],
    "auth_errors":            ["auth_errors", "auth_failure", "authentication_failed", "auth_denied"],
    "backend_pool_unhealthy": ["backend_pool_unhealthy", "unhealthy_backend", "pool_error",
                               "backend_failure"],
    "user_timeout":           ["user_timeout", "session_timeout", "request_timeout"],
    "route_flap":             ["route_flap", "bgp_flap", "ospf_flap"],
    "dependency_error":       ["dependency_error", "service_unavailable", "upstream_failure"],
}


def classify_alert(alert_type: str) -> str:
    """Map raw alert_type string to a canonical bucket name."""
    at = alert_type.lower().replace("-", "_").replace(" ", "_")
    for bucket, patterns in _AT_BUCKETS.items():
        if any(p in at for p in patterns):
            return bucket
    return "other"


# Known propagation chains (alert-type sequences in chronological order)
PROPAGATION_PATTERNS: list[tuple[str, ...]] = [
    # Physical degradation chain
    ("link_errors", "packet_drop"),
    ("link_errors", "packet_drop", "connection_timeout"),
    ("link_errors", "packet_drop", "latency"),
    ("link_errors", "latency"),
    # Packet loss downstream
    ("packet_drop", "connection_timeout"),
    ("packet_drop", "latency"),
    ("packet_drop", "connection_timeout", "user_timeout"),
    # CPU pressure chain
    ("cpu", "latency"),
    ("cpu", "latency", "user_timeout"),
    ("cpu", "connection_timeout"),
    # Auth → connection chain
    ("auth_errors", "connection_timeout"),
    ("auth_errors", "user_timeout"),
    # Backend health chain
    ("backend_pool_unhealthy", "latency"),
    ("backend_pool_unhealthy", "connection_timeout"),
    ("backend_pool_unhealthy", "user_timeout"),
    # Generic timeout chain
    ("latency", "user_timeout"),
    ("connection_timeout", "user_timeout"),
    # Route instability
    ("route_flap", "packet_drop"),
    ("route_flap", "link_errors"),
    ("route_flap", "latency"),
    # Dependency cascade
    ("dependency_error", "latency"),
    ("dependency_error", "connection_timeout"),
]


def is_subsequence(seq: list[str], pattern: tuple[str, ...]) -> bool:
    """Return True if every element of pattern appears in seq in order."""
    it = iter(seq)
    return all(p in it for p in pattern)
