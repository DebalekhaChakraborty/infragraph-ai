"""
event_correlation — Pre-RCA event clustering and causal evidence layer.

Correlates observable alert events into coherent clusters before RCA scoring.
No root-cause labels, remediation steps, or evaluation fields are produced.

Public API
----------
correlate_events(events, graph, mode)
    → list of raw cluster dicts (call build_causal_evidence + make_cluster to finalise)

build_causal_evidence(raw_events, roles, dims, diagram_scope, mode)
    → list of causal evidence items

make_cluster(...)
    → validated cluster dict

make_cluster_output(case_id, scenario_id, mode, clusters)
    → top-level output wrapper

load_case_for_correlation(lib_root, repo_root, case_id, mode=None)
    → (events, graph, mode, scenario_id, diagram_id)

write_cluster_output(output, case_id, scenario_id, mode, out_dir)
    → Path to written file

load_cluster_file(cluster_path)
    → dict | None

find_cluster_for_case(cluster_data, case_id, scenario_id="")
    → dict | None

FORBIDDEN_KEYS
    frozenset of keys that must never appear in cluster output
"""

from .correlator import correlate_events
from .evidence import build_causal_evidence
from .io import find_cluster_for_case, load_case_for_correlation, load_cluster_file, write_cluster_output
from .schema import (
    FORBIDDEN_KEYS,
    make_causal_evidence_item,
    make_cluster,
    make_cluster_output,
    make_event_in_cluster,
)

__all__ = [
    "correlate_events",
    "build_causal_evidence",
    "load_case_for_correlation",
    "write_cluster_output",
    "load_cluster_file",
    "find_cluster_for_case",
    "FORBIDDEN_KEYS",
    "make_cluster",
    "make_cluster_output",
    "make_event_in_cluster",
    "make_causal_evidence_item",
]
