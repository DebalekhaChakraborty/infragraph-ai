"""
evidence.py — Causal evidence trail builder.

Produces deterministic, hypothesis-labelled evidence items from cluster
scoring data.  No root-cause labels, remediation steps, or evaluation fields.
"""
from __future__ import annotations

from ._patterns import PROPAGATION_PATTERNS, classify_alert, is_subsequence
from .schema import make_causal_evidence_item


def build_causal_evidence(
    raw_events: list[dict],
    roles: list[str],
    dims: dict[str, float],
    diagram_scope: list[str],
    mode: str,
) -> list[dict]:
    """
    Build a list of causal evidence items for one cluster.

    Each item covers one correlation stage and is grounded in observable
    event data only.  Stages: temporal_correlation, alert_sequence,
    topology_proximity, cross_diagram_correlation, propagation_hypothesis.
    """
    evidence: list[dict] = []
    ev_ids    = [e.get("event_id", f"EVT-{i:03d}") for i, e in enumerate(raw_events)]
    node_ids  = list(dict.fromkeys(e.get("node", "") for e in raw_events if e.get("node")))

    counter = 1

    # Stage 1 — temporal_correlation
    temporal_conf = dims.get("temporal", 0.0)
    if temporal_conf > 0 and raw_events:
        times    = [e.get("time_offset_min", 0) for e in raw_events]
        span_min = int(max(times) - min(times))
        if span_min == 0:
            summary = (
                f"{len(raw_events)} alert(s) co-occurred at t={int(min(times))} min, "
                f"indicating a simultaneous or burst-origin fault."
            )
        else:
            summary = (
                f"{len(raw_events)} alert(s) within a {span_min}-minute window "
                f"(t={int(min(times))}..{int(max(times))} min), consistent with a propagating fault."
            )
        evidence.append(make_causal_evidence_item(
            evidence_id=f"CE-{counter:03d}",
            stage="temporal_correlation",
            summary=summary,
            supporting_events=ev_ids,
            supporting_nodes=node_ids,
            confidence=temporal_conf,
        ))
        counter += 1

    # Stage 2 — alert_sequence
    seq     = [classify_alert(e.get("alert_type", "")) for e in raw_events]
    matched = [p for p in PROPAGATION_PATTERNS if is_subsequence(seq, p)]
    if matched:
        best       = matched[0]
        chain_str  = " -> ".join(best)
        at_conf    = dims.get("alert_type_seq", 0.0)
        evidence.append(make_causal_evidence_item(
            evidence_id=f"CE-{counter:03d}",
            stage="alert_sequence",
            summary=(
                f"Alert type sequence matches propagation chain: {chain_str}. "
                f"This ordering is consistent with fault propagation from the initiating node."
            ),
            supporting_events=ev_ids,
            supporting_nodes=node_ids,
            confidence=at_conf,
        ))
        counter += 1

    # Stage 3 — topology_proximity
    topo_conf = dims.get("topology", 0.0)
    if topo_conf > 0 and len(node_ids) >= 2:
        node_sample = node_ids[:4]
        suffix      = "..." if len(node_ids) > 4 else ""
        evidence.append(make_causal_evidence_item(
            evidence_id=f"CE-{counter:03d}",
            stage="topology_proximity",
            summary=(
                f"Alerted nodes ({', '.join(node_sample)}{suffix}) are topologically proximate "
                f"(topology score {topo_conf:.2f}), suggesting fault propagation through "
                f"connected infrastructure."
            ),
            supporting_events=ev_ids,
            supporting_nodes=node_ids,
            confidence=topo_conf,
        ))
        counter += 1

    # Stage 4 — cross_diagram_correlation (enterprise, multi-diagram only)
    cross_conf = dims.get("cross_diagram", 0.0)
    if mode == "enterprise_gnn_rca" and len(diagram_scope) > 1 and cross_conf > 0:
        evidence.append(make_causal_evidence_item(
            evidence_id=f"CE-{counter:03d}",
            stage="cross_diagram_correlation",
            summary=(
                f"Events span {len(diagram_scope)} diagrams "
                f"({', '.join(diagram_scope)}). "
                f"Cross-diagram correlation score {cross_conf:.2f} indicates the fault "
                f"may have propagated across topology boundaries."
            ),
            supporting_events=ev_ids,
            supporting_nodes=node_ids,
            confidence=cross_conf,
        ))
        counter += 1

    # Stage 5 — propagation_hypothesis (always included)
    seed_events = [ev_ids[i] for i, r in enumerate(roles) if r == "cluster_seed"]
    prop_events = [ev_ids[i] for i, r in enumerate(roles) if r == "propagation_signal"]
    signal_nodes = list(dict.fromkeys(
        raw_events[i].get("node", "")
        for i, r in enumerate(roles)
        if r in {"cluster_seed", "propagation_signal"} and raw_events[i].get("node")
    ))

    if prop_events:
        seed_node = raw_events[0].get("node", "?") if raw_events else "?"
        seed_at   = raw_events[0].get("alert_type", "?") if raw_events else "?"
        hyp_summary = (
            f"Hypothesis: fault initiated at {seed_node} (alert: {seed_at}) "
            f"and propagated to {len(prop_events)} downstream node(s) "
            f"matching known propagation patterns."
        )
    else:
        hyp_summary = (
            "Hypothesis: fault originated at the cluster seed node and may have propagated "
            "downstream through connected infrastructure."
        )

    hyp_conf = min(1.0, max(0.0, (dims.get("temporal", 0.0) + dims.get("topology", 0.0)) / 2))
    evidence.append(make_causal_evidence_item(
        evidence_id=f"CE-{counter:03d}",
        stage="propagation_hypothesis",
        summary=hyp_summary,
        supporting_events=seed_events + prop_events,
        supporting_nodes=signal_nodes,
        confidence=round(hyp_conf, 4),
    ))

    return evidence
