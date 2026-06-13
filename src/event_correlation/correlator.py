"""
correlator.py — Deterministic event correlation engine.

Groups observable alert events into coherent clusters using five scoring
dimensions: temporal proximity, topology proximity, alert-type sequence,
source/peer context, and cross-diagram correlation.

No root-cause labels, remediation steps, or evaluation fields are produced.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ._patterns import (
    PROPAGATION_PATTERNS,
    classify_alert,
    is_subsequence,
)

# ── Optional networkx import ────────────────────────────────────────────────────

try:
    import networkx as nx  # type: ignore
    _NX = True
except ImportError:
    _NX = False


# ── Deduplication ───────────────────────────────────────────────────────────────

def _dedup_key(event: dict) -> tuple:
    return (
        event.get("node", ""),
        event.get("alert_type", ""),
        event.get("severity", ""),
        event.get("time_offset_min", 0),
        event.get("diagram_id", ""),
    )


def _dedup_events(events: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    result: list[dict] = []
    for ev in events:
        k = _dedup_key(ev)
        if k not in seen:
            seen.add(k)
            result.append(ev)
    return result


# ── Graph builder ───────────────────────────────────────────────────────────────

def _build_graph(graph: dict | None) -> Any:
    if not _NX or graph is None:
        return None
    G: Any = nx.Graph()
    for node in graph.get("nodes", []):
        nid = node.get("id", "")
        if nid:
            G.add_node(nid)
    for edge in graph.get("edges", []):
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src and tgt and src != tgt:
            G.add_edge(src, tgt)
    for edge in graph.get("cross_diagram_edges", []):
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src and tgt and src != tgt:
            G.add_edge(src, tgt)
    return G if G.number_of_nodes() > 0 else None


def _node_to_diagram(graph: dict | None) -> dict[str, str]:
    """Build node_id → diagram_id lookup from graph node metadata."""
    if graph is None:
        return {}
    result: dict[str, str] = {}
    for node in graph.get("nodes", []):
        nid  = node.get("id", "")
        diag = node.get("diagram_id", "")
        if nid and diag:
            result[nid] = diag
    return result


# ── Score components ────────────────────────────────────────────────────────────

def _temporal_score(time_span_min: float) -> float:
    if time_span_min <= 15:
        return 1.0
    if time_span_min <= 30:
        return 0.85
    if time_span_min <= 60:
        return 0.65
    return 0.40


def _hop_score(hops: int) -> float:
    return {1: 1.0, 2: 0.85, 3: 0.65, 4: 0.45}.get(hops, 0.0)


def _topology_score(G: Any, alerted_nodes: list[str]) -> float:
    if not _NX or G is None or len(alerted_nodes) < 2:
        return 0.0
    node_set = set(G.nodes())
    scores: list[float] = []
    for i in range(len(alerted_nodes)):
        for j in range(i + 1, len(alerted_nodes)):
            a, b = alerted_nodes[i], alerted_nodes[j]
            if a == b or a not in node_set or b not in node_set:
                continue
            try:
                length = nx.shortest_path_length(G, a, b)
                scores.append(_hop_score(length))
            except Exception:
                scores.append(0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _alert_type_seq_score(events: list[dict]) -> float:
    seq = [classify_alert(e.get("alert_type", "")) for e in events]
    matches = sum(1 for p in PROPAGATION_PATTERNS if is_subsequence(seq, p))
    if matches == 0:
        return 0.20
    return min(1.0, 0.55 + 0.15 * matches)


def _source_peer_score(events: list[dict]) -> float:
    if len(events) < 2:
        return 0.0
    total = 0
    score = 0.0
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            d_i = events[i].get("diagram_id", "")
            d_j = events[j].get("diagram_id", "")
            total += 1
            score += 0.70 if (d_i and d_i == d_j) else 0.30
    return score / total if total > 0 else 0.0


def _cross_diagram_score(events: list[dict], graph: dict | None, mode: str) -> float:
    if mode != "enterprise_gnn_rca":
        return 0.0
    alerted_diagrams = sorted({e.get("diagram_id", "") for e in events if e.get("diagram_id")})
    if len(alerted_diagrams) <= 1:
        return 0.0
    if graph is None:
        return 0.40

    node_diag = _node_to_diagram(graph)
    alerted_set = set(alerted_diagrams)

    for edge in graph.get("cross_diagram_edges", []):
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if not src or not tgt:
            continue
        src_diag = (edge.get("source_diagram") or edge.get("from_diagram") or node_diag.get(src, ""))
        tgt_diag = (edge.get("target_diagram") or edge.get("to_diagram") or node_diag.get(tgt, ""))
        if src_diag and tgt_diag and src_diag != tgt_diag:
            if src_diag in alerted_set and tgt_diag in alerted_set:
                return 1.0

    # Alerted nodes span diagrams but no explicit connecting edges found
    return 0.60


# ── Weights per mode ─────────────────────────────────────────────────────────────

_WEIGHTS_TOPOLOGY: dict[str, float] = {
    "temporal":       0.30,
    "topology":       0.35,
    "alert_type_seq": 0.20,
    "source_peer":    0.15,
    "cross_diagram":  0.00,
}

_WEIGHTS_ENTERPRISE: dict[str, float] = {
    "temporal":       0.25,
    "topology":       0.30,
    "alert_type_seq": 0.20,
    "source_peer":    0.10,
    "cross_diagram":  0.15,
}


def _cluster_score_from_dims(dims: dict[str, float], mode: str) -> float:
    weights = _WEIGHTS_ENTERPRISE if mode == "enterprise_gnn_rca" else _WEIGHTS_TOPOLOGY
    total   = sum(weights[k] * dims.get(k, 0.0) for k in weights)
    return round(min(1.0, max(0.0, total)), 4)


# ── Correlation role assignment ─────────────────────────────────────────────────

# Pre-build: seed alert_type → patterns that start with it
_PATTERNS_BY_SEED: dict[str, list[tuple[str, ...]]] = defaultdict(list)
for _pat in PROPAGATION_PATTERNS:
    _PATTERNS_BY_SEED[_pat[0]].append(_pat)


def _assign_roles(events: list[dict]) -> list[str]:
    """
    Assign one correlation_role per event (index-aligned).

    First event (lowest time_offset_min) → cluster_seed.
    Subsequent events:
      propagation_signal  if alert type continues a pattern from the seed.
      peer_signal         if same diagram as seed (but not propagation).
      noise_candidate     otherwise.
    """
    if not events:
        return []
    roles: list[str] = ["noise_candidate"] * len(events)
    roles[0] = "cluster_seed"

    seed_at   = classify_alert(events[0].get("alert_type", ""))
    seed_diag = events[0].get("diagram_id", "")
    seen_ats: list[str] = [seed_at]

    for i, ev in enumerate(events[1:], start=1):
        ev_at   = classify_alert(ev.get("alert_type", ""))
        ev_diag = ev.get("diagram_id", "")

        # Check if ev_at continues any propagation chain started by seen_ats
        is_propagation = False
        for pat in PROPAGATION_PATTERNS:
            n = len(seen_ats)
            if len(pat) > n and list(pat[:n]) == seen_ats and pat[n] == ev_at:
                is_propagation = True
                break

        if is_propagation:
            roles[i] = "propagation_signal"
        elif ev_diag and ev_diag == seed_diag:
            roles[i] = "peer_signal"
        # else: noise_candidate (default)

        seen_ats.append(ev_at)

    return roles


# ── Temporal windowing ──────────────────────────────────────────────────────────

_WINDOW_MAX_MIN: int = 60  # events within 60 min of window start form one cluster


def _form_windows(events: list[dict]) -> list[list[dict]]:
    """Greedy 60-min temporal windows.  Input must be sorted by time_offset_min."""
    if not events:
        return []
    clusters: list[list[dict]] = []
    current: list[dict]        = [events[0]]
    window_start: int          = events[0].get("time_offset_min", 0)

    for ev in events[1:]:
        t = ev.get("time_offset_min", 0)
        if t - window_start <= _WINDOW_MAX_MIN:
            current.append(ev)
        else:
            clusters.append(current)
            current      = [ev]
            window_start = t

    if current:
        clusters.append(current)
    return clusters


# ── Public API ──────────────────────────────────────────────────────────────────

def correlate_events(
    events: list[dict],
    graph: dict | None,
    mode: str,
) -> list[dict]:
    """
    Correlate alert events into clusters.

    Returns a list of raw cluster dicts.  Each dict contains:
      raw_events    — deduplicated, time-sorted event list
      roles         — correlation_role per event (index-aligned)
      dims          — dict of five scoring dimensions
      cluster_score — weighted composite score
      reasons       — list of human-readable correlation reason strings
      diagram_scope — sorted list of diagram_ids present in cluster
    """
    deduped = _dedup_events(events)
    deduped.sort(key=lambda e: e.get("time_offset_min", 0))

    G       = _build_graph(graph)
    windows = _form_windows(deduped)

    raw_clusters: list[dict] = []
    for window_events in windows:
        times    = [e.get("time_offset_min", 0) for e in window_events]
        time_span = max(times) - min(times) if len(times) > 1 else 0

        alerted_nodes = list(dict.fromkeys(
            e.get("node", "") for e in window_events if e.get("node")
        ))
        diagrams = sorted({e.get("diagram_id", "") for e in window_events if e.get("diagram_id")})

        dims: dict[str, float] = {
            "temporal":       _temporal_score(time_span),
            "topology":       _topology_score(G, alerted_nodes),
            "alert_type_seq": _alert_type_seq_score(window_events),
            "source_peer":    _source_peer_score(window_events),
            "cross_diagram":  _cross_diagram_score(window_events, graph, mode),
        }
        score = _cluster_score_from_dims(dims, mode)
        roles = _assign_roles(window_events)

        # Build human-readable reasons
        reasons: list[str] = []
        t_lo, t_hi = int(min(times)), int(max(times))
        if time_span == 0:
            reasons.append(f"{len(window_events)} event(s) co-occurred at t={t_lo} min")
        else:
            reasons.append(
                f"{len(window_events)} event(s) span {int(time_span)} min "
                f"(t={t_lo}..{t_hi})"
            )

        seq = [classify_alert(e.get("alert_type", "")) for e in window_events]
        matched = [p for p in PROPAGATION_PATTERNS if is_subsequence(seq, p)]
        if matched:
            best = matched[0]
            reasons.append(f"Propagation pattern detected: {' -> '.join(best)}")

        if len(diagrams) > 1:
            reasons.append(f"Events span {len(diagrams)} diagrams: {', '.join(diagrams)}")

        if dims["topology"] >= 0.85:
            reasons.append("Alerted nodes are directly adjacent (1-2 hops)")
        elif dims["topology"] >= 0.55:
            reasons.append("Alerted nodes are topologically proximate (2-3 hops)")

        raw_clusters.append({
            "raw_events":    window_events,
            "roles":         roles,
            "dims":          dims,
            "cluster_score": score,
            "reasons":       reasons,
            "diagram_scope": diagrams,
        })

    return raw_clusters
