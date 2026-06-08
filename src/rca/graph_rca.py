"""
Rule-based graph-traversal Root Cause Analysis.

Given a topology graph and a list of alerts, walks upstream from alerted nodes
and scores each candidate root cause by criticality, alert severity, and
graph centrality.
"""

import networkx as nx


_SEVERITY_SCORE = {"critical": 3, "major": 2, "warning": 1}
_CLASS_CRITICALITY = {
    "router": 5, "firewall": 5, "database": 4,
    "load_balancer": 3, "switch": 3, "server": 2, "cloud_or_wan": 2,
}


def run_rca(
    G: nx.Graph,
    alerts: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Score every alerted node and return the top-k root-cause candidates.

    Parameters
    ----------
    G:       Topology graph (nodes must have ``class_name`` attribute).
    alerts:  List of alert dicts with at minimum ``node`` and ``severity`` keys.
    top_k:   Number of top candidates to return.

    Returns
    -------
    List of dicts ``{node, class_name, score, reasons}`` sorted descending by score.
    """
    alerted_nodes = {a["node"] for a in alerts if a["node"] in G}
    if not alerted_nodes:
        return []

    # Accumulate alert severity per node
    node_severity: dict[str, float] = {}
    for a in alerts:
        n = a["node"]
        if n in G:
            node_severity[n] = node_severity.get(n, 0) + _SEVERITY_SCORE.get(a.get("severity", "warning"), 1)

    # Betweenness centrality — high-centrality nodes are more likely root causes
    centrality = nx.betweenness_centrality(G)

    scores = []
    for n in alerted_nodes:
        cls      = G.nodes[n].get("class_name", "server")
        sev_sc   = node_severity.get(n, 0)
        cent_sc  = centrality.get(n, 0) * 10
        crit_sc  = _CLASS_CRITICALITY.get(cls, 1)

        # Upstream bonus: does this node have alerted downstream neighbours?
        downstream_alerted = sum(
            1 for nb in G.neighbors(n) if nb in alerted_nodes and nb != n
        )

        total = sev_sc + cent_sc + crit_sc + downstream_alerted * 1.5
        scores.append({
            "node":       n,
            "class_name": cls,
            "score":      round(total, 3),
            "reasons": {
                "alert_severity":       sev_sc,
                "betweenness_centrality": round(cent_sc, 3),
                "device_criticality":   crit_sc,
                "downstream_alerted":   downstream_alerted,
            },
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]
