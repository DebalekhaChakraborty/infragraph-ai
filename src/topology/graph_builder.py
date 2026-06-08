"""
Build a NetworkX topology graph from YOLO detections and detected connector lines.
"""

import math
import networkx as nx


def _centre(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _closest_node(
    px: float, py: float,
    nodes: list[dict],
    max_dist: float = 60.0,
) -> str | None:
    best_id, best_d = None, max_dist
    for n in nodes:
        cx, cy = _centre(n["bbox"])
        d = math.hypot(px - cx, py - cy)
        if d < best_d:
            best_d, best_id = d, n["id"]
    return best_id


def build_graph(
    detections: list[dict],
    lines: list[tuple[int, int, int, int]],
    max_snap_dist: float = 60.0,
) -> nx.Graph:
    """Assemble a NetworkX graph from detector output and line segments.

    Parameters
    ----------
    detections:    List of dicts with keys ``id``, ``class_name``, ``bbox``
                   (x1, y1, x2, y2 in pixel space) and optionally ``label``.
    lines:         List of (x1, y1, x2, y2) connector line segments.
    max_snap_dist: Maximum pixel distance to snap a line endpoint to a node centre.

    Returns
    -------
    G : nx.Graph  — nodes carry ``class_name`` and ``bbox`` attributes;
                    edges carry ``weight`` (pixel distance between centres).
    """
    G = nx.Graph()

    for det in detections:
        G.add_node(
            det["id"],
            class_name=det["class_name"],
            label=det.get("label", det["id"]),
            bbox=det["bbox"],
        )

    for x1, y1, x2, y2 in lines:
        src = _closest_node(x1, y1, detections, max_snap_dist)
        dst = _closest_node(x2, y2, detections, max_snap_dist)
        if src and dst and src != dst and not G.has_edge(src, dst):
            cx1, cy1 = _centre(G.nodes[src]["bbox"])
            cx2, cy2 = _centre(G.nodes[dst]["bbox"])
            G.add_edge(src, dst, weight=round(math.hypot(cx2 - cx1, cy2 - cy1), 1))

    return G


def graph_to_dict(G: nx.Graph) -> dict:
    """Serialise graph to a JSON-compatible dict."""
    return {
        "nodes": [
            {"id": n, **G.nodes[n]} for n in G.nodes
        ],
        "edges": [
            {"source": u, "target": v, **d} for u, v, d in G.edges(data=True)
        ],
    }
