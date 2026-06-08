"""
Visualization helpers: draw YOLO boxes, overlay topology graphs, plot RCA scores.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


_CLASS_COLORS = {
    "router":        "#1565C0",
    "switch":        "#2E7D32",
    "firewall":      "#C62828",
    "server":        "#455A64",
    "database":      "#6A1B9A",
    "load_balancer": "#E65100",
    "cloud_or_wan":  "#0277BD",
}


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw YOLO bounding boxes and class labels on *image* (BGR numpy array).

    Parameters
    ----------
    detections: List of dicts with ``bbox`` (x1,y1,x2,y2) and ``class_name``.
    """
    import cv2
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls  = det.get("class_name", "?")
        conf = det.get("confidence", None)
        hex_col = _CLASS_COLORS.get(cls, "#FFFFFF")
        r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
        color = (b, g, r)  # BGR
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = cls if conf is None else f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    return out


def plot_topology_graph(
    G: nx.Graph,
    title: str = "Topology Graph",
    highlight_nodes: list[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot a NetworkX topology graph with per-class colours.

    Parameters
    ----------
    highlight_nodes: Node IDs to mark with a red ring (e.g., root-cause candidates).
    """
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42, k=1.5)

    node_colors = [_CLASS_COLORS.get(G.nodes[n].get("class_name", "server"), "#888888")
                   for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color="#AAAAAA")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=600)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: G.nodes[n].get("label", n) for n in G.nodes()},
                            font_size=7, font_color="white", font_weight="bold")

    if highlight_nodes:
        hl_pos = {n: pos[n] for n in highlight_nodes if n in pos}
        nx.draw_networkx_nodes(G, hl_pos, ax=ax, nodelist=list(hl_pos),
                               node_color="none", edgecolors="red",
                               node_size=750, linewidths=3)

    legend = [mpatches.Patch(color=c, label=cls.replace("_", " ").title())
              for cls, c in _CLASS_COLORS.items()]
    ax.legend(handles=legend, loc="upper left", fontsize=7, framealpha=0.85)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_rca_scores(
    candidates: list[dict],
    title: str = "RCA Candidate Scores",
    figsize: tuple[int, int] = (8, 4),
) -> plt.Figure:
    """Bar chart of root-cause candidate scores."""
    labels = [f"{c['node']}\n({c['class_name']})" for c in candidates]
    scores = [c["score"] for c in candidates]
    colors = [_CLASS_COLORS.get(c["class_name"], "#888888") for c in candidates]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel("Score")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.2 if scores else 1)
    fig.tight_layout()
    return fig
