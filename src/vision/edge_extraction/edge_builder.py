"""
edge_builder.py — High-level vision connector extraction entry point.

extract_edges_from_diagram(image_path, detected_nodes, min_confidence) -> dict

Orchestrates:
  1. detect_connector_segments()  — Hough line detection
  2. match_segments_to_nodes()    — endpoint-to-bbox matching
  3. confidence filtering
"""
from __future__ import annotations

from pathlib import Path

from .line_detector    import detect_connector_segments
from .endpoint_matcher import match_segments_to_nodes


def extract_edges_from_diagram(
    image_path:     "str | Path",
    detected_nodes: list[dict],
    min_confidence: float = 0.35,
) -> dict:
    """
    Run the full vision connector extraction pipeline.

    Parameters
    ----------
    image_path     : path to the original diagram image
    detected_nodes : list of nodes with bbox (from RF-DETR or annotation ingestion)
    min_confidence : minimum connector_confidence to include an edge

    Returns
    -------
    dict
        ok            : bool   — True only if at least one edge passes confidence gate
        edges         : list   — filtered edge candidates
        segments      : list   — all detected segments (for debug)
        source        : str    — "vision_connector_extraction" | "unavailable" | ...
        edge_count    : int
        segment_count : int
        warning       : str    — empty on success; honest message otherwise
    """
    seg_result = detect_connector_segments(image_path)

    if not seg_result.get("ok"):
        return {
            "ok":            False,
            "edges":         [],
            "segments":      [],
            "source":        seg_result.get("source", "unavailable"),
            "edge_count":    0,
            "segment_count": 0,
            "warning":       seg_result.get("warning", ""),
        }

    segments   = seg_result.get("segments", [])
    candidates = match_segments_to_nodes(segments, detected_nodes)

    edges = [
        e for e in candidates
        if e.get("connector_confidence", 0.0) >= min_confidence
    ]

    if not edges:
        warning = (
            "Vision connector extraction did not meet confidence threshold "
            "— using verified graph metadata fallback."
        )
    elif seg_result.get("warning"):
        warning = seg_result["warning"]
    else:
        warning = ""

    return {
        "ok":            bool(edges),
        "edges":         edges,
        "segments":      segments,
        "source":        "vision_connector_extraction",
        "edge_count":    len(edges),
        "segment_count": len(segments),
        "warning":       warning,
    }
