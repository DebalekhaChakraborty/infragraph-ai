"""
edge_extraction — Vision-based connector/edge extraction for infrastructure diagrams.

Public API
----------
extract_edges_from_diagram(image_path, detected_nodes, min_confidence) -> dict
    Full pipeline: line detection → endpoint matching → edge candidates.

detect_connector_segments(image_path) -> dict
    Raw Hough line segment detection (cv2 optional).

match_segments_to_nodes(segments, detected_nodes, max_endpoint_distance) -> list[dict]
    Match segments to detected device bounding boxes.

render_connector_debug_overlay(image_path, segments, edges, output_path) -> str
    Optional debug overlay (cv2 optional, no-ops gracefully).
"""
from .edge_builder     import extract_edges_from_diagram
from .line_detector    import detect_connector_segments
from .endpoint_matcher import match_segments_to_nodes
from .debug_render     import render_connector_debug_overlay

__all__ = [
    "extract_edges_from_diagram",
    "detect_connector_segments",
    "match_segments_to_nodes",
    "render_connector_debug_overlay",
]
