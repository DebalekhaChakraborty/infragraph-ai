"""
endpoint_matcher.py — Match detected line segments to detected device nodes.

match_segments_to_nodes(segments, detected_nodes, max_endpoint_distance) -> list[dict]

For each segment, both endpoints are matched to the nearest node bounding box.
If the two endpoints map to different nodes within max_endpoint_distance pixels,
a candidate edge is emitted.  Self-loops and duplicates are removed; when multiple
segments map to the same pair the highest-confidence candidate wins.
"""
from __future__ import annotations

import math


def _bbox_from_node(node: dict) -> "list[float] | None":
    """
    Extract [x1, y1, x2, y2] from a node dict.

    Accepts:
      bbox = [x1, y1, x2, y2]  (xyxy)
      bbox = [x, y, w, h]       (detected via bbox_format)
      x / y / width / height
    Returns None when no usable bbox is found.
    """
    bbox = node.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        a, b, c, d = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        fmt = (node.get("bbox_format") or "").lower()
        if fmt == "xywh":
            x1, y1, x2, y2 = a, b, a + c, b + d
        else:
            x1, y1 = min(a, c), min(b, d)
            x2, y2 = max(a, c), max(b, d)
        if (x2 - x1) >= 1 and (y2 - y1) >= 1:
            return [x1, y1, x2, y2]

    if all(k in node for k in ("x", "y", "width", "height")):
        x = float(node["x"]); y = float(node["y"])
        return [x, y, x + float(node["width"]), y + float(node["height"])]

    return None


def _point_to_bbox_dist(px: float, py: float, bbox: list[float]) -> float:
    """Minimum Euclidean distance from point (px, py) to the nearest point inside bbox."""
    x1, y1, x2, y2 = bbox
    cx = max(x1, min(x2, px))
    cy = max(y1, min(y2, py))
    return math.hypot(px - cx, py - cy)


def match_segments_to_nodes(
    segments: list[dict],
    detected_nodes: list[dict],
    max_endpoint_distance: float = 80.0,
) -> list[dict]:
    """
    Match line segments to detected device bounding boxes.

    Parameters
    ----------
    segments              : from detect_connector_segments()
    detected_nodes        : from RF-DETR / annotation ingestion; each needs a bbox
    max_endpoint_distance : max pixels from segment endpoint to bbox boundary

    Returns
    -------
    list[dict] — deduplicated candidate edges, highest confidence kept per pair.
        source                   : str  — node_id
        target                   : str  — node_id
        segment_id               : str
        endpoint_distance_source : float
        endpoint_distance_target : float
        connector_confidence     : float
        source_type              : "vision_connector_extraction"
    """
    # Build (node_id, bbox) list — skip nodes without a usable bbox
    node_bboxes: list[tuple[str, list[float]]] = []
    for node in detected_nodes:
        nid = (
            node.get("node_id")
            or node.get("id")
            or node.get("object_id")
            or node.get("canonical_id")
            or ""
        )
        if not nid:
            continue
        bbox = _bbox_from_node(node)
        if bbox:
            node_bboxes.append((nid, bbox))

    if not node_bboxes or not segments:
        return []

    def _nearest(px: float, py: float) -> "tuple[str, float]":
        best_id, best_d = "", float("inf")
        for nid, bbox in node_bboxes:
            d = _point_to_bbox_dist(px, py, bbox)
            if d < best_d:
                best_d, best_id = d, nid
        return best_id, best_d

    # (canonical_pair) -> best candidate
    edge_map: dict[tuple[str, str], dict] = {}

    for seg in segments:
        x1, y1 = float(seg["x1"]), float(seg["y1"])
        x2, y2 = float(seg["x2"]), float(seg["y2"])

        src_id, dist_src = _nearest(x1, y1)
        tgt_id, dist_tgt = _nearest(x2, y2)

        # Both endpoints must be close enough to a node
        if dist_src > max_endpoint_distance or dist_tgt > max_endpoint_distance:
            continue
        # No self-loops
        if not src_id or not tgt_id or src_id == tgt_id:
            continue

        # Canonical (undirected) pair for deduplication
        pair = (min(src_id, tgt_id), max(src_id, tgt_id))

        # Confidence: segment score penalised by endpoint distance
        dist_penalty   = (dist_src + dist_tgt) / (2.0 * max_endpoint_distance)
        conn_confidence = round(
            seg.get("confidence", 0.5) * max(0.0, 1.0 - 0.35 * dist_penalty),
            3,
        )

        candidate = {
            "source":                   src_id,
            "target":                   tgt_id,
            "segment_id":               seg["segment_id"],
            "endpoint_distance_source": round(dist_src, 1),
            "endpoint_distance_target": round(dist_tgt, 1),
            "connector_confidence":     conn_confidence,
            "source_type":              "vision_connector_extraction",
        }

        if pair not in edge_map or conn_confidence > edge_map[pair]["connector_confidence"]:
            edge_map[pair] = candidate

    return list(edge_map.values())
