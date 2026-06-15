"""
debug_render.py — Optional debug overlay for vision connector extraction.

render_connector_debug_overlay(image_path, segments, edges, output_path) -> str

Draws:
  • All detected line segments in cyan (thin)
  • Matched edges in green with endpoint circles
  • Edge label "source → target" and confidence score at midpoint
  • Footer banner with segment/edge counts

No-ops gracefully when OpenCV is unavailable; returns "" in that case.
"""
from __future__ import annotations

import math
from pathlib import Path


def render_connector_debug_overlay(
    image_path:  "str | Path",
    segments:    list[dict],
    edges:       list[dict],
    output_path: "str | Path",
) -> str:
    """
    Write a debug overlay image.

    Returns the output path as a string on success, "" on any failure.
    """
    output_path = Path(output_path)
    image_path  = Path(image_path)

    try:
        import cv2         # type: ignore
        import numpy as np # type: ignore
    except ImportError:
        return ""

    if not image_path.exists():
        return ""

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return ""

        h, w = img.shape[:2]
        overlay = img.copy()

        # ── 1. Draw all detected segments (cyan, thin) ────────────────────────
        for seg in segments:
            cv2.line(
                overlay,
                (seg["x1"], seg["y1"]),
                (seg["x2"], seg["y2"]),
                color=(0, 220, 220),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        # ── 2. Draw matched edges (green) + endpoint circles ──────────────────
        seg_map = {s["segment_id"]: s for s in segments}
        for edge in edges:
            seg = seg_map.get(edge.get("segment_id", ""))
            if not seg:
                continue
            x1, y1, x2, y2 = seg["x1"], seg["y1"], seg["x2"], seg["y2"]

            # Green thick line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 200, 80), 2, cv2.LINE_AA)

            # Endpoint circles
            cv2.circle(overlay, (x1, y1), 5, (0, 230, 90), -1)
            cv2.circle(overlay, (x2, y2), 5, (0, 230, 90), -1)

            # Label at midpoint
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            label  = f"{edge['source']} → {edge['target']}"
            conf   = f"{edge.get('connector_confidence', 0):.2f}"
            ly1 = max(12, my - 4)
            ly2 = max(24, my + 10)
            cv2.putText(overlay, label, (max(0, mx - 30), ly1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 140, 50), 1, cv2.LINE_AA)
            cv2.putText(overlay, conf,  (max(0, mx - 30), ly2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (30, 100, 30), 1, cv2.LINE_AA)

        # ── 3. Blend overlay with original ────────────────────────────────────
        result = cv2.addWeighted(img, 0.45, overlay, 0.55, 0)

        # ── 4. Footer banner ──────────────────────────────────────────────────
        cv2.rectangle(result, (0, h - 22), (w, h), (240, 240, 245), -1)
        cv2.putText(
            result,
            f"Vision Connector Debug | {len(segments)} segments | {len(edges)} edges matched",
            (8, h - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.37,
            (30, 80, 200), 1, cv2.LINE_AA,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        return str(output_path)

    except Exception:
        return ""
