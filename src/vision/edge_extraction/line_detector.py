"""
line_detector.py — Vision-based connector/line segment detection.

detect_connector_segments(image_path) -> dict

Uses OpenCV probabilistic Hough transform to detect line segments that likely
represent network connectors/edges in infrastructure diagrams.

Degrades gracefully when OpenCV is unavailable — returns ok=False with a clear
warning; callers must fall back to annotation/local-graph edges.
"""
from __future__ import annotations

import math
from pathlib import Path


def detect_connector_segments(image_path: "str | Path") -> dict:
    """
    Detect connector line segments in an infrastructure diagram image.

    Parameters
    ----------
    image_path : path to the diagram image (PNG / JPG / BMP).

    Returns
    -------
    dict
        ok            : bool
        segments      : list[dict] — segment_id, x1/y1/x2/y2, length, angle, confidence
        source        : str        — "hough_line_detector" | "opencv_unavailable" | "error"
        segment_count : int
        warning       : str
    """
    try:
        import cv2           # type: ignore
        import numpy as np   # type: ignore
    except ImportError:
        return {
            "ok":            False,
            "segments":      [],
            "source":        "opencv_unavailable",
            "segment_count": 0,
            "warning":       (
                "OpenCV not installed — install opencv-python-headless for "
                "vision connector extraction"
            ),
        }

    image_path = Path(image_path)
    if not image_path.exists():
        return {
            "ok":            False,
            "segments":      [],
            "source":        "image_not_found",
            "segment_count": 0,
            "warning":       f"Image not found: {image_path}",
        }

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return {
                "ok":            False,
                "segments":      [],
                "source":        "image_load_failed",
                "segment_count": 0,
                "warning":       f"cv2.imread returned None for {image_path.name}",
            }

        h, w = img.shape[:2]
        img_diagonal = math.hypot(w, h)

        # ── Grayscale ───────────────────────────────────────────────────────────
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # ── Canny edges (conservative — avoid false positives inside device boxes)
        canny = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3)

        # ── Adaptive threshold (helps with thin/light connectors on white backgrounds)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=6,
        )
        kern        = np.ones((2, 2), np.uint8)
        thresh_thin = cv2.erode(thresh, kern, iterations=1)

        # Merge both edge maps
        combined = cv2.bitwise_or(canny, thresh_thin)

        # ── Probabilistic Hough transform ────────────────────────────────────────
        # Conservative thresholds — rather miss edges than add phantom ones.
        min_line_length = max(25, int(img_diagonal * 0.028))
        max_line_gap    = max(10, int(img_diagonal * 0.012))

        lines = cv2.HoughLinesP(
            combined,
            rho=1,
            theta=math.pi / 180,
            threshold=20,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        segments: list[dict] = []
        if lines is not None:
            for i, line in enumerate(lines, 1):
                x1, y1, x2, y2 = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[0][3])
                length = math.hypot(x2 - x1, y2 - y1)
                if length < min_line_length:
                    continue

                angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

                # Heuristic confidence: longer lines → higher confidence
                len_ratio  = min(length / img_diagonal, 1.0)
                confidence = 0.30 + 0.55 * len_ratio          # 0.30 – 0.85

                # Slight boost for near-axis lines (most connectors are H/V)
                angle_mod = angle % 90
                if angle_mod < 8 or angle_mod > 82:
                    confidence = min(confidence + 0.05, 0.92)

                segments.append({
                    "segment_id": f"SEG-{i:03d}",
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "length":     round(length, 1),
                    "angle":      round(angle, 1),
                    "confidence": round(confidence, 3),
                })

        # Sort longest first (longest = most likely real connector)
        segments.sort(key=lambda s: s["length"], reverse=True)

        return {
            "ok":            True,
            "segments":      segments,
            "source":        "hough_line_detector",
            "segment_count": len(segments),
            "warning":       "" if segments else "No line segments detected above threshold",
        }

    except Exception as exc:
        return {
            "ok":            False,
            "segments":      [],
            "source":        "error",
            "segment_count": 0,
            "warning":       f"Connector segment detection error: {exc}",
        }
