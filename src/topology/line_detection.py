"""
Detect connector lines between network-device bounding boxes using
probabilistic Hough transform.
"""

import cv2
import numpy as np


def detect_lines(
    image: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 50,
    min_line_length: int = 40,
    max_line_gap: int = 20,
) -> list[tuple[int, int, int, int]]:
    """Return a list of (x1, y1, x2, y2) segments found in *image*.

    Parameters
    ----------
    image:           BGR or grayscale image (numpy array).
    rho:             Distance resolution of the accumulator in pixels.
    theta:           Angle resolution in radians.
    threshold:       Accumulator threshold — only lines > threshold are returned.
    min_line_length: Minimum number of pixels making up a line.
    max_line_gap:    Maximum gap in pixels between line segments to treat them
                     as a single line.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    raw = cv2.HoughLinesP(
        edges, rho, theta, threshold,
        minLineLength=min_line_length, maxLineGap=max_line_gap,
    )
    if raw is None:
        return []
    return [tuple(seg[0]) for seg in raw]


def filter_lines_by_bbox(
    lines: list[tuple[int, int, int, int]],
    bboxes: list[tuple[int, int, int, int]],
    margin: int = 15,
) -> list[tuple[int, int, int, int]]:
    """Keep only lines whose endpoints are close to a device bounding box.

    Parameters
    ----------
    lines:   List of (x1, y1, x2, y2) segments.
    bboxes:  List of (x1, y1, x2, y2) device bounding boxes.
    margin:  Pixel tolerance for endpoint proximity.
    """
    def near_any_bbox(px, py):
        for bx1, by1, bx2, by2 in bboxes:
            if (bx1 - margin) <= px <= (bx2 + margin) and \
               (by1 - margin) <= py <= (by2 + margin):
                return True
        return False

    return [
        (x1, y1, x2, y2) for x1, y1, x2, y2 in lines
        if near_any_bbox(x1, y1) or near_any_bbox(x2, y2)
    ]
