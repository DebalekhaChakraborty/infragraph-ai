"""
YOLO label file I/O and bounding-box conversion helpers.
"""

from pathlib import Path


CLASSES = [
    "router", "switch", "firewall", "server",
    "database", "load_balancer", "cloud_or_wan",
]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def load_labels(label_path: str | Path) -> list[dict]:
    """Read a YOLO .txt label file and return a list of annotation dicts.

    Each dict has keys: ``class_id``, ``class_name``, ``cx``, ``cy``,
    ``w``, ``h``  (all normalised 0-1).
    """
    rows = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            rows.append({
                "class_id":   cid,
                "class_name": CLASSES[cid] if cid < len(CLASSES) else str(cid),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w":  float(parts[3]),
                "h":  float(parts[4]),
            })
    return rows


def yolo_to_pixel(
    cx: float, cy: float, w: float, h: float,
    img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    """Convert normalised YOLO coords to pixel (x1, y1, x2, y2)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def pixel_to_yolo(
    x1: int, y1: int, x2: int, y2: int,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """Convert pixel (x1, y1, x2, y2) to normalised YOLO (cx, cy, w, h)."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def save_labels(annotations: list[dict], label_path: str | Path) -> None:
    """Write a list of annotation dicts back to a YOLO .txt label file."""
    lines = []
    for a in annotations:
        cid = a.get("class_id", CLASS_ID.get(a.get("class_name", ""), 0))
        lines.append(f"{cid} {a['cx']:.6f} {a['cy']:.6f} {a['w']:.6f} {a['h']:.6f}")
    Path(label_path).write_text("\n".join(lines))
