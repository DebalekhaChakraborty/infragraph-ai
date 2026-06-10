#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

COLORS = {
    "router": "#2563eb",
    "switch": "#16a34a",
    "firewall": "#dc2626",
    "server": "#475569",
    "database": "#7c3aed",
    "load_balancer": "#ea580c",
    "cloud_or_wan": "#0284c7",
    "service": "#0d9488",
}

def get_bbox(obj):
    for key in ["bbox", "box", "xyxy"]:
        if key in obj and obj[key]:
            b = obj[key]
            if len(b) == 4:
                return b
    if all(k in obj for k in ["x", "y", "width", "height"]):
        return [obj["x"], obj["y"], obj["x"] + obj["width"], obj["y"] + obj["height"]]
    if all(k in obj for k in ["x1", "y1", "x2", "y2"]):
        return [obj["x1"], obj["y1"], obj["x2"], obj["y2"]]
    return None

def normalize_bbox(b):
    x1, y1, x2, y2 = map(float, b)
    # COCO xywh guard
    if x2 <= x1 or y2 <= y1:
        x2 = x1 + abs(x2)
        y2 = y1 + abs(y2)
    return int(x1), int(y1), int(x2), int(y2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    image_path = Path(args.image)
    ann_path = Path(args.annotation)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    ann = json.loads(ann_path.read_text())

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = None
        small = None

    # Draw connectors first
    for c in ann.get("connectors", []):
        pts = c.get("points") or c.get("polyline") or []
        if len(pts) >= 2:
            pts2 = []
            for p in pts:
                if isinstance(p, dict):
                    pts2.append((int(p.get("x", 0)), int(p.get("y", 0))))
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts2.append((int(p[0]), int(p[1])))
            if len(pts2) >= 2:
                draw.line(pts2, fill="#06b6d4", width=3)

    # Draw boxes
    for obj in ann.get("objects", []):
        b = get_bbox(obj)
        if not b:
            continue
        x1, y1, x2, y2 = normalize_bbox(b)
        cls = obj.get("type") or obj.get("class") or obj.get("label") or obj.get("category") or "device"
        node_id = obj.get("id") or obj.get("node_id") or obj.get("name") or cls
        color = COLORS.get(cls, "#facc15")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        label = f"{node_id} | {cls}"
        if "confidence" in obj:
            label += f" {float(obj['confidence']):.2f}"

        tb = draw.textbbox((x1, y1), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x1, max(0, y1 - th - 8), x1 + tw + 10, y1], fill=color)
        draw.text((x1 + 5, max(0, y1 - th - 5)), label, fill="white", font=font)

    # Footer watermark
    footer = "Verified Annotation Overlay — rendered from V3 metadata"
    draw.rectangle([0, img.height - 34, img.width, img.height], fill="#0f172a")
    draw.text((14, img.height - 26), footer, fill="#e2e8f0", font=small)

    img.save(out_path)
    print(out_path)

if __name__ == "__main__":
    main()
