#!/usr/bin/env python3
"""Render a small clean-overlay review sheet from V3 gallery records."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from runtime_ingestion import render_v3_annotation_preview  # noqa: E402


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / value


def _select_records(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("source_dataset") != "v3":
            continue
        if not record.get("image_path") or not record.get("annotation_path"):
            continue
        dtype = str(record.get("source_diagram_id") or record.get("diagram_type") or "unknown")
        by_type[dtype].append(record)

    selected: list[dict[str, Any]] = []
    type_names = sorted(by_type)
    while len(selected) < count and any(by_type.values()):
        for dtype in type_names:
            if by_type[dtype]:
                selected.append(by_type[dtype].pop(0))
                if len(selected) >= count:
                    break
    return selected


def _make_contact_sheet(items: list[tuple[Path, dict[str, Any]]], out_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow is not installed; contact sheet skipped.")
        return

    if not items:
        return

    thumb_w, thumb_h = 520, 360
    pad = 18
    label_h = 58
    cols = 2
    rows = (len(items) + cols - 1) // cols
    sheet_w = cols * thumb_w + (cols + 1) * pad
    sheet_h = rows * (thumb_h + label_h) + (rows + 1) * pad + 54
    sheet = Image.new("RGB", (sheet_w, sheet_h), (248, 250, 252))
    draw = ImageDraw.Draw(sheet)
    try:
        title_font = ImageFont.truetype("arial.ttf", 22)
        label_font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        title_font = ImageFont.load_default()
        label_font = title_font

    draw.text((pad, 18), "InfraGraph V3 Clean Annotation Overlay Review", fill=(15, 23, 42), font=title_font)

    for idx, (img_path, meta) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = pad + col * (thumb_w + pad)
        y = 54 + pad + row * (thumb_h + label_h + pad)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
        frame = Image.new("RGB", (thumb_w, thumb_h), (255, 255, 255))
        ox = (thumb_w - img.width) // 2
        oy = (thumb_h - img.height) // 2
        frame.paste(img, (ox, oy))
        sheet.paste(frame, (x, y))
        draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=(203, 213, 225), width=1)
        label = (
            f"{meta.get('gallery_id', '')} | {meta.get('diagram_type', '')} | "
            f"boxes={meta.get('boxes_rendered', 0)} skipped={meta.get('boxes_skipped', 0)}"
        )
        draw.text((x, y + thumb_h + 10), label, fill=(30, 41, 59), font=label_font)
        draw.text(
            (x, y + thumb_h + 30),
            f"connectors_rendered={meta.get('connectors_rendered', 0)} mode={meta.get('overlay_mode', 'clean')}",
            fill=(71, 85, 105),
            font=label_font,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render clean V3 annotation overlay samples from the gallery manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", default=str(REPO_ROOT / "assets" / "gallery" / "manifest.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "annotation_overlays_review"))
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_root = Path(args.out)
    records = _load_json(manifest_path)
    if not isinstance(records, list):
        print(f"Manifest is not a list: {manifest_path}")
        return 1

    selected = _select_records(records, args.count)
    if not selected:
        print("No V3 gallery records with image and annotation paths were found.")
        return 1

    rendered: list[tuple[Path, dict[str, Any]]] = []
    print("Rendering clean V3 overlays:")
    for record in selected:
        gallery_id = str(record.get("gallery_id") or "unknown")
        diagram_type = str(record.get("source_diagram_id") or "unknown")
        image_path = _resolve_repo_path(str(record.get("image_path", "")))
        annotation_path = _resolve_repo_path(str(record.get("annotation_path", "")))
        out_dir = out_root / gallery_id
        out_path = out_dir / "detected.png"
        meta = render_v3_annotation_preview(
            image_path,
            annotation_path,
            out_path,
            overlay_mode="clean",
            draw_connectors=False,
        )
        meta["gallery_id"] = gallery_id
        meta["diagram_type"] = diagram_type
        rendered.append((out_path, meta))
        print(
            f"  {gallery_id:8s} {diagram_type:28s} "
            f"boxes={meta.get('boxes_rendered', 0):2d} "
            f"skipped={meta.get('boxes_skipped', 0):2d} "
            f"connectors={meta.get('connectors_rendered', 0):2d} "
            f"mode={meta.get('overlay_mode')}"
        )

    contact_sheet = out_root / "contact_sheet.png"
    _make_contact_sheet(rendered, contact_sheet)
    print()
    print(f"Contact sheet: {contact_sheet}")
    print(f"Output root:   {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
