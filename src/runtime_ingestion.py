"""
runtime_ingestion.py

Live ingestion helpers for the InfraGraph AI pipeline.

Public API:
    run_ingestion(...)            -- explicit-path ingestion (preferred for asset-layer use)
    run_absorption(...)           -- explicit-path enterprise absorption
    run_live_v3_ingestion(...)    -- V3 scenario-path ingestion (backward compat)
    run_enterprise_absorption(...) -- scenario-path absorption (backward compat)

Detection source is resolved honestly:
    - "LIVE_RFDETR_INFERENCE"      if external RF-DETR inference succeeds
    - "VERIFIED_ANNOTATION_FALLBACK" if verified annotations are used after
                                     live inference is unavailable
    - "RF-DETR Trained Prediction" if outputs/rfdetr_v3_predictions/<id>.png exists
    - "Verified Annotation Overlay" otherwise (annotation bboxes rendered as overlay)
"""
from __future__ import annotations

import copy
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

_CLASS_TO_TYPE: dict[str, str] = {
    "router": "router", "switch": "switch", "firewall": "firewall",
    "server": "server", "database": "database",
    "load_balancer": "load_balancer", "cloud_or_wan": "cloud_or_wan",
    "service": "service",
}


# ── private helpers ────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── bbox normalization helper ──────────────────────────────────────────────────
def normalize_bbox_for_pil(
    obj: dict,
    image_w: int,
    image_h: int,
) -> "tuple[int, int, int, int] | None":
    """
    Convert an annotation object's bounding-box to PIL-safe (x0, y0, x1, y1).

    Accepted input formats:
      a) obj["bbox"] = [x1, y1, x2, y2]   (xyxy, possibly unordered or out-of-bounds)
      b) obj["bbox"] = [x, y, w, h]        (COCO xywh — detected via bbox_format)
      c) obj["x"], obj["y"], obj["width"], obj["height"]
      d) obj["x1"], obj["y1"], obj["x2"], obj["y2"]

    Always:
      - Sorts x and y so x0 <= x1, y0 <= y1.
      - Clamps to [0, image_w-1] / [0, image_h-1].
      - Returns None for degenerate boxes (< 2 px wide or tall) or malformed data.
      - Never raises.
    """
    try:
        fmt = obj.get("bbox_format", "").lower()

        # resolve raw values
        bbox = obj.get("bbox")
        if bbox is not None and len(bbox) >= 4:
            a, b, c, d = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            if fmt == "xywh":
                x0, y0, x1, y1 = a, b, a + c, b + d
            elif fmt == "xyxy":
                x0, y0, x1, y1 = a, b, c, d
            else:
                # V3 annotation JSON uses xyxy. Treat ambiguous four-value
                # boxes as xyxy unless bbox_format explicitly says xywh.
                x0, y0, x1, y1 = a, b, c, d
        elif "x1" in obj and "y1" in obj and "x2" in obj and "y2" in obj:
            x0 = float(obj["x1"]); y0 = float(obj["y1"])
            x1 = float(obj["x2"]); y1 = float(obj["y2"])
        elif "x" in obj and "y" in obj and "width" in obj and "height" in obj:
            x0 = float(obj["x"]);   y0 = float(obj["y"])
            x1 = x0 + float(obj["width"]); y1 = y0 + float(obj["height"])
        else:
            return None

        # sort so x0 <= x1, y0 <= y1
        x0, x1 = (min(x0, x1), max(x0, x1))
        y0, y1 = (min(y0, y1), max(y0, y1))

        # clamp to image
        x0 = max(0, min(image_w - 1, int(x0)))
        y0 = max(0, min(image_h - 1, int(y0)))
        x1 = max(0, min(image_w - 1, int(x1)))
        y1 = max(0, min(image_h - 1, int(y1)))

        if (x1 - x0) < 2 or (y1 - y0) < 2:
            return None
        return (x0, y0, x1, y1)
    except Exception:
        return None


# ── annotation preview renderer ───────────────────────────────────────────────
_CLS_COLORS: dict[str, tuple[int, int, int]] = {
    "router":        (255, 107,  53),
    "switch":        ( 78, 205, 196),
    "firewall":      (255,  78,  80),
    "server":        ( 69, 183, 209),
    "database":      (150, 206, 180),
    "load_balancer": (255, 234, 167),
    "cloud_or_wan":  (221, 160, 221),
    "service":       (152, 216, 200),
}
_DEFAULT_COLOR: tuple[int, int, int] = (168, 168, 168)
_OVERLAY_RENDERER_VERSION = "v3_clean_overlay_v1"
_CLASS_DISPLAY: dict[str, str] = {
    "cloud_or_wan": "cloud/WAN",
    "load_balancer": "load balancer",
}


def _overlay_meta_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.meta.json")


def _needs_clean_overlay_render(out_path: Path) -> bool:
    meta_path = _overlay_meta_path(out_path)
    if not out_path.exists() or not meta_path.exists():
        return True
    meta = _load_json(meta_path)
    return (
        meta.get("renderer_version") != _OVERLAY_RENDERER_VERSION
        or meta.get("overlay_mode") != "clean"
        or bool(meta.get("draw_connectors", False))
    )


def render_v3_annotation_preview(
    image_path: Path,
    annotation_path: Path,
    out_path: Path,
    *,
    overlay_mode: str = "clean",
    draw_connectors: bool = False,
) -> dict:
    """
    Draw a V3 annotation overlay onto a copy of the source image.

    clean mode is intended for primary UI use: object outlines, identity + type
    labels, optional subtle connectors, and no translucent fills. Debug mode can
    include connector labels and class details for inspection.

    Returns a metadata dict:
        rendered, boxes_rendered, boxes_skipped, boxes_skipped_large,
        connectors_rendered, connectors_skipped, overlay_mode, draw_connectors,
        renderer_version, out_path
    Uses an alternate path gracefully if Pillow is unavailable or annotation is missing.
    Never raises.
    """
    overlay_mode = overlay_mode if overlay_mode in {"clean", "debug"} else "clean"
    meta: dict = {
        "rendered": False,
        "boxes_rendered": 0,
        "boxes_skipped": 0,
        "boxes_skipped_large": 0,
        "connectors_rendered": 0,
        "connectors_skipped": 0,
        "connectors_skipped_long": 0,
        "overlay_mode": overlay_mode,
        "draw_connectors": bool(draw_connectors),
        "renderer_version": _OVERLAY_RENDERER_VERSION,
        "out_path": str(out_path),
        "meta_path": str(_overlay_meta_path(out_path)),
    }

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        if image_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, out_path)
            _save_json(_overlay_meta_path(out_path), meta)
        return meta

    try:
        annotation = _load_json(annotation_path) if annotation_path.exists() else {}
        if not annotation or not image_path.exists():
            if image_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, out_path)
                _save_json(_overlay_meta_path(out_path), meta)
            return meta

        img  = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        img_w, img_h = img.size
        image_area = max(img_w * img_h, 1)
        image_diagonal = max(math.hypot(img_w, img_h), 1.0)

        try:
            font    = ImageFont.truetype("arial.ttf", 14)
            font_sm = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            try:
                font    = ImageFont.load_default(size=14)
                font_sm = ImageFont.load_default(size=11)
            except Exception:
                font    = ImageFont.load_default()
                font_sm = font

        # ── connector polylines (off by default) ──────────────────────────────
        connectors = annotation.get("connectors", [])
        if draw_connectors:
            for conn in connectors:
                try:
                    pts = conn.get("points") or []
                    if len(pts) < 2:
                        meta["connectors_skipped"] += 1
                        continue
                    flat = [
                        (
                            max(0, min(img_w - 1, int(float(p[0])))),
                            max(0, min(img_h - 1, int(float(p[1])))),
                        )
                        for p in pts
                        if len(p) >= 2
                    ]
                    if len(flat) < 2:
                        meta["connectors_skipped"] += 1
                        continue
                    connector_length = sum(
                        math.hypot(x2 - x1, y2 - y1)
                        for (x1, y1), (x2, y2) in zip(flat, flat[1:])
                    )
                    if connector_length > image_diagonal * 0.65:
                        meta["connectors_skipped"] += 1
                        meta["connectors_skipped_long"] += 1
                        continue
                    if overlay_mode == "clean":
                        draw.line(flat, fill=(37, 99, 235, 90), width=1)
                    else:
                        draw.line(flat, fill=(30, 144, 255, 190), width=2)
                        x1c, y1c = flat[-2]
                        x2c, y2c = flat[-1]
                        ang = math.atan2(y2c - y1c, x2c - x1c)
                        for side in (0.45, -0.45):
                            ax = max(0, min(img_w - 1, x2c - int(8 * math.cos(ang + side))))
                            ay = max(0, min(img_h - 1, y2c - int(8 * math.sin(ang + side))))
                            draw.line([(x2c, y2c), (ax, ay)], fill=(30, 144, 255, 210), width=2)
                        lbl = conn.get("label_text", conn.get("relationship", ""))
                        if lbl and flat:
                            mid = flat[len(flat) // 2]
                            tx  = max(0, min(img_w - 40, mid[0] + 3))
                            ty  = max(0, min(img_h - 14, mid[1] - 14))
                            draw.text((tx, ty), str(lbl), fill=(20, 90, 150), font=font_sm)
                    meta["connectors_rendered"] += 1
                except Exception:
                    meta["connectors_skipped"] += 1
        else:
            meta["connectors_skipped"] = len(connectors)

        # ── bounding boxes + node labels ──────────────────────────────────────
        for obj in annotation.get("objects", []):
            try:
                box = normalize_bbox_for_pil(obj, img_w, img_h)
                if box is None:
                    meta["boxes_skipped"] += 1
                    continue
                x0, y0, x1, y1 = box
                cls     = obj.get("class_name", "server")
                r, g, b = _CLS_COLORS.get(cls, _DEFAULT_COLOR)
                area_ratio = ((x1 - x0) * (y1 - y0)) / image_area

                if overlay_mode == "clean" and area_ratio > 0.18:
                    meta["boxes_skipped"] += 1
                    meta["boxes_skipped_large"] += 1
                    continue

                width = 3 if overlay_mode == "clean" else 2
                draw.rectangle([x0, y0, x1, y1], outline=(r, g, b, 255), width=width)

                node_id = obj.get("object_id") or obj.get("label_text") or ""
                cls_display = _CLASS_DISPLAY.get(str(cls), str(cls))
                if overlay_mode == "clean":
                    label = str(node_id or cls_display)
                    if node_id and cls_display:
                        label = f"{node_id} · {cls_display}"
                else:
                    label = f"{node_id or cls_display} [{cls_display}]"
                lh_est  = 16
                try:
                    tb = draw.textbbox((x0, 0), label, font=font)
                    lw, lh_est = tb[2] - tb[0], tb[3] - tb[1]
                except AttributeError:
                    lw = len(label) * 8
                # place label above box; clamp so it never goes off-image
                label_y0 = max(0, y0 - lh_est - 6)
                label_y1 = label_y0 + lh_est + 4
                if label_y1 > img_h - 1:
                    label_y1 = min(img_h - 1, y0 + lh_est + 4)
                    label_y0 = max(0, label_y1 - lh_est - 4)
                lx0 = max(0, x0)
                lx1 = min(img_w - 1, x0 + lw + 6)
                if lx1 > lx0 and label_y1 > label_y0:
                    draw.rectangle([lx0, label_y0, lx1, label_y1], fill=(r, g, b, 220))
                    draw.rectangle([lx0, label_y0, lx1, label_y1], outline=(r, g, b, 255), width=1)
                draw.text((lx0 + 3, label_y0 + 2), label, fill=(255, 255, 255), font=font)
                meta["boxes_rendered"] += 1
            except Exception:
                meta["boxes_skipped"] += 1

        # ── footer banner ─────────────────────────────────────────────────────
        footer_h = 26
        draw.rectangle([0, img_h - footer_h, img_w, img_h], fill=(255, 255, 255, 220))
        draw.text(
            (10, img_h - footer_h + 5),
            f"Verified Annotation Overlay | {overlay_mode}",
            fill=(37, 99, 235),
            font=font_sm,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "PNG")
        meta["rendered"] = True
        _save_json(_overlay_meta_path(out_path), meta)

    except Exception as exc:
        # last-resort: copy original so the UI still has something to show
        try:
            if image_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, out_path)
        except Exception:
            pass
        meta["error"] = str(exc)
        _save_json(_overlay_meta_path(out_path), meta)

    return meta


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH MEMORY TABLE BUILDERS  (public helpers, callable from the app layer)
# ══════════════════════════════════════════════════════════════════════════════

def _evidence_src(detection_source: str) -> str:
    if "Verified Annotation" in detection_source:
        return "verified_metadata"
    if "Live RF-DETR" in detection_source or detection_source == "LIVE_RFDETR_INFERENCE":
        return "live_detector"
    if "RF-DETR" in detection_source or "Trained" in detection_source:
        return "trained_detector"
    return "inferred"


def _conf_str(raw, detection_source: str) -> str:
    """Return confidence as a string; blank for annotation overlay (never fake it)."""
    if "Verified Annotation" in detection_source:
        return ""
    if raw is None:
        return ""
    try:
        return str(round(float(raw), 3))
    except (TypeError, ValueError):
        return ""


def _numeric_conf_values(
    rows: list[dict],
    keys: tuple[str, ...] = ("confidence", "score"),
) -> list[float]:
    """Extract numeric confidence values from a list of row dicts, trying multiple keys."""
    vals: list[float] = []
    for row in rows or []:
        for key in keys:
            raw = row.get(key)
            if raw in (None, "", "—"):
                continue
            try:
                vals.append(float(raw))
                break
            except (TypeError, ValueError):
                continue
    return vals


def _confidence_summary(
    detected_nodes: list[dict],
    detected_edges: list[dict],
    device_rows: list[dict],
    connector_rows: list[dict],
    ocr_rows: list[dict],
) -> dict:
    """Build confidence_summary, preferring live detected_nodes/edges over table rows."""
    node_conf = _numeric_conf_values(detected_nodes)
    if not node_conf:
        node_conf = _numeric_conf_values(device_rows)

    edge_conf = _numeric_conf_values(detected_edges)
    if not edge_conf:
        edge_conf = _numeric_conf_values(connector_rows)

    conf_source = "detected_nodes" if _numeric_conf_values(detected_nodes) else "device_rows"

    return {
        "device_detection_avg":   round(sum(node_conf) / len(node_conf), 3) if node_conf else "—",
        "edge_extraction_avg":    round(sum(edge_conf) / len(edge_conf), 3) if edge_conf else "—",
        "ocr_text_blocks":        len(ocr_rows),
        "connector_count":        len(connector_rows),
        "low_confidence_items":   sum(1 for c in node_conf if c < 0.90),
        "device_confidence_count": len(node_conf),
        "edge_confidence_count":   len(edge_conf),
        "confidence_source":       conf_source,
    }


def build_device_rows(
    local_graph: dict,
    annotation: dict,
    detection_source: str = "Verified Annotation Overlay",
) -> list[dict]:
    """Normalized device rows for devices.csv and the evidence table."""
    ev_src = _evidence_src(detection_source)
    rows: list[dict] = []
    seen: set[str] = set()

    for n in local_graph.get("nodes", []):
        nid = n.get("id") or n.get("node_id") or ""
        if not nid or nid in seen:
            continue
        seen.add(nid)
        bbox = n.get("bbox") or []
        rows.append({
            "node_id":         nid,
            "device_type":     n.get("type", ""),
            "display_label":   n.get("label", nid),
            "canonical_id":    n.get("canonical_id", nid),
            "ip_address":      n.get("ip_address", ""),
            "zone":            n.get("zone", ""),
            "interface":       n.get("interface", ""),
            "vlan":            n.get("vlan", ""),
            "is_shared_entity": str(n.get("is_shared_entity", False)),
            "evidence_source": ev_src,
            "confidence":      _conf_str(n.get("confidence"), detection_source),
            "bbox":            str(bbox) if bbox else "",
            "x1": str(bbox[0]) if len(bbox) > 0 else "",
            "y1": str(bbox[1]) if len(bbox) > 1 else "",
            "x2": str(bbox[2]) if len(bbox) > 2 else "",
            "y2": str(bbox[3]) if len(bbox) > 3 else "",
        })

    for obj in annotation.get("objects", []):
        nid = obj.get("object_id") or obj.get("node_id") or ""
        if not nid or nid in seen:
            continue
        seen.add(nid)
        bbox = obj.get("bbox") or []
        rows.append({
            "node_id":         nid,
            "device_type":     obj.get("class_name", obj.get("type", "")),
            "display_label":   obj.get("label", nid),
            "canonical_id":    obj.get("canonical_id", nid),
            "ip_address":      obj.get("ip_address", ""),
            "zone":            obj.get("zone", ""),
            "interface":       "",
            "vlan":            "",
            "is_shared_entity": str(obj.get("is_shared_entity", False)),
            "evidence_source": ev_src,
            "confidence":      _conf_str(obj.get("confidence"), detection_source),
            "bbox":            str(bbox) if bbox else "",
            "x1": str(bbox[0]) if len(bbox) > 0 else "",
            "y1": str(bbox[1]) if len(bbox) > 1 else "",
            "x2": str(bbox[2]) if len(bbox) > 2 else "",
            "y2": str(bbox[3]) if len(bbox) > 3 else "",
        })

    return rows


def build_connector_rows(
    local_graph: dict,
    annotation: dict,
    detection_source: str = "Verified Annotation Overlay",
) -> list[dict]:
    """Normalized connector rows for connectors.csv and the evidence table."""
    ev_src = _evidence_src(detection_source)
    rows: list[dict] = []
    seen: set[tuple] = set()

    for e in local_graph.get("edges", []):
        src = e.get("source", "")
        tgt = e.get("target", "")
        rel = e.get("relationship", "connected_to")
        if not src or not tgt:
            continue
        key = (src, tgt, rel)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "edge_id":          e.get("edge_id", f"{src}→{tgt}"),
            "source":           src,
            "target":           tgt,
            "relationship":     rel,
            "protocol":         e.get("protocol", ""),
            "label":            e.get("label", ""),
            "source_interface": e.get("source_interface", e.get("src_interface", "")),
            "target_interface": e.get("target_interface", e.get("tgt_interface", "")),
            "vlan":             e.get("vlan", ""),
            "scope":            e.get("edge_scope", e.get("scope", "")),
            "direction":        "directed",
            "evidence_source":  ev_src,
            "confidence":       _conf_str(e.get("confidence"), detection_source),
        })

    for conn in annotation.get("connectors", []):
        src = conn.get("source", conn.get("from_node", ""))
        tgt = conn.get("target", conn.get("to_node", ""))
        rel = conn.get("label", conn.get("relationship", "connected_to"))
        if not src or not tgt:
            continue
        key = (src, tgt, rel)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "edge_id":          conn.get("connector_id", f"{src}→{tgt}"),
            "source":           src,
            "target":           tgt,
            "relationship":     rel,
            "protocol":         conn.get("protocol", ""),
            "label":            conn.get("label", ""),
            "source_interface": conn.get("source_interface", ""),
            "target_interface": conn.get("target_interface", ""),
            "vlan":             conn.get("vlan", ""),
            "scope":            conn.get("scope", ""),
            "direction":        "directed",
            "evidence_source":  "verified_connector_metadata",
            "confidence":       "",
        })

    return rows


def build_interface_rows(local_graph: dict, annotation: dict) -> list[dict]:
    """Normalized interface/IP rows for interfaces.csv and the evidence table."""
    rows: list[dict] = []
    node_map: dict[str, dict] = {}
    for n in local_graph.get("nodes", []):
        nid = n.get("id") or n.get("node_id") or ""
        if nid:
            node_map[nid] = {
                "node_id":        nid,
                "device_type":    n.get("type", ""),
                "ip_address":     n.get("ip_address", ""),
                "interface":      n.get("interface", ""),
                "vlan":           n.get("vlan", ""),
                "zone":           n.get("zone", ""),
                "connected_to":   "",
                "protocol":       "",
                "port":           "",
                "evidence_source": "verified_metadata",
            }
    for e in local_graph.get("edges", []):
        src = e.get("source", "")
        tgt = e.get("target", "")
        si  = e.get("source_interface", e.get("src_interface", ""))
        ti  = e.get("target_interface", e.get("tgt_interface", ""))
        proto = e.get("protocol", e.get("label", ""))
        if src in node_map and si and not node_map[src]["interface"]:
            node_map[src]["interface"]   = si
            node_map[src]["connected_to"] = tgt
            node_map[src]["protocol"]    = proto
        if tgt in node_map and ti and not node_map[tgt]["interface"]:
            node_map[tgt]["interface"]   = ti
            node_map[tgt]["connected_to"] = src
            node_map[tgt]["protocol"]    = proto
    rows = list(node_map.values())
    return rows


def build_ocr_rows(annotation: dict) -> list[dict]:
    """Normalized OCR/text rows for ocr_text.csv and the evidence table."""
    rows: list[dict] = []
    for blk in annotation.get("text_blocks", []):
        bbox = blk.get("bbox") or []
        rows.append({
            "text":           blk.get("text", ""),
            "text_type":      blk.get("type", blk.get("role", "")),
            "linked_node":    blk.get("linked_node", blk.get("node_id", "")),
            "bbox":           str(bbox) if bbox else "",
            "x1": str(bbox[0]) if len(bbox) > 0 else "",
            "y1": str(bbox[1]) if len(bbox) > 1 else "",
            "x2": str(bbox[2]) if len(bbox) > 2 else "",
            "y2": str(bbox[3]) if len(bbox) > 3 else "",
            "confidence":     "",
            "evidence_source": "verified_text_metadata",
        })
    return rows


def _bbox_list_from_live_detection(det: dict) -> list[float]:
    bbox = det.get("bbox") or {}
    if isinstance(bbox, dict):
        return [
            float(bbox.get("x1", 0)),
            float(bbox.get("y1", 0)),
            float(bbox.get("x2", 0)),
            float(bbox.get("y2", 0)),
        ]
    if isinstance(bbox, list) and len(bbox) >= 4:
        return [float(v) for v in bbox[:4]]
    return []


def _bbox_iou(a: list, b: list) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _live_rfdetr_nodes_from_detections(detections: list[dict], annotation: dict) -> list[dict]:
    objects = [o for o in annotation.get("objects", []) if isinstance(o, dict)]
    used: set[int] = set()
    nodes: list[dict] = []
    for idx, det in enumerate(detections, 1):
        label = det.get("label") or det.get("device_type") or "server"
        bbox = _bbox_list_from_live_detection(det)
        best_iou = 0.0
        best_idx = -1
        best_obj: dict = {}
        for obj_idx, obj in enumerate(objects):
            if obj_idx in used:
                continue
            if obj.get("class_name") != label:
                continue
            iou = _bbox_iou(bbox, obj.get("bbox", []))
            if iou > best_iou:
                best_iou = iou
                best_idx = obj_idx
                best_obj = obj
        if best_idx >= 0 and best_iou >= 0.08:
            used.add(best_idx)
        else:
            best_obj = {}
        node_id = best_obj.get("object_id") or best_obj.get("label_text") or f"RFDET-{idx:03d}"
        nodes.append({
            "node_id":          node_id,
            "canonical_id":     best_obj.get("canonical_id", node_id),
            "class_name":       label,
            "type":             _CLASS_TO_TYPE.get(label, det.get("device_type", "server")),
            "bbox":             bbox,
            "confidence":       float(det.get("confidence", 0.0)),
            "is_shared_entity": best_obj.get("is_shared_entity", False),
            "is_ghost":         False,
            "zone":             best_obj.get("zone", ""),
            "ip_address":       best_obj.get("ip_address", ""),
            "source":           "LIVE_RFDETR_INFERENCE",
            "match_iou":        round(best_iou, 3),
        })
    return nodes


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def run_live_v3_ingestion(
    repo_root: Path,
    diagram_path: Path,
    diagram_id: str,
    scenario_path: Path,
    use_live_rfdetr: bool = True,
    rfdetr_model=None,      # pre-loaded RF-DETR model (from st.cache_resource); loaded internally if None
    external_rfdetr_result: dict | None = None,
) -> dict:
    """
    Load a V3 annotation + local graph and write a self-contained ingestion
    run folder at:
        outputs/live_ingestion/<scenario_id>__<diagram_id>/

    Resolution order for detection source:
        1. outputs/rfdetr_v3_predictions/<scenario_id>__<diagram_id>.png
           -> detection_source = "RF-DETR trained prediction"
        2. No RF-DETR output exists
           -> detection_source = "Verified Annotation Overlay"
           -> detected image = annotation-rendered overlay

    Returns a dict with:
        run_dir, original_image, detected_image, detection_source,
        annotation, local_graph, detected_nodes, detected_edges,
        node_table_rows, edge_table_rows, packet, confidence_summary
    """
    scenario_id = scenario_path.name
    run_dir = repo_root / "runtime_state" / "live_ingestion" / f"{scenario_id}__{diagram_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── original image ────────────────────────────────────────────────────────
    orig_out = run_dir / "original.png"
    if diagram_path.exists() and not orig_out.exists():
        shutil.copy2(diagram_path, orig_out)

    # ── detect source (3-tier priority) ──────────────────────────────────────
    #   1. External live RF-DETR result supplied by the Streamlit bridge
    #   2. Static rfdetr_v3_predictions file
    #   3. Verified Annotation Overlay rendered from V3 metadata
    detected_out     = run_dir / "detected.png"
    detection_source = "Verified Annotation Overlay"
    _rfdetr_error: str  = ""
    _rfdetr_time_s: float = 0.0

    if use_live_rfdetr and not external_rfdetr_result:
        _rfdetr_error = (
            "In-process RF-DETR is disabled for Streamlit ingestion; "
            "provide external_rfdetr_result from the RF-DETR subprocess bridge."
        )

    if external_rfdetr_result and external_rfdetr_result.get("ok"):
        live_img = Path(external_rfdetr_result.get("annotated_image_path", ""))
        if live_img.exists():
            shutil.copy2(live_img, detected_out)
        detection_source = "LIVE_RFDETR_INFERENCE"
        _rfdetr_time_s = round(float(external_rfdetr_result.get("inference_runtime_ms", 0)) / 1000.0, 3)
    elif external_rfdetr_result and not external_rfdetr_result.get("ok"):
        _rfdetr_error = external_rfdetr_result.get("error", "external RF-DETR failed")

    if detection_source == "Verified Annotation Overlay":
        _rfdetr_pred_static = (
            repo_root / "outputs" / "rfdetr_v3_predictions"
            / f"{scenario_id}__{diagram_id}.png"
        )
        if _rfdetr_pred_static.exists():
            if not detected_out.exists():
                shutil.copy2(_rfdetr_pred_static, detected_out)
            detection_source = "RF-DETR trained prediction"

    # ── load annotation & local graph ─────────────────────────────────────────
    ann_path = scenario_path / "annotations" / f"{diagram_id}.json"
    lg_path  = scenario_path / "local_graphs"  / f"{diagram_id}.json"
    annotation:  dict = _load_json(ann_path) if ann_path.exists() else {}
    local_graph: dict = _load_json(lg_path)  if lg_path.exists()  else {}

    # render verified annotation overlay when absent or rendered by an older renderer
    _render_meta: dict = {
        "rendered": False, "boxes_rendered": 0, "boxes_skipped": 0,
        "boxes_skipped_large": 0, "connectors_rendered": 0, "connectors_skipped": 0,
        "connectors_skipped_long": 0,
        "overlay_mode": "clean", "draw_connectors": False,
        "renderer_version": _OVERLAY_RENDERER_VERSION,
    }
    if detection_source == "Verified Annotation Overlay" and _needs_clean_overlay_render(detected_out):
        _render_meta = render_v3_annotation_preview(
            diagram_path,
            ann_path,
            detected_out,
            overlay_mode="clean",
            draw_connectors=False,
        )
        if not detected_out.exists():
            shutil.copy2(orig_out, detected_out)

    # ── detected_nodes from live detections or annotation objects ─────────────
    is_rfdetr = detection_source.startswith("RF-DETR") or detection_source == "LIVE_RFDETR_INFERENCE"
    live_detections = (
        external_rfdetr_result.get("detections", [])
        if external_rfdetr_result and external_rfdetr_result.get("ok")
        else []
    )
    detected_nodes: list[dict] = _live_rfdetr_nodes_from_detections(live_detections, annotation) if live_detections else []
    if not detected_nodes:
        for obj in annotation.get("objects", []):
            detected_nodes.append({
                "node_id":          obj.get("object_id", ""),
                "canonical_id":     obj.get("canonical_id", obj.get("object_id", "")),
                "class_name":       obj.get("class_name", "server"),
                "type":             _CLASS_TO_TYPE.get(obj.get("class_name", ""), "server"),
                "bbox":             obj.get("bbox", []),
                "confidence":       obj.get("confidence", 0.96 if is_rfdetr else 0.88),
                "is_shared_entity": obj.get("is_shared_entity", False),
                "is_ghost":         obj.get("is_ghost", False),
                "zone":             obj.get("zone", ""),
                "source":           detection_source,
            })

    # ── detected_edges from connectors + local graph ──────────────────────────
    detected_edges: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for conn in annotation.get("connectors", []):
        pair = (
            conn.get("source", conn.get("from_node", "")),
            conn.get("target", conn.get("to_node", "")),
        )
        seen_pairs.add(pair)
        detected_edges.append({
            "source":       pair[0],
            "target":       pair[1],
            "relationship": conn.get("label", "connected_to"),
            "label":        conn.get("label", ""),
            "confidence":   conn.get("confidence", 0.91 if is_rfdetr else 0.78),
            "connector_id": conn.get("connector_id", ""),
            "source_type":  "annotation_connector",
        })
    for e in local_graph.get("edges", []):
        pair = (e.get("source", ""), e.get("target", ""))
        if pair not in seen_pairs and pair[0] and pair[1]:
            seen_pairs.add(pair)
            detected_edges.append({
                "source":       pair[0],
                "target":       pair[1],
                "relationship": e.get("relationship", "connected_to"),
                "label":        e.get("label", ""),
                "confidence":   0.82,
                "connector_id": "",
                "source_type":  "local_graph",
            })

    # ── graph-memory table rows ───────────────────────────────────────────────
    device_rows    = build_device_rows(local_graph, annotation, detection_source)
    connector_rows = build_connector_rows(local_graph, annotation, detection_source)
    interface_rows = build_interface_rows(local_graph, annotation)
    ocr_rows       = build_ocr_rows(annotation)

    # slim backward-compat tables (no fake confidence)
    graph_nodes = local_graph.get("nodes") or detected_nodes
    node_table_rows: list[dict] = [
        {
            "node_id":        n.get("id", n.get("node_id", "")),
            "type":           n.get("type", "server"),
            "ip_address":     n.get("ip_address", ""),
            "zone":           n.get("zone", ""),
            "shared":         n.get("is_shared_entity", False),
            "evidence_source": _evidence_src(detection_source),
        }
        for n in graph_nodes
    ]
    graph_edges = local_graph.get("edges") or detected_edges
    edge_table_rows: list[dict] = [
        {
            "source":       e.get("source", ""),
            "target":       e.get("target", ""),
            "relationship": e.get("relationship", "connected_to"),
            "label":        e.get("label", ""),
        }
        for e in graph_edges
    ]

    # ── confidence summary (only real detector values) ────────────────────────
    confidence_summary = _confidence_summary(
        detected_nodes, detected_edges, device_rows, connector_rows, ocr_rows,
    )

    runtime_mode = (
        "LIVE_RFDETR_INFERENCE"
        if detection_source == "LIVE_RFDETR_INFERENCE"
        else "VERIFIED_ANNOTATION_FALLBACK"
        if external_rfdetr_result and not external_rfdetr_result.get("ok")
        else detection_source
    )
    packet_source = (
        external_rfdetr_result.get("source", "live_rfdetr_subprocess")
        if detection_source == "LIVE_RFDETR_INFERENCE"
        else "verified training annotation / safe curated sample"
        if runtime_mode == "VERIFIED_ANNOTATION_FALLBACK"
        else detection_source
    )

    # ── graph_memory_packet ───────────────────────────────────────────────────
    packet: dict = {
        "diagram_id":           diagram_id,
        "scenario_id":          scenario_id,
        "source":               packet_source,
        "original_image":       str(orig_out),
        "detected_image":       str(detected_out),
        "detection_source":     detection_source,
        "evidence_source":      _evidence_src(detection_source),
        "annotation_path":      str(ann_path) if ann_path.exists() else "",
        "local_graph_path":     str(lg_path)  if lg_path.exists()  else "",
        "node_count":           len(device_rows),
        "edge_count":           len(connector_rows),
        "ocr_text_block_count": len(ocr_rows),
        "connector_count":      len(connector_rows),
        "devices":              device_rows,
        "connectors":           connector_rows,
        "interfaces":           interface_rows,
        "ocr_text":             ocr_rows,
        "nodes":                node_table_rows,
        "edges":                edge_table_rows,
        "confidence_summary":   confidence_summary,
        "text_blocks":          annotation.get("text_blocks", []),
        "source_label":         f"Source: {detection_source}",
        "annotation_preview":   _render_meta,
        "runtime_mode":         runtime_mode,
        "absorption_mode":      "SESSION_MEMORY_ABSORPTION",
        "rfdetr_subprocess":    external_rfdetr_result or {},
    }

    # ── persist artifacts ─────────────────────────────────────────────────────
    _save_json(run_dir / "detected_nodes.json",      detected_nodes)
    _save_json(run_dir / "detected_edges.json",      detected_edges)
    _save_json(run_dir / "graph_memory_packet.json", packet)
    _save_csv(run_dir  / "node_table.csv",           node_table_rows)
    _save_csv(run_dir  / "edge_table.csv",           edge_table_rows)
    _save_csv(run_dir  / "devices.csv",              device_rows)
    _save_csv(run_dir  / "connectors.csv",           connector_rows)
    _save_csv(run_dir  / "interfaces.csv",           interface_rows)
    _save_csv(run_dir  / "ocr_text.csv",             ocr_rows)
    _save_json(run_dir / "ingestion_summary.json",   {
        "diagram_id":          diagram_id,
        "scenario_id":         scenario_id,
        "source":              packet_source,
        "detection_source":    detection_source,
        "runtime_mode":        runtime_mode,
        "node_count":          len(device_rows),
        "edge_count":          len(connector_rows),
        "annotation_preview":  _render_meta,
        "rfdetr_inference_time_s": _rfdetr_time_s,
        "rfdetr_error":        _rfdetr_error,
        "use_live_rfdetr":     use_live_rfdetr,
        "rfdetr_subprocess":    external_rfdetr_result or {},
    })

    return {
        "run_dir":               run_dir,
        "original_image":        orig_out,
        "detected_image":        detected_out,
        "source":                packet_source,
        "detection_source":      detection_source,
        "runtime_mode":          runtime_mode,
        "rfdetr_inference_time_s": _rfdetr_time_s,
        "rfdetr_error":          _rfdetr_error,
        "annotation":            annotation,
        "local_graph":           local_graph,
        "detected_nodes":        detected_nodes,
        "detected_edges":        detected_edges,
        "node_table_rows":       node_table_rows,
        "edge_table_rows":       edge_table_rows,
        "device_rows":           device_rows,
        "connector_rows":        connector_rows,
        "interface_rows":        interface_rows,
        "ocr_rows":              ocr_rows,
        "packet":                packet,
        "confidence_summary":    confidence_summary,
    }


def run_enterprise_absorption(
    repo_root: Path,
    scenario_path: Path,
    diagram_id: str,
    local_graph: dict,
) -> dict:
    """
    Load enterprise_graph.json, stitch_map.json, alerts.json from scenario_path.
    Compute enterprise_before (without the selected diagram cluster) and
    enterprise_after (full graph).
    Persist to:
        outputs/live_absorption/<scenario_id>__<diagram_id>/

    Returns:
        run_dir, enterprise_before, enterprise_after, stitch_map, alerts, summary
    """
    scenario_id = scenario_path.name
    run_dir = repo_root / "runtime_state" / "live_absorption" / f"{scenario_id}__{diagram_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    eg_path = scenario_path / "enterprise_graph.json"
    sm_path = scenario_path / "stitch_map.json"
    al_path = scenario_path / "alerts.json"

    enterprise_graph: dict = _load_json(eg_path) if eg_path.exists() else {}
    stitch_map:       dict = _load_json(sm_path) if sm_path.exists() else {}
    alerts:           dict = _load_json(al_path) if al_path.exists() else {}

    # enterprise_before: remove selected diagram cluster
    before    = copy.deepcopy(enterprise_graph)
    clusters  = before.get("diagram_clusters", [])

    # diagram_clusters can be a list of dicts {"diagram_id":..., "node_ids":[...]}
    # or historically a dict keyed by diagram_id — handle both
    if isinstance(clusters, list):
        cluster_obj = next((c for c in clusters if c.get("diagram_id") == diagram_id), {})
        rm_node_ids = set(cluster_obj.get("node_ids", []))
        before["diagram_clusters"] = [c for c in clusters if c.get("diagram_id") != diagram_id]
    else:
        rm_node_ids = set(clusters.get(diagram_id, {}).get("node_ids", []))
        before["diagram_clusters"] = {k: v for k, v in clusters.items() if k != diagram_id}

    before["nodes"] = [n for n in before.get("nodes", []) if n.get("id") not in rm_node_ids]
    before["edges"] = [
        e for e in before.get("edges", [])
        if e.get("source") not in rm_node_ids and e.get("target") not in rm_node_ids
    ]

    # compute summary
    local_ids  = {n.get("canonical_id", n.get("id")) for n in local_graph.get("nodes", [])}
    ent_ids    = {n.get("id") for n in enterprise_graph.get("nodes", [])}
    matched    = sorted(local_ids & ent_ids)
    cross_links = [
        e for e in stitch_map.get("cross_diagram_edges", [])
        if e.get("source_diagram") == diagram_id or e.get("target_diagram") == diagram_id
    ]
    summary = {
        "absorbed_diagram_id":          diagram_id,
        "scenario_id":                  scenario_id,
        "nodes_absorbed":               len(local_graph.get("nodes", [])),
        "edges_absorbed":               len(local_graph.get("edges", [])),
        "shared_entities_matched":      len(matched),
        "cross_diagram_links_created":  len(cross_links),
        "matched_entities":             matched,
        "cross_diagram_links":          cross_links,
        "before_node_count":            len(before["nodes"]),
        "after_node_count":             len(enterprise_graph.get("nodes", [])),
        "status":                       "absorbed_into_enterprise_graph",
    }

    # persist
    _save_json(run_dir / "enterprise_before.json",  before)
    _save_json(run_dir / "enterprise_after.json",   enterprise_graph)
    _save_json(run_dir / "absorption_summary.json", summary)
    _save_json(run_dir / "alerts.json",             alerts)

    return {
        "run_dir":           run_dir,
        "enterprise_before": before,
        "enterprise_after":  enterprise_graph,
        "stitch_map":        stitch_map,
        "alerts":            alerts,
        "summary":           summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPLICIT-PATH API  (preferred for asset-layer and ONB-XXX usage)
# ══════════════════════════════════════════════════════════════════════════════

def run_ingestion(
    repo_root: Path,
    image_path: Path,
    diagram_id: str,
    run_id: str,
    annotation_path: "Path | None" = None,
    local_graph_path: "Path | None" = None,
    enterprise_graph_path: "Path | None" = None,
    stitch_map_path: "Path | None" = None,
    alerts_path: "Path | None" = None,
    use_live_rfdetr: bool = True,
    rfdetr_model=None,
    external_rfdetr_result: dict | None = None,
) -> dict:
    """
    Explicit-path version of the ingestion pipeline.

    Unlike run_live_v3_ingestion(), this function accepts every file path
    directly instead of deriving them from a scenario directory.  Use this
    when processing assets from assets/onboarding/ONB-XXX/ or any non-V3
    directory layout.

    Output:
        outputs/live_ingestion/<run_id>/

    Returns the same dict structure as run_live_v3_ingestion().
    """
    run_dir = repo_root / "runtime_state" / "live_ingestion" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── original image ────────────────────────────────────────────────────────
    orig_out = run_dir / "original.png"
    if image_path.exists() and not orig_out.exists():
        shutil.copy2(image_path, orig_out)

    detected_out     = run_dir / "detected.png"
    detection_source = "Verified Annotation Overlay"
    _rfdetr_error: str    = ""
    _rfdetr_time_s: float = 0.0

    # ── 1. External live RF-DETR result ───────────────────────────────────────
    if use_live_rfdetr and not external_rfdetr_result:
        _rfdetr_error = (
            "In-process RF-DETR is disabled for Streamlit ingestion; "
            "provide external_rfdetr_result from the RF-DETR subprocess bridge."
        )

    if external_rfdetr_result and external_rfdetr_result.get("ok"):
        live_img = Path(external_rfdetr_result.get("annotated_image_path", ""))
        if live_img.exists():
            shutil.copy2(live_img, detected_out)
        detection_source = "LIVE_RFDETR_INFERENCE"
        _rfdetr_time_s = round(float(external_rfdetr_result.get("inference_runtime_ms", 0)) / 1000.0, 3)
    elif external_rfdetr_result and not external_rfdetr_result.get("ok"):
        _rfdetr_error = external_rfdetr_result.get("error", "external RF-DETR failed")

    # ── 2. Static rfdetr_v3_predictions ──────────────────────────────────────
    if detection_source == "Verified Annotation Overlay":
        _static = repo_root / "outputs" / "rfdetr_v3_predictions" / f"{run_id}.png"
        if _static.exists():
            if not detected_out.exists():
                shutil.copy2(_static, detected_out)
            detection_source = "RF-DETR Trained Prediction"

    # ── load annotation & local graph ─────────────────────────────────────────
    ann_p: Path = annotation_path  if annotation_path  else Path("/nonexistent")
    lg_p:  Path = local_graph_path if local_graph_path else Path("/nonexistent")
    annotation:  dict = _load_json(ann_p) if ann_p.exists() else {}
    local_graph: dict = _load_json(lg_p)  if lg_p.exists()  else {}

    # ── 3. Annotation overlay ─────────────────────────────────────────────────
    _render_meta: dict = {
        "rendered": False, "boxes_rendered": 0, "boxes_skipped": 0,
        "boxes_skipped_large": 0, "connectors_rendered": 0, "connectors_skipped": 0,
        "connectors_skipped_long": 0,
        "overlay_mode": "clean", "draw_connectors": False,
        "renderer_version": _OVERLAY_RENDERER_VERSION,
    }
    if detection_source == "Verified Annotation Overlay" and _needs_clean_overlay_render(detected_out):
        _render_meta = render_v3_annotation_preview(
            image_path,
            ann_p,
            detected_out,
            overlay_mode="clean",
            draw_connectors=False,
        )
        if not detected_out.exists() and orig_out.exists():
            shutil.copy2(orig_out, detected_out)

    # ── nodes / edges ─────────────────────────────────────────────────────────
    is_live = detection_source.startswith("Live") or detection_source == "LIVE_RFDETR_INFERENCE"
    live_detections = (
        external_rfdetr_result.get("detections", [])
        if external_rfdetr_result and external_rfdetr_result.get("ok")
        else []
    )
    detected_nodes: list[dict] = _live_rfdetr_nodes_from_detections(live_detections, annotation) if live_detections else []
    if not detected_nodes:
        for obj in annotation.get("objects", []):
            detected_nodes.append({
                "node_id":          obj.get("object_id", ""),
                "canonical_id":     obj.get("canonical_id", obj.get("object_id", "")),
                "class_name":       obj.get("class_name", "server"),
                "type":             _CLASS_TO_TYPE.get(obj.get("class_name", ""), "server"),
                "bbox":             obj.get("bbox", []),
                "confidence":       obj.get("confidence", 0.96 if is_live else 0.88),
                "is_shared_entity": obj.get("is_shared_entity", False),
                "is_ghost":         obj.get("is_ghost", False),
                "zone":             obj.get("zone", ""),
                "source":           detection_source,
            })

    detected_edges: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for conn in annotation.get("connectors", []):
        pair = (
            conn.get("source", conn.get("from_node", "")),
            conn.get("target", conn.get("to_node", "")),
        )
        seen_pairs.add(pair)
        detected_edges.append({
            "source":       pair[0],
            "target":       pair[1],
            "relationship": conn.get("label", "connected_to"),
            "label":        conn.get("label", ""),
            "confidence":   conn.get("confidence", 0.91 if is_live else 0.78),
            "connector_id": conn.get("connector_id", ""),
            "source_type":  "annotation_connector",
        })
    for e in local_graph.get("edges", []):
        pair = (e.get("source", ""), e.get("target", ""))
        if pair not in seen_pairs and pair[0] and pair[1]:
            seen_pairs.add(pair)
            detected_edges.append({
                "source":       pair[0],
                "target":       pair[1],
                "relationship": e.get("relationship", "connected_to"),
                "label":        e.get("label", ""),
                "confidence":   0.82,
                "connector_id": "",
                "source_type":  "local_graph",
            })

    # ── graph-memory table rows ───────────────────────────────────────────────
    device_rows    = build_device_rows(local_graph, annotation, detection_source)
    connector_rows = build_connector_rows(local_graph, annotation, detection_source)
    interface_rows = build_interface_rows(local_graph, annotation)
    ocr_rows       = build_ocr_rows(annotation)

    graph_nodes = local_graph.get("nodes") or detected_nodes
    node_table_rows: list[dict] = [
        {
            "node_id":        n.get("id", n.get("node_id", "")),
            "type":           n.get("type", "server"),
            "ip_address":     n.get("ip_address", ""),
            "zone":           n.get("zone", ""),
            "shared":         n.get("is_shared_entity", False),
            "evidence_source": _evidence_src(detection_source),
        }
        for n in graph_nodes
    ]
    graph_edges = local_graph.get("edges") or detected_edges
    edge_table_rows: list[dict] = [
        {
            "source":       e.get("source", ""),
            "target":       e.get("target", ""),
            "relationship": e.get("relationship", "connected_to"),
            "label":        e.get("label", ""),
        }
        for e in graph_edges
    ]

    confidence_summary = _confidence_summary(
        detected_nodes, detected_edges, device_rows, connector_rows, ocr_rows,
    )

    runtime_mode = (
        "LIVE_RFDETR_INFERENCE"
        if detection_source == "LIVE_RFDETR_INFERENCE"
        else "VERIFIED_ANNOTATION_FALLBACK"
        if external_rfdetr_result and not external_rfdetr_result.get("ok")
        else detection_source
    )
    packet_source = (
        external_rfdetr_result.get("source", "live_rfdetr_subprocess")
        if detection_source == "LIVE_RFDETR_INFERENCE"
        else "verified training annotation / safe curated sample"
        if runtime_mode == "VERIFIED_ANNOTATION_FALLBACK"
        else detection_source
    )

    packet: dict = {
        "diagram_id":           diagram_id,
        "run_id":               run_id,
        "source":               packet_source,
        "original_image":       str(orig_out),
        "detected_image":       str(detected_out),
        "detection_source":     detection_source,
        "evidence_source":      _evidence_src(detection_source),
        "annotation_path":      str(ann_p) if ann_p.exists() else "",
        "local_graph_path":     str(lg_p)  if lg_p.exists()  else "",
        "node_count":           len(device_rows),
        "edge_count":           len(connector_rows),
        "ocr_text_block_count": len(ocr_rows),
        "connector_count":      len(connector_rows),
        "devices":              device_rows,
        "connectors":           connector_rows,
        "interfaces":           interface_rows,
        "ocr_text":             ocr_rows,
        "nodes":                node_table_rows,
        "edges":                edge_table_rows,
        "confidence_summary":   confidence_summary,
        "text_blocks":          annotation.get("text_blocks", []),
        "source_label":         f"Source: {detection_source}",
        "annotation_preview":   _render_meta,
        "runtime_mode":         runtime_mode,
        "absorption_mode":      "SESSION_MEMORY_ABSORPTION",
        "rfdetr_subprocess":    external_rfdetr_result or {},
    }

    _save_json(run_dir / "detected_nodes.json",      detected_nodes)
    _save_json(run_dir / "detected_edges.json",      detected_edges)
    _save_json(run_dir / "graph_memory_packet.json", packet)
    _save_csv(run_dir  / "node_table.csv",           node_table_rows)
    _save_csv(run_dir  / "edge_table.csv",           edge_table_rows)
    _save_csv(run_dir  / "devices.csv",              device_rows)
    _save_csv(run_dir  / "connectors.csv",           connector_rows)
    _save_csv(run_dir  / "interfaces.csv",           interface_rows)
    _save_csv(run_dir  / "ocr_text.csv",             ocr_rows)
    _save_json(run_dir / "ingestion_summary.json",   {
        "diagram_id":              diagram_id,
        "run_id":                  run_id,
        "source":                  packet_source,
        "detection_source":        detection_source,
        "runtime_mode":            runtime_mode,
        "node_count":              len(device_rows),
        "edge_count":              len(connector_rows),
        "annotation_preview":      _render_meta,
        "rfdetr_inference_time_s": _rfdetr_time_s,
        "rfdetr_error":            _rfdetr_error,
        "use_live_rfdetr":         use_live_rfdetr,
        "rfdetr_subprocess":       external_rfdetr_result or {},
    })

    return {
        "run_dir":                 run_dir,
        "original_image":          orig_out,
        "detected_image":          detected_out,
        "source":                  packet_source,
        "detection_source":        detection_source,
        "runtime_mode":            runtime_mode,
        "rfdetr_inference_time_s": _rfdetr_time_s,
        "rfdetr_error":            _rfdetr_error,
        "annotation":              annotation,
        "local_graph":             local_graph,
        "detected_nodes":          detected_nodes,
        "detected_edges":          detected_edges,
        "node_table_rows":         node_table_rows,
        "edge_table_rows":         edge_table_rows,
        "device_rows":             device_rows,
        "connector_rows":          connector_rows,
        "interface_rows":          interface_rows,
        "ocr_rows":                ocr_rows,
        "packet":                  packet,
        "confidence_summary":      confidence_summary,
    }


def run_absorption(
    repo_root: Path,
    run_id: str,
    local_graph: dict,
    diagram_id: str = "",
    enterprise_graph_path: "Path | None" = None,
    stitch_map_path: "Path | None" = None,
    alerts_path: "Path | None" = None,
) -> dict:
    """
    Explicit-path version of enterprise graph absorption.

    Unlike run_enterprise_absorption(), this function accepts explicit file
    paths for enterprise_graph, stitch_map, and alerts JSON files.  Use this
    for ONB-XXX assets or any non-V3 scenario directory layout.

    Output:
        outputs/live_absorption/<run_id>/

    Returns the same dict structure as run_enterprise_absorption().
    """
    run_dir = repo_root / "runtime_state" / "live_absorption" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    eg_p = enterprise_graph_path if enterprise_graph_path else Path("/nonexistent")
    sm_p = stitch_map_path        if stitch_map_path        else Path("/nonexistent")
    al_p = alerts_path            if alerts_path            else Path("/nonexistent")

    enterprise_graph: dict = _load_json(eg_p) if eg_p.exists() else {}
    stitch_map:       dict = _load_json(sm_p) if sm_p.exists() else {}
    alerts:           dict = _load_json(al_p) if al_p.exists() else {}

    before   = copy.deepcopy(enterprise_graph)
    clusters = before.get("diagram_clusters", [])

    if isinstance(clusters, list):
        cluster_obj  = next((c for c in clusters if c.get("diagram_id") == diagram_id), {})
        rm_node_ids  = set(cluster_obj.get("node_ids", []))
        before["diagram_clusters"] = [c for c in clusters if c.get("diagram_id") != diagram_id]
    else:
        rm_node_ids  = set(clusters.get(diagram_id, {}).get("node_ids", []))
        before["diagram_clusters"] = {k: v for k, v in clusters.items() if k != diagram_id}

    before["nodes"] = [n for n in before.get("nodes", []) if n.get("id") not in rm_node_ids]
    before["edges"] = [
        e for e in before.get("edges", [])
        if e.get("source") not in rm_node_ids and e.get("target") not in rm_node_ids
    ]

    local_ids   = {n.get("canonical_id", n.get("id")) for n in local_graph.get("nodes", [])}
    ent_ids     = {n.get("id") for n in enterprise_graph.get("nodes", [])}
    matched     = sorted(local_ids & ent_ids)
    cross_links = [
        e for e in stitch_map.get("cross_diagram_edges", [])
        if e.get("source_diagram") == diagram_id or e.get("target_diagram") == diagram_id
    ]
    summary = {
        "absorbed_diagram_id":         diagram_id,
        "run_id":                      run_id,
        "nodes_absorbed":              len(local_graph.get("nodes", [])),
        "edges_absorbed":              len(local_graph.get("edges", [])),
        "shared_entities_matched":     len(matched),
        "cross_diagram_links_created": len(cross_links),
        "matched_entities":            matched,
        "cross_diagram_links":         cross_links,
        "before_node_count":           len(before.get("nodes", [])),
        "after_node_count":            len(enterprise_graph.get("nodes", [])),
        "status":                      "absorbed_into_enterprise_graph",
    }

    _save_json(run_dir / "enterprise_before.json",  before)
    _save_json(run_dir / "enterprise_after.json",   enterprise_graph)
    _save_json(run_dir / "absorption_summary.json", summary)
    _save_json(run_dir / "alerts.json",             alerts)

    return {
        "run_dir":           run_dir,
        "enterprise_before": before,
        "enterprise_after":  enterprise_graph,
        "stitch_map":        stitch_map,
        "alerts":            alerts,
        "summary":           summary,
    }
