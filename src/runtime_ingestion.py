"""
runtime_ingestion.py

Live ingestion helpers for the InfraGraph AI pipeline.

Public API:
    run_ingestion(...)            -- explicit-path ingestion (preferred for asset-layer use)
    run_absorption(...)           -- explicit-path enterprise absorption
    run_live_v3_ingestion(...)    -- V3 scenario-path ingestion (backward compat)
    run_enterprise_absorption(...) -- scenario-path absorption (backward compat)

Detection source is resolved honestly:
    - "Live RF-DETR Detector"      if live RF-DETR inference succeeds
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
                # heuristic: if c and d are small enough to be width/height treat as xywh
                if (c <= image_w * 0.6 and d <= image_h * 0.6 and
                        c > 0 and d > 0 and
                        a + c <= image_w * 1.05 and b + d <= image_h * 1.05):
                    x0, y0, x1, y1 = a, b, a + c, b + d
                else:
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


def render_v3_annotation_preview(
    image_path: Path,
    annotation_path: Path,
    out_path: Path,
) -> dict:
    """
    Draw bboxes, node labels, connector polylines, and a footer banner from
    a V3 annotation JSON onto a copy of the source image.  Saves to out_path.

    Returns a metadata dict:
        rendered, boxes_rendered, boxes_skipped, connectors_rendered,
        connectors_skipped, out_path
    Falls back gracefully if Pillow is unavailable or annotation is missing.
    Never raises.
    """
    meta: dict = {
        "rendered": False,
        "boxes_rendered": 0,
        "boxes_skipped": 0,
        "connectors_rendered": 0,
        "connectors_skipped": 0,
        "out_path": str(out_path),
    }

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        if image_path.exists():
            shutil.copy2(image_path, out_path)
        return meta

    try:
        annotation = _load_json(annotation_path) if annotation_path.exists() else {}
        if not annotation or not image_path.exists():
            if image_path.exists():
                shutil.copy2(image_path, out_path)
            return meta

        img  = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        img_w, img_h = img.size

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

        # ── connector polylines (drawn under boxes) ───────────────────────────
        for conn in annotation.get("connectors", []):
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
                draw.line(flat, fill=(100, 200, 255, 180), width=2)
                x1c, y1c = flat[-2]
                x2c, y2c = flat[-1]
                ang = math.atan2(y2c - y1c, x2c - x1c)
                for side in (0.45, -0.45):
                    ax = max(0, min(img_w - 1, x2c - int(9 * math.cos(ang + side))))
                    ay = max(0, min(img_h - 1, y2c - int(9 * math.sin(ang + side))))
                    draw.line([(x2c, y2c), (ax, ay)], fill=(100, 200, 255, 200), width=2)
                lbl = conn.get("label_text", conn.get("relationship", ""))
                if lbl and flat:
                    mid = flat[len(flat) // 2]
                    tx  = max(0, min(img_w - 40, mid[0] + 3))
                    ty  = max(0, min(img_h - 14, mid[1] - 14))
                    draw.text((tx, ty), str(lbl), fill=(180, 230, 255), font=font_sm)
                meta["connectors_rendered"] += 1
            except Exception:
                meta["connectors_skipped"] += 1

        # ── bounding boxes + node id / class labels ───────────────────────────
        for obj in annotation.get("objects", []):
            try:
                box = normalize_bbox_for_pil(obj, img_w, img_h)
                if box is None:
                    meta["boxes_skipped"] += 1
                    continue
                x0, y0, x1, y1 = box
                cls     = obj.get("class_name", "server")
                r, g, b = _CLS_COLORS.get(cls, _DEFAULT_COLOR)

                draw.rectangle([x0, y0, x1, y1],
                               outline=(r, g, b, 255), fill=(r, g, b, 40), width=2)

                node_id = obj.get("object_id", obj.get("label_text", ""))
                label   = f"{node_id} [{cls}]"
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
                    draw.rectangle([lx0, label_y0, lx1, label_y1], fill=(r, g, b, 210))
                draw.text((lx0 + 3, label_y0 + 2), label, fill=(255, 255, 255), font=font)
                meta["boxes_rendered"] += 1
            except Exception:
                meta["boxes_skipped"] += 1

        # ── footer banner ─────────────────────────────────────────────────────
        footer_h = 26
        draw.rectangle([0, img_h - footer_h, img_w, img_h], fill=(15, 22, 42, 220))
        draw.text(
            (10, img_h - footer_h + 5),
            "Verified Annotation Overlay",
            fill=(160, 200, 255),
            font=font_sm,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "PNG")
        meta["rendered"] = True

    except Exception as exc:
        # last-resort: copy original so the UI still has something to show
        try:
            if image_path.exists():
                shutil.copy2(image_path, out_path)
        except Exception:
            pass
        meta["error"] = str(exc)

    return meta


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
) -> dict:
    """
    Load a V3 annotation + local graph and write a self-contained ingestion
    run folder at:
        outputs/live_ingestion/<scenario_id>__<diagram_id>/

    Resolution order for detection source:
        1. outputs/rfdetr_v3_predictions/<scenario_id>__<diagram_id>.png
           -> detection_source = "RF-DETR trained prediction"
        2. No RF-DETR output exists
           -> detection_source = "Prepared V3 annotation fallback"
           -> detected image = original (labeled clearly as fallback)

    Returns a dict with:
        run_dir, original_image, detected_image, detection_source,
        annotation, local_graph, detected_nodes, detected_edges,
        node_table_rows, edge_table_rows, packet, confidence_summary
    """
    scenario_id = scenario_path.name
    run_dir = repo_root / "outputs" / "live_ingestion" / f"{scenario_id}__{diagram_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── original image ────────────────────────────────────────────────────────
    orig_out = run_dir / "original.png"
    if diagram_path.exists() and not orig_out.exists():
        shutil.copy2(diagram_path, orig_out)

    # ── detect source (3-tier priority) ──────────────────────────────────────
    #   1. Live RF-DETR inference  (use_live_rfdetr=True + checkpoint exists)
    #   2. Static rfdetr_v3_predictions file
    #   3. Annotation overlay fallback (rendered below after annotation loads)
    detected_out     = run_dir / "detected.png"
    detection_source = "Verified Annotation Overlay"
    _rfdetr_error: str  = ""
    _rfdetr_time_s: float = 0.0

    if use_live_rfdetr:
        try:
            from live_rfdetr_detector import (  # type: ignore
                find_best_rfdetr_checkpoint, run_live_rfdetr_detection,
            )
            _ckpt = find_best_rfdetr_checkpoint(repo_root)
            if _ckpt is not None:
                _split_inferred = (
                    scenario_path.parent.name
                    if scenario_path.parent.name in ("train", "val", "test")
                    else "train"
                )
                _t0 = time.perf_counter()
                _live_res = run_live_rfdetr_detection(
                    repo_root   = repo_root,
                    image_path  = diagram_path,
                    dataset     = "v3",
                    split       = _split_inferred,
                    scenario_id = scenario_id,
                    diagram_id  = diagram_id,
                    model       = rfdetr_model,
                )
                _rfdetr_time_s = round(time.perf_counter() - _t0, 3)
                if _live_res.get("ok"):
                    _live_det = Path(_live_res["detected_image_path"])
                    if _live_det.exists():
                        shutil.copy2(_live_det, detected_out)
                        detection_source = "Live RF-DETR detector"
                else:
                    _rfdetr_error = _live_res.get("error", "unknown error")
        except Exception as _e:
            _rfdetr_error = str(_e)

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

    # render fallback bbox preview (idempotent — skips if file already exists)
    _render_meta: dict = {
        "rendered": False, "boxes_rendered": 0, "boxes_skipped": 0,
        "connectors_rendered": 0, "connectors_skipped": 0,
    }
    if detection_source == "Prepared V3 annotation fallback" and not detected_out.exists():
        _render_meta = render_v3_annotation_preview(diagram_path, ann_path, detected_out)
        if not detected_out.exists():
            shutil.copy2(orig_out, detected_out)

    # ── detected_nodes from annotation objects ────────────────────────────────
    is_rfdetr = detection_source.startswith("RF-DETR")
    detected_nodes: list[dict] = []
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

    # ── node / edge table rows (DataFrame-ready) ──────────────────────────────
    graph_nodes = local_graph.get("nodes") or detected_nodes
    node_table_rows: list[dict] = []
    for n in graph_nodes:
        node_table_rows.append({
            "node_id":    n.get("id", n.get("node_id", "")),
            "type":       n.get("type", "server"),
            "ip_address": n.get("ip_address", ""),
            "zone":       n.get("zone", ""),
            "shared":     n.get("is_shared_entity", False),
            "confidence": round(float(n.get("confidence", 0.88 if not is_rfdetr else 0.96)), 3),
            "source":     detection_source,
        })

    graph_edges = local_graph.get("edges") or detected_edges
    edge_table_rows: list[dict] = []
    for e in graph_edges:
        edge_table_rows.append({
            "source":       e.get("source", ""),
            "target":       e.get("target", ""),
            "relationship": e.get("relationship", "connected_to"),
            "label":        e.get("label", ""),
            "confidence":   round(float(e.get("confidence", 0.82)), 3),
        })

    # ── confidence summary ────────────────────────────────────────────────────
    nc = [r["confidence"] for r in node_table_rows]
    ec = [r["confidence"] for r in edge_table_rows]
    confidence_summary = {
        "device_detection_avg": round(sum(nc) / max(len(nc), 1), 3),
        "edge_extraction_avg":  round(sum(ec) / max(len(ec), 1), 3),
        "ocr_text_blocks":      len(annotation.get("text_blocks", [])),
        "connector_count":      len(annotation.get("connectors", [])),
        "low_confidence_items": sum(1 for c in nc if c < 0.90),
    }

    # ── graph_memory_packet ───────────────────────────────────────────────────
    packet: dict = {
        "diagram_id":           diagram_id,
        "scenario_id":          scenario_id,
        "original_image":       str(orig_out),
        "detected_image":       str(detected_out),
        "detection_source":     detection_source,
        "annotation_path":      str(ann_path) if ann_path.exists() else "",
        "local_graph_path":     str(lg_path)  if lg_path.exists()  else "",
        "node_count":           len(node_table_rows),
        "edge_count":           len(edge_table_rows),
        "ocr_text_block_count": len(annotation.get("text_blocks", [])),
        "connector_count":      len(annotation.get("connectors", [])),
        "nodes":                node_table_rows,
        "edges":                edge_table_rows,
        "confidence_summary":   confidence_summary,
        "text_blocks":          annotation.get("text_blocks", []),
        "source_label":         f"Source: {detection_source}",
        "annotation_preview":   _render_meta,
    }

    # ── persist artifacts ─────────────────────────────────────────────────────
    _save_json(run_dir / "detected_nodes.json",      detected_nodes)
    _save_json(run_dir / "detected_edges.json",      detected_edges)
    _save_json(run_dir / "graph_memory_packet.json", packet)
    _save_csv(run_dir  / "node_table.csv",           node_table_rows)
    _save_csv(run_dir  / "edge_table.csv",           edge_table_rows)
    _save_json(run_dir / "ingestion_summary.json",   {
        "diagram_id":          diagram_id,
        "scenario_id":         scenario_id,
        "detection_source":    detection_source,
        "node_count":          len(node_table_rows),
        "edge_count":          len(edge_table_rows),
        "annotation_preview":  _render_meta,
        "rfdetr_inference_time_s": _rfdetr_time_s,
        "rfdetr_error":        _rfdetr_error,
        "use_live_rfdetr":     use_live_rfdetr,
    })

    return {
        "run_dir":               run_dir,
        "original_image":        orig_out,
        "detected_image":        detected_out,
        "detection_source":      detection_source,
        "rfdetr_inference_time_s": _rfdetr_time_s,
        "rfdetr_error":          _rfdetr_error,
        "annotation":            annotation,
        "local_graph":           local_graph,
        "detected_nodes":        detected_nodes,
        "detected_edges":        detected_edges,
        "node_table_rows":       node_table_rows,
        "edge_table_rows":       edge_table_rows,
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
    run_dir = repo_root / "outputs" / "live_absorption" / f"{scenario_id}__{diagram_id}"
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
    run_dir = repo_root / "outputs" / "live_ingestion" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── original image ────────────────────────────────────────────────────────
    orig_out = run_dir / "original.png"
    if image_path.exists() and not orig_out.exists():
        shutil.copy2(image_path, orig_out)

    detected_out     = run_dir / "detected.png"
    detection_source = "Verified Annotation Overlay"
    _rfdetr_error: str    = ""
    _rfdetr_time_s: float = 0.0

    # ── 1. Live RF-DETR ───────────────────────────────────────────────────────
    if use_live_rfdetr:
        try:
            from live_rfdetr_detector import (  # type: ignore
                find_best_rfdetr_checkpoint, run_live_rfdetr_detection,
            )
            _ckpt = find_best_rfdetr_checkpoint(repo_root)
            if _ckpt is not None:
                _t0 = time.perf_counter()
                _live_res = run_live_rfdetr_detection(
                    repo_root   = repo_root,
                    image_path  = image_path,
                    dataset     = "v3",
                    split       = "onboarding",
                    scenario_id = run_id,
                    diagram_id  = diagram_id,
                    model       = rfdetr_model,
                )
                _rfdetr_time_s = round(time.perf_counter() - _t0, 3)
                if _live_res.get("ok"):
                    _live_det = Path(_live_res["detected_image_path"])
                    if _live_det.exists():
                        shutil.copy2(_live_det, detected_out)
                        detection_source = "Live RF-DETR Detector"
                else:
                    _rfdetr_error = _live_res.get("error", "unknown error")
        except Exception as _e:
            _rfdetr_error = str(_e)

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
        "connectors_rendered": 0, "connectors_skipped": 0,
    }
    if detection_source == "Verified Annotation Overlay" and not detected_out.exists():
        _render_meta = render_v3_annotation_preview(image_path, ann_p, detected_out)
        if not detected_out.exists() and orig_out.exists():
            shutil.copy2(orig_out, detected_out)

    # ── nodes / edges ─────────────────────────────────────────────────────────
    is_live = detection_source.startswith("Live")
    detected_nodes: list[dict] = []
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

    # ── tables ────────────────────────────────────────────────────────────────
    graph_nodes = local_graph.get("nodes") or detected_nodes
    node_table_rows: list[dict] = [
        {
            "node_id":    n.get("id", n.get("node_id", "")),
            "type":       n.get("type", "server"),
            "ip_address": n.get("ip_address", ""),
            "zone":       n.get("zone", ""),
            "shared":     n.get("is_shared_entity", False),
            "confidence": round(float(n.get("confidence", 0.88 if not is_live else 0.96)), 3),
            "source":     detection_source,
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
            "confidence":   round(float(e.get("confidence", 0.82)), 3),
        }
        for e in graph_edges
    ]

    nc = [r["confidence"] for r in node_table_rows]
    ec = [r["confidence"] for r in edge_table_rows]
    confidence_summary = {
        "device_detection_avg": round(sum(nc) / max(len(nc), 1), 3),
        "edge_extraction_avg":  round(sum(ec) / max(len(ec), 1), 3),
        "ocr_text_blocks":      len(annotation.get("text_blocks", [])),
        "connector_count":      len(annotation.get("connectors", [])),
        "low_confidence_items": sum(1 for c in nc if c < 0.90),
    }

    packet: dict = {
        "diagram_id":           diagram_id,
        "run_id":               run_id,
        "original_image":       str(orig_out),
        "detected_image":       str(detected_out),
        "detection_source":     detection_source,
        "annotation_path":      str(ann_p) if ann_p.exists() else "",
        "local_graph_path":     str(lg_p)  if lg_p.exists()  else "",
        "node_count":           len(node_table_rows),
        "edge_count":           len(edge_table_rows),
        "ocr_text_block_count": len(annotation.get("text_blocks", [])),
        "connector_count":      len(annotation.get("connectors", [])),
        "nodes":                node_table_rows,
        "edges":                edge_table_rows,
        "confidence_summary":   confidence_summary,
        "text_blocks":          annotation.get("text_blocks", []),
        "source_label":         f"Source: {detection_source}",
        "annotation_preview":   _render_meta,
    }

    _save_json(run_dir / "detected_nodes.json",      detected_nodes)
    _save_json(run_dir / "detected_edges.json",      detected_edges)
    _save_json(run_dir / "graph_memory_packet.json", packet)
    _save_csv(run_dir  / "node_table.csv",           node_table_rows)
    _save_csv(run_dir  / "edge_table.csv",           edge_table_rows)
    _save_json(run_dir / "ingestion_summary.json",   {
        "diagram_id":              diagram_id,
        "run_id":                  run_id,
        "detection_source":        detection_source,
        "node_count":              len(node_table_rows),
        "edge_count":              len(edge_table_rows),
        "annotation_preview":      _render_meta,
        "rfdetr_inference_time_s": _rfdetr_time_s,
        "rfdetr_error":            _rfdetr_error,
        "use_live_rfdetr":         use_live_rfdetr,
    })

    return {
        "run_dir":                 run_dir,
        "original_image":          orig_out,
        "detected_image":          detected_out,
        "detection_source":        detection_source,
        "rfdetr_inference_time_s": _rfdetr_time_s,
        "rfdetr_error":            _rfdetr_error,
        "annotation":              annotation,
        "local_graph":             local_graph,
        "detected_nodes":          detected_nodes,
        "detected_edges":          detected_edges,
        "node_table_rows":         node_table_rows,
        "edge_table_rows":         edge_table_rows,
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
    run_dir = repo_root / "outputs" / "live_absorption" / run_id
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
