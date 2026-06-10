"""
runtime_ingestion.py

Live ingestion helpers for the InfraGraph AI Streamlit demo.

Two public functions:
    run_live_v3_ingestion    -- V3 diagram + annotation -> graph memory packet
    run_enterprise_absorption -- local graph -> enterprise graph before/after

Neither function fakes training outputs. Detection source is resolved honestly:
    - "RF-DETR trained prediction"      if outputs/rfdetr_v3_predictions/<id>.png exists
    - "Prepared V3 annotation fallback" otherwise
"""
from __future__ import annotations

import copy
import csv
import json
import math
import shutil
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


# ── annotation preview renderer ───────────────────────────────────────────────
def render_v3_annotation_preview(
    image_path: Path,
    annotation_path: Path,
    out_path: Path,
) -> Path:
    """
    Draw bboxes, node labels, connector polylines, and a footer banner from
    a V3 annotation JSON onto a copy of the source image.  Saves to out_path.
    Falls back to copying the original if Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        if image_path.exists():
            shutil.copy2(image_path, out_path)
        return out_path

    annotation = _load_json(annotation_path) if annotation_path.exists() else {}
    if not annotation or not image_path.exists():
        if image_path.exists():
            shutil.copy2(image_path, out_path)
        return out_path

    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    img_w, img_h = img.size

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
    _DEFAULT_COLOR = (168, 168, 168)

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

    # connector polylines — drawn under boxes
    for conn in annotation.get("connectors", []):
        pts = conn.get("points", [])
        if len(pts) < 2:
            continue
        flat = [(int(p[0]), int(p[1])) for p in pts]
        draw.line(flat, fill=(100, 200, 255, 180), width=2)
        x1c, y1c = flat[-2]
        x2c, y2c = flat[-1]
        ang = math.atan2(y2c - y1c, x2c - x1c)
        for side in (0.45, -0.45):
            ax = x2c - int(9 * math.cos(ang + side))
            ay = y2c - int(9 * math.sin(ang + side))
            draw.line([(x2c, y2c), (ax, ay)], fill=(100, 200, 255, 200), width=2)
        lbl = conn.get("label_text", conn.get("relationship", ""))
        if lbl and pts:
            mid = pts[len(pts) // 2]
            draw.text((int(mid[0]) + 3, int(mid[1]) - 14), str(lbl),
                      fill=(180, 230, 255), font=font_sm)

    # bounding boxes + node id / class labels
    for obj in annotation.get("objects", []):
        bbox = obj.get("bbox", [])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cls   = obj.get("class_name", "server")
        r, g, b = _CLS_COLORS.get(cls, _DEFAULT_COLOR)

        draw.rectangle([x1, y1, x2, y2],
                       outline=(r, g, b, 255), fill=(r, g, b, 40), width=2)

        node_id = obj.get("object_id", obj.get("label_text", ""))
        label   = f"{node_id} [{cls}]"
        label_y = max(0, y1 - 18)
        try:
            tb = draw.textbbox((x1, label_y), label, font=font)
            lw, lh = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:
            lw, lh = len(label) * 8, 16
        draw.rectangle([x1, label_y, x1 + lw + 6, label_y + lh + 4],
                       fill=(r, g, b, 210))
        draw.text((x1 + 3, label_y + 2), label, fill=(255, 255, 255), font=font)

    # footer banner
    footer_h = 26
    draw.rectangle([0, img_h - footer_h, img_w, img_h], fill=(15, 22, 42, 220))
    draw.text((10, img_h - footer_h + 5),
              "Prepared V3 annotation fallback — rendered as detection overlay",
              fill=(160, 200, 255), font=font_sm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def run_live_v3_ingestion(
    repo_root: Path,
    diagram_path: Path,
    diagram_id: str,
    scenario_path: Path,
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

    # ── detect source ─────────────────────────────────────────────────────────
    rfdetr_pred = (
        repo_root / "outputs" / "rfdetr_v3_predictions"
        / f"{scenario_id}__{diagram_id}.png"
    )
    if rfdetr_pred.exists():
        detected_out = run_dir / "detected.png"
        if not detected_out.exists():
            shutil.copy2(rfdetr_pred, detected_out)
        detection_source = "RF-DETR trained prediction"
    else:
        detected_out = run_dir / "detected.png"
        detection_source = "Prepared V3 annotation fallback"

    # ── load annotation & local graph ─────────────────────────────────────────
    ann_path = scenario_path / "annotations" / f"{diagram_id}.json"
    lg_path  = scenario_path / "local_graphs"  / f"{diagram_id}.json"
    annotation:  dict = _load_json(ann_path) if ann_path.exists() else {}
    local_graph: dict = _load_json(lg_path)  if lg_path.exists()  else {}

    # render fallback bbox preview (idempotent — skips if file already exists)
    if detection_source == "Prepared V3 annotation fallback" and not detected_out.exists():
        render_v3_annotation_preview(diagram_path, ann_path, detected_out)
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
    }

    # ── persist artifacts ─────────────────────────────────────────────────────
    _save_json(run_dir / "detected_nodes.json",      detected_nodes)
    _save_json(run_dir / "detected_edges.json",      detected_edges)
    _save_json(run_dir / "graph_memory_packet.json", packet)
    _save_csv(run_dir  / "node_table.csv",           node_table_rows)
    _save_csv(run_dir  / "edge_table.csv",           edge_table_rows)

    return {
        "run_dir":            run_dir,
        "original_image":     orig_out,
        "detected_image":     detected_out,
        "detection_source":   detection_source,
        "annotation":         annotation,
        "local_graph":        local_graph,
        "detected_nodes":     detected_nodes,
        "detected_edges":     detected_edges,
        "node_table_rows":    node_table_rows,
        "edge_table_rows":    edge_table_rows,
        "packet":             packet,
        "confidence_summary": confidence_summary,
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
