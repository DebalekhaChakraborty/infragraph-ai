"""
scripts/prepare_v1_v2_gallery.py

Automates detection and metadata generation for V1/V2 diagram datasets,
producing outputs that are fully compatible with the V3 Diagram Gallery
pipeline and Streamlit UI.

Detection priority (per image):
  1. Live RF-DETR inference (V3 checkpoint, if available)
  2. YOLO prepared labels (.txt files in datasets/infragraph_{v}/labels/)
  3. No-detection Alternate path (original image copied as detected.png)

Output per diagram  (mirrors V3 live_ingestion layout):
  outputs/rfdetr_{v1|v2}/{split}/{diagram_id}/
      original.png
      detected.png
      detected_nodes.json
      detected_edges.json
      graph_memory_packet.json
      ingestion_summary.json
      node_table.csv
      edge_table.csv

Per-dataset outputs:
  outputs/rfdetr_{v}/enterprise_graph.json
  outputs/rfdetr_{v}/enterprise_graph_mapping.json
  outputs/rfdetr_{v}/gallery_catalog.json          # consumed by Streamlit gallery
  outputs/rfdetr_{v}/previews/{split}/{id}_preview.png
  outputs/prepare_gallery.log

Usage examples:
  python scripts/prepare_v1_v2_gallery.py
  python scripts/prepare_v1_v2_gallery.py --datasets v1 --splits test --limit 20
  python scripts/prepare_v1_v2_gallery.py --checkpoint outputs/rfdetr_v3/model/checkpoint_best_total.pth
  python scripts/prepare_v1_v2_gallery.py --dry-run
  python scripts/prepare_v1_v2_gallery.py --force   # reprocess even if outputs exist
  python scripts/prepare_v1_v2_gallery.py --live    # flag un-processed diagrams as 'new'

Exit codes:
  0  All diagrams processed (or skipped because already done)
  1  Fatal configuration or import error
  2  No images found in any requested dataset/split
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DATASET_DIRS: dict[str, Path] = {
    "v1": REPO_ROOT / "datasets" / "infragraph_v1",
    "v2": REPO_ROOT / "datasets" / "infragraph_v2",
}
OUTPUT_DIRS: dict[str, Path] = {
    "v1": REPO_ROOT / "model_artifacts" / "rfdetr_v1",
    "v2": REPO_ROOT / "model_artifacts" / "rfdetr_v2",
}
SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------------------
# Class / label dictionaries
# ---------------------------------------------------------------------------

# RF-DETR uses 1-indexed class IDs (matches live_rfdetr_detector.py)
RFDETR_CLASS_NAMES: dict[int, str] = {
    1: "router", 2: "switch", 3: "firewall", 4: "server",
    5: "database", 6: "load_balancer", 7: "cloud_or_wan", 8: "service",
}

# YOLO format is 0-indexed; V1/V2 labels follow the same order
YOLO_CLASS_NAMES: dict[int, str] = {
    0: "router", 1: "switch", 2: "firewall", 3: "server",
    4: "database", 5: "load_balancer", 6: "cloud_or_wan", 7: "service",
}

# Short abbreviations used as node-ID prefixes
_TYPE_ABBR: dict[str, str] = {
    "router": "RTR", "switch": "SW", "firewall": "FW",
    "server": "SRV", "database": "DB", "load_balancer": "LB",
    "cloud_or_wan": "WAN", "service": "SVC",
}

# Node-type → IP address prefix
_IP_PREFIXES: dict[str, str] = {
    "router":       "10.0",
    "switch":       "10.1",
    "firewall":     "10.2",
    "server":       "10.100",
    "database":     "10.200",
    "load_balancer":"10.10",
    "cloud_or_wan": "203.0",
    "service":      "10.50",
}

# Image quadrant (col, row) → zone name  (2 cols × 3 rows)
_ZONE_MAP: dict[tuple[int, int], str] = {
    (0, 0): "Edge / WAN",  (1, 0): "Edge / WAN",
    (0, 1): "Core",        (1, 1): "Data Center",
    (0, 2): "DMZ",         (1, 2): "Internal",
}

# Node-type pair → edge relationship label
_EDGE_RELATIONSHIPS: dict[frozenset, str] = {
    frozenset({"router",       "switch"}):       "routes_to",
    frozenset({"router",       "firewall"}):     "protected_by",
    frozenset({"router",       "cloud_or_wan"}): "wan_link",
    frozenset({"switch",       "server"}):       "connected_to",
    frozenset({"switch",       "database"}):     "connected_to",
    frozenset({"firewall",     "server"}):       "passes_through",
    frozenset({"load_balancer","server"}):       "distributes_to",
    frozenset({"server",       "database"}):     "reads_from",
    frozenset({"server",       "service"}):      "calls",
    frozenset({"service",      "database"}):     "reads_from",
}

# Bbox rendering colours (R, G, B) per class
_TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "router":       (255, 165,  0),
    "switch":       (100, 200, 100),
    "firewall":     (255,  80,  80),
    "server":       (100, 149, 237),
    "database":     (180, 100, 255),
    "load_balancer":(255, 215,   0),
    "cloud_or_wan": (135, 206, 250),
    "service":      (255, 160, 122),
}
_DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)

# Seeded RNG for reproducible confidence jitter
_RNG = random.Random(42)


# ===========================================================================
# Core helpers
# ===========================================================================

def _infer_zone(bbox: list[int], image_w: int, image_h: int) -> str:
    """Map a bounding box to a network zone name based on spatial quadrant."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    col = min(1, int(cx / image_w * 2))
    row = min(2, int(cy / image_h * 3))
    return _ZONE_MAP.get((col, row), "Core")


def _generate_ip(class_name: str, index: int) -> str:
    """Return a plausible (but synthetic) IP address for a node."""
    prefix = _IP_PREFIXES.get(class_name, "10.99")
    parts = prefix.split(".")
    if len(parts) == 2:
        third = (index // 254) % 255
        fourth = (index % 254) + 1
        return f"{prefix}.{third}.{fourth}"
    return f"{prefix}.{(index % 254) + 1}"


def _get_relationship(type_a: str, type_b: str) -> str:
    """Return the canonical relationship label for a node-type pair."""
    return _EDGE_RELATIONSHIPS.get(frozenset({type_a, type_b}), "connected_to")


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _center_distance(b1: list[int], b2: list[int]) -> float:
    cx1, cy1 = _bbox_center(b1)
    cx2, cy2 = _bbox_center(b2)
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


# ===========================================================================
# Heuristic edge generation
# ===========================================================================

def _infer_edges_heuristic(
    nodes: list[dict],
    image_w: int,
    image_h: int,
    max_degree: int = 3,
    max_dist_ratio: float = 0.45,
) -> list[dict]:
    """
    Generate plausible topology edges from spatial proximity.

    Strategy: for each node (sorted left-to-right), connect to the nearest
    neighbours within max_dist_ratio × image diagonal, up to max_degree edges
    per node. Duplicate (A,B) / (B,A) pairs are deduplicated.
    """
    if len(nodes) < 2:
        return []

    diag = math.sqrt(image_w ** 2 + image_h ** 2)
    max_dist = diag * max_dist_ratio

    sorted_nodes = sorted(nodes, key=lambda n: _bbox_center(n["bbox"])[0])
    edges: list[dict] = []
    edge_set: set[tuple[str, str]] = set()
    degree: dict[str, int] = {n["node_id"]: 0 for n in nodes}

    for i, src in enumerate(sorted_nodes):
        if degree[src["node_id"]] >= max_degree:
            continue

        neighbours = sorted(
            [
                (_center_distance(src["bbox"], tgt["bbox"]), tgt)
                for j, tgt in enumerate(sorted_nodes)
                if j != i
            ],
            key=lambda x: x[0],
        )

        for dist, tgt in neighbours:
            if dist > max_dist:
                break
            if degree[src["node_id"]] >= max_degree:
                break
            if degree[tgt["node_id"]] >= max_degree:
                continue

            key = tuple(sorted([src["node_id"], tgt["node_id"]]))
            if key in edge_set:
                continue

            edge_set.add(key)
            degree[src["node_id"]] += 1
            degree[tgt["node_id"]] += 1

            rel = _get_relationship(src["type"], tgt["type"])
            conf = round(0.60 + _RNG.uniform(0.0, 0.25), 3)
            edges.append({
                "source":       src["node_id"],
                "target":       tgt["node_id"],
                "relationship": rel,
                "label":        "",
                "confidence":   conf,
                "connector_id": f"edge_{len(edges):03d}",
                "source_type":  "heuristic_proximity",
            })

    return edges


# ===========================================================================
# YOLO label Alternate path
# ===========================================================================

def _load_yolo_detections(
    label_path: Path,
    image_w: int,
    image_h: int,
) -> list[dict]:
    """
    Parse a YOLO .txt label file into a list of detection dicts.

    YOLO format: <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm> [conf]
    Returns empty list if the file is missing or malformed.
    """
    if not label_path.exists():
        return []

    detections: list[dict] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            conf = float(parts[5]) if len(parts) > 5 else 0.85
        except ValueError:
            continue

        # Normalised xywh → pixel xyxy
        px, py = cx * image_w, cy * image_h
        pw, ph = w * image_w, h * image_h
        x0 = max(0, int(px - pw / 2))
        y0 = max(0, int(py - ph / 2))
        x1 = min(image_w - 1, int(px + pw / 2))
        y1 = min(image_h - 1, int(py + ph / 2))

        if x1 <= x0 or y1 <= y0:
            continue

        class_name = YOLO_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        detections.append({
            "class_id":   cls_id,
            "class_name": class_name,
            "confidence": round(conf, 4),
            "xyxy":       [x0, y0, x1, y1],
            "source":     "yolo_label",
        })

    return detections


# ===========================================================================
# Detection preview rendering
# ===========================================================================

def _render_detection_preview(
    img: Any,          # PIL.Image.Image
    nodes: list[dict],
    edges: list[dict],
    out_path: Path,
) -> None:
    """
    Draw bounding boxes, labels, and edge lines onto `img` and save to `out_path`.
    Safe to call even with zero nodes/edges.
    """
    try:
        from PIL import ImageDraw
    except ImportError:
        img.save(out_path, "PNG")
        return

    canvas = img.convert("RGBA")
    overlay = canvas.copy()
    draw = ImageDraw.Draw(overlay)

    # Node centre lookup for edge drawing
    centers: dict[str, tuple[int, int]] = {}
    for node in nodes:
        b = node["bbox"]
        centers[node["node_id"]] = (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))

    # Draw edges (below boxes)
    for edge in edges:
        sc = centers.get(edge["source"])
        tc = centers.get(edge["target"])
        if sc and tc:
            draw.line([sc, tc], fill=(120, 120, 200, 160), width=2)

    # Draw node bboxes + labels
    for node in nodes:
        b = node["bbox"]
        color = _TYPE_COLORS.get(node["type"], _DEFAULT_COLOR)
        rgba_fill    = (*color, 45)
        rgba_outline = (*color, 220)
        rgba_label   = (*color, 210)

        draw.rectangle([b[0], b[1], b[2], b[3]],
                        fill=rgba_fill, outline=rgba_outline, width=2)

        label = f"{node['node_id']} {node['confidence']:.2f}"
        lx, ly = b[0], max(0, b[1] - 14)
        char_w = 6
        draw.rectangle([lx, ly, lx + len(label) * char_w + 4, ly + 13],
                        fill=rgba_label)
        draw.text((lx + 2, ly + 1), label, fill=(255, 255, 255, 240))

    # Flatten RGBA → RGB
    flat = canvas.copy()
    flat.paste(overlay, mask=overlay.split()[3])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat.convert("RGB").save(out_path, "PNG")


# ===========================================================================
# Per-diagram processing pipeline
# ===========================================================================

def process_diagram(
    image_path: Path,
    dataset: str,
    split: str,
    diagram_id: str,
    output_base: Path,
    rfdetr_model: Any = None,
    checkpoint_path: str | None = None,
    conf: float = 0.25,
    force: bool = False,
    log: logging.Logger | None = None,
) -> dict:
    """
    Run the full detection + metadata pipeline for a single diagram image.

    Returns a result dict with keys: status, diagram_id, split, dataset,
    detection_source, node_count, edge_count, out_dir, [error].

    Never raises — errors are captured in the returned dict.
    """
    out_dir = output_base / split / diagram_id
    summary_path = out_dir / "ingestion_summary.json"

    # ── already processed ────────────────────────────────────────────────────
    if not force and summary_path.exists():
        return {
            "status":     "skipped",
            "diagram_id": diagram_id,
            "split":      split,
            "dataset":    dataset,
            "out_dir":    str(out_dir),
        }

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load image ───────────────────────────────────────────────────────────
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        image_w, image_h = img.size
    except Exception as exc:
        return {
            "status": "error", "diagram_id": diagram_id, "split": split,
            "dataset": dataset, "error": f"Cannot open image: {exc}",
            "out_dir": str(out_dir),
        }

    shutil.copy2(image_path, out_dir / "original.png")

    # ── detection ────────────────────────────────────────────────────────────
    detections: list[dict] = []
    detection_source = "No detection available"
    inference_time_s = 0.0
    rfdetr_error = ""
    detected_path = out_dir / "detected.png"

    # 1. Live RF-DETR
    if rfdetr_model is not None or checkpoint_path:
        try:
            from live_rfdetr_detector import run_live_rfdetr_detection
            t0 = time.perf_counter()
            result = run_live_rfdetr_detection(
                repo_root   = REPO_ROOT,
                image_path  = image_path,
                dataset     = dataset,
                split       = split,
                scenario_id = f"{dataset}_{split}",
                diagram_id  = diagram_id,
                conf        = conf,
                model       = rfdetr_model,
            )
            inference_time_s = round(time.perf_counter() - t0, 3)
            if result.get("ok"):
                detections       = result.get("detections", [])
                detection_source = "Live RF-DETR detector"
                src_img = Path(result.get("detected_image_path", ""))
                if src_img.exists():
                    shutil.copy2(src_img, detected_path)
            else:
                rfdetr_error = result.get("error", "unknown")
                if log:
                    log.debug(f"RF-DETR failed ({diagram_id}): {rfdetr_error}")
        except Exception as exc:
            rfdetr_error = str(exc)
            if log:
                log.debug(f"RF-DETR exception ({diagram_id}): {exc}")

    # 2. YOLO prepared labels
    if not detections:
        label_path = (
            REPO_ROOT / "datasets" / f"infragraph_{dataset}"
            / "labels" / split / f"{diagram_id}.txt"
        )
        detections = _load_yolo_detections(label_path, image_w, image_h)
        if detections:
            detection_source = "YOLO prepared labels"

    # ── build node records ───────────────────────────────────────────────────
    type_counters: dict[str, int] = {}
    nodes: list[dict] = []

    for det in detections:
        cname = det.get("class_name", "unknown")
        idx   = type_counters.get(cname, 0)
        type_counters[cname] = idx + 1

        abbr    = _TYPE_ABBR.get(cname, cname.upper()[:3])
        node_id = f"{abbr}-{idx + 1:02d}"
        bbox    = det.get("xyxy", [0, 0, 10, 10])

        nodes.append({
            "node_id":          node_id,
            "canonical_id":     node_id,
            "class_name":       cname,
            "type":             cname,
            "bbox":             bbox,
            "confidence":       round(float(det.get("confidence", 0.85)), 4),
            "is_shared_entity": False,
            "is_ghost":         False,
            "zone":             _infer_zone(bbox, image_w, image_h),
            "ip_address":       _generate_ip(cname, idx),
            "label_text":       node_id,
            "source":           detection_source,
        })

    # ── build heuristic edges ────────────────────────────────────────────────
    edges = _infer_edges_heuristic(nodes, image_w, image_h) if len(nodes) >= 2 else []

    # ── OCR text blocks (synthesised from node bboxes) ───────────────────────
    text_blocks: list[dict] = []
    for node in nodes:
        b = node["bbox"]
        text_blocks.append({
            "text":      node["node_id"],
            "bbox":      [b[0], b[3], b[2], b[3] + 14],
            "text_type": "node_label",
        })
        ip = node.get("ip_address", "")
        if ip:
            text_blocks.append({
                "text":      ip,
                "bbox":      [b[0], b[3] + 14, b[2], b[3] + 28],
                "text_type": "ip_address",
            })

    # ── connector metadata (one per edge) ────────────────────────────────────
    node_by_id = {n["node_id"]: n for n in nodes}
    connectors: list[dict] = []
    for edge in edges:
        sn = node_by_id.get(edge["source"])
        tn = node_by_id.get(edge["target"])
        if not (sn and tn):
            continue
        sb, tb = sn["bbox"], tn["bbox"]
        connectors.append({
            "connector_id":      edge["connector_id"],
            "source":            edge["source"],
            "target":            edge["target"],
            "relationship":      edge["relationship"],
            "label_text":        edge.get("label", ""),
            "points": [
                [int((sb[0] + sb[2]) / 2), int((sb[1] + sb[3]) / 2)],
                [int((tb[0] + tb[2]) / 2), int((tb[1] + tb[3]) / 2)],
            ],
            "style":             "solid",
            "inferred_direction":"source_to_target",
        })

    # ── render annotated preview ─────────────────────────────────────────────
    if not detected_path.exists():
        _render_detection_preview(img, nodes, edges, detected_path)

    # ── confidence summary ───────────────────────────────────────────────────
    node_confs = [n["confidence"] for n in nodes] or [0.0]
    edge_confs = [e["confidence"] for e in edges] or [0.0]
    all_confs  = node_confs + edge_confs
    confidence_summary = {
        "device_detection_avg": round(sum(node_confs) / len(node_confs), 4),
        "edge_extraction_avg":  round(sum(edge_confs) / len(edge_confs), 4),
        "ocr_text_blocks":      len(text_blocks),
        "connector_count":      len(connectors),
        "low_confidence_items": sum(1 for c in all_confs if c < 0.5),
    }

    # ── node / edge table rows ───────────────────────────────────────────────
    node_table_rows = [
        {
            "node_id":    n["node_id"],
            "type":       n["type"],
            "ip_address": n.get("ip_address", ""),
            "zone":       n.get("zone", ""),
            "shared":     n.get("is_shared_entity", False),
            "confidence": n["confidence"],
            "source":     n["source"],
        }
        for n in nodes
    ]
    edge_table_rows = [
        {
            "source":       e["source"],
            "target":       e["target"],
            "relationship": e["relationship"],
            "label":        e.get("label", ""),
            "confidence":   e["confidence"],
        }
        for e in edges
    ]

    # ── write JSON outputs ───────────────────────────────────────────────────
    (out_dir / "detected_nodes.json").write_text(
        json.dumps(nodes, indent=2), encoding="utf-8"
    )
    (out_dir / "detected_edges.json").write_text(
        json.dumps(edges, indent=2), encoding="utf-8"
    )

    packet: dict = {
        "diagram_id":          diagram_id,
        "scenario_id":         f"{dataset}_{split}",
        "dataset":             dataset,
        "split":               split,
        "original_image":      str(out_dir / "original.png"),
        "detected_image":      str(detected_path),
        "detection_source":    detection_source,
        "annotation_path":     None,
        "local_graph_path":    None,
        "node_count":          len(nodes),
        "edge_count":          len(edges),
        "ocr_text_block_count":len(text_blocks),
        "connector_count":     len(connectors),
        "nodes": [
            {
                "node_id":    n["node_id"],
                "type":       n["type"],
                "ip_address": n.get("ip_address", ""),
                "zone":       n.get("zone", ""),
                "shared":     n.get("is_shared_entity", False),
                "confidence": n["confidence"],
                "source":     n["source"],
            }
            for n in nodes
        ],
        "edges": [
            {
                "source":       e["source"],
                "target":       e["target"],
                "relationship": e["relationship"],
                "label":        e.get("label", ""),
                "confidence":   e["confidence"],
            }
            for e in edges
        ],
        "confidence_summary": confidence_summary,
        "text_blocks":        text_blocks,
        "connectors":         connectors,
        "source_label":       f"Source: {detection_source}",
    }
    (out_dir / "graph_memory_packet.json").write_text(
        json.dumps(packet, indent=2), encoding="utf-8"
    )

    ingestion_summary: dict = {
        "diagram_id":          diagram_id,
        "dataset":             dataset,
        "split":               split,
        "detection_source":    detection_source,
        "node_count":          len(nodes),
        "edge_count":          len(edges),
        "rfdetr_inference_time_s": inference_time_s,
        "rfdetr_error":        rfdetr_error,
        "use_live_rfdetr":     rfdetr_model is not None or bool(checkpoint_path),
        "is_new":              False,
    }
    (out_dir / "ingestion_summary.json").write_text(
        json.dumps(ingestion_summary, indent=2), encoding="utf-8"
    )

    # ── CSV tables ───────────────────────────────────────────────────────────
    if node_table_rows:
        _write_csv(out_dir / "node_table.csv", node_table_rows)
    if edge_table_rows:
        _write_csv(out_dir / "edge_table.csv", edge_table_rows)

    return {
        "status":           "ok",
        "diagram_id":       diagram_id,
        "split":            split,
        "dataset":          dataset,
        "detection_source": detection_source,
        "node_count":       len(nodes),
        "edge_count":       len(edges),
        "out_dir":          str(out_dir),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# Enterprise graph builder
# ===========================================================================

def build_enterprise_graph(
    all_packets: list[dict],
    dataset: str,
    output_base: Path,
) -> dict:
    """
    Aggregate all processed diagram packets into a single enterprise graph JSON.

    Node IDs are prefixed with the diagram ID to avoid collisions across
    multiple diagrams in the same dataset (V1/V2 have no cross-diagram
    shared-entity information, so shared_entities is always empty).

    Saved to:  outputs/rfdetr_{v}/enterprise_graph.json
    """
    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    diagram_clusters: list[dict] = []
    seen_ids: set[str] = set()

    for packet in all_packets:
        diagram_id = packet["diagram_id"]
        split      = packet.get("split", "unknown")
        cluster_ids: list[str] = []

        for node in packet.get("nodes", []):
            global_id = f"{diagram_id}__{node['node_id']}"
            if global_id not in seen_ids:
                seen_ids.add(global_id)
                all_nodes.append({
                    **node,
                    "id":               global_id,
                    "label":            node["node_id"],
                    "diagram_id":       diagram_id,
                    "diagram_type":     None,
                    "canonical_id":     global_id,
                    "is_shared_entity": False,
                })
                cluster_ids.append(global_id)

        for edge in packet.get("edges", []):
            all_edges.append({
                **edge,
                "source":         f"{diagram_id}__{edge['source']}",
                "target":         f"{diagram_id}__{edge['target']}",
                "edge_type":      "local",
                "source_diagram": diagram_id,
                "target_diagram": diagram_id,
            })

        diagram_clusters.append({
            "diagram_id":   diagram_id,
            "split":        split,
            "dataset":      dataset,
            "diagram_type": None,
            "node_ids":     cluster_ids,
        })

    enterprise_graph: dict = {
        "dataset":         dataset,
        "diagram_types":   sorted({c["diagram_id"] for c in diagram_clusters}),
        "nodes":           all_nodes,
        "edges":           all_edges,
        "diagram_clusters":diagram_clusters,
        "shared_entities": [],
        "node_count":      len(all_nodes),
        "edge_count":      len(all_edges),
        "diagram_count":   len(diagram_clusters),
        "generated_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (output_base / "enterprise_graph.json").write_text(
        json.dumps(enterprise_graph, indent=2), encoding="utf-8"
    )
    return enterprise_graph


def build_enterprise_graph_mapping(
    all_packets: list[dict],
    enterprise_graph: dict,
    dataset: str,
    output_base: Path,
) -> None:
    """
    Write a mapping file linking local diagram node IDs to enterprise graph IDs.

    Saved to:  outputs/rfdetr_{v}/enterprise_graph_mapping.json

    Schema consumed by the Streamlit gallery Enterprise Graph Brain tab.
    """
    mapping: dict = {
        "dataset":       dataset,
        "generated_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "diagram_count": len(all_packets),
        "total_nodes":   enterprise_graph["node_count"],
        "total_edges":   enterprise_graph["edge_count"],
        "diagrams":      [],
    }

    for packet in all_packets:
        diagram_id = packet["diagram_id"]
        entries: list[dict] = []
        for node in packet.get("nodes", []):
            entries.append({
                "local_node_id":      node["node_id"],
                "enterprise_node_id": f"{diagram_id}__{node['node_id']}",
                "type":               node["type"],
                "zone":               node.get("zone", ""),
                "confidence":         node["confidence"],
            })
        mapping["diagrams"].append({
            "diagram_id":    diagram_id,
            "split":         packet.get("split", "unknown"),
            "node_count":    len(entries),
            "node_mappings": entries,
        })

    (output_base / "enterprise_graph_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )


# ===========================================================================
# Gallery preview sheet
# ===========================================================================

def build_gallery_preview_sheet(
    image_path: Path,
    detected_path: Path,
    out_path: Path,
    title: str = "",
) -> None:
    """
    Render a side-by-side contact sheet (original | detection) and save as PNG.
    Silently skips if PIL is unavailable or the source image is missing.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    if not image_path.exists():
        return

    SHEET_W  = 1400
    SHEET_H  = 520
    PAD      = 20
    HEADER_H = 44

    try:
        orig = Image.open(image_path).convert("RGB")
        det  = Image.open(detected_path).convert("RGB") if detected_path.exists() else orig.copy()
    except Exception:
        return

    cell_w = (SHEET_W - PAD * 3) // 2
    cell_h = SHEET_H - HEADER_H - PAD * 2

    sheet = Image.new("RGB", (SHEET_W, SHEET_H), (15, 23, 42))
    draw  = ImageDraw.Draw(sheet)

    # Header bar
    draw.rectangle([0, 0, SHEET_W, HEADER_H - 4], fill=(20, 30, 55))
    if title:
        draw.text((PAD, 6),  title,                     fill=(190, 215, 245))
    draw.text((PAD,           HEADER_H - 16), "Original",          fill=(100, 150, 200))
    draw.text((PAD + cell_w + PAD, HEADER_H - 16), "Detection Preview", fill=(100, 200, 120))

    # Paste thumbnails
    y_off = HEADER_H
    for pil_img, x_off in [(orig, PAD), (det, PAD + cell_w + PAD)]:
        thumb = pil_img.copy()
        thumb.thumbnail((cell_w, cell_h), Image.LANCZOS)
        sheet.paste(thumb, (x_off, y_off))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "PNG")


# ===========================================================================
# Gallery catalog (consumed by Streamlit UI)
# ===========================================================================

def write_gallery_catalog(
    all_results: list[dict],
    dataset: str,
    dataset_dir: Path,
    output_base: Path,
    enterprise_graph_path: Path,
    live_mode: bool = False,
) -> None:
    """
    Write gallery_catalog.json for the Streamlit Diagram Gallery tab.

    Each record uses the same schema as _build_diagram_catalog() in
    app/streamlit_app.py so the UI can load V1/V2 exactly like V3.

    Key fields:
      prediction_path  — annotated detected.png (V1/V2 use this, not annotation_path)
      local_graph_path — graph_memory_packet.json (closest V3 equivalent)
      is_new           — True when --live is set and no detection exists yet
      is_v3            — always False for V1/V2
    """
    records: list[dict] = []

    for r in all_results:
        status     = r.get("status", "error")
        diagram_id = r.get("diagram_id", "")
        split      = r.get("split", "unknown")
        out_dir    = Path(r.get("out_dir", output_base / split / diagram_id))

        src_image    = dataset_dir / "images" / split / f"{diagram_id}.png"
        detected_png = out_dir / "detected.png"
        packet_json  = out_dir / "graph_memory_packet.json"
        summary_json = out_dir / "ingestion_summary.json"

        # In --live mode, un-processed diagrams are flagged as "new"
        is_new = live_mode and not summary_json.exists()

        if status == "error" and not is_new:
            continue

        records.append({
            "display_name":         f"{dataset.upper()} / {split} / {diagram_id}",
            "dataset":              dataset,
            "split":                split,
            "diagram_id":           diagram_id,
            "diagram_type":         None,
            "image_path":           str(src_image),
            "prediction_path":      str(detected_png)     if detected_png.exists()  else None,
            "annotation_path":      None,
            "local_graph_path":     str(packet_json)      if packet_json.exists()   else None,
            "scenario_id":          f"{dataset}_{split}",
            "enterprise_graph_path":str(enterprise_graph_path)
                                    if enterprise_graph_path.exists() else None,
            "alerts_path":          None,
            "is_v3":                False,
            "is_new":               is_new,
        })

    (output_base / "gallery_catalog.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


# ===========================================================================
# Main
# ===========================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare V1/V2 diagram datasets for InfraGraph AI Diagram Gallery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--datasets", nargs="+", choices=["v1", "v2"], default=["v1", "v2"],
        help="Dataset versions to process",
    )
    p.add_argument(
        "--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS),
        help="Dataset splits to include",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="RF-DETR checkpoint .pth path (overrides auto-discovery from "
             "outputs/rfdetr_v3/model/)",
    )
    p.add_argument(
        "--conf", type=float, default=0.25,
        help="RF-DETR detection confidence threshold",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Max images per split per dataset (0 = all)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Reprocess diagrams even when outputs already exist",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without writing any files",
    )
    p.add_argument(
        "--live", action="store_true",
        help="Flag un-processed diagrams as 'new' in gallery catalog "
             "(Presentation-ready mode: UI will show Run Live Detector button)",
    )
    p.add_argument(
        "--output-base", type=str, default=None,
        help="Override base output directory (default: outputs/rfdetr_{v})",
    )
    p.add_argument(
        "--log-file", type=str, default=None,
        help="Log file path (default: outputs/prepare_gallery.log)",
    )
    p.add_argument(
        "--no-previews", action="store_true",
        help="Skip generating side-by-side preview contact sheets",
    )
    return p


def _setup_logging(log_file: str | None) -> logging.Logger:
    log_path = (
        Path(log_file)
        if log_file
        else REPO_ROOT / "outputs" / "prepare_gallery.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


def main() -> int:  # noqa: C901
    args = _build_arg_parser().parse_args()
    log  = _setup_logging(args.log_file)

    log.info(
        "InfraGraph AI — V1/V2 Gallery Prep  |  "
        f"datasets={args.datasets}  splits={args.splits}  "
        f"dry_run={args.dry_run}  live={args.live}  force={args.force}"
    )

    # ── RF-DETR model load ────────────────────────────────────────────────────
    rfdetr_model   = None
    checkpoint_path: str | None = args.checkpoint

    if not args.dry_run:
        try:
            from live_rfdetr_detector import (  # type: ignore
                find_best_rfdetr_checkpoint,
                load_rfdetr_model,
            )
            if checkpoint_path is None:
                ckpt = find_best_rfdetr_checkpoint(REPO_ROOT)
                checkpoint_path = str(ckpt) if ckpt else None

            if checkpoint_path and Path(checkpoint_path).exists():
                log.info(f"Loading RF-DETR checkpoint: {checkpoint_path}")
                rfdetr_model = load_rfdetr_model(checkpoint_path)
                log.info("RF-DETR model ready.")
            else:
                log.warning(
                    "No RF-DETR checkpoint found — "
                    "falling back to YOLO labels / no-detection."
                )
        except Exception as exc:
            log.warning(f"RF-DETR unavailable ({exc}); continuing without live detection.")

    # ── per-dataset loop ──────────────────────────────────────────────────────
    total_images_seen = 0

    for dataset in args.datasets:
        dataset_dir = DATASET_DIRS[dataset]
        if not dataset_dir.exists():
            log.warning(f"Dataset directory not found, skipping: {dataset_dir}")
            continue

        output_base = (
            Path(args.output_base) / dataset
            if args.output_base
            else OUTPUT_DIRS[dataset]
        )
        if not args.dry_run:
            output_base.mkdir(parents=True, exist_ok=True)

        log.info(f"\n{'='*64}")
        log.info(f"  Dataset: {dataset.upper()}   src={dataset_dir}")
        log.info(f"  Output:  {output_base}")
        log.info(f"{'='*64}")

        all_results: list[dict] = []
        all_packets: list[dict] = []

        # ── per-split loop ────────────────────────────────────────────────────
        for split in args.splits:
            img_dir = dataset_dir / "images" / split
            if not img_dir.exists():
                log.debug(f"  [{split}] image dir not found: {img_dir}")
                continue

            images = sorted(img_dir.glob("*.png"))
            if args.limit > 0:
                images = images[: args.limit]

            if not images:
                log.debug(f"  [{split}] no PNG images found")
                continue

            total_images_seen += len(images)
            log.info(f"  [{split}]  {len(images)} diagram(s)")

            for img_path in images:
                diagram_id = img_path.stem

                if args.dry_run:
                    out_dir = output_base / split / diagram_id
                    done    = (out_dir / "ingestion_summary.json").exists()
                    tag     = "done" if done else "NEW"
                    print(f"    [{tag:4s}] {dataset}/{split}/{diagram_id}")
                    all_results.append({
                        "status":     "skipped" if done else "pending",
                        "diagram_id": diagram_id,
                        "split":      split,
                        "dataset":    dataset,
                        "out_dir":    str(out_dir),
                    })
                    continue

                try:
                    result = process_diagram(
                        image_path      = img_path,
                        dataset         = dataset,
                        split           = split,
                        diagram_id      = diagram_id,
                        output_base     = output_base,
                        rfdetr_model    = rfdetr_model,
                        checkpoint_path = checkpoint_path,
                        conf            = args.conf,
                        force           = args.force,
                        log             = log,
                    )
                except Exception as exc:
                    result = {
                        "status":     "error",
                        "diagram_id": diagram_id,
                        "split":      split,
                        "dataset":    dataset,
                        "error":      str(exc),
                        "out_dir":    str(output_base / split / diagram_id),
                    }
                    log.error(
                        f"    ✗ FATAL {diagram_id}: {exc}\n"
                        + traceback.format_exc()
                    )

                all_results.append(result)

                status = result.get("status", "error")
                if status == "ok":
                    log.info(
                        f"    ✓ {diagram_id:30s}  "
                        f"{result.get('detection_source','?'):32s}  "
                        f"nodes={result.get('node_count',0):2d}  "
                        f"edges={result.get('edge_count',0):2d}"
                    )
                    pkt_path = Path(result["out_dir"]) / "graph_memory_packet.json"
                    if pkt_path.exists():
                        all_packets.append(
                            json.loads(pkt_path.read_text(encoding="utf-8"))
                        )
                elif status == "skipped":
                    log.debug(f"    — {diagram_id}: already processed")
                    pkt_path = Path(result["out_dir"]) / "graph_memory_packet.json"
                    if pkt_path.exists():
                        all_packets.append(
                            json.loads(pkt_path.read_text(encoding="utf-8"))
                        )
                else:
                    log.error(f"    ✗ {diagram_id}: {result.get('error')}")

            # ── preview contact sheets for this split ─────────────────────────
            if not args.dry_run and not args.no_previews:
                previews_dir = output_base / "previews" / split
                previews_dir.mkdir(parents=True, exist_ok=True)
                built = 0
                for r in all_results:
                    if r.get("split") != split:
                        continue
                    if r.get("status") not in ("ok", "skipped"):
                        continue
                    did = r["diagram_id"]
                    src = dataset_dir / "images" / split / f"{did}.png"
                    det = Path(r["out_dir"]) / "detected.png"
                    out = previews_dir / f"{did}_preview.png"
                    if not out.exists() or args.force:
                        build_gallery_preview_sheet(
                            src, det, out,
                            title=f"{dataset.upper()} / {split} / {did}",
                        )
                        built += 1
                if built:
                    log.info(f"  [{split}] Built {built} preview sheet(s) → {previews_dir}")

        # end per-split loop

        if args.dry_run:
            n_new  = sum(1 for r in all_results if r.get("status") == "pending")
            n_done = sum(1 for r in all_results if r.get("status") == "skipped")
            log.info(f"\n  DRY-RUN {dataset.upper()}: {n_new} new, {n_done} already processed")
            continue

        # ── enterprise graph ──────────────────────────────────────────────────
        if all_packets:
            log.info(
                f"\n  Building enterprise graph "
                f"({len(all_packets)} diagram packet(s))…"
            )
            enterprise_graph = build_enterprise_graph(all_packets, dataset, output_base)
            build_enterprise_graph_mapping(
                all_packets, enterprise_graph, dataset, output_base
            )
            log.info(
                f"  Enterprise graph: "
                f"{enterprise_graph['node_count']} nodes, "
                f"{enterprise_graph['edge_count']} edges, "
                f"{enterprise_graph['diagram_count']} diagrams"
            )
        else:
            log.warning("  No packets collected — enterprise graph not built.")
            enterprise_graph = {}

        # ── gallery catalog ───────────────────────────────────────────────────
        eg_path = output_base / "enterprise_graph.json"
        write_gallery_catalog(
            all_results          = all_results,
            dataset              = dataset,
            dataset_dir          = dataset_dir,
            output_base          = output_base,
            enterprise_graph_path= eg_path,
            live_mode            = args.live,
        )
        catalog_path = output_base / "gallery_catalog.json"
        n_catalog = len(json.loads(catalog_path.read_text(encoding="utf-8")))
        log.info(f"  Gallery catalog: {n_catalog} record(s) → {catalog_path}")

        # ── per-dataset summary ───────────────────────────────────────────────
        n_ok   = sum(1 for r in all_results if r.get("status") == "ok")
        n_skip = sum(1 for r in all_results if r.get("status") == "skipped")
        n_err  = sum(1 for r in all_results if r.get("status") == "error")
        log.info(
            f"\n  {dataset.upper()} COMPLETE — "
            f"processed={n_ok}  skipped={n_skip}  errors={n_err}"
        )
        log.info(f"  Outputs: {output_base}\n")

    # ── global exit ───────────────────────────────────────────────────────────
    if total_images_seen == 0 and not args.dry_run:
        log.error(
            "No images found in any requested dataset/split. "
            "Check that datasets/infragraph_v1/ and/or datasets/infragraph_v2/ exist."
        )
        return 2

    log.info("Gallery preparation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

