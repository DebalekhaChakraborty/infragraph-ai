#!/usr/bin/env python3
"""
onboard_diagram.py — Live diagram onboarding for InfraGraph AI.

Runs YOLO detection, builds a local topology graph, and registers the result
into graph_memory/index.json.

Usage:
    python scripts/onboard_diagram.py \
        --image datasets/infragraph_v2/images/test/diagram_0373.png \
        --diagram-id demo_onboard_0373 \
        --out outputs/onboarded_diagrams
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).parent.parent

# ── Optional visual libs ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    from PIL import Image as _PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Candidate YOLO model paths, tried in order
_MODEL_CANDIDATES = [
    REPO_ROOT / "training_runs" / "infragraph_yolo_v2" / "weights" / "best.pt",
    REPO_ROOT / "training_runs" / "infragraph_yolo_v1" / "weights" / "best.pt",
]

GRAPH_MEMORY_DIR   = REPO_ROOT / "graph_memory"
GRAPH_MEMORY_INDEX = GRAPH_MEMORY_DIR / "index.json"

# Presentation Alternate path paths (used when YOLO unavailable and image is diagram_0373)
_DEMO_STEM           = "diagram_0373"
_DEMO_DETECTED_NODES = REPO_ROOT / "outputs" / "topology_demo" / "diagram_0373_detected_nodes.json"
_DEMO_DETECTED_IMG   = REPO_ROOT / "outputs" / "v2_test_predictions_cpu" / "diagram_0373.jpg"

# Device-type → canonical ID prefix
TYPE_PREFIX: dict[str, str] = {
    "firewall":      "FW",
    "router":        "RTR",
    "switch":        "SW",
    "server":        "APP",
    "database":      "DB",
    "load_balancer": "LB",
    "cloud_or_wan":  "WAN",
    "service":       "SVC",
}

# Zone bands by vertical fraction of image height
_ZONE_BANDS = [(0.25, "wan"), (0.50, "perimeter"), (0.75, "core"), (1.01, "server")]

# Which types each device type can connect to (ordered by preference)
DOWNSTREAM: dict[str, list[str]] = {
    "cloud_or_wan":  ["router", "firewall"],
    "router":        ["firewall", "switch"],
    "firewall":      ["switch", "load_balancer"],
    "switch":        ["load_balancer", "server", "database"],
    "load_balancer": ["server"],
    "server":        ["database"],
    "database":      [],
}

RELATIONSHIP: dict[tuple[str, str], str] = {
    ("cloud_or_wan", "router"):    "routes_to",
    ("cloud_or_wan", "firewall"):  "routes_to",
    ("router",        "firewall"): "routes_to",
    ("router",        "switch"):   "routes_to",
    ("firewall",      "switch"):         "secured_by",
    ("firewall",      "load_balancer"):  "secured_by",
    ("switch",        "load_balancer"):  "connected_to",
    ("switch",        "server"):         "connected_to",
    ("switch",        "database"):       "connected_to",
    ("load_balancer", "server"):         "serves",
    ("server",        "database"):       "depends_on",
}

NODE_TYPE_COLOR: dict[str, str] = {
    "router":        "#60a5fa",
    "switch":        "#818cf8",
    "firewall":      "#ef4444",
    "server":        "#22d3ee",
    "database":      "#10b981",
    "load_balancer": "#f59e0b",
    "cloud_or_wan":  "#94a3b8",
    "service":       "#a78bfa",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _infer_zone(bbox: dict, img_h: int) -> str:
    cy = (bbox["y1"] + bbox["y2"]) / 2
    frac = cy / max(1, img_h)
    for threshold, name in _ZONE_BANDS:
        if frac <= threshold:
            return name
    return "server"


def _canonical_name(device_type: str, count: int) -> str:
    return f"{TYPE_PREFIX.get(device_type, 'DEV')}-{count:03d}"


def _center(bbox: dict) -> tuple[float, float]:
    return ((bbox["x1"] + bbox["x2"]) / 2, (bbox["y1"] + bbox["y2"]) / 2)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _img_dimensions(image_path: Path) -> tuple[int, int]:
    """Return (width, height) of image, falling back to defaults."""
    if HAS_PIL:
        try:
            with _PILImage.open(str(image_path)) as img:
                return img.size
        except Exception:
            pass
    return 1280, 960


# ══════════════════════════════════════════════════════════════════════════════
# YOLO INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _try_yolo(
    image_path: Path,
    model_path: Path,
) -> tuple[list[dict], Any] | None:
    """
    Attempt YOLO inference. Returns (raw_detections, results_obj) or None.
    raw_detections: list of {"type", "confidence", "bbox":{x1,y1,x2,y2}}
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return None

    # Try given path; if absent, walk candidates
    resolved = model_path if model_path.exists() else None
    if resolved is None:
        for cand in _MODEL_CANDIDATES:
            if cand.exists():
                resolved = cand
                break
    if resolved is None:
        return None

    try:
        model = YOLO(str(resolved))
        results = model.predict(str(image_path), device="cpu", save=False, verbose=False)
        if not results:
            return None
        r = results[0]
        detections: list[dict] = []
        for box in r.boxes:
            cls_idx = int(box.cls.item())
            detections.append({
                "type":       r.names[cls_idx],
                "confidence": float(box.conf.item()),
                "bbox": {
                    "x1": float(box.xyxy[0][0].item()),
                    "y1": float(box.xyxy[0][1].item()),
                    "x2": float(box.xyxy[0][2].item()),
                    "y2": float(box.xyxy[0][3].item()),
                },
            })
        if not detections:
            return None
        return detections, r
    except Exception as exc:
        print(f"  [warn] YOLO inference error: {exc}", file=sys.stderr)
        return None


def _save_annotated(results_obj: Any, out_path: Path) -> bool:
    """Save YOLO-annotated image. Returns True on success."""
    try:
        annotated = results_obj.plot()  # BGR numpy array
        if HAS_PIL:
            img = _PILImage.fromarray(annotated[..., ::-1])   # BGR → RGB
            img.save(str(out_path))
        else:
            import numpy as np  # noqa: F401
            import cv2
            cv2.imwrite(str(out_path), annotated)
        return True
    except Exception as exc:
        print(f"  [warn] Could not save annotated image: {exc}", file=sys.stderr)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MOCK Alternate path (diagram_0373 only)
# ══════════════════════════════════════════════════════════════════════════════

def _load_mock_detections(image_path: Path) -> list[dict] | str:
    """
    Load pre-existing detected nodes for diagram_0373.
    Returns raw detections list or an error string.
    """
    if _DEMO_STEM not in str(image_path):
        return (
            f"YOLO model unavailable and '{image_path.name}' is not diagram_0373. "
            "Provide a trained model via --model to onboard new diagrams."
        )
    if not _DEMO_DETECTED_NODES.exists():
        return f"Presentation detected_nodes not found at {_DEMO_DETECTED_NODES}"

    raw = json.loads(_DEMO_DETECTED_NODES.read_text(encoding="utf-8"))
    detections: list[dict] = []
    for n in raw:
        bp = n.get("bbox_pixel", [0, 0, 100, 100])
        detections.append({
            "type":       n.get("type", "server"),
            "confidence": float(n.get("confidence", 0.80)),
            "bbox":       {"x1": float(bp[0]), "y1": float(bp[1]),
                           "x2": float(bp[2]), "y2": float(bp[3])},
        })
    return detections


# ══════════════════════════════════════════════════════════════════════════════
# NODE TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_detected_nodes(
    raw_detections: list[dict],
    img_w: int,
    img_h: int,
) -> list[dict]:
    """
    Convert raw detections into the onboarding detected_nodes schema.
    Assigns canonical names (FW-001, SW-001, …) and infers zones.
    """
    type_counts: dict[str, int] = {}
    nodes: list[dict] = []
    for det in raw_detections:
        dtype = det["type"]
        type_counts[dtype] = type_counts.get(dtype, 0) + 1
        count = type_counts[dtype]
        bbox  = det["bbox"]
        cx, cy = _center(bbox)
        nodes.append({
            "node_id":        _canonical_name(dtype, count),
            "detected_type":  dtype,
            "confidence":     round(det["confidence"], 4),
            "bbox":           {k: round(v) for k, v in bbox.items()},
            "bbox_center":    {"x": round(cx), "y": round(cy)},
            "zone":           _infer_zone(bbox, img_h),
            "canonical_name": _canonical_name(dtype, count),
            "graph_status":   "added",
        })
    return nodes


# ══════════════════════════════════════════════════════════════════════════════
# EDGE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def infer_edges(nodes: list[dict]) -> list[dict]:
    """
    Infer edges from node types and spatial layout using nearest-neighbour rules.

    Rules:
    - cloud_or_wan  → nearest router or firewall
    - router        → nearest firewall or switch
    - firewall      → nearest switch or load_balancer
    - switch        → nearest load_balancer, server, or database
    - load_balancer → nearest servers (up to 3)
    - server        → nearest database
    """
    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()

    centers  = {n["node_id"]: (n["bbox_center"]["x"], n["bbox_center"]["y"]) for n in nodes}
    by_type: dict[str, list[dict]] = {}
    for n in nodes:
        by_type.setdefault(n["detected_type"], []).append(n)

    for src in nodes:
        downstream_types = DOWNSTREAM.get(src["detected_type"], [])
        if not downstream_types:
            continue
        src_c = centers[src["node_id"]]

        # Gather all candidate targets from downstream types
        candidates: list[tuple[float, dict]] = []
        for dt in downstream_types:
            for dst in by_type.get(dt, []):
                if dst["node_id"] == src["node_id"]:
                    continue
                ek  = (src["node_id"], dst["node_id"])
                rk  = (dst["node_id"], src["node_id"])
                if ek in seen or rk in seen:
                    continue
                candidates.append((_dist(src_c, centers[dst["node_id"]]), dst))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])
        # load_balancer fans out to up to 3 servers; others connect to 1–2 nearest
        max_conn = 3 if src["detected_type"] == "load_balancer" else 2

        for _, dst in candidates[:max_conn]:
            ek = (src["node_id"], dst["node_id"])
            seen.add(ek)
            rel = RELATIONSHIP.get(
                (src["detected_type"], dst["detected_type"]), "connected_to"
            )
            edges.append({
                "source":       src["node_id"],
                "target":       dst["node_id"],
                "relationship": rel,
            })

    return edges


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_local_graph(diagram_id: str, nodes: list[dict], edges: list[dict]) -> dict:
    graph_nodes = [
        {
            "id":         n["node_id"],
            "label":      f"{n['detected_type'].replace('_', ' ').title()} {n['node_id']}",
            "type":       n["detected_type"],
            "zone":       n["zone"],
            "confidence": n["confidence"],
            "bbox_center": n["bbox_center"],
        }
        for n in nodes
    ]
    return {
        "diagram_id":         diagram_id,
        "nodes":              graph_nodes,
        "edges":              edges,
        "graph_build_method": "layout_inference_v1",
        "notes": (
            "Edges are inferred from detected device types and spatial layout for MVP onboarding. "
            "Node names are generated during MVP onboarding unless metadata/OCR is available."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH PREVIEW
# ══════════════════════════════════════════════════════════════════════════════

def draw_graph_preview(
    graph: dict,
    out_path: Path,
) -> bool:
    """Draw a networkx + matplotlib preview of the local graph. Returns True on success."""
    if not (HAS_MPL and HAS_NX):
        return False

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    diagram_id = graph.get("diagram_id", "")

    if not nodes:
        return False

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"], ntype=n["type"])
    for e in edges:
        if G.has_node(e["source"]) and G.has_node(e["target"]):
            G.add_edge(e["source"], e["target"], rel=e.get("relationship", ""))

    # Use spatial positions from bbox_center (flip y for image coords)
    pos: dict[str, tuple[float, float]] = {}
    for n in nodes:
        bc = n.get("bbox_center", {})
        if bc:
            pos[n["id"]] = (float(bc.get("x", 0)), -float(bc.get("y", 0)))
    if len(pos) < len(nodes):
        pos = nx.spring_layout(G, seed=42, k=2.5)

    colors = [NODE_TYPE_COLOR.get(G.nodes[n].get("ntype", "server"), "#94a3b8") for n in G.nodes]
    labels = {n: n for n in G.nodes}

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_facecolor("#0b0f1c")
    fig.patch.set_facecolor("#0b0f1c")

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=colors, node_size=1300,
        labels=labels, font_size=7, font_color="white",
        edge_color="#334155", arrows=True, arrowsize=15, width=1.5,
        connectionstyle="arc3,rad=0.08",
    )

    present_types = list({G.nodes[n].get("ntype", "server") for n in G.nodes})
    legend = [
        mpatches.Patch(color=NODE_TYPE_COLOR.get(t, "#94a3b8"),
                       label=t.replace("_", " ").title())
        for t in present_types
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=7,
              facecolor="#0d1220", edgecolor="#334155", labelcolor="white")
    ax.set_title(
        f"Local Graph — {diagram_id} — {len(nodes)} nodes · {len(edges)} edges",
        color="#e2e8f0", fontsize=10, fontweight="bold", pad=8,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH MEMORY INDEX
# ══════════════════════════════════════════════════════════════════════════════

def update_graph_memory_index(entry: dict) -> None:
    """Add or update one entry in graph_memory/index.json."""
    GRAPH_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    if GRAPH_MEMORY_INDEX.exists():
        try:
            index = json.loads(GRAPH_MEMORY_INDEX.read_text(encoding="utf-8"))
        except Exception:
            index = []
    # Replace any existing entry for this diagram_id
    index = [e for e in index if e.get("diagram_id") != entry["diagram_id"]]
    index.append(entry)
    GRAPH_MEMORY_INDEX.write_text(json.dumps(index, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ONBOARDING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_onboarding(
    image_path: str | Path,
    diagram_id: str,
    model_path: str | Path | None = None,
    out_dir: str | Path = "assets/preloaded/onboarded_diagrams",
    on_step: Callable[[str], None] | None = None,
) -> dict:
    """
    Onboard one topology diagram into InfraGraph AI graph memory.

    Parameters
    ----------
    image_path  : path to the input PNG/JPG
    diagram_id  : unique identifier for this diagram
    model_path  : YOLO model weights; auto-detected if None
    out_dir     : root output directory
    on_step     : optional callback(msg) for progress reporting

    Returns
    -------
    dict with keys: success, error, diagram_id, detection_method, paths, stats, nodes, edges
    """

    def _step(msg: str) -> None:
        if on_step:
            on_step(msg)

    image_path = Path(image_path)
    out_dir    = Path(out_dir)
    sc_dir     = out_dir / diagram_id
    sc_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Receive image ──────────────────────────────────────────────────────
    _step("Image received")
    shutil.copy2(str(image_path), str(sc_dir / "original.png"))
    img_w, img_h = _img_dimensions(image_path)

    # ── 2. Vision detection ───────────────────────────────────────────────────
    _step("Running vision detection...")

    resolved_model: Path | None = None
    if model_path:
        resolved_model = Path(model_path)
        if not resolved_model.exists():
            resolved_model = None
    if resolved_model is None:
        for cand in _MODEL_CANDIDATES:
            if cand.exists():
                resolved_model = cand
                break

    raw_detections: list[dict] | None = None
    yolo_obj: Any = None
    detection_method = "yolo_v8"

    if resolved_model:
        yolo_result = _try_yolo(image_path, resolved_model)
        if yolo_result:
            raw_detections, yolo_obj = yolo_result

    if raw_detections is None:
        detection_method = "mock_fallback"
        mock = _load_mock_detections(image_path)
        if isinstance(mock, str):
            return {"success": False, "error": mock, "paths": {}, "stats": {}}
        raw_detections = mock

    # Save detection image
    detected_img = sc_dir / "detected.png"
    if yolo_obj is not None:
        if not _save_annotated(yolo_obj, detected_img):
            shutil.copy2(str(image_path), str(detected_img))
    elif _DEMO_DETECTED_IMG.exists():
        shutil.copy2(str(_DEMO_DETECTED_IMG), str(detected_img))
    else:
        shutil.copy2(str(image_path), str(detected_img))

    _step("Vision detection complete")

    # ── 3. Node table ─────────────────────────────────────────────────────────
    nodes = build_detected_nodes(raw_detections, img_w, img_h)
    nodes_path = sc_dir / "detected_nodes.json"
    nodes_path.write_text(json.dumps(nodes, indent=2), encoding="utf-8")
    _step("Node table generated")

    # ── 4. Local graph ────────────────────────────────────────────────────────
    edges      = infer_edges(nodes)
    local_graph = build_local_graph(diagram_id, nodes, edges)
    graph_path  = sc_dir / "local_graph.json"
    graph_path.write_text(json.dumps(local_graph, indent=2), encoding="utf-8")
    _step("Local graph created")

    # ── 5. Graph preview ──────────────────────────────────────────────────────
    preview_path = sc_dir / "graph_preview.png"
    draw_graph_preview(local_graph, preview_path)

    # ── 6. Graph memory index ─────────────────────────────────────────────────
    memory_entry = {
        "diagram_id":          diagram_id,
        "source_image":        str(image_path),
        "detected_nodes_path": str(nodes_path),
        "local_graph_path":    str(graph_path),
        "graph_preview_path":  str(preview_path) if preview_path.exists() else None,
        "node_count":          len(nodes),
        "edge_count":          len(edges),
        "detection_method":    detection_method,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "status":              "processed",
    }
    update_graph_memory_index(memory_entry)
    _step("Graph memory updated")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    type_dist: dict[str, int] = {}
    for n in nodes:
        type_dist[n["detected_type"]] = type_dist.get(n["detected_type"], 0) + 1

    summary = {
        "diagram_id":              diagram_id,
        "source_image":            str(image_path),
        "onboarding_timestamp":    datetime.now(timezone.utc).isoformat(),
        "detection_method":        detection_method,
        "node_count":              len(nodes),
        "edge_count":              len(edges),
        "detected_types":          type_dist,
        "graph_memory_status":     "registered",
        "output_dir":              str(sc_dir),
    }
    summary_path = sc_dir / "onboarding_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "success":          True,
        "error":            None,
        "diagram_id":       diagram_id,
        "detection_method": detection_method,
        "paths": {
            "original":       str(sc_dir / "original.png"),
            "detected":       str(detected_img),
            "detected_nodes": str(nodes_path),
            "local_graph":    str(graph_path),
            "graph_preview":  str(preview_path),
            "summary":        str(summary_path),
        },
        "stats": {
            "node_count":    len(nodes),
            "edge_count":    len(edges),
            "detected_types": type_dist,
        },
        "nodes":  nodes,
        "edges":  edges,
        "graph":  local_graph,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Onboard a topology diagram into InfraGraph AI graph memory"
    )
    p.add_argument("--image",      required=True,
                   help="Path to input diagram image (PNG or JPG)")
    p.add_argument("--diagram-id", required=True,
                   help="Unique identifier for this diagram")
    p.add_argument(
        "--model",
        default=str(REPO_ROOT / "training_runs" / "infragraph_yolo_v2" / "weights" / "best.pt"),
        help="YOLO model weights path (default: infragraph_yolo_v2/best.pt, uses v1)",
    )
    p.add_argument("--out", default="assets/preloaded/onboarded_diagrams",
                   help="Root output directory (default: assets/preloaded/onboarded_diagrams)")
    args = p.parse_args()

    _p = lambda s: print(s, file=sys.stdout)  # noqa: E731
    _p("\nInfraGraph AI -- Diagram Onboarding")
    _p(f"   Image     : {args.image}")
    _p(f"   Diagram ID: {args.diagram_id}")
    _p(f"   Model     : {args.model}")
    _p(f"   Output    : {args.out}\n")

    if not HAS_MPL or not HAS_NX:
        _p("   [warn] matplotlib/networkx not found -- graph_preview.png will be skipped")
    if not HAS_PIL:
        _p("   [warn] Pillow not found -- image dimension detection using defaults")

    result = run_onboarding(
        image_path=args.image,
        diagram_id=args.diagram_id,
        model_path=args.model,
        out_dir=args.out,
        on_step=lambda msg: _p(f"   -> {msg}"),
    )

    if result["success"]:
        stats = result["stats"]
        _p("\n   [OK] Onboarding complete")
        _p(f"   Nodes : {stats['node_count']}")
        _p(f"   Edges : {stats['edge_count']}")
        _p(f"   Method: {result['detection_method']}")
        _p(f"   Output: {Path(result['paths']['original']).parent}")
        _p(f"   Graph memory: {GRAPH_MEMORY_INDEX}")
        _p("\n   Output files:")
        for key, path in result["paths"].items():
            if path:
                pp = Path(path)
                if pp.exists():
                    kb = pp.stat().st_size // 1024
                    _p(f"     {key:<18s} {path}  ({kb} KB)" if kb else f"     {key:<18s} {path}")
        _p("")
    else:
        _p(f"\n   [FAIL] Onboarding failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()


