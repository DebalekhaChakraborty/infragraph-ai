"""
test_connector_extraction.py

Smoke-test for the vision connector extraction pipeline.

Usage:
    cd infragraph-ai
    python scripts/test_connector_extraction.py [--diagram <path>] [--annotation <path>]

If no --diagram is given, picks the first .png in
datasets/infragraph_v3/scenarios/val/<first_scenario>/diagrams/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── repo root on sys.path ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── CLI args ──────────────────────────────────────────────────────────────────
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision connector extraction smoke-test")
    p.add_argument("--diagram",    type=str, default="", help="Path to diagram image")
    p.add_argument("--annotation", type=str, default="", help="Path to annotation JSON")
    return p.parse_args()


def _find_sample() -> tuple[Path, Path | None]:
    """Return (diagram_path, annotation_path | None) from the dataset."""
    scenarios_root = REPO_ROOT / "datasets" / "infragraph_v3" / "scenarios" / "val"
    for scenario_dir in sorted(scenarios_root.iterdir()):
        diag_dir = scenario_dir / "diagrams"
        ann_dir  = scenario_dir / "annotations"
        for img in sorted(diag_dir.glob("*.png")):
            ann = ann_dir / (img.stem + ".json")
            return img, (ann if ann.exists() else None)
    return Path(), None


def _nodes_from_annotation(ann_path: Path) -> list[dict]:
    """Extract detected_nodes in the runtime_ingestion format from an annotation file."""
    ann = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    _class_to_type = {
        "router": "router", "switch": "switch", "firewall": "firewall",
        "server": "server", "load_balancer": "load_balancer",
        "database": "database", "dns_server": "dns", "cloud": "cloud",
        "wlan_ap": "wlan_ap", "vpn_gateway": "vpn_gateway",
    }
    nodes: list[dict] = []
    for obj in ann.get("objects", []):
        nodes.append({
            "node_id":      obj.get("object_id", obj.get("canonical_id", "")),
            "canonical_id": obj.get("canonical_id", obj.get("object_id", "")),
            "class_name":   obj.get("class_name", "server"),
            "type":         _class_to_type.get(obj.get("class_name", ""), "server"),
            "bbox":         obj.get("bbox", []),
            "confidence":   obj.get("confidence", 0.88),
        })
    return nodes


def main() -> None:
    args = _parse()

    # ── resolve diagram + annotation ──────────────────────────────────────────
    if args.diagram:
        diagram_path = Path(args.diagram)
        ann_path     = Path(args.annotation) if args.annotation else None
    else:
        diagram_path, ann_path = _find_sample()

    if not diagram_path or not diagram_path.exists():
        print(f"[ERROR] Diagram not found: {diagram_path}")
        sys.exit(1)

    print(f"Diagram  : {diagram_path}")
    print(f"Annotation: {ann_path or '(none)'}")

    # ── load detected_nodes ───────────────────────────────────────────────────
    detected_nodes: list[dict] = []
    if ann_path and ann_path.exists():
        detected_nodes = _nodes_from_annotation(ann_path)
    print(f"Nodes loaded: {len(detected_nodes)}")

    # ── run extraction ────────────────────────────────────────────────────────
    from vision.edge_extraction.edge_builder import extract_edges_from_diagram
    from vision.edge_extraction.debug_render import render_connector_debug_overlay

    result = extract_edges_from_diagram(diagram_path, detected_nodes)

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\n--- Vision Connector Extraction ---")
    print(f"ok            : {result['ok']}")
    print(f"source        : {result['source']}")
    print(f"segment_count : {result['segment_count']}")
    print(f"edge_count    : {result['edge_count']}")
    if result.get("warning"):
        print(f"warning       : {result['warning']}")

    if result.get("segments"):
        print(f"\nTop-5 segments (by length):")
        for seg in result["segments"][:5]:
            print(
                f"  {seg['segment_id']}  ({seg['x1']},{seg['y1']})→({seg['x2']},{seg['y2']})"
                f"  len={seg['length']}  conf={seg['confidence']}"
            )

    if result.get("edges"):
        print(f"\nMatched edges ({result['edge_count']} total):")
        for e in result["edges"]:
            print(
                f"  {e['source']} ↔ {e['target']}"
                f"  conf={e['connector_confidence']}"
                f"  seg={e.get('segment_id', '')}"
            )
    else:
        print("\nNo edges matched — fallback to annotation/local-graph edges.")

    # ── debug overlay ─────────────────────────────────────────────────────────
    out_dir  = REPO_ROOT / "outputs" / "connector_extraction_debug"
    out_path = out_dir / f"{diagram_path.stem}_connectors_debug.png"
    written  = render_connector_debug_overlay(
        diagram_path,
        result.get("segments", []),
        result.get("edges", []),
        out_path,
    )
    if written:
        print(f"\nDebug overlay written to: {written}")
    else:
        print("\nDebug overlay skipped (OpenCV unavailable or no content).")


if __name__ == "__main__":
    main()
