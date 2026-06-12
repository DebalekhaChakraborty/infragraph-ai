#!/usr/bin/env python3
"""Quality checks for InfraGraph V3 annotation JSON files."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VALID_CLASSES = {
    "router",
    "switch",
    "firewall",
    "server",
    "database",
    "load_balancer",
    "cloud_or_wan",
    "service",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _to_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        return [float(value[i]) for i in range(4)]
    except Exception:
        return None


def _clip_box(raw: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = raw
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return (
        max(0.0, min(float(width - 1), x1)),
        max(0.0, min(float(height - 1), y1)),
        max(0.0, min(float(width - 1), x2)),
        max(0.0, min(float(height - 1), y2)),
    )


def _point_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except Exception:
        return None


def _connector_length(points: list[Any]) -> float:
    parsed = [_point_pair(p) for p in points]
    clean = [p for p in parsed if p is not None]
    if len(clean) < 2:
        return 0.0
    total = 0.0
    for (x1, y1), (x2, y2) in zip(clean, clean[1:]):
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _possible_cross_diagram_connector(conn: dict[str, Any]) -> bool:
    text = " ".join(
        str(conn.get(k, ""))
        for k in (
            "relationship",
            "label",
            "label_text",
            "edge_scope",
            "source_diagram",
            "target_diagram",
            "connector_id",
        )
    ).lower()
    return any(term in text for term in ("cross", "enterprise", "shared", "wan_dependency"))


def _iter_annotation_paths(dataset_root: Path) -> list[Path]:
    paths: list[Path] = []
    for split in ("train", "val", "test"):
        split_root = dataset_root / "scenarios" / split
        if not split_root.exists():
            continue
        for scenario_dir in sorted(split_root.iterdir()):
            ann_dir = scenario_dir / "annotations"
            if ann_dir.exists():
                paths.extend(sorted(ann_dir.glob("*.json")))
    return paths


def _image_path_for_annotation(annotation: dict[str, Any], ann_path: Path) -> Path:
    image_path = annotation.get("image_path")
    if image_path:
        candidate = Path(str(image_path))
        if candidate.exists():
            return candidate
    scenario_dir = ann_path.parent.parent
    return scenario_dir / "diagrams" / f"{ann_path.stem}.png"


def inspect_annotation(ann_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    annotation = _load_json(ann_path)
    scenario_dir = ann_path.parent.parent
    split = scenario_dir.parent.name
    scenario_id = annotation.get("scenario_id") or scenario_dir.name
    diagram_id = annotation.get("diagram_id") or ann_path.stem
    diagram_type = annotation.get("diagram_type") or diagram_id
    image_path = _image_path_for_annotation(annotation, ann_path)

    width = int(annotation.get("width") or 0)
    height = int(annotation.get("height") or 0)
    image_exists = image_path.exists()
    image_area = max(width * height, 1)
    diagonal = math.hypot(width, height) if width > 0 and height > 0 else 1.0

    objects = annotation.get("objects", [])
    connectors = annotation.get("connectors", [])

    row: dict[str, Any] = {
        "split": split,
        "scenario_id": scenario_id,
        "diagram_id": diagram_id,
        "diagram_type": diagram_type,
        "annotation_path": str(ann_path),
        "image_path": str(image_path),
        "image_exists": image_exists,
        "annotation_exists": ann_path.exists(),
        "width": width,
        "height": height,
        "object_count": len(objects) if isinstance(objects, list) else 0,
        "connector_count": len(connectors) if isinstance(connectors, list) else 0,
        "invalid_class_count": 0,
        "malformed_bbox_count": 0,
        "out_of_bounds_bbox_count": 0,
        "tiny_bbox_count": 0,
        "suspicious_large_bbox_count": 0,
        "very_large_bbox_count": 0,
        "suspicious_wide_bbox_count": 0,
        "suspicious_tall_bbox_count": 0,
        "missing_object_id_count": 0,
        "missing_label_text_count": 0,
        "malformed_connector_count": 0,
        "suspicious_long_connector_count": 0,
        "possible_cross_diagram_connector_count": 0,
        "max_bbox_area_ratio": 0.0,
        "max_connector_length_ratio": 0.0,
        "suspicious_score": 0,
    }
    suspicious_rows: list[dict[str, Any]] = []

    if not width or not height:
        row["suspicious_score"] += 5
        suspicious_rows.append({**row, "item_type": "diagram", "item_id": diagram_id, "issue": "missing_width_height"})
    if not image_exists:
        row["suspicious_score"] += 5
        suspicious_rows.append({**row, "item_type": "diagram", "item_id": diagram_id, "issue": "image_missing"})

    if not isinstance(objects, list):
        objects = []
        row["suspicious_score"] += 5
        suspicious_rows.append({**row, "item_type": "objects", "item_id": diagram_id, "issue": "objects_not_list"})

    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            row["malformed_bbox_count"] += 1
            row["suspicious_score"] += 2
            suspicious_rows.append({**row, "item_type": "object", "item_id": f"object_{idx}", "issue": "object_not_dict"})
            continue

        object_id = obj.get("object_id", "")
        label_text = obj.get("label_text", "")
        class_name = obj.get("class_name", "")
        item_id = object_id or label_text or f"object_{idx}"

        if class_name not in VALID_CLASSES:
            row["invalid_class_count"] += 1
            row["suspicious_score"] += 2
            suspicious_rows.append({**row, "item_type": "object", "item_id": item_id, "issue": "invalid_class"})
        if not object_id:
            row["missing_object_id_count"] += 1
            row["suspicious_score"] += 1
            suspicious_rows.append({**row, "item_type": "object", "item_id": item_id, "issue": "missing_object_id"})
        if not label_text:
            row["missing_label_text_count"] += 1
            row["suspicious_score"] += 1
            suspicious_rows.append({**row, "item_type": "object", "item_id": item_id, "issue": "missing_label_text"})

        raw = _to_float_list(obj.get("bbox"))
        if raw is None or width <= 0 or height <= 0:
            row["malformed_bbox_count"] += 1
            row["suspicious_score"] += 3
            suspicious_rows.append({**row, "item_type": "object", "item_id": item_id, "issue": "malformed_bbox"})
            continue

        clipped = _clip_box(raw, width, height)
        x1, y1, x2, y2 = clipped
        clipped_w = x2 - x1
        clipped_h = y2 - y1
        if clipped != tuple(raw):
            row["out_of_bounds_bbox_count"] += 1
            row["suspicious_score"] += 1
            suspicious_rows.append({**row, "item_type": "object", "item_id": item_id, "issue": "bbox_clipped_to_image"})
        if clipped_w <= 2 or clipped_h <= 2:
            row["malformed_bbox_count"] += 1
            row["suspicious_score"] += 3
            suspicious_rows.append({**row, "item_type": "object", "item_id": item_id, "issue": "degenerate_bbox"})
            continue

        area_ratio = (clipped_w * clipped_h) / image_area
        row["max_bbox_area_ratio"] = max(float(row["max_bbox_area_ratio"]), area_ratio)
        flags: list[str] = []
        if area_ratio > 0.12:
            row["suspicious_large_bbox_count"] += 1
            row["suspicious_score"] += 3
            flags.append("suspicious_large_bbox")
        if area_ratio > 0.18:
            row["very_large_bbox_count"] += 1
            row["suspicious_score"] += 5
            flags.append("very_large_bbox")
        if area_ratio < 0.00005:
            row["tiny_bbox_count"] += 1
            row["suspicious_score"] += 2
            flags.append("tiny_bbox")
        if clipped_w > width * 0.40:
            row["suspicious_wide_bbox_count"] += 1
            row["suspicious_score"] += 2
            flags.append("suspicious_wide_bbox")
        if clipped_h > height * 0.40:
            row["suspicious_tall_bbox_count"] += 1
            row["suspicious_score"] += 2
            flags.append("suspicious_tall_bbox")
        for flag in flags:
            suspicious_rows.append({
                **row,
                "item_type": "object",
                "item_id": item_id,
                "issue": flag,
                "bbox_area_ratio": round(area_ratio, 6),
            })

    if not isinstance(connectors, list):
        connectors = []
        row["suspicious_score"] += 3
        suspicious_rows.append({**row, "item_type": "connectors", "item_id": diagram_id, "issue": "connectors_not_list"})

    for idx, conn in enumerate(connectors):
        if not isinstance(conn, dict):
            row["malformed_connector_count"] += 1
            row["suspicious_score"] += 2
            suspicious_rows.append({**row, "item_type": "connector", "item_id": f"connector_{idx}", "issue": "connector_not_dict"})
            continue
        connector_id = conn.get("connector_id") or f"connector_{idx}"
        source = conn.get("source") or conn.get("from_node")
        target = conn.get("target") or conn.get("to_node")
        points = conn.get("points")
        malformed = not source or not target or not isinstance(points, list) or len(points) < 2
        if malformed:
            row["malformed_connector_count"] += 1
            row["suspicious_score"] += 2
            suspicious_rows.append({**row, "item_type": "connector", "item_id": connector_id, "issue": "malformed_connector"})
            continue
        length_ratio = _connector_length(points) / max(diagonal, 1.0)
        row["max_connector_length_ratio"] = max(float(row["max_connector_length_ratio"]), length_ratio)
        if length_ratio > 0.65:
            row["suspicious_long_connector_count"] += 1
            row["suspicious_score"] += 2
            suspicious_rows.append({
                **row,
                "item_type": "connector",
                "item_id": connector_id,
                "issue": "suspicious_long_connector",
                "connector_length_ratio": round(length_ratio, 6),
            })
        if _possible_cross_diagram_connector(conn):
            row["possible_cross_diagram_connector_count"] += 1
            suspicious_rows.append({
                **row,
                "item_type": "connector",
                "item_id": connector_id,
                "issue": "possible_cross_diagram_connector",
                "connector_length_ratio": round(length_ratio, 6),
            })

    row["max_bbox_area_ratio"] = round(float(row["max_bbox_area_ratio"]), 6)
    row["max_connector_length_ratio"] = round(float(row["max_connector_length_ratio"]), 6)
    return row, suspicious_rows


def make_recommendation(suspicious_large_rate: float) -> str:
    if suspicious_large_rate < 0.05:
        return "DISPLAY_ONLY_FIX"
    if suspicious_large_rate <= 0.15:
        return "MANUAL_REVIEW_REQUIRED"
    return "ANNOTATION_REGENERATION_RECOMMENDED"


def run_qa(dataset_root: Path, out_dir: Path) -> dict[str, Any]:
    ann_paths = _iter_annotation_paths(dataset_root)
    diagram_rows: list[dict[str, Any]] = []
    suspicious_rows: list[dict[str, Any]] = []
    class_distribution: Counter[str] = Counter()
    diagram_type_distribution: Counter[str] = Counter()

    for ann_path in ann_paths:
        row, suspicious = inspect_annotation(ann_path)
        diagram_rows.append(row)
        suspicious_rows.extend(suspicious)
        ann = _load_json(ann_path)
        diagram_type_distribution[str(row.get("diagram_type", ""))] += 1
        for obj in ann.get("objects", []) if isinstance(ann.get("objects", []), list) else []:
            if isinstance(obj, dict):
                class_distribution[str(obj.get("class_name", ""))] += 1

    total_diagrams = len(diagram_rows)
    total_objects = sum(int(r["object_count"]) for r in diagram_rows)
    total_connectors = sum(int(r["connector_count"]) for r in diagram_rows)
    suspicious_large = sum(int(r["suspicious_large_bbox_count"]) for r in diagram_rows)
    malformed_boxes = sum(int(r["malformed_bbox_count"]) for r in diagram_rows)
    suspicious_large_rate = suspicious_large / max(total_objects, 1)
    malformed_bbox_rate = malformed_boxes / max(total_objects, 1)
    recommendation = make_recommendation(suspicious_large_rate)

    top_problem_diagrams = sorted(
        diagram_rows,
        key=lambda r: (
            int(r["suspicious_score"]),
            int(r["very_large_bbox_count"]),
            int(r["malformed_bbox_count"]),
        ),
        reverse=True,
    )[:20]

    report = {
        "dataset_root": str(dataset_root),
        "total_diagrams": total_diagrams,
        "total_objects": total_objects,
        "total_connectors": total_connectors,
        "suspicious_large_bbox_count": suspicious_large,
        "suspicious_large_bbox_rate": round(suspicious_large_rate, 6),
        "malformed_bbox_count": malformed_boxes,
        "malformed_bbox_rate": round(malformed_bbox_rate, 6),
        "class_distribution": dict(class_distribution),
        "diagram_type_distribution": dict(diagram_type_distribution),
        "recommendation": recommendation,
        "top_problem_diagrams": top_problem_diagrams,
    }

    summary_fields = [
        "split",
        "scenario_id",
        "diagram_id",
        "diagram_type",
        "image_exists",
        "width",
        "height",
        "object_count",
        "connector_count",
        "invalid_class_count",
        "malformed_bbox_count",
        "out_of_bounds_bbox_count",
        "tiny_bbox_count",
        "suspicious_large_bbox_count",
        "very_large_bbox_count",
        "suspicious_wide_bbox_count",
        "suspicious_tall_bbox_count",
        "missing_object_id_count",
        "missing_label_text_count",
        "malformed_connector_count",
        "suspicious_long_connector_count",
        "possible_cross_diagram_connector_count",
        "max_bbox_area_ratio",
        "max_connector_length_ratio",
        "suspicious_score",
        "annotation_path",
        "image_path",
    ]
    suspicious_fields = summary_fields + ["item_type", "item_id", "issue", "bbox_area_ratio", "connector_length_ratio"]

    _write_json(out_dir / "annotation_quality_report.json", report)
    _write_csv(out_dir / "annotation_quality_summary.csv", diagram_rows, summary_fields)
    _write_csv(out_dir / "suspicious_annotations.csv", suspicious_rows, suspicious_fields)
    _write_csv(out_dir / "top_problem_diagrams.csv", top_problem_diagrams, summary_fields)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate InfraGraph V3 annotation quality without modifying source data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", default="./datasets/infragraph_v3")
    parser.add_argument("--out", default="./reports/v3_annotation_qa")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out)
    report = run_qa(dataset_root, out_dir)

    print("InfraGraph V3 Annotation QA")
    print(f"  dataset_root: {dataset_root}")
    print(f"  output_dir:    {out_dir}")
    print(f"  total diagrams:   {report['total_diagrams']}")
    print(f"  total objects:    {report['total_objects']}")
    print(f"  total connectors: {report['total_connectors']}")
    print(f"  suspicious large boxes: {report['suspicious_large_bbox_rate'] * 100:.2f}%")
    print(f"  malformed boxes:        {report['malformed_bbox_rate'] * 100:.2f}%")
    print(f"  recommendation: {report['recommendation']}")
    print()
    print("Top problem diagrams:")
    for row in report["top_problem_diagrams"][:20]:
        print(
            f"  {row['split']}/{row['scenario_id']}/{row['diagram_id']} "
            f"score={row['suspicious_score']} "
            f"large={row['suspicious_large_bbox_count']} "
            f"malformed={row['malformed_bbox_count']}"
        )
    print()
    print(f"Report: {out_dir / 'annotation_quality_report.json'}")
    print(f"Summary CSV: {out_dir / 'annotation_quality_summary.csv'}")
    print(f"Suspicious CSV: {out_dir / 'suspicious_annotations.csv'}")
    print(f"Top diagrams CSV: {out_dir / 'top_problem_diagrams.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
