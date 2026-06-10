#!/usr/bin/env python3
"""
prepare_rfdetr_dataset.py

Reads V3 scenario annotations and builds RF-DETR COCO-format dataset with
full traceability back to local graphs and enterprise graphs.

Usage:
    python scripts/prepare_rfdetr_dataset.py \
        --dataset-root ./datasets/diagram_v3_enterprise \
        --out ./datasets/diagram_v3_enterprise/rfdetr
"""

import argparse
import json
import shutil
from pathlib import Path

COCO_CATS = [
    {"id": 1, "name": "router",        "supercategory": "network_device"},
    {"id": 2, "name": "switch",        "supercategory": "network_device"},
    {"id": 3, "name": "firewall",      "supercategory": "network_device"},
    {"id": 4, "name": "server",        "supercategory": "network_device"},
    {"id": 5, "name": "database",      "supercategory": "network_device"},
    {"id": 6, "name": "load_balancer", "supercategory": "network_device"},
    {"id": 7, "name": "cloud_or_wan",  "supercategory": "network_device"},
    {"id": 8, "name": "service",       "supercategory": "network_device"},
]
CAT_ID = {c["name"]: c["id"] for c in COCO_CATS}


def _coco_info():
    return {
        "description": "InfraGraph AI V3 Diagram Intelligence Dataset",
        "url": "",
        "version": "3.0",
        "year": 2026,
        "contributor": "InfraGraph AI",
        "date_created": "2026-06-10",
    }


def build_coco_split(dataset_root, split, out_images_dir, image_id_start, ann_id_start):
    """
    Process one split. Returns (coco_dict, diagram_metadata_list,
    next_image_id, next_ann_id).
    """
    scenarios_dir = dataset_root / "scenarios" / split
    if not scenarios_dir.exists():
        empty = {"info": _coco_info(), "licenses": [], "images": [],
                 "annotations": [], "categories": COCO_CATS}
        return empty, [], image_id_start, ann_id_start

    image_id = image_id_start
    ann_id = ann_id_start
    images_list = []
    annotations_list = []
    diagram_metadata = []

    for scenario_dir in sorted(scenarios_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        scenario_id = scenario_dir.name
        ann_dir = scenario_dir / "annotations"
        if not ann_dir.exists():
            continue

        enterprise_graph_path = scenario_dir / "enterprise_graph.json"
        alerts_path           = scenario_dir / "alerts.json"

        for ann_file in sorted(ann_dir.glob("*.json")):
            with open(ann_file) as f:
                ann = json.load(f)

            diagram_id   = ann.get("diagram_id", ann_file.stem)
            diagram_type = ann.get("diagram_type", "")
            img_w        = ann.get("width", 1280)
            img_h        = ann.get("height", 960)

            src_img = scenario_dir / "diagrams" / f"{diagram_id}.png"
            if not src_img.exists():
                continue

            dst_name = f"{scenario_id}__{diagram_id}.png"
            dst_img  = out_images_dir / dst_name
            shutil.copy2(src_img, dst_img)

            images_list.append({
                "id":        image_id,
                "file_name": dst_name,
                "width":     img_w,
                "height":    img_h,
                "scenario_id": scenario_id,
                "diagram_id":  diagram_id,
                "diagram_type": diagram_type,
            })

            local_graph_path = scenario_dir / "local_graphs" / f"{diagram_id}.json"

            diagram_metadata.append({
                "image_id":              image_id,
                "file_name":             dst_name,
                "scenario_id":           scenario_id,
                "diagram_id":            diagram_id,
                "diagram_type":          diagram_type,
                "source_annotation_path": str(ann_file),
                "local_graph_path":       str(local_graph_path) if local_graph_path.exists() else "",
                "enterprise_graph_path":  str(enterprise_graph_path) if enterprise_graph_path.exists() else "",
                "alerts_path":            str(alerts_path) if alerts_path.exists() else "",
                "split":                  split,
                "width":                  img_w,
                "height":                 img_h,
                "num_objects":            len(ann.get("objects", [])),
            })

            for obj in ann.get("objects", []):
                cat_id = CAT_ID.get(obj.get("class_name", ""))
                if cat_id is None:
                    continue
                x1, y1, x2, y2 = obj["bbox"]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)
                bw = x2 - x1; bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue
                annotations_list.append({
                    "id":          ann_id,
                    "image_id":    image_id,
                    "category_id": cat_id,
                    "bbox":        [float(x1), float(y1), float(bw), float(bh)],
                    "area":        float(bw * bh),
                    "iscrowd":     0,
                    "is_shared_entity": obj.get("is_shared_entity", False),
                    "canonical_id":     obj.get("canonical_id", obj.get("object_id", "")),
                })
                ann_id += 1

            image_id += 1

    coco = {
        "info":        _coco_info(),
        "licenses":    [],
        "images":      images_list,
        "annotations": annotations_list,
        "categories":  COCO_CATS,
    }
    return coco, diagram_metadata, image_id, ann_id


def main():
    p = argparse.ArgumentParser(description="Prepare RF-DETR COCO dataset from V3 scenarios")
    p.add_argument("--dataset-root", type=str, default="./datasets/diagram_v3_enterprise")
    p.add_argument("--out",          type=str, default="./datasets/diagram_v3_enterprise/rfdetr")
    args = p.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_root     = Path(args.out).resolve()

    if not dataset_root.exists():
        print(f"[FAIL] dataset-root not found: {dataset_root}")
        print("       Run generate_diagram_v3_enterprise_dataset.py first.")
        return 1

    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {dataset_root}")
    print(f"Writing to:   {out_root}")

    image_id = 0
    ann_id   = 0
    all_meta = []

    for split in ["train", "val", "test"]:
        print(f"  Processing split: {split}")
        coco, meta, image_id, ann_id = build_coco_split(
            dataset_root, split,
            out_root / "images" / split,
            image_id, ann_id,
        )
        coco_path = out_root / "annotations" / f"instances_{split}.json"
        with open(coco_path, "w") as f:
            json.dump(coco, f, indent=2)
        all_meta.extend(meta)
        print(f"    images={len(coco['images'])}  annotations={len(coco['annotations'])}")

    meta_path = out_root / "diagram_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    total_img = sum(1 for m in all_meta)
    total_ann = ann_id
    print(f"Done. total_images={total_img}  total_annotations={total_ann}")
    print(f"COCO annotations: {out_root / 'annotations'}")
    print(f"Diagram metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
