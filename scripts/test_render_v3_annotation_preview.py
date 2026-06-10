"""
scripts/test_render_v3_annotation_preview.py

Iterate over V3 annotation files and attempt to render bbox previews.
Prints a summary of success / failure / skipped boxes.
Exits non-zero only when render_v3_annotation_preview itself raises an
unhandled exception (it should never do so after the robustness fix).

Usage:
    python scripts/test_render_v3_annotation_preview.py
    python scripts/test_render_v3_annotation_preview.py --limit 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from runtime_ingestion import render_v3_annotation_preview  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after this many annotation files (0 = all)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "debug_v3_annotation_previews"),
                        help="Directory to write rendered previews")
    args = parser.parse_args()

    ann_root = REPO_ROOT / "datasets" / "infragraph_v3" / "scenarios"
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_root.glob("*/*/annotations/*.json"))
    if args.limit > 0:
        ann_files = ann_files[: args.limit]

    total = len(ann_files)
    success = 0
    failed  = 0
    total_boxes_rendered  = 0
    total_boxes_skipped   = 0
    total_conns_rendered  = 0
    total_conns_skipped   = 0
    unexpected_exceptions = 0

    for ann_path in ann_files:
        # derive diagram image path from annotation path
        # annotations/<diagram_id>.json  ->  diagrams/<diagram_id>.png
        diagram_id   = ann_path.stem
        scenario_dir = ann_path.parent.parent
        img_path     = scenario_dir / "diagrams" / f"{diagram_id}.png"
        split        = scenario_dir.parent.name   # train / val / test
        scenario_id  = scenario_dir.name

        out_path = out_dir / split / scenario_id / f"{diagram_id}.png"

        try:
            meta = render_v3_annotation_preview(img_path, ann_path, out_path)
        except Exception as exc:
            print(f"  UNEXPECTED EXCEPTION  {split}/{scenario_id}/{diagram_id}: {exc}")
            unexpected_exceptions += 1
            failed += 1
            continue

        if meta.get("rendered"):
            success += 1
        else:
            failed += 1
            err = meta.get("error", "unknown")
            print(f"  RENDER FAILED  {split}/{scenario_id}/{diagram_id}: {err}")

        total_boxes_rendered  += meta.get("boxes_rendered",  0)
        total_boxes_skipped   += meta.get("boxes_skipped",   0)
        total_conns_rendered  += meta.get("connectors_rendered", 0)
        total_conns_skipped   += meta.get("connectors_skipped",  0)

        if meta.get("boxes_skipped", 0) > 0:
            print(f"  skipped {meta['boxes_skipped']} box(es)  "
                  f"{split}/{scenario_id}/{diagram_id}")

    print()
    print("=" * 60)
    print(f"Total annotation files : {total}")
    print(f"Rendered successfully  : {success}")
    print(f"Failed to render       : {failed}")
    print(f"Boxes rendered         : {total_boxes_rendered}")
    print(f"Boxes skipped          : {total_boxes_skipped}")
    print(f"Connectors rendered    : {total_conns_rendered}")
    print(f"Connectors skipped     : {total_conns_skipped}")
    print(f"Unexpected exceptions  : {unexpected_exceptions}")
    print("=" * 60)

    return 1 if unexpected_exceptions > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
