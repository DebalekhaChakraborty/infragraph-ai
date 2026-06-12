#!/usr/bin/env python3
"""
migrate_outputs_structure.py — Move legacy outputs/ subfolders into the new
canonical directory structure.

New layout
----------
runtime_state/   live_ingestion, live_absorption, incident_runs,
                 vector_memory, global_graph_memory
demo_assets/     demo_hero, enterprise_gnn_rca, gnn_rca, mlp_rca,
                 qwen_explanation, annotation_overlays,
                 annotation_overlays_review, onboarded_diagrams
model_artifacts/ rfdetr_v3, rfdetr_v3_smoke
reports/         val_eval, v3_annotation_qa

Usage
-----
# Preview what would be moved (no writes):
python scripts/migrate_outputs_structure.py --dry-run

# Apply the migration:
python scripts/migrate_outputs_structure.py --apply

# Apply and overwrite existing destination files:
python scripts/migrate_outputs_structure.py --apply --overwrite
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Migration map: (source_rel, dest_rel) ─────────────────────────────────────
# Each tuple is relative to REPO_ROOT.
MIGRATION_MAP: list[tuple[str, str]] = [
    # runtime_state/
    ("outputs/live_ingestion",              "runtime_state/live_ingestion"),
    ("outputs/live_absorption",             "runtime_state/live_absorption"),
    ("outputs/incident_runs",               "runtime_state/incident_runs"),
    ("outputs/vector_memory",               "runtime_state/vector_memory"),
    ("outputs/global_graph_memory",         "runtime_state/global_graph_memory"),
    # demo_assets/
    ("outputs/demo_hero",                   "demo_assets/demo_hero"),
    ("outputs/enterprise_gnn_rca",          "demo_assets/enterprise_gnn_rca"),
    ("outputs/gnn_rca",                     "demo_assets/gnn_rca"),
    ("outputs/mlp_rca",                     "demo_assets/mlp_rca"),
    ("outputs/qwen_explanation",            "demo_assets/qwen_explanation"),
    ("outputs/annotation_overlays",         "demo_assets/annotation_overlays"),
    ("outputs/annotation_overlays_review",  "demo_assets/annotation_overlays_review"),
    ("outputs/onboarded_diagrams",          "demo_assets/onboarded_diagrams"),
    # model_artifacts/
    ("outputs/rfdetr_v3",                   "model_artifacts/rfdetr_v3"),
    ("outputs/rfdetr_v3_smoke",             "model_artifacts/rfdetr_v3_smoke"),
    # reports/
    ("outputs/val_eval",                    "reports/val_eval"),
    ("outputs/v3_annotation_qa",            "reports/v3_annotation_qa"),
]

# Hydra/date-stamped run dirs under outputs/2026-*/  -> reports/hydra_runs/
_HYDRA_SRC_GLOB = "outputs/2026-*"
_HYDRA_DST_BASE = "reports/hydra_runs"


def _move_tree(
    src: Path,
    dst: Path,
    overwrite: bool,
    dry_run: bool,
) -> int:
    """
    Recursively copy src into dst, then remove src.
    Returns count of files moved.
    """
    moved = 0
    for src_file in sorted(src.rglob("*")):
        if src_file.is_dir():
            continue
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        if dst_file.exists() and not overwrite:
            print(f"  SKIP  (exists) {dst_file.relative_to(REPO_ROOT)}")
            continue
        if not dry_run:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
        action = "DRY-RUN" if dry_run else "MOVED"
        print(f"  {action}  {src_file.relative_to(REPO_ROOT)}  ->  {dst_file.relative_to(REPO_ROOT)}")
        moved += 1
    if not dry_run and moved > 0:
        shutil.rmtree(src, ignore_errors=True)
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate legacy outputs/ structure to new canonical layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="Print what would be moved without making any changes.")
    mode.add_argument("--apply",   action="store_true",
                      help="Actually move the files.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing destination files (default: skip).")
    args = parser.parse_args()

    dry_run = args.dry_run
    overwrite = args.overwrite

    print(f"Repository root: {REPO_ROOT}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'APPLY'}")
    if overwrite:
        print("Overwrite: YES")
    print()

    total_moved = 0
    total_skipped = 0

    # ── Static migration map ───────────────────────────────────────────────────
    for src_rel, dst_rel in MIGRATION_MAP:
        src = REPO_ROOT / src_rel
        dst = REPO_ROOT / dst_rel
        if not src.exists():
            print(f"  SKIP  (not found) {src_rel}")
            total_skipped += 1
            continue
        print(f"Moving {src_rel}  ->  {dst_rel}")
        moved = _move_tree(src, dst, overwrite=overwrite, dry_run=dry_run)
        total_moved += moved
        if moved == 0:
            print(f"  (nothing to move in {src_rel})")

    # ── Hydra / dated run dirs ─────────────────────────────────────────────────
    for hydra_src in sorted(REPO_ROOT.glob(_HYDRA_SRC_GLOB)):
        if not hydra_src.is_dir():
            continue
        dst_rel = f"{_HYDRA_DST_BASE}/{hydra_src.name}"
        dst = REPO_ROOT / dst_rel
        print(f"Moving {hydra_src.relative_to(REPO_ROOT)}  ->  {dst_rel}")
        moved = _move_tree(hydra_src, dst, overwrite=overwrite, dry_run=dry_run)
        total_moved += moved

    # ── Keep outputs/ with .gitkeep if it is now empty ────────────────────────
    outputs_dir = REPO_ROOT / "outputs"
    if outputs_dir.exists() and not dry_run:
        remaining = [p for p in outputs_dir.iterdir() if p.name != ".gitkeep"]
        if not remaining:
            gitkeep = outputs_dir / ".gitkeep"
            gitkeep.touch(exist_ok=True)
            print(f"\noutputs/ is empty — kept outputs/.gitkeep")

    print()
    print(f"{'Would move' if dry_run else 'Moved'}: {total_moved} file(s)")
    print(f"Skipped (source absent or dest exists): {total_skipped}")

    if dry_run:
        print()
        print("Run with --apply to execute the migration.")

    print()
    print("To execute:")
    print("  python scripts/migrate_outputs_structure.py --apply")


if __name__ == "__main__":
    main()
