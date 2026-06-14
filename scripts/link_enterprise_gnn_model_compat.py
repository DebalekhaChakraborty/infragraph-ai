#!/usr/bin/env python3
"""
link_enterprise_gnn_model_compat.py

Create compatibility symlinks (or copies on Windows) so that legacy paths
used by older scripts and the Streamlit UI can locate the trained GNN model.

Source (canonical):
    model_artifacts/enterprise_gnn_rca/enterprise_gnn_rca.pt

Targets created:
    outputs/enterprise_gnn_rca/enterprise_gnn_model.pt
    assets/preloaded/enterprise_gnn_rca/enterprise_gnn_model.pt
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCE = REPO_ROOT / "model_artifacts" / "enterprise_gnn_rca" / "enterprise_gnn_rca.pt"

TARGETS = [
    REPO_ROOT / "outputs"       / "enterprise_gnn_rca" / "enterprise_gnn_model.pt",
    REPO_ROOT / "assets" / "preloaded" / "enterprise_gnn_rca" / "enterprise_gnn_model.pt",
]


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
        return f"symlink → {src}"
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        return f"copy    ← {src}"


def main() -> int:
    if not SOURCE.exists():
        print(
            f"[ERROR] Canonical model not found: {SOURCE}\n"
            "Train first:\n"
            "  python scripts/build_enterprise_gnn_dataset.py\n"
            "  python scripts/train_enterprise_gnn_rca.py \\\n"
            "      --graphs      data/rca/enterprise_gnn/graphs.pt \\\n"
            "      --index       data/rca/enterprise_gnn/graph_index.json \\\n"
            "      --out-dir     model_artifacts/enterprise_gnn_rca \\\n"
            "      --report-dir  reports/enterprise_gnn_rca \\\n"
            "      --epochs      80"
        )
        return 1

    print(f"Source: {SOURCE}  ({SOURCE.stat().st_size:,} bytes)\n")
    for dst in TARGETS:
        result = _link_or_copy(SOURCE, dst)
        print(f"  {result}\n  → {dst}")

    print("\nDone. Compatibility paths are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
