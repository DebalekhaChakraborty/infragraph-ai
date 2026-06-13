#!/usr/bin/env python3
"""
build_kb_index.py — Build the ChromaDB vector index for SOP/KB documents.

Reads KB documents from assets/kb/ (or --kb-root) and builds a persistent
ChromaDB index in runtime_state/kb_index/ (or --index-dir).

Writes a build summary to reports/kb_index/build_summary.json.

Usage:
  python scripts/build_kb_index.py
  python scripts/build_kb_index.py --reset
  python scripts/build_kb_index.py --kb-root assets/kb --index-dir runtime_state/kb_index
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from kb_retrieval.schema import DEFAULT_COLLECTION, DEFAULT_INDEX_DIR, DEFAULT_KB_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB vector index for SOP/KB/known-resolution documents."
    )
    parser.add_argument(
        "--kb-root", default=DEFAULT_KB_ROOT, metavar="DIR",
        help=f"KB document root directory (default: {DEFAULT_KB_ROOT}).",
    )
    parser.add_argument(
        "--index-dir", default=DEFAULT_INDEX_DIR, metavar="DIR",
        help=f"ChromaDB persistent store directory (default: {DEFAULT_INDEX_DIR}).",
    )
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION, metavar="NAME",
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete the existing collection before rebuilding.",
    )
    args = parser.parse_args()

    kb_root  = (REPO_ROOT / args.kb_root).resolve()
    idx_dir  = (REPO_ROOT / args.index_dir).resolve()

    print("====================================================")
    print(" InfraGraph AI — Build KB / SOP Index")
    print("====================================================")
    print(f"KB root      : {kb_root.relative_to(REPO_ROOT)}")
    print(f"Index dir    : {idx_dir.relative_to(REPO_ROOT)}")
    print(f"Collection   : {args.collection}")
    print(f"Reset        : {args.reset}")
    print()

    if not kb_root.exists():
        print(f"[ERROR] KB root directory does not exist: {kb_root}")
        print("        Create it and add SOP/runbook/known_resolution markdown files first.")
        sys.exit(1)

    try:
        from kb_retrieval.indexer import build_kb_index
    except ImportError as exc:
        print(f"[ERROR] Could not import kb_retrieval: {exc}")
        sys.exit(1)

    t0 = time.monotonic()
    try:
        summary = build_kb_index(
            kb_root=kb_root,
            index_dir=idx_dir,
            collection_name=args.collection,
            reset=args.reset,
        )
    except RuntimeError as exc:
        print(f"[ERROR] Index build failed: {exc}")
        sys.exit(1)
    elapsed = time.monotonic() - t0

    summary["elapsed_seconds"] = round(elapsed, 2)
    summary["kb_root"]         = str(kb_root.relative_to(REPO_ROOT))

    print(f"Documents loaded  : {summary['documents_loaded']}")
    print(f"Chunks indexed    : {summary['chunks_indexed']}")
    print(f"Embed model       : {summary.get('embed_model', '—')}")
    print(f"Elapsed           : {elapsed:.1f}s")
    print()

    # Write build summary
    report_dir = REPO_ROOT / "reports" / "kb_index"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "build_summary.json"
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Build summary     : {report_path.relative_to(REPO_ROOT)}")
    print()
    print("[PASS] KB index build complete.")


if __name__ == "__main__":
    main()
