"""
find_lora_adapter_artifacts.py — Search a vERL run directory for LoRA adapter files.

Usage:
    python training/verl_grpo/find_lora_adapter_artifacts.py \\
        --run-dir training/verl_grpo/runs/qwen3_4b_grpo_lora_amd

Prints a table of every adapter-related file found and suggests the
INFRAGRAPH_LORA_ADAPTER_PATH environment variable if a usable adapter
directory is detected.

Exit codes:
    0  — at least one adapter file found
    1  — no adapter files found (training checkpoint may need --save-freq)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ADAPTER_FILES = {
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
}

_CHECKPOINT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "adapter_config.json",
    "config.json",
    "tokenizer*.json",
    "special_tokens_map.json",
]

_VERL_CKPT_DIRS = [
    "actor",
    "actor_optimizer",
    "critic",
    "global_step_*",
    "step_*",
    "epoch_*",
    "checkpoint*",
    "latest",
]


def _find_files(run_dir: Path) -> list[Path]:
    """Return all files under run_dir that look like adapter or checkpoint artifacts."""
    found: list[Path] = []
    if not run_dir.exists():
        return found
    for root, dirs, files in os.walk(run_dir):
        root_path = Path(root)
        dirs.sort()
        for fname in sorted(files):
            fpath = root_path / fname
            if (
                fname in _ADAPTER_FILES
                or fname.endswith(".safetensors")
                or fname.endswith(".bin")
                or fname == "adapter_config.json"
                or fname == "config.json"
                or "tokenizer" in fname
            ):
                found.append(fpath)
    return found


def _find_adapter_dirs(run_dir: Path, files: list[Path]) -> list[Path]:
    """Return directories that contain both adapter_config.json and a weights file."""
    dirs_with_config: set[Path] = set()
    dirs_with_weights: set[Path] = set()
    for f in files:
        if f.name == "adapter_config.json":
            dirs_with_config.add(f.parent)
        if f.name in ("adapter_model.safetensors", "adapter_model.bin"):
            dirs_with_weights.add(f.parent)
    return sorted(dirs_with_config & dirs_with_weights)


def _sizeof(path: Path) -> str:
    try:
        size = path.stat().st_size
        if size >= 1_073_741_824:
            return f"{size / 1_073_741_824:.1f} GB"
        if size >= 1_048_576:
            return f"{size / 1_048_576:.1f} MB"
        if size >= 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size} B"
    except OSError:
        return "?"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a vERL run directory for LoRA adapter artifacts."
    )
    parser.add_argument(
        "--run-dir",
        default="training/verl_grpo/runs/qwen3_4b_grpo_lora_amd",
        help="Path to the vERL run/output directory (default: %(default)s)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()

    print(f"Scanning: {run_dir}")
    print()

    if not run_dir.exists():
        print(f"[ERROR] Directory does not exist: {run_dir}")
        print("  Run training with INFRAGRAPH_RUN_REAL_VERL=1 first, or check --run-dir.")
        sys.exit(1)

    files = _find_files(run_dir)

    if not files:
        print("[RESULT] No adapter or checkpoint files found.")
        print()
        print("Possible reasons:")
        print("  1. Training has not run yet (INFRAGRAPH_RUN_REAL_VERL=1 was not set).")
        print("  2. save_freq was not set — vERL may not have written checkpoints.")
        print("     Re-run with: SAVE_FREQ=8 TEST_FREQ=8 INFRAGRAPH_RUN_REAL_VERL=1 \\")
        print(f"       RUN_DIR={args.run_dir} \\")
        print("       bash training/verl_grpo/train_qwen3_grpo.sh")
        print("  3. vERL's FSDP strategy writes sharded actor weights that require")
        print("     export_lora_adapter.py to convert to PEFT format.")
        sys.exit(1)

    print(f"Found {len(files)} artifact file(s):")
    print()
    col_w = max(len(str(f.relative_to(run_dir))) for f in files) + 2
    print(f"  {'Path':<{col_w}}  {'Size':>8}")
    print(f"  {'-' * col_w}  {'--------':>8}")
    for f in files:
        rel = str(f.relative_to(run_dir))
        print(f"  {rel:<{col_w}}  {_sizeof(f):>8}")

    print()
    adapter_dirs = _find_adapter_dirs(run_dir, files)

    if adapter_dirs:
        print("PEFT-compatible adapter director(ies) found:")
        for d in adapter_dirs:
            print(f"  {d}")
        print()
        best = adapter_dirs[-1]
        rel_best = best.relative_to(run_dir.parent.parent.parent) if run_dir.parent.parent.parent.exists() else best
        print("Suggested environment variable:")
        print(f"  export INFRAGRAPH_LORA_ADAPTER_PATH={rel_best}")
        print()
        print("Load in the app:")
        print("  streamlit run app/streamlit_app.py")
        sys.exit(0)
    else:
        print("[NOTE] No directory contains both adapter_config.json + adapter_model.*")
        print("  The checkpoint may be in vERL's FSDP sharded format.")
        print("  Run export_lora_adapter.py to convert:")
        print(f"    python training/verl_grpo/export_lora_adapter.py --run-dir {args.run_dir}")
        sys.exit(0)


if __name__ == "__main__":
    main()
