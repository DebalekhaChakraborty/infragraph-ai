"""
write_training_summary.py — Generate a post-run training summary.

Creates:
    training/verl_grpo/runs/qwen3_4b_grpo_lora_amd/training_summary.md

Run after a real vERL training pass, or call with --dry-run to document
a scaffold/dry-run attempt honestly.

Usage
-----
python training/verl_grpo/write_training_summary.py [--run-dir <path>] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR    = Path(__file__).resolve().parent
REPO_ROOT     = SCRIPT_DIR.parent.parent


def _display_path(p: Path) -> str:
    """Return repo-relative path when possible, otherwise absolute path.

    Training runs may live outside the repo, for example under /tmp,
    to avoid filling /workspace/shared. pathlib.relative_to() raises
    ValueError for those paths, so summary generation must handle both.
    """
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p.resolve())
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "qwen3_4b_grpo_lora_amd"
DATA_DIR        = SCRIPT_DIR / "data"


# ── Hardware detection ────────────────────────────────────────────────────────

def _torch_info() -> dict[str, object]:
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = ""
        try:
            if cuda_avail:
                device_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return {
            "torch_version":    torch.__version__,
            "torch_hip_version": getattr(torch.version, "hip", None),
            "cuda_available":   cuda_avail,
            "device_name":      device_name,
        }
    except ImportError:
        return {
            "torch_version":    "not installed",
            "torch_hip_version": None,
            "cuda_available":   False,
            "device_name":      "",
        }


# ── Artifact detection ────────────────────────────────────────────────────────

def _find_artifacts(run_dir: Path) -> dict[str, list[str]]:
    """Walk run_dir for checkpoint and adapter files."""
    ckpt_patterns = ["*.pt", "*.safetensors", "*.bin", "adapter_model.*", "pytorch_model.*"]
    config_patterns = ["adapter_config.json", "config.json", "training_args.json"]

    checkpoints: list[str] = []
    configs: list[str] = []

    if run_dir.exists():
        for pat in ckpt_patterns:
            checkpoints.extend(_display_path(p) for p in run_dir.rglob(pat))
        for pat in config_patterns:
            configs.extend(_display_path(p) for p in run_dir.rglob(pat))

    return {"checkpoints": sorted(checkpoints), "configs": sorted(configs)}


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


# ── Markdown generation ───────────────────────────────────────────────────────

def _md_table(rows: list[tuple[str, str]]) -> str:
    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(r[1]) for r in rows)
    lines = [
        f"| {'Field':<{w1}} | {'Value':<{w2}} |",
        f"|{'-' * (w1 + 2)}|{'-' * (w2 + 2)}|",
    ]
    for k, v in rows:
        lines.append(f"| {k:<{w1}} | {v:<{w2}} |")
    return "\n".join(lines)


def _md_list(items: list[str], empty_msg: str = "_none found_") -> str:
    if not items:
        return empty_msg
    return "\n".join(f"- `{i}`" for i in items)


def write_summary(run_dir: Path, dry_run: bool) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)

    hw      = _torch_info()
    arts    = _find_artifacts(run_dir)
    n_train = _count_jsonl(DATA_DIR / "rca_remediation_rl_train.jsonl")
    n_eval  = _count_jsonl(DATA_DIR / "rca_remediation_rl_eval.jsonl")

    parq_train = DATA_DIR / "verl_train.parquet"
    parq_eval  = DATA_DIR / "verl_eval.parquet"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run_status = "Scaffold / dry-run only" if dry_run else "Real vERL training run completed"

    training_claim = (
        "> **Honest status:** This summary documents a scaffold and reward-evaluated alignment "
        "dataset only.  No LoRA adapter checkpoint was produced in this repository.  "
        "The claim _'LoRA fine-tuned Qwen3-4B with GRPO using vERL on AMD GPUs'_ requires a "
        "real training run that writes adapter files to this `runs/` directory."
    ) if dry_run else (
        "> **Honest status:** A real training run completed.  "
        "Verify adapter/checkpoint files below before making fine-tuning claims."
        + ("\n>\n> ⚠️  No adapter files detected — verify run completed successfully."
           if not arts["checkpoints"] else "")
    )

    # Reward function summary
    reward_rows = [
        ("json_format",                   "16%  — valid JSON with all required keys"),
        ("root_cause_match",              "18%  — probable_root_cause names correct node"),
        ("grounded_node",                 "14%  — impacted nodes cited in response"),
        ("no_hallucinated_device",        "14%  — no device IDs outside valid set"),
        ("validation_before_remediation", "12%  — validation steps precede remediation"),
        ("rollback_safety",               "12%  — rollback notes + do_not_execute safeguards"),
        ("enterprise_escalation",         " 8%  — escalation for cross-diagram incidents"),
        ("itsm_ticket_summary",            " 6%  — structured ITSM dict present"),
    ]

    md = f"""# InfraGraph AI — Qwen3-4B LoRA + GRPO/vERL Training Summary

Generated: {ts}

---

## Run Status

**{run_status}**

{training_claim}

---

## Configuration

{_md_table([
    ("Base model",           os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")),
    ("Method",               "LoRA (rank=16, alpha=32, target=all-linear)"),
    ("Algorithm",            "GRPO via verl.trainer.main_ppo (algorithm.adv_estimator=grpo)"),
    ("Framework",            "vERL (https://github.com/volcengine/verl)"),
    ("Rollout backend",      "vLLM"),
    ("Actor strategy",       "FSDP"),
    ("Run directory",        _display_path(run_dir)),
])}

---

## Hardware

{_md_table([
    ("PyTorch version",  str(hw["torch_version"])),
    ("CUDA available",   str(hw["cuda_available"])),
    ("HIP version",      str(hw["torch_hip_version"] or "—")),
    ("Device name",      str(hw["device_name"] or "—")),
])}

---

## Dataset

{_md_table([
    ("Train JSONL records",  str(n_train)),
    ("Eval JSONL records",   str(n_eval)),
    ("Train parquet exists", "yes" if parq_train.exists() else "no"),
    ("Eval parquet exists",  "yes" if parq_eval.exists() else "no"),
    ("Ability tag",          "graph_grounded_remediation"),
    ("Data source",          "infragraph_rca_remediation"),
])}

---

## Reward Functions

| Component | Weight | Description |
|-----------|--------|-------------|
""" + "\n".join(
        f"| `{name}` | {desc} |"
        for name, desc in reward_rows
    ) + f"""

Reward entry point: `training/verl_grpo/verl_reward.py::compute_score`

---

## Adapter / Checkpoint Artifacts

{_md_list(arts["checkpoints"], "_No checkpoint files found in run directory._")}

### Config files

{_md_list(arts["configs"], "_No config files found._")}

---

## Honest Claims

Only make the following claims after the corresponding evidence exists:

| Claim | Requires |
|-------|----------|
| "Reward-evaluated alignment dataset built" | `verl_train.parquet` + `verl_eval.parquet` exist |
| "GRPO training scaffold implemented" | `train_qwen3_grpo.sh` runs without error |
| "LoRA fine-tuned Qwen3-4B with GRPO/vERL" | Adapter checkpoint files in `runs/` |
| "Tested on AMD GPU (ROCm)" | `torch_hip_version` non-null AND adapter files exist |
"""

    out_path = run_dir / "training_summary.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Training summary written: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write post-run training summary.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR),
                        help="Run directory to scan for artifacts and write summary into")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mark summary as scaffold/dry-run (no real training completed)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out = write_summary(run_dir, dry_run=args.dry_run)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
