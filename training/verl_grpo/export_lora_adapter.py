"""
export_lora_adapter.py — Convert a vERL GRPO actor checkpoint to a PEFT LoRA adapter.

vERL's FSDP actor strategy saves per-rank sharded weight files rather than
standard PEFT adapter_model.safetensors.  This script attempts to:

  1. Locate the latest actor checkpoint under --run-dir.
  2. Try to load it as a PEFT PeftModel directly (works if vERL already merged
     adapter weights into the checkpoint).
  3. If that fails, attempt to merge sharded FSDP state dicts with
     torch.distributed.fsdp utilities and extract only LoRA delta weights.
  4. Save adapter_model.safetensors + adapter_config.json to --output-dir.

Fails gracefully with a clear message if the checkpoint format is not
recognised — it does NOT fabricate adapter files.

Usage:
    python training/verl_grpo/export_lora_adapter.py \\
        --run-dir training/verl_grpo/runs/qwen3_4b_grpo_lora_amd \\
        --base-model Qwen/Qwen3-4B \\
        --output-dir training/verl_grpo/exported_adapter

Exit codes:
    0  — adapter exported successfully
    1  — export failed or checkpoint format not supported
    2  — prerequisites not installed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── Prerequisite check ────────────────────────────────────────────────────────

def _check_prerequisites() -> None:
    missing = []
    for pkg in ("torch", "transformers", "peft"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"        pip install {' '.join(missing)}")
        sys.exit(2)


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def _find_latest_actor_dir(run_dir: Path) -> Path | None:
    """Return the most recent actor checkpoint directory under run_dir."""
    candidates: list[tuple[int, Path]] = []
    for d in run_dir.rglob("actor"):
        if d.is_dir():
            # Parent name is typically global_step_N or step_N
            parent_name = d.parent.name
            step = 0
            for part in parent_name.replace("global_step_", "").replace("step_", "").split("_"):
                try:
                    step = int(part)
                    break
                except ValueError:
                    pass
            candidates.append((step, d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _list_weight_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.safetensors")) + sorted(directory.glob("*.bin"))
    return files


# ── Export paths ──────────────────────────────────────────────────────────────

def _try_peft_load(actor_dir: Path, base_model: str, output_dir: Path) -> bool:
    """Attempt to load actor_dir directly as a PeftModel and save adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    weight_files = _list_weight_files(actor_dir)
    if not weight_files:
        return False

    # Check if actor_config.json or adapter_config.json exists
    has_adapter_config = (actor_dir / "adapter_config.json").exists()
    if not has_adapter_config:
        return False

    print(f"  adapter_config.json found — attempting PeftModel load from {actor_dir} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(actor_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"  Saved PEFT adapter to {output_dir}")
        return True
    except Exception as exc:
        print(f"  PeftModel.from_pretrained failed: {exc}")
        return False


def _try_fsdp_merge(actor_dir: Path, base_model: str, output_dir: Path) -> bool:
    """
    Attempt to merge FSDP sharded state dicts and extract LoRA weights.

    vERL's FSDP actor saves one file per GPU rank.  This path:
      - Loads all rank shards with torch.load
      - Looks for keys matching typical LoRA patterns (lora_A, lora_B)
      - Saves only the LoRA delta keys to adapter_model.safetensors

    This is a best-effort heuristic — it may fail on unusual shard layouts.
    """
    import torch
    try:
        from safetensors.torch import save_file as safetensors_save
    except ImportError:
        safetensors_save = None

    weight_files = _list_weight_files(actor_dir)
    if not weight_files:
        return False

    print(f"  Attempting FSDP shard merge from {len(weight_files)} file(s) ...")
    merged: dict[str, "torch.Tensor"] = {}
    for wf in weight_files:
        try:
            if wf.suffix == ".safetensors":
                from safetensors.torch import load_file
                shard = load_file(str(wf), device="cpu")
            else:
                shard = torch.load(str(wf), map_location="cpu", weights_only=True)
            if isinstance(shard, dict):
                # state_dict may be nested under "module" or "model"
                if "module" in shard and isinstance(shard["module"], dict):
                    shard = shard["module"]
                elif "model" in shard and isinstance(shard["model"], dict):
                    shard = shard["model"]
                merged.update(shard)
        except Exception as exc:
            print(f"    [warn] could not load {wf.name}: {exc}")

    if not merged:
        print("  No tensors loaded from shards.")
        return False

    lora_keys = {k: v for k, v in merged.items() if "lora_A" in k or "lora_B" in k}
    if not lora_keys:
        print(f"  Merged {len(merged)} keys but none match LoRA patterns (lora_A/lora_B).")
        print("  vERL FSDP may have saved full model weights rather than LoRA deltas.")
        print("  Manual inspection required — see 'Limitations' in export_lora_adapter.py.")
        return False

    print(f"  Found {len(lora_keys)} LoRA weight tensors.")

    # Write adapter_config.json (minimal PEFT v2 format)
    import json
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_config = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 16,
        "target_modules": "all-linear",
        "task_type": "CAUSAL_LM",
    }
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    if safetensors_save is not None:
        safetensors_save(lora_keys, str(output_dir / "adapter_model.safetensors"))
        print(f"  Saved adapter_model.safetensors ({len(lora_keys)} tensors) to {output_dir}")
    else:
        torch.save(lora_keys, str(output_dir / "adapter_model.bin"))
        print(f"  Saved adapter_model.bin ({len(lora_keys)} tensors) to {output_dir}")
        print("  (Install safetensors for the preferred format: pip install safetensors)")

    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a vERL GRPO actor checkpoint to PEFT LoRA adapter format."
    )
    parser.add_argument(
        "--run-dir",
        default="training/verl_grpo/runs/qwen3_4b_grpo_lora_amd",
        help="vERL run/output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="HuggingFace base model ID (default: reads MODEL_ID env var or Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the exported adapter (default: <run-dir>/exported_adapter)",
    )
    args = parser.parse_args()

    _check_prerequisites()

    import os
    base_model = args.base_model or os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "exported_adapter"

    print(f"Run directory : {run_dir}")
    print(f"Base model    : {base_model}")
    print(f"Output dir    : {output_dir}")
    print()

    if not run_dir.exists():
        print(f"[ERROR] Run directory does not exist: {run_dir}")
        print("  Ensure training completed with INFRAGRAPH_RUN_REAL_VERL=1.")
        sys.exit(1)

    actor_dir = _find_latest_actor_dir(run_dir)
    if actor_dir is None:
        print("[ERROR] No 'actor' checkpoint directory found under the run directory.")
        print("  Expected layout: <run-dir>/global_step_N/actor/")
        print()
        print("  Possible causes:")
        print("  1. save_freq was not set — re-run with SAVE_FREQ=8.")
        print("  2. Training did not complete enough steps to trigger a save.")
        print("  3. vERL used a different checkpoint layout for this version.")
        print()
        print("  Directory contents:")
        for p in sorted(run_dir.rglob("*"))[:40]:
            print(f"    {p.relative_to(run_dir)}")
        sys.exit(1)

    print(f"Actor checkpoint: {actor_dir}")
    print()

    # Try PeftModel load first (cleanest path)
    if _try_peft_load(actor_dir, base_model, output_dir):
        print()
        print("[SUCCESS] Adapter exported via PeftModel.from_pretrained.")
        _print_next_steps(output_dir)
        sys.exit(0)

    # Fall back to FSDP shard merge
    if _try_fsdp_merge(actor_dir, base_model, output_dir):
        print()
        print("[SUCCESS] Adapter exported via FSDP shard merge (heuristic).")
        print("  Verify the adapter loads cleanly before use:")
        print(f"    python -c \"from peft import PeftModel; print('OK')\"")
        _print_next_steps(output_dir)
        sys.exit(0)

    print()
    print("[FAILED] Could not export adapter from the checkpoint.")
    print()
    print("Limitations of this script:")
    print("  - Requires LoRA delta weights to be present as separate tensors.")
    print("  - FSDP may have merged LoRA weights into base model weights.")
    print("    In that case, the full model must be loaded and LoRA re-extracted.")
    print("  - Consult vERL documentation for checkpoint format details:")
    print("    https://github.com/volcengine/verl")
    sys.exit(1)


def _print_next_steps(output_dir: Path) -> None:
    print()
    print("Next steps:")
    print(f"  export INFRAGRAPH_LORA_ADAPTER_PATH={output_dir}")
    print("  streamlit run app/streamlit_app.py")
    print()
    print("  Or verify loading:")
    print("  python -c \"")
    print(f"    from peft import PeftModel")
    print(f"    from transformers import AutoModelForCausalLM")
    print(f"    import torch")
    print(f"    base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B', device_map='cpu')")
    print(f"    model = PeftModel.from_pretrained(base, '{output_dir}')")
    print(f"    print('Adapter loaded successfully')\"")


if __name__ == "__main__":
    main()
