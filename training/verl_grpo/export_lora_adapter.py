"""
export_lora_adapter.py — Convert a vERL GRPO actor checkpoint to a PEFT LoRA adapter.

vERL's FSDP actor strategy saves per-rank sharded weight files (.pt) rather than
standard PEFT adapter_model.safetensors.  This script:

  1. Locates the latest actor checkpoint under --run-dir.
  2. Tries to load it as a PeftModel directly (works if vERL already wrote
     adapter_config.json alongside the weights).
  3. If that fails, loads all .pt/.safetensors/.bin rank shards, extracts
     keys matching LoRA patterns (lora_A / lora_B), normalises them to PEFT
     format, and saves adapter_model.safetensors + adapter_config.json.

Fails with a clear message if no LoRA tensors are found — does NOT fabricate
adapter files.

Usage:
    python training/verl_grpo/export_lora_adapter.py \\
        --run-dir  /tmp/infragraph_grpo_runs/qwen3_4b_grpo_lora_amd_saved \\
        --base-model Qwen/Qwen3-4B \\
        --output-dir /tmp/infragraph_qwen3_grpo_lora_adapter

Exit codes:
    0  — adapter exported successfully
    1  — export failed or checkpoint format not supported
    2  — prerequisites not installed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── Prerequisite check ────────────────────────────────────────────────────────

def _check_prerequisites() -> None:
    missing = []
    for pkg in ("torch", "safetensors"):
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
    """Return the actor directory from the highest global_step_N checkpoint."""
    candidates: list[tuple[int, Path]] = []
    for d in run_dir.rglob("actor"):
        if d.is_dir():
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
    """Return all weight files (.pt, .safetensors, .bin) in directory."""
    files: list[Path] = []
    files += sorted(directory.glob("*.pt"))
    files += sorted(directory.glob("*.safetensors"))
    files += sorted(directory.glob("*.bin"))
    return files


# ── Tensor loading helpers ────────────────────────────────────────────────────

def _load_pt_file(path: Path) -> dict:
    """Load a .pt checkpoint, trying mmap first then weights_only fallback."""
    import torch
    try:
        state = torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:
        # older torch versions do not support mmap=
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:
            state = torch.load(str(path), map_location="cpu")
    return state


def _unwrap_nested(state: object) -> dict:
    """Unwrap common nesting keys: module, model, state_dict."""
    if not isinstance(state, dict):
        return {}
    for key in ("module", "model", "state_dict"):
        if key in state and isinstance(state[key], dict):
            return state[key]
    return state


def _load_shard(path: Path) -> dict:
    """Load one weight shard regardless of format."""
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return _unwrap_nested(_load_pt_file(path))


# ── Key normalisation ─────────────────────────────────────────────────────────

def _normalise_lora_key(key: str) -> str:
    """Normalise vERL LoRA key to PEFT format.

    vERL stores:  ...lora_A.default.weight
    PEFT expects: ...lora_A.weight
    """
    return key.replace(".default.weight", ".weight")


def _is_lora_key(key: str) -> bool:
    return "lora_A" in key or "lora_B" in key


# ── Strategy 1: PeftModel direct load ────────────────────────────────────────

def _try_peft_load(actor_dir: Path, base_model: str, output_dir: Path) -> bool:
    """Attempt PeftModel.from_pretrained if adapter_config.json is present."""
    if not (actor_dir / "adapter_config.json").exists():
        return False
    if not _list_weight_files(actor_dir):
        return False

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        print(f"  [skip] peft/transformers not available: {exc}")
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


# ── Strategy 2: FSDP shard extraction ────────────────────────────────────────

def _try_fsdp_merge(actor_dir: Path, base_model: str, output_dir: Path) -> bool:
    """Extract LoRA tensors from vERL FSDP shards and build a PEFT adapter."""
    from safetensors.torch import save_file as safetensors_save

    weight_files = _list_weight_files(actor_dir)
    if not weight_files:
        print(f"  No weight files found in {actor_dir}")
        return False

    print(f"  Loading {len(weight_files)} shard file(s) from {actor_dir} ...")

    merged: dict = {}
    for wf in weight_files:
        try:
            shard = _load_shard(wf)
            if isinstance(shard, dict):
                merged.update(shard)
                print(f"    loaded {wf.name}: {len(shard)} keys")
            else:
                print(f"    [warn] {wf.name}: unexpected type {type(shard)}")
        except Exception as exc:
            print(f"    [warn] could not load {wf.name}: {exc}")

    if not merged:
        print("  No tensors loaded from shards.")
        return False

    print(f"  Total keys in merged state: {len(merged)}")

    # Extract and normalise LoRA tensors
    lora_tensors: dict = {}
    for key, tensor in merged.items():
        if _is_lora_key(key):
            normalised = _normalise_lora_key(key)
            lora_tensors[normalised] = tensor

    if not lora_tensors:
        print(f"  Merged {len(merged)} keys but none match LoRA patterns.")
        print("  Sample keys:")
        for k in list(merged.keys())[:10]:
            print(f"    {k}")
        return False

    print(f"  Extracted {len(lora_tensors)} LoRA tensors.")

    # Infer rank from first lora_A tensor shape
    rank = 16
    for key, tensor in lora_tensors.items():
        if "lora_A" in key and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            rank = int(tensor.shape[0])
            break

    # Write adapter_config.json
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_config = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "task_type": "CAUSAL_LM",
        "use_rslora": False,
    }
    with open(output_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)

    # Write adapter_model.safetensors
    safetensors_save(lora_tensors, str(output_dir / "adapter_model.safetensors"))

    # Write README.md
    readme = (
        f"# InfraGraph GRPO LoRA Adapter\n\n"
        f"Exported from a vERL/FSDP actor checkpoint.\n\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| Base model | `{base_model}` |\n"
        f"| Rank (r) | {rank} |\n"
        f"| Alpha | 32 |\n"
        f"| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |\n"
        f"| Tensor count | {len(lora_tensors)} |\n\n"
        f"## vLLM serving\n\n"
        f"```bash\n"
        f"vllm serve {base_model} \\\\\n"
        f"  --served-model-name Qwen3-4B \\\\\n"
        f"  --enable-lora \\\\\n"
        f"  --lora-modules infragraph={output_dir} \\\\\n"
        f"  --host 0.0.0.0 --port 8000 \\\\\n"
        f"  --gpu-memory-utilization 0.55 --max-model-len 2048\n"
        f"```\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"  Saved adapter_model.safetensors ({len(lora_tensors)} tensors) to {output_dir}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a vERL GRPO actor checkpoint to PEFT LoRA adapter format."
    )
    parser.add_argument(
        "--run-dir",
        default="/tmp/infragraph_grpo_runs/qwen3_4b_grpo_lora_amd_saved",
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
        help="Where to write the exported adapter (default: /tmp/infragraph_qwen3_grpo_lora_adapter)",
    )
    args = parser.parse_args()

    _check_prerequisites()

    import os
    base_model = args.base_model or os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")
    run_dir    = Path(args.run_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path("/tmp/infragraph_qwen3_grpo_lora_adapter")
    )

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
        print("  Directory contents:")
        for p in sorted(run_dir.rglob("*"))[:40]:
            print(f"    {p.relative_to(run_dir)}")
        sys.exit(1)

    print(f"Actor checkpoint : {actor_dir}")
    print()

    # Strategy 1: PeftModel direct load (only if adapter_config.json present)
    if _try_peft_load(actor_dir, base_model, output_dir):
        _print_success(output_dir)
        sys.exit(0)

    # Strategy 2: FSDP shard extraction
    if _try_fsdp_merge(actor_dir, base_model, output_dir):
        _print_success(output_dir)
        sys.exit(0)

    print()
    print("[FAILED] Could not export LoRA adapter from the checkpoint.")
    print()
    print("Possible causes:")
    print("  - The checkpoint contains full merged weights, not separate LoRA deltas.")
    print("  - The shard files could not be read (corrupted or unexpected format).")
    print("  - vERL used a different checkpoint layout for this version.")
    print()
    print("Manual inspection:")
    print(f"  python -c \"import torch; sd = torch.load('{actor_dir}/model_world_size_1_rank_0.pt', map_location='cpu'); print(list(sd.keys())[:20])\"")
    sys.exit(1)


def _print_success(output_dir: Path) -> None:
    from safetensors.torch import load_file
    weights = load_file(str(output_dir / "adapter_model.safetensors"))
    tensor_count = len(weights)

    print()
    print("[SUCCESS] Exported PEFT LoRA adapter")
    print(f"  adapter_config.json       : {output_dir / 'adapter_config.json'}")
    print(f"  adapter_model.safetensors : {output_dir / 'adapter_model.safetensors'}")
    print(f"  tensor_count              : {tensor_count}")
    print()
    print("Next steps:")
    print(f"  export INFRAGRAPH_LORA_ADAPTER_PATH={output_dir}")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
