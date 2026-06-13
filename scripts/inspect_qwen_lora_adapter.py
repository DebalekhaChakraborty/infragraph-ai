#!/usr/bin/env python3
"""
inspect_qwen_lora_adapter.py

Inspect a saved Qwen LoRA adapter produced by train_qwen_sop_lora.py.

Checks:
  - Adapter directory exists
  - adapter_config.json present and valid
  - Adapter weights present (adapter_model.safetensors or adapter_model.bin)
  - Tokenizer files present
  - training_summary.json present and readable
  - Prints adapter size and LoRA config summary

Usage:
  python scripts/inspect_qwen_lora_adapter.py \\
      --adapter-dir model_artifacts/qwen_lora/infragraph_sop_grounded_smoke

  python scripts/inspect_qwen_lora_adapter.py \\
      --adapter-dir model_artifacts/qwen_lora/infragraph_sop_grounded
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _check(label: str, ok: bool, msg: str = "") -> bool:
    status = "[PASS]" if ok else "[FAIL]"
    print(f"  {status} {label}" + (f" — {msg}" if msg else ""))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a saved Qwen LoRA adapter."
    )
    parser.add_argument(
        "--adapter-dir", required=True, metavar="DIR",
        help="Path to the adapter directory (absolute or relative to repo root).",
    )
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.is_absolute():
        adapter_dir = (REPO_ROOT / adapter_dir).resolve()

    print("=" * 60)
    print(" InfraGraph AI -- Qwen LoRA Adapter Inspector")
    print("=" * 60)
    print(f"  Adapter dir: {adapter_dir}")
    print()

    all_pass = True

    # ── Directory ─────────────────────────────────────────────────────────────
    dir_ok = _check("Directory exists", adapter_dir.exists())
    all_pass = all_pass and dir_ok
    if not dir_ok:
        print()
        print("[ERROR] Adapter directory not found. Run training first:")
        print("  python scripts/train_qwen_sop_lora.py --smoke --bf16")
        sys.exit(1)
    print()

    # ── adapter_config.json ───────────────────────────────────────────────────
    cfg_path = adapter_dir / "adapter_config.json"
    cfg_ok = _check("adapter_config.json", cfg_path.exists())
    all_pass = all_pass and cfg_ok
    if cfg_ok:
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            print(f"    peft_type       : {cfg.get('peft_type', '—')}")
            print(f"    base_model      : {cfg.get('base_model_name_or_path', '—')}")
            print(f"    r               : {cfg.get('r', '—')}")
            print(f"    lora_alpha      : {cfg.get('lora_alpha', '—')}")
            print(f"    lora_dropout    : {cfg.get('lora_dropout', '—')}")
            target = cfg.get("target_modules", [])
            print(f"    target_modules  : {', '.join(target) if target else '—'}")
            print(f"    task_type       : {cfg.get('task_type', '—')}")
        except Exception as exc:
            print(f"    [WARN] Could not parse adapter_config.json: {exc}")
    print()

    # ── Adapter weights ───────────────────────────────────────────────────────
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path         = adapter_dir / "adapter_model.bin"

    if safetensors_path.exists():
        size_mb = safetensors_path.stat().st_size / (1024 * 1024)
        _check("adapter_model.safetensors", True, f"{size_mb:.2f} MB")
    elif bin_path.exists():
        size_mb = bin_path.stat().st_size / (1024 * 1024)
        _check("adapter_model.bin", True, f"{size_mb:.2f} MB")
    else:
        _check("Adapter weights", False, "neither .safetensors nor .bin found")
        all_pass = False
    print()

    # ── Tokenizer files ───────────────────────────────────────────────────────
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    found_tokenizer = [f for f in tokenizer_files if (adapter_dir / f).exists()]
    if found_tokenizer:
        _check("Tokenizer files", True, ", ".join(found_tokenizer))
    else:
        _check("Tokenizer files", False, "no tokenizer_config.json / tokenizer.json")
        # Soft warning — adapter can be used without tokenizer files if loaded separately
        print("    [NOTE] Tokenizer files can be loaded directly from the base model.")
    print()

    # ── training_summary.json ─────────────────────────────────────────────────
    summary_path = adapter_dir / "training_summary.json"
    summary_ok = _check("training_summary.json", summary_path.exists())
    all_pass = all_pass and summary_ok
    if summary_ok:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            fields = [
                ("model_name",      "model"),
                ("train_records",   "train records"),
                ("val_records",     "val records"),
                ("train_samples",   "train samples (after tokenisation)"),
                ("epochs",          "epochs"),
                ("batch_size",      "batch size"),
                ("grad_accum",      "grad accum"),
                ("learning_rate",   "learning rate"),
                ("lora_r",          "lora r"),
                ("lora_alpha",      "lora alpha"),
                ("max_length",      "max length"),
                ("smoke",           "smoke mode"),
                ("timestamp",       "timestamp"),
            ]
            for key, label in fields:
                if key in summary:
                    print(f"    {label:40s}: {summary[key]}")
            print()
            if summary.get("final_train_loss") is not None:
                print(f"    {'final train loss':40s}: {summary['final_train_loss']:.4f}")
            if summary.get("final_eval_loss") is not None:
                print(f"    {'final eval loss':40s}: {summary['final_eval_loss']:.4f}")
            if summary.get("note"):
                print(f"    note: {summary['note']}")
        except Exception as exc:
            print(f"    [WARN] Could not parse training_summary.json: {exc}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    if all_pass:
        print("[PASS] Adapter inspection complete — all required files present.")
    else:
        print("[WARN] Some checks failed — see above for details.")
    print()
    print("To load this adapter for inference:")
    print("  from peft import PeftModel")
    print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
    rel = adapter_dir.relative_to(REPO_ROOT) if adapter_dir.is_relative_to(REPO_ROOT) else adapter_dir
    print(f'  model = AutoModelForCausalLM.from_pretrained("<base_model>")')
    print(f'  model = PeftModel.from_pretrained(model, "{rel}")')


if __name__ == "__main__":
    main()
