#!/usr/bin/env python3
"""
train_qwen_sop_lora.py

LoRA supervised fine-tuning of a Qwen model on the SOP-grounded
InfraGraph AI remediation dataset.

This trains for BEHAVIOR ALIGNMENT — the model learns correct output
schema, KB-*/CE-* citation discipline, and SOP-grounded response posture.
It does NOT inject SOP knowledge into model weights.  SOP knowledge lives
in the KB vector index; add new SOPs there via build_kb_index.py.

Usage:
  # Smoke test (8 records, 1 epoch):
  python scripts/train_qwen_sop_lora.py --smoke --gradient-checkpointing --bf16

  # Full demo training:
  python scripts/train_qwen_sop_lora.py \\
      --epochs 3 --batch-size 1 --grad-accum 8 \\
      --learning-rate 2e-4 --gradient-checkpointing --bf16

No bitsandbytes / 4-bit quantization — use standard float32/bf16/fp16.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Constants ─────────────────────────────────────────────────────────────────

IGNORE_INDEX = -100

_DEFAULT_MODEL     = "Qwen/Qwen3-4B"
_DEFAULT_TRAIN     = "data/qwen_sop_grounded_expanded/train.jsonl"
_DEFAULT_VAL       = "data/qwen_sop_grounded_expanded/val.jsonl"
_DEFAULT_OUTPUT    = "model_artifacts/qwen_lora/infragraph_sop_grounded"
_SMOKE_OUTPUT      = "model_artifacts/qwen_lora/infragraph_sop_grounded_smoke"
_SMOKE_RECORDS     = 8
_SMOKE_VAL_RECORDS = 4

_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _manual_chat_format(messages: list[dict]) -> str:
    """Fallback: format messages when tokenizer.apply_chat_template is unavailable."""
    parts = []
    for msg in messages:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}")
    return "\n".join(parts)


def _manual_prompt_format(messages: list[dict]) -> str:
    """Format all but last message and append assistant prefix for loss masking."""
    parts = []
    for msg in messages[:-1]:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def tokenize_record(record: dict, tokenizer, max_length: int) -> dict | None:
    """
    Tokenize one training record with assistant-only loss masking.

    Loss is computed only on assistant response tokens.
    System + user tokens are masked with IGNORE_INDEX (-100).

    Returns None if the record is malformed or if assistant tokens are
    absent (e.g., the full sequence is truncated before the response).
    """
    messages = record.get("messages", [])
    if len(messages) != 3:
        return None

    # Build full sequence text and prompt prefix text
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_text   = _manual_chat_format(messages)
            prompt_text = _manual_prompt_format(messages)
    else:
        full_text   = _manual_chat_format(messages)
        prompt_text = _manual_prompt_format(messages)

    full_enc = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    prompt_enc = tokenizer(
        prompt_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
    )

    input_ids      = full_enc["input_ids"][0]
    attention_mask = full_enc["attention_mask"][0]
    prompt_len     = len(prompt_enc["input_ids"][0])

    if prompt_len >= len(input_ids):
        # No assistant tokens remain after truncation — skip this record
        return None

    labels = input_ids.clone()
    labels[:prompt_len] = IGNORE_INDEX  # mask prompt; train only on assistant

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


# ── Dataset ──────────────────────────────────────────────────────────────────

import torch  # noqa: E402 — imported after repo path is set
from torch.utils.data import Dataset


class SopQwenDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int) -> None:
        self.samples: list[dict] = []
        skipped = 0
        for record in records:
            sample = tokenize_record(record, tokenizer, max_length)
            if sample is not None:
                self.samples.append(sample)
            else:
                skipped += 1
        if skipped:
            print(f"  [WARN] {skipped} record(s) skipped (malformed or fully truncated).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def _collate(batch: list[dict], pad_token_id: int) -> dict:
    from torch.nn.utils.rnn import pad_sequence  # local import to avoid issues if torch absent
    return {
        "input_ids": pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "attention_mask": pad_sequence(
            [b["attention_mask"] for b in batch],
            batch_first=True,
            padding_value=0,
        ),
        "labels": pad_sequence(
            [b["labels"] for b in batch],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        ),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA SFT training for SOP-grounded Qwen InfraGraph AI remediation."
    )
    p.add_argument("--model-name",   default=_DEFAULT_MODEL,  help=f"Base model (default: {_DEFAULT_MODEL})")
    p.add_argument("--train-file",   default=_DEFAULT_TRAIN,  help=f"Train JSONL (default: {_DEFAULT_TRAIN})")
    p.add_argument("--val-file",     default=_DEFAULT_VAL,    help=f"Val JSONL (default: {_DEFAULT_VAL})")
    p.add_argument("--output-dir",   default=_DEFAULT_OUTPUT, help=f"Adapter output dir (default: {_DEFAULT_OUTPUT})")
    p.add_argument("--max-length",   type=int,   default=2048, help="Max token length per sequence (default: 2048)")
    p.add_argument("--epochs",       type=int,   default=3,    help="Training epochs (default: 3)")
    p.add_argument("--batch-size",   type=int,   default=1,    help="Per-device batch size (default: 1)")
    p.add_argument("--grad-accum",   type=int,   default=8,    help="Gradient accumulation steps (default: 8)")
    p.add_argument("--learning-rate",type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    p.add_argument("--lora-r",       type=int,   default=16,   help="LoRA rank r (default: 16)")
    p.add_argument("--lora-alpha",   type=int,   default=32,   help="LoRA alpha (default: 32)")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    p.add_argument("--seed",         type=int,   default=42,   help="Random seed (default: 42)")
    p.add_argument("--bf16",         action="store_true", help="Use bfloat16 mixed precision")
    p.add_argument("--fp16",         action="store_true", help="Use float16 mixed precision")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce VRAM")
    p.add_argument("--smoke",        action="store_true",
                   help=f"Smoke test: train {_SMOKE_RECORDS} records for 1 epoch, "
                        f"output to {_SMOKE_OUTPUT}")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.bf16 and args.fp16:
        print("[ERROR] Cannot use both --bf16 and --fp16. Choose one.")
        sys.exit(1)

    # Smoke mode overrides
    if args.smoke:
        args.epochs     = 1
        args.output_dir = _SMOKE_OUTPUT

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" InfraGraph AI -- Qwen LoRA SFT Training")
    print("=" * 60)
    print(f"  model          : {args.model_name}")
    print(f"  train file     : {args.train_file}")
    print(f"  val file       : {args.val_file}")
    print(f"  output dir     : {args.output_dir}")
    print(f"  epochs         : {args.epochs}")
    print(f"  batch size     : {args.batch_size}")
    print(f"  grad accum     : {args.grad_accum}")
    print(f"  eff. batch     : {args.batch_size * args.grad_accum}")
    print(f"  learning rate  : {args.learning_rate}")
    print(f"  lora r/alpha   : {args.lora_r} / {args.lora_alpha}")
    print(f"  max length     : {args.max_length}")
    print(f"  bf16/fp16      : {args.bf16} / {args.fp16}")
    print(f"  grad ckpt      : {args.gradient_checkpointing}")
    print(f"  smoke          : {args.smoke}")
    print()

    # ── Imports (deferred to allow --help without torch) ──────────────────────
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        print(f"[ERROR] Required package not installed: {exc}")
        print("        Run: pip install -r requirements/requirements-qwen-lora.txt")
        sys.exit(1)

    set_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cpu":
        print("[WARN] No GPU detected. Training will be very slow on CPU.")
        print("       Disable --bf16/--fp16 if you see dtype errors on CPU.")
        print()

    # ── Load data ─────────────────────────────────────────────────────────────
    train_path = (REPO_ROOT / args.train_file).resolve()
    val_path   = (REPO_ROOT / args.val_file).resolve()

    if not train_path.exists():
        print(f"[ERROR] Train file not found: {train_path}")
        print("        Run: python scripts/expand_sop_grounded_qwen_training_data.py --strict-kb")
        sys.exit(1)
    if not val_path.exists():
        print(f"[ERROR] Val file not found: {val_path}")
        sys.exit(1)

    train_records = load_jsonl(train_path)
    val_records   = load_jsonl(val_path)

    if args.smoke:
        train_records = train_records[:_SMOKE_RECORDS]
        val_records   = val_records[:_SMOKE_VAL_RECORDS]

    print(f"  Records loaded : {len(train_records)} train, {len(val_records)} val")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = "auto"

    print(f"Loading model   : {args.model_name}  (dtype={torch_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    if args.gradient_checkpointing:
        # Required to backpropagate through frozen base model weights with PEFT
        model.enable_input_require_grads()

    model.print_trainable_parameters()
    print()

    # ── Tokenize datasets ─────────────────────────────────────────────────────
    print("Tokenizing training records...")
    train_dataset = SopQwenDataset(train_records, tokenizer, args.max_length)
    print("Tokenizing validation records...")
    val_dataset   = SopQwenDataset(val_records,   tokenizer, args.max_length)

    print(f"  Samples after tokenization: {len(train_dataset)} train, {len(val_dataset)} val")
    if len(train_dataset) == 0:
        print("[ERROR] No training samples after tokenization. Check data files.")
        sys.exit(1)

    # ── Trainer ───────────────────────────────────────────────────────────────
    gc_kwargs = {"use_reentrant": False} if args.gradient_checkpointing else None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    collate_fn = partial(_collate, pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting training...")
    train_result = trainer.train()
    print()

    # ── Save adapter ──────────────────────────────────────────────────────────
    print(f"Saving LoRA adapter to: {output_dir.relative_to(REPO_ROOT)}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ── Extract loss metrics ──────────────────────────────────────────────────
    train_loss: float | None = getattr(train_result, "training_loss", None)
    eval_loss: float | None = None
    for log in reversed(trainer.state.log_history):
        if "eval_loss" in log and eval_loss is None:
            eval_loss = log["eval_loss"]
        if train_loss is None and "loss" in log:
            train_loss = log["loss"]
        if train_loss is not None and eval_loss is not None:
            break

    # ── Training summary ──────────────────────────────────────────────────────
    summary: dict = {
        "model_name":           args.model_name,
        "train_records":        len(train_records),
        "val_records":          len(val_records),
        "train_samples":        len(train_dataset),
        "val_samples":          len(val_dataset),
        "epochs":               args.epochs,
        "batch_size":           args.batch_size,
        "grad_accum":           args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate":        args.learning_rate,
        "lora_r":               args.lora_r,
        "lora_alpha":           args.lora_alpha,
        "lora_dropout":         args.lora_dropout,
        "max_length":           args.max_length,
        "bf16":                 args.bf16,
        "fp16":                 args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "smoke":                args.smoke,
        "output_dir":           str(output_dir.relative_to(REPO_ROOT)),
        "timestamp":            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "final_train_loss":     train_loss,
        "final_eval_loss":      eval_loss,
        "note": (
            "Demo-scale SOP-grounded LoRA alignment. "
            "Not production-scale. "
            "SOP updates require KB re-indexing, not retraining."
        ),
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("[PASS] Training complete.")
    print(f"  Adapter    : {output_dir.relative_to(REPO_ROOT)}/")
    if train_loss is not None:
        print(f"  Train loss : {train_loss:.4f}")
    if eval_loss is not None:
        print(f"  Eval loss  : {eval_loss:.4f}")
    print(f"  Summary    : {summary_path.relative_to(REPO_ROOT)}")
    print()
    print("Inspect adapter:")
    print(f"  python scripts/inspect_qwen_lora_adapter.py --adapter-dir {output_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
