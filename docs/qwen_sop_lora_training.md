# Qwen SOP-Grounded LoRA Training

Supervised fine-tuning pipeline for behavior alignment of the InfraGraph AI Qwen remediation model.

---

## Purpose

This LoRA adapter trains Qwen to:

- Follow the InfraGraph AI remediation output schema exactly
- Cite KB-* and CE-* evidence IDs from the supplied context
- Apply domain-specific SOP constraints (load balancer, database, WAN, firewall)
- Never invent commands, IP addresses, owners, or device facts
- Return strict JSON responses

This is **behavior alignment**, not SOP memorization.

### What this training does NOT do

- It does not inject SOP knowledge into model weights.
- It does not replace the KB vector index.
- It does not train the model to answer questions about specific network devices.

SOP knowledge lives in the KB vector index (`runtime_state/kb_index/`). When new SOPs are added or existing ones are revised, rebuild the KB index — not the LoRA adapter:

```bash
python scripts/build_kb_index.py --reset
```

---

## When to retrain vs when to re-index

| Change | Action |
|--------|--------|
| New SOP or runbook added to `assets/kb/` | Rebuild KB index only |
| Existing SOP content revised | Rebuild KB index only |
| Model outputs wrong schema fields | Retrain LoRA adapter |
| Model ignores KB-*/CE-* citation discipline | Retrain LoRA adapter |
| Model invents device names or commands | Retrain LoRA adapter |
| More scenario coverage needed | Generate more records, retrain |
| Model output quality improved by more data | Generate more records, retrain |

---

## Dataset scale

**Current state:** 100 synthetic records (85 train / 15 val) across 4 base scenarios.

This is demo-scale, not production-scale. 100 records can teach schema discipline and citation posture, but real generalization requires more coverage.

| Scale | Records | Expected outcome |
|-------|---------|-----------------|
| Current (demo) | ~100 | Schema alignment, citation discipline on covered domains |
| Small production | 500–1000 | Better generalization across alert variations |
| Production | 2000+ | Robust behavior across new scenarios and domain combinations |

To scale up:
1. Add enterprise scenarios to `scenario_library/enterprise_gnn_rca/`
2. Add domain SOPs to `assets/kb/`
3. Rebuild KB index
4. Re-run `expand_sop_grounded_qwen_training_data.py --records-per-scenario 100`
5. Retrain LoRA

---

## Setup

```bash
# Install LoRA training dependencies (in addition to base requirements)
pip install -r requirements/requirements-qwen-lora.txt

# Note: torch must be installed separately for your hardware:
# CUDA:     pip install torch
# AMD ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

No bitsandbytes. No 4-bit quantization. Standard float32/bfloat16/float16 only.

---

## Generate training data first

The KB index must be built before generating training data with strict-kb:

```bash
python scripts/build_kb_index.py --reset
python scripts/expand_sop_grounded_qwen_training_data.py --strict-kb --records-per-scenario 25
```

---

## Smoke training (quick check)

Trains on 8 records for 1 epoch. Verifies the pipeline end-to-end before committing to a full run.

```bash
python scripts/train_qwen_sop_lora.py \
    --smoke \
    --gradient-checkpointing \
    --bf16
```

Output:
```
model_artifacts/qwen_lora/infragraph_sop_grounded_smoke/
  adapter_config.json
  adapter_model.safetensors
  tokenizer.json
  tokenizer_config.json
  training_summary.json
```

Inspect smoke adapter:
```bash
python scripts/inspect_qwen_lora_adapter.py \
    --adapter-dir model_artifacts/qwen_lora/infragraph_sop_grounded_smoke
```

---

## Full demo training

```bash
python scripts/train_qwen_sop_lora.py \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 8 \
    --learning-rate 2e-4 \
    --gradient-checkpointing \
    --bf16
```

Effective batch size: 8 (1 × 8 grad accum). For a 4B model this fits in ~16–24 GB VRAM with gradient checkpointing and BF16.

Inspect:
```bash
python scripts/inspect_qwen_lora_adapter.py \
    --adapter-dir model_artifacts/qwen_lora/infragraph_sop_grounded
```

---

## LoRA configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| r | 16 | LoRA rank — higher = more capacity, more VRAM |
| alpha | 32 | LoRA alpha — scaling factor (alpha/r = 2.0) |
| dropout | 0.05 | Small dropout to reduce overfitting on 100 records |
| target_modules | q/k/v/o/gate/up/down | All attention + FFN projections |
| bias | none | Standard — no bias adaptation |

Trained modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.

---

## Loss masking

The training script masks the system prompt and user message tokens so loss is computed only on the assistant response. This is implemented in `tokenize_record()`:

1. Format the full sequence (system + user + assistant) with the chat template.
2. Format the prompt-only prefix (system + user + `<|assistant|>` start).
3. Tokenize both; the prompt length is used to set labels to -100 (ignored).
4. The model trains to predict only the assistant JSON response.

This ensures the model learns to produce the correct JSON output format, not to reproduce the prompt.

---

## Training artifacts

```
model_artifacts/qwen_lora/
  infragraph_sop_grounded/
    adapter_config.json         LoRA adapter configuration
    adapter_model.safetensors   LoRA adapter weights
    tokenizer.json              Tokenizer (copied from base model)
    tokenizer_config.json
    special_tokens_map.json
    training_summary.json       Run metadata and loss metrics
    checkpoint-*/               Intermediate checkpoints (auto-cleaned to last 1)

  infragraph_sop_grounded_smoke/
    (same structure, smaller adapter from smoke run)
```

Adapter weights are gitignored by default (`model_artifacts/qwen_lora/` in `.gitignore`).

---

## Loading adapter for inference

### With Hugging Face (local inference)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = "Qwen/Qwen3-4B"
adapter_dir = "model_artifacts/qwen_lora/infragraph_sop_grounded"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()
```

### With vLLM (production serving)

vLLM supports LoRA adapters via `LoRARequest`. The adapter must be accessible at the path specified.

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="Qwen/Qwen3-4B",
    enable_lora=True,
    max_lora_rank=16,        # must match --lora-r
)

lora_request = LoRARequest(
    lora_name="infragraph_sop",
    lora_int_id=1,
    lora_local_path="model_artifacts/qwen_lora/infragraph_sop_grounded",
)

outputs = llm.generate(
    prompts=[prompt],
    sampling_params=SamplingParams(temperature=0.0, max_tokens=2048),
    lora_request=lora_request,
)
```

**Note:** vLLM requires the ROCm build on AMD hardware. See `requirements/requirements-amd-rocm.txt` and `scripts/amd_rocm/bootstrap_grpo_env.sh`.

---

## Integrity constraints

- Never reads `labels.json` or evaluation outputs.
- Only trains on `data/qwen_sop_grounded_expanded/train.jsonl` and `val.jsonl`.
- No forbidden keys (`expected_root_cause`, `ground_truth_node`, etc.) in training data.
- Adapter weights are not committed to git by default.
- Model retraining does not modify the KB index, RCA pipeline, or GNN checkpoint.
