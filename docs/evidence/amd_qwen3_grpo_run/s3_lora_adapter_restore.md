# InfraGraph GRPO LoRA Adapter — S3 Location and Restore Instructions

> **Note — GRPO adapter only.** These instructions are for the old GRPO/vERL
> reinforcement learning adapter (`/tmp/infragraph_qwen3_grpo_lora_adapter`).
> This is NOT the current SOP-grounded SFT LoRA adapter. The GRPO adapter
> settings below (`--max-model-len 2048`, `INFRAGRAPH_QWEN_MAX_TOKENS=900`)
> are specific to the GRPO adapter and must NOT be used with the SOP-grounded
> adapter.
>
> For the current SOP-grounded adapter and reset workflow, see:
> `docs/amd_rocm_qwen_sop_lora_reset.md`

The exported PEFT LoRA adapter is stored in S3, not in Git.
Git stores code, evidence documents, and these restore instructions.

## S3 adapter location

```
s3://my-hackathons/infragraph-ai/model_artifacts/qwen3_grpo_lora_adapter/
```

Contents:

| File | Description |
|------|-------------|
| `adapter_config.json` | PEFT LoRA config (rank 16, alpha 32, target modules) |
| `adapter_model.safetensors` | 504 extracted LoRA weight tensors |
| `README.md` | Adapter provenance and vLLM serving notes |

## Restore adapter locally

```bash
aws s3 sync s3://my-hackathons/infragraph-ai/model_artifacts/qwen3_grpo_lora_adapter/ \
    /tmp/infragraph_qwen3_grpo_lora_adapter/ --no-progress
```

## Publish updated adapter to S3

After a new training run completes:

```bash
scripts/amd_rocm/publish_lora_adapter_to_s3.sh
```

## Serve with vLLM (AMD ROCm) — GRPO adapter only

> These settings are for the GRPO adapter. The SOP-grounded SFT adapter
> requires `--max-model-len 8192` and `INFRAGRAPH_QWEN_MAX_TOKENS=1400`.
> See `docs/amd_rocm_qwen_sop_lora_reset.md` for the current adapter.

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-4B \
  --served-model-name Qwen3-4B \
  --enable-lora \
  --lora-modules infragraph=/tmp/infragraph_qwen3_grpo_lora_adapter \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.55 \
  --max-model-len 2048
```

## Streamlit env vars (GRPO adapter — set after vLLM is serving)

> GRPO adapter settings. For SOP-grounded adapter use
> `INFRAGRAPH_QWEN_MAX_TOKENS=1400` and `INFRAGRAPH_QWEN_TIMEOUT=240`.

```bash
export INFRAGRAPH_QWEN_BASE_URL="http://127.0.0.1:8000/v1"
export INFRAGRAPH_QWEN_MODEL="infragraph"
export INFRAGRAPH_QWEN_TIMEOUT="120"
export INFRAGRAPH_QWEN_MAX_TOKENS="900"
export INFRAGRAPH_LORA_ADAPTER_PATH="/tmp/infragraph_qwen3_grpo_lora_adapter"

streamlit run app/streamlit_app.py
```
