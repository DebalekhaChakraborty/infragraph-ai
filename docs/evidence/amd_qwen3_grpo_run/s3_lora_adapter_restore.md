# InfraGraph GRPO LoRA Adapter — S3 Location and Restore Instructions

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

## Serve with vLLM (AMD ROCm)

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

## Streamlit env vars (set after vLLM is serving)

```bash
export INFRAGRAPH_QWEN_BASE_URL="http://127.0.0.1:8000/v1"
export INFRAGRAPH_QWEN_MODEL="infragraph"
export INFRAGRAPH_QWEN_TIMEOUT="120"
export INFRAGRAPH_QWEN_MAX_TOKENS="900"
export INFRAGRAPH_LORA_ADAPTER_PATH="/tmp/infragraph_qwen3_grpo_lora_adapter"

streamlit run app/streamlit_app.py
```
