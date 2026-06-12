# Live LoRA vLLM Verification

## Status

The InfraGraph LoRA adapter was exported from the completed vERL/GRPO checkpoint and loaded into vLLM with `--enable-lora`.

## Runtime model listing

`/v1/models` showed two models:

- `Qwen3-4B` — base model, root `Qwen/Qwen3-4B`
- `infragraph` — LoRA adapter model, root `/tmp/infragraph_qwen3_grpo_lora_adapter`, parent `Qwen3-4B`

## Inference test

A `/v1/chat/completions` request was successfully executed with:

```json
{
  "model": "infragraph"
}
