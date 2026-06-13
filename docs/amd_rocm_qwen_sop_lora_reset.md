# AMD ROCm — Qwen SOP-Grounded LoRA Reset Workflow

Post-reset procedure for serving the SOP-grounded SFT LoRA adapter and regenerating remediation outputs on AMD ROCm Jupyter.

---

## Two LoRA flows in this repo

### Old GRPO/vERL LoRA (archived)

- Trained with `training/verl_grpo/train_qwen3_grpo.sh` via vERL/GRPO reinforcement learning
- Adapter exported to `/tmp/infragraph_qwen3_grpo_lora_adapter` (GRPO convention)
- Published to S3: `s3://my-hackathons/infragraph-ai/model_artifacts/qwen3_grpo_lora_adapter/`
- Served with `--max-model-len 2048` and `--gpu-memory-utilization 0.55` (GRPO adapter settings)
- `INFRAGRAPH_QWEN_MAX_TOKENS=900` (GRPO adapter setting)
- Evidence: `docs/evidence/amd_qwen3_grpo_run/`

### Current SOP-Grounded SFT LoRA

- Trained with `scripts/train_qwen_sop_lora.py` (standard SFT, no vERL/GRPO required)
- Adapter output: `model_artifacts/qwen_lora/infragraph_sop_grounded/`
- Git-ignored by default; rebuild with training script
- Served with `--max-model-len 8192` and `--gpu-memory-utilization 0.75`
- `INFRAGRAPH_QWEN_MAX_TOKENS=1400`
- `INFRAGRAPH_QWEN_MODEL=infragraph` (the vLLM `--lora-modules` alias)
- `INFRAGRAPH_KB_TOP_K=3` (optimal KB retrieval for this adapter)

The two adapters are **not interchangeable**. The SOP-grounded adapter requires `--max-model-len 8192` to avoid context-length errors at inference time.

---

## When to run each GRPO/vERL script

These scripts are for the old GRPO/vERL path. Run them only if you are reproducing the GRPO training experiment — **not** for normal remediation demo resets.

| Script | When to run |
|--------|------------|
| `scripts/amd_rocm/bootstrap_grpo_env.sh` | Only when setting up the vERL/GRPO training environment from scratch |
| `scripts/amd_rocm/patch_verl_runtime_for_rocm.sh` | Only before running GRPO training via vERL |
| `scripts/amd_rocm/run_qwen3_grpo_success_path.sh` | Only when running a full GRPO training pass |
| `scripts/amd_rocm/publish_lora_adapter_to_s3.sh` | Only after a new GRPO adapter is produced and verified |

**Normal reset after Jupyter/GPU restart does NOT require running any of these.** The GRPO training stack is not needed to serve the SOP-grounded adapter.

---

## Normal reset flow (SOP-grounded adapter)

### Terminal 1 — start vLLM

```bash
bash scripts/amd_rocm/start_qwen_sop_lora_vllm.sh
```

Wait for the log line:
```
INFO:     Application startup complete.
```

Verify the server is up:
```bash
curl http://127.0.0.1:8000/v1/models
```

Expected output includes `"id": "infragraph"`.

### Terminal 2 — generate remediation

```bash
bash scripts/amd_rocm/generate_qwen_sop_remediation_after_reset.sh
```

This script:
1. Checks vLLM is reachable (fails early if not)
2. Rebuilds the SOP/KB vector index
3. Validates existing RCA outputs
4. Generates SOP-grounded remediation for all 4 enterprise scenarios
5. Validates remediation outputs
6. Runs quality inspection
7. Prints a compact source/model/ok summary

Expected final output:
```
enterprise_v3_0000.json source= qwen_vllm model= infragraph ok= True
enterprise_v3_0072.json source= qwen_vllm model= infragraph ok= True
enterprise_v3_0073.json source= qwen_vllm model= infragraph ok= True
enterprise_v3_0074.json source= qwen_vllm model= infragraph ok= True
```

---

## Streamlit launch

After Terminal 2 completes:

```bash
export INFRAGRAPH_QWEN_BASE_URL="http://127.0.0.1:8000/v1"
export INFRAGRAPH_QWEN_MODEL="infragraph"
export INFRAGRAPH_QWEN_MAX_TOKENS="1400"
export INFRAGRAPH_QWEN_TIMEOUT="240"
export INFRAGRAPH_QWEN_TEMPERATURE="0.0"

python -m streamlit run app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false \
  > streamlit.log 2>&1 &
```

Open via Jupyter proxy:
```
https://<your-jupyter-host>/proxy/8501/
```

---

## Environment variable reference

### vLLM serving (`start_qwen_sop_lora_vllm.sh`)

| Variable | Default | Notes |
|----------|---------|-------|
| `INFRAGRAPH_LORA_ADAPTER_PATH` | `model_artifacts/qwen_lora/infragraph_sop_grounded` | Override to point at a different adapter dir |
| `INFRAGRAPH_USE_TMP_ADAPTER_SYMLINK` | `1` | Creates `/tmp/infragraph_sop_grounded_lora` symlink. Set to `0` to use real path directly |
| `INFRAGRAPH_BASE_MODEL` | `Qwen/Qwen3-4B` | Hugging Face model ID |
| `INFRAGRAPH_SERVED_MODEL_NAME` | `Qwen3-4B` | The `--served-model-name` value |
| `INFRAGRAPH_VLLM_HOST` | `0.0.0.0` | vLLM bind host |
| `INFRAGRAPH_VLLM_PORT` | `8000` | vLLM bind port |
| `INFRAGRAPH_VLLM_GPU_MEMORY_UTILIZATION` | `0.75` | GPU memory fraction for vLLM |
| `INFRAGRAPH_VLLM_MAX_MODEL_LEN` | `8192` | Context length. **Must be 8192 for SOP-grounded adapter** |
| `VLLM_USE_TRITON_FLASH_ATTN` | `0` | Set to 0 on AMD ROCm to avoid triton flash attention issues |

### Why `INFRAGRAPH_QWEN_MODEL=infragraph`

The vLLM `--lora-modules` flag registers the adapter under the alias `infragraph`. The InfraGraph AI generation code sends this model name in API requests. If `INFRAGRAPH_QWEN_MODEL` is set to anything else (e.g. `Qwen3-4B`), vLLM will route to the base model without the LoRA adapter.

### Why `INFRAGRAPH_VLLM_MAX_MODEL_LEN=8192`

The SOP-grounded adapter's prompt (RCA context + KB evidence + structured JSON output) exceeds 2048 tokens. Using the old GRPO setting of `--max-model-len 2048` causes HTTP 400 context-length errors. The SOP-grounded adapter requires 8192.

### Why `INFRAGRAPH_QWEN_MAX_TOKENS=1400`

The SOP-grounded adapter generates rich structured JSON remediation output with multiple list fields. 900 tokens (old GRPO setting) is insufficient and causes incomplete responses that fail the required-field validation check. 1400 tokens reliably fits the full schema output.

### Why `INFRAGRAPH_KB_TOP_K=3`

3 KB evidence chunks keeps the prompt within the 8192 token budget while providing sufficient SOP context. Higher values (5+) risk exceeding the context window when combined with the full RCA evidence.

### `/tmp` symlink

When `INFRAGRAPH_USE_TMP_ADAPTER_SYMLINK=1` (default), the script creates:
```
/tmp/infragraph_sop_grounded_lora -> model_artifacts/qwen_lora/infragraph_sop_grounded
```

vLLM on some AMD ROCm setups requires the adapter path to be under `/tmp`. The symlink accommodates this without moving the actual adapter. Set `INFRAGRAPH_USE_TMP_ADAPTER_SYMLINK=0` to use the real path directly.

---

## Remediation generation (`generate_qwen_sop_remediation_after_reset.sh`)

| Variable | Default | Notes |
|----------|---------|-------|
| `INFRAGRAPH_QWEN_BASE_URL` | `http://127.0.0.1:8000/v1` | vLLM endpoint |
| `INFRAGRAPH_QWEN_MODEL` | `infragraph` | Must match the `--lora-modules` alias |
| `INFRAGRAPH_QWEN_TIMEOUT` | `240` | HTTP timeout in seconds |
| `INFRAGRAPH_QWEN_MAX_TOKENS` | `1400` | Max output tokens |
| `INFRAGRAPH_QWEN_TEMPERATURE` | `0.0` | Deterministic output |
| `INFRAGRAPH_KB_TOP_K` | `3` | KB evidence chunks per scenario |
| `INFRAGRAPH_INCLUDE_RAW` | `1` | Include `raw_model_output` field in output JSON |
| `INFRAGRAPH_BUILD_KB_INDEX` | `1` | Rebuild KB index before generation |

---

## Fallback behavior

If Qwen/vLLM fails (HTTP error, timeout, invalid JSON, missing required fields), the generation script automatically falls back to deterministic template remediation:

- `remediation_source` becomes `"template_fallback"` (not `"qwen_vllm"`)
- `qwen_error` is preserved in the output envelope
- The file still passes `validate_remediation_outputs.py` (template output is complete)

The fallback is silent in `--prefer-qwen` mode (logs the error but continues). Use `--strict-qwen` to disable fallback and fail the scenario on Qwen errors.

---

## Rebuilding the SOP-grounded adapter

If the adapter was lost after a session reset:

```bash
# On AMD ROCm machine — does not require vERL or GRPO
python scripts/build_kb_index.py --reset
python scripts/expand_sop_grounded_qwen_training_data.py --strict-kb --records-per-scenario 25
python scripts/train_qwen_sop_lora.py --epochs 3 --bf16
```

Then start the vLLM server and regenerate:
```bash
bash scripts/amd_rocm/start_qwen_sop_lora_vllm.sh  # Terminal 1
bash scripts/amd_rocm/generate_qwen_sop_remediation_after_reset.sh  # Terminal 2
```

---

## What NOT to do on a normal reset

- Do NOT run `bootstrap_grpo_env.sh` — that installs the vERL/GRPO training stack, which is not needed to serve the SOP-grounded adapter
- Do NOT run `patch_verl_runtime_for_rocm.sh` — that patches the vERL runtime, not vLLM serving
- Do NOT run `run_qwen3_grpo_success_path.sh` — that triggers GRPO retraining
- Do NOT use `--max-model-len 2048` with the SOP-grounded adapter
- Do NOT set `INFRAGRAPH_QWEN_MAX_TOKENS=900` with the SOP-grounded adapter
- Do NOT use the old GRPO adapter path `/tmp/infragraph_qwen3_grpo_lora_adapter` with the SOP-grounded scripts
