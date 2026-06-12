# AMD ROCm Qwen3-4B GRPO Training — Completion Evidence

## Status

A real Qwen/Qwen3-4B GRPO/vERL training pass completed on AMD ROCm GPU.
32/32 training steps completed successfully.

## Run configuration

| Field | Value |
|-------|-------|
| Model | Qwen/Qwen3-4B |
| Method | LoRA rank 16, alpha 32, target all-linear |
| Algorithm | GRPO via `verl.trainer.main_ppo algorithm.adv_estimator=grpo` |
| Rollout backend | vLLM |
| Actor strategy | FSDP |
| Hardware | AMD ROCm GPU |
| PyTorch | 2.8.0+gitb2fb688 |
| HIP version | 7.0.51831-a3e329ad8 |
| Python | 3.12.11 |
| Training steps | 32/32 completed |
| VRAM usage observed | ~42% |
| GPU power observed | ~278 W |

## Runtime patches applied

- `actor_rollout_ref.rollout.free_cache_engine=False` — disables unsupported
  `--enable_sleep_mode` flag that vLLM passes on ROCm.
- `actor_rollout_ref.rollout.enforce_eager=True` — skips CUDA graph capture
  (safe path on ROCm).
- `extra_info` and `reward_model` written as nested struct columns in parquet
  via `datasets.Dataset.from_list(...).to_parquet(...)`, not JSON strings,
  so vERL's `rl_dataset.py` can access `row_dict["extra_info"]["index"]` directly.
- Source-level patch to `verl/utils/dataset/rl_dataset.py` as belt-and-suspenders
  backup (see `scripts/amd_rocm/patch_verl_runtime_for_rocm.sh`).

## Adapter checkpoint status

No persisted LoRA adapter checkpoint files (`adapter_model.safetensors`,
`adapter_config.json`, etc.) were detected in the committed run evidence.

Do NOT claim a reusable fine-tuned adapter is available unless checkpoint files
are present in `training/verl_grpo/runs/qwen3_4b_grpo_lora_amd/` or documented
as an external artifact.

The training pass demonstrates that the full AMD ROCm GRPO pipeline is
functional end-to-end with this model and configuration.

## Reproducibility

```bash
# 1. Bootstrap environment (AMD ROCm machine)
bash scripts/amd_rocm/bootstrap_grpo_env.sh

# 2. Apply runtime patches
bash scripts/amd_rocm/patch_verl_runtime_for_rocm.sh

# 3. Run full training pipeline
INFRAGRAPH_RUN_REAL_VERL=1 bash training/verl_grpo/train_qwen3_grpo.sh
```

See `scripts/amd_rocm/run_qwen3_grpo_success_path.sh` for the exact command
sequence used in the successful run.
