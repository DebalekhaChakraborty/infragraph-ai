# AMD ROCm Qwen3-4B GRPO Training Evidence

Status: Completed

Evidence:
- Training reached 32/32 steps.
- Training completed successfully.
- Summary generated at `training/verl_grpo/runs/qwen3_4b_grpo_lora_amd/training_summary.md`.
- Run used Qwen/Qwen3-4B, LoRA rank 16, GRPO via vERL, vLLM rollout backend, and AMD ROCm GPU.

Observed runtime:
- GPU utilization reached 100%.
- VRAM usage observed around 42%.
- Power observed around 278W.

Note:
- Runtime ROCm workaround was applied to disable unsupported vLLM sleep mode.
