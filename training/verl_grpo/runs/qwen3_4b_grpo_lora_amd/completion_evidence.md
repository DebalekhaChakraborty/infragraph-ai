# InfraGraph AI - Qwen3-4B GRPO/vERL Training Completion Evidence

Status: Completed

Model:
- Qwen/Qwen3-4B

Training:
- LoRA rank: 16
- GRPO via vERL main_ppo
- vLLM rollout backend
- Total training steps: 32/32
- Training completed successfully

Hardware:
- AMD ROCm GPU
- GPU utilization observed at 100%
- VRAM observed around 42%
- Power observed around 278W

Key runtime notes:
- vLLM sleep mode disabled for ROCm compatibility
- vERL dataset reader patched to parse extra_info JSON strings
- agent.num_workers reduced to 1 for single-GPU batch chunking stability

Primary summary:
training/verl_grpo/runs/qwen3_4b_grpo_lora_amd/training_summary.md
