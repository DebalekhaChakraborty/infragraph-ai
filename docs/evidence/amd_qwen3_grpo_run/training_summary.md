# InfraGraph AI — Qwen3-4B LoRA + GRPO/vERL Training Summary

Generated: 2026-06-12 13:57 UTC

---

## Run Status

**Real vERL training run completed**

> **Honest status:** A real training run completed.  Verify adapter/checkpoint files below before making fine-tuning claims.

---

## Configuration

| Field           | Value                                                         |
|-----------------|---------------------------------------------------------------|
| Base model      | Qwen/Qwen3-4B                                                 |
| Method          | LoRA (rank=16, alpha=32, target=all-linear)                   |
| Algorithm       | GRPO via verl.trainer.main_ppo (algorithm.adv_estimator=grpo) |
| Framework       | vERL (https://github.com/volcengine/verl)                     |
| Rollout backend | vLLM                                                          |
| Actor strategy  | FSDP                                                          |
| Run directory   | /tmp/infragraph_grpo_runs/qwen3_4b_grpo_lora_amd_saved        |

---

## Hardware

| Field           | Value               |
|-----------------|---------------------|
| PyTorch version | 2.8.0+gitb2fb688    |
| CUDA available  | True                |
| HIP version     | 7.0.51831-a3e329ad8 |
| Device name     | —                   |

---

## Dataset

| Field                | Value                      |
|----------------------|----------------------------|
| Train JSONL records  | 64                         |
| Eval JSONL records   | 16                         |
| Train parquet exists | yes                        |
| Eval parquet exists  | yes                        |
| Ability tag          | graph_grounded_remediation |
| Data source          | infragraph_rca_remediation |

---

## Reward Functions

| Component | Weight | Description |
|-----------|--------|-------------|
| `json_format` | 16%  — valid JSON with all required keys |
| `root_cause_match` | 18%  — probable_root_cause names correct node |
| `grounded_node` | 14%  — impacted nodes cited in response |
| `no_hallucinated_device` | 14%  — no device IDs outside valid set |
| `validation_before_remediation` | 12%  — validation steps precede remediation |
| `rollback_safety` | 12%  — rollback notes + do_not_execute safeguards |
| `enterprise_escalation` |  8%  — escalation for cross-diagram incidents |
| `servicenow_summary` |  6%  — structured ServiceNow dict present |

Reward entry point: `training/verl_grpo/verl_reward.py::compute_score`

---

## Adapter / Checkpoint Artifacts

- `/tmp/infragraph_grpo_runs/qwen3_4b_grpo_lora_amd_saved/global_step_32/actor/extra_state_world_size_1_rank_0.pt`
- `/tmp/infragraph_grpo_runs/qwen3_4b_grpo_lora_amd_saved/global_step_32/actor/model_world_size_1_rank_0.pt`
- `/tmp/infragraph_grpo_runs/qwen3_4b_grpo_lora_amd_saved/global_step_32/actor/optim_world_size_1_rank_0.pt`
- `/tmp/infragraph_grpo_runs/qwen3_4b_grpo_lora_amd_saved/global_step_32/data.pt`

### Config files

- `/tmp/infragraph_grpo_runs/qwen3_4b_grpo_lora_amd_saved/global_step_32/actor/huggingface/config.json`

---

## Honest Claims

Only make the following claims after the corresponding evidence exists:

| Claim | Requires |
|-------|----------|
| "Reward-evaluated alignment dataset built" | `verl_train.parquet` + `verl_eval.parquet` exist |
| "GRPO training scaffold implemented" | `train_qwen3_grpo.sh` runs without error |
| "LoRA fine-tuned Qwen3-4B with GRPO/vERL" | Adapter checkpoint files in `runs/` |
| "Tested on AMD GPU (ROCm)" | `torch_hip_version` non-null AND adapter files exist |
