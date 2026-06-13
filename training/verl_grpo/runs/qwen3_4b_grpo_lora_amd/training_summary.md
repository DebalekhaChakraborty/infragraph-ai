# InfraGraph AI — Qwen3-4B LoRA + GRPO/vERL Training Summary

Generated: 2026-06-11 23:26 UTC

---

## Run Status

**Scaffold / dry-run only**

> **Honest status:** This summary documents a scaffold and reward-evaluated alignment dataset only.  No LoRA adapter checkpoint was produced in this repository.  The claim _'LoRA fine-tuned Qwen3-4B with GRPO using vERL on AMD GPUs'_ requires a real training run that writes adapter files to this `runs/` directory.

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
| Run directory   | training\verl_grpo\runs\qwen3_4b_grpo_lora_amd                |

---

## Hardware

| Field           | Value         |
|-----------------|---------------|
| PyTorch version | not installed |
| CUDA available  | False         |
| HIP version     | —             |
| Device name     | —             |

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

_No checkpoint files found in run directory._

### Config files

_No config files found._

---

## Honest Claims

Only make the following claims after the corresponding evidence exists:

| Claim | Requires |
|-------|----------|
| "Reward-evaluated alignment dataset built" | `verl_train.parquet` + `verl_eval.parquet` exist |
| "GRPO training scaffold implemented" | `train_qwen3_grpo.sh` runs without error |
| "LoRA fine-tuned Qwen3-4B with GRPO/vERL" | Adapter checkpoint files in `runs/` |
| "Tested on AMD GPU (ROCm)" | `torch_hip_version` non-null AND adapter files exist |
