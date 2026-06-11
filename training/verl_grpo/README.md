# InfraGraph AI Qwen3 LoRA + GRPO/vERL Alignment Scaffold

This folder turns InfraGraph graph RCA records into sample alignment data for a Qwen remediation agent.

It exists to support the hackathon story: graph memory and Enterprise GNN RCA identify the root-cause evidence, then Qwen3 is aligned to produce safe, graph-grounded resolution plans instead of generic runbook text.

## What Is Implemented

- `build_rca_rl_dataset.py` scans V3 scenarios, alert timelines, enterprise graphs, and GNN result files when present.
- It writes:
  - `training/verl_grpo/data/rca_remediation_rl_train.jsonl`
  - `training/verl_grpo/data/rca_remediation_rl_eval.jsonl`
- Each record has prompt, graph evidence, chosen safe response, rejected unsafe response, and reward tags.
- `reward_functions.py` evaluates deterministic reward signals for JSON validity, root-cause match, graph grounding, no hallucinated devices, validation-before-remediation, rollback safety, enterprise escalation, and ServiceNow readiness.
- `train_qwen3_grpo.sh` is safe to run. If `verl` is not installed, it prints install guidance and the exact training command instead of pretending training happened.

## Planned

- Full Qwen3 LoRA/GRPO training on AMD GPUs.
- Larger generated preference dataset across more V3 scenarios.
- vERL integration with rollout workers and reward hooks in a dedicated training environment.

## Commands

Build sample train/eval records:

```bash
python training/verl_grpo/build_rca_rl_dataset.py
```

Evaluate chosen vs rejected responses:

```bash
python training/verl_grpo/reward_functions.py \
  --data training/verl_grpo/data/rca_remediation_rl_eval.jsonl \
  --out training/verl_grpo/reward_eval_report.json
```

Run the safe training scaffold:

```bash
bash training/verl_grpo/train_qwen3_grpo.sh
```

## AMD GPU Notes

The intended training path is Qwen/Qwen3-4B-Instruct with LoRA adapters and GRPO/vERL on ROCm-capable AMD GPUs. Use small batches first, verify vLLM rollout stability, and keep adapter outputs under `training/verl_grpo/runs/`.

No checkpoint in this repository should be interpreted as completed alignment training unless a real training run writes it.
