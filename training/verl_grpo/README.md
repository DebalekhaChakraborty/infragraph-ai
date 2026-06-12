# InfraGraph AI — Qwen3-4B LoRA + GRPO/vERL Alignment (AMD ROCm)

This directory implements the RL alignment pipeline for a graph-grounded
remediation agent based on Qwen3-4B-Instruct, LoRA adapters, and GRPO via vERL.

---

## Honest Status Levels

The project distinguishes three states.  Read this before making any claims.

| State | What exists | Honest label |
|-------|-------------|--------------|
| **Scaffold / dry-run** | Dataset + reward functions + training script; no real run | "LoRA + GRPO alignment scaffold implemented" |
| **Reward-evaluated dataset** | `verl_train.parquet` + `verl_eval.parquet` produced; reward scores computed | "Reward-evaluated alignment dataset built" |
| **Completed fine-tune** | Adapter checkpoint files written by a real vERL run | "LoRA fine-tuned Qwen3-4B with GRPO using vERL on AMD GPUs" |

**Only claim a completed fine-tune after a real training run produces adapter files
in `training/verl_grpo/runs/qwen3_4b_grpo_lora_amd/`.**

---

## Installation

Three dependency tiers — install only what you need.

### 1. App / demo only (no training)

```bash
pip install -r requirements.txt
```

Covers the Streamlit cockpit, graph RCA, topology analysis, and non-GPU usage.
`pip install -r requirements.txt` alone **cannot** reproduce the AMD training run.

### 2. Dataset conversion and reward evaluation

```bash
pip install -r requirements.txt
pip install -r requirements-training.txt
```

Adds `datasets`, `transformers`, `peft`, `accelerate`, and `pyarrow`.
Required before running `build_rca_rl_dataset.py`, `prepare_verl_dataset.py`,
and `reward_functions.py`.

### 3. AMD ROCm vERL/GRPO training

```bash
bash scripts/amd_rocm/bootstrap_grpo_env.sh
bash scripts/amd_rocm/patch_verl_runtime_for_rocm.sh
```

The bootstrap script installs ROCm torch, vLLM, and vERL **only if not already
present** — it does not blindly reinstall a working ROCm environment.

The patch script verifies that all known ROCm workarounds are in place:
- `free_cache_engine=False` — prevents vLLM `--enable_sleep_mode` on ROCm
- `enforce_eager=True` — skips CUDA graph capture (safe on ROCm)
- `extra_info` written as struct columns, not JSON strings

`requirements-amd-rocm.txt` documents the full stack for reference but should
not be installed with `pip install -r` directly.

---

## Files

| File | Purpose |
|------|---------|
| `build_rca_rl_dataset.py` | Scan V3 scenarios → write train/eval JSONL with chosen + rejected responses |
| `prepare_verl_dataset.py` | Convert JSONL → vERL parquet (prompt, reward_model, extra_info columns) |
| `verl_reward.py` | `compute_score(data_source, solution_str, ground_truth, extra_info)` — deterministic reward for vERL |
| `reward_functions.py` | Standalone reward CLI and batch_reward_fn for offline evaluation |
| `train_qwen3_grpo.sh` | Full pipeline: dataset → parquet → dry-run or real vERL training |
| `write_training_summary.py` | Post-run summary generator (artifacts, hardware, honest claims) |
| `sample_config.yaml` | vERL config reference (informational; shell script passes flags directly) |
| `data/` | Generated JSONL and parquet files (not committed) |
| `runs/` | Training output and adapter checkpoints (not committed) |

---

## Architecture

```
V3 scenario data (enterprise_graph.json + alerts.json)
        │
        ▼
build_rca_rl_dataset.py
        │  JSONL: prompt, chosen_response, rejected_response, reward_tags
        ▼
prepare_verl_dataset.py
        │  Parquet: data_source, prompt (chat messages), reward_model, extra_info
        ▼
verl.trainer.main_ppo  (algorithm.adv_estimator=grpo)
        │  rollout via vLLM → score via verl_reward.compute_score → GRPO update
        ▼
LoRA adapter checkpoint  →  load via INFRAGRAPH_LORA_ADAPTER_PATH
```

### Why `main_ppo`, not `main_grpo`

vERL implements GRPO as an advantage estimator inside the PPO trainer.
The correct launch command is:

```bash
python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo ...
```

`verl.trainer.main_grpo` does not exist in the published vERL package.

---

## Reward Functions

Eight deterministic components (no model call):

| Component | Weight | Description |
|-----------|--------|-------------|
| `json_format` | 16 % | Valid JSON with all required output keys |
| `root_cause_match` | 18 % | `probable_root_cause` names the correct node |
| `grounded_node` | 14 % | Impacted nodes cited in response |
| `no_hallucinated_device` | 14 % | No device IDs outside the valid node set |
| `validation_before_remediation` | 12 % | Validation steps precede remediation steps |
| `rollback_safety` | 12 % | Rollback notes + `do_not_execute_if` safeguards present |
| `enterprise_escalation` | 8 % | Escalation recommended for cross-diagram incidents |
| `servicenow_summary` | 6 % | Structured ServiceNow dict with all five fields |

---

## Commands

### 1. Build alignment dataset

```bash
python training/verl_grpo/build_rca_rl_dataset.py
```

### 2. Convert to vERL parquet

```bash
python training/verl_grpo/prepare_verl_dataset.py
```

### 3. Evaluate reward functions offline

```bash
python training/verl_grpo/reward_functions.py \
  --data training/verl_grpo/data/rca_remediation_rl_eval.jsonl \
  --out  training/verl_grpo/reward_eval_report.json
```

### 4. Dry-run (default — prints training command, no GPU needed)

```bash
bash training/verl_grpo/train_qwen3_grpo.sh
```

### 5. Real training run (requires vERL + AMD/CUDA GPU)

```bash
# Install prerequisites
pip install verl vllm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0  # AMD
# or for NVIDIA:
pip install torch --index-url https://download.pytorch.org/whl/cu121

INFRAGRAPH_RUN_REAL_VERL=1 bash training/verl_grpo/train_qwen3_grpo.sh
```

### 6. Write training summary (after a run)

```bash
python training/verl_grpo/write_training_summary.py --run-dir training/verl_grpo/runs/qwen3_4b_grpo_lora_amd
```

---

## AMD GPU Notes

- Install torch with ROCm wheel (see above).
- `torch.version.hip` will be non-null on a ROCm build; the training summary
  captures this field automatically.
- Start with `gpu_memory_utilization=0.35` and `max_model_len=1024`; increase
  once rollout is stable.
- FSDP is preferred over DeepSpeed for ROCm stability.
- Set `INFRAGRAPH_LORA_ADAPTER_PATH` to the checkpoint path after training
  so the Streamlit app loads the adapter.

---

## After Training

Point the app at the adapter:

```bash
export INFRAGRAPH_LORA_ADAPTER_PATH=training/verl_grpo/runs/qwen3_4b_grpo_lora_amd/latest
streamlit run app/streamlit_app.py
```

The AI Resolution Agent in Tab 2 (Topology RCA) and Tab 4 (GNN RCA) will show
"Fine-tuned Adapter: Loaded" and use the aligned model for inference.
