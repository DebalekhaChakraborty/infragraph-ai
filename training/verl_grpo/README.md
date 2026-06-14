# InfraGraph AI — Qwen3-4B LoRA + GRPO/vERL Alignment (AMD ROCm)

This directory implements the RL alignment pipeline for a graph-grounded
remediation agent based on Qwen3-4B, LoRA adapters, and GRPO via vERL.

---

## Honest Status Levels

The project distinguishes four states.  Read this before making any claims.

| State | What exists | Honest label |
|-------|-------------|--------------|
| **Scaffold / dry-run** | Dataset + reward functions + training script; no real run | "LoRA + GRPO alignment scaffold implemented" |
| **Reward-evaluated dataset** | `verl_train.parquet` + `verl_eval.parquet` produced; reward scores computed | "Reward-evaluated alignment dataset built" |
| **Real training pass completed** | 32/32 steps ran on AMD ROCm GPU; no persisted adapter detected in committed evidence | "Real Qwen/Qwen3-4B GRPO training pass completed on AMD ROCm (adapter checkpoint not committed)" |
| **Persisted adapter available** | `adapter_model.safetensors` / `adapter_config.json` in `runs/` or documented external path | "LoRA fine-tuned Qwen3-4B with GRPO using vERL on AMD GPU — adapter available" |

**Current status: Real training pass completed (32/32 steps, AMD ROCm, HIP 7.0).**
No adapter checkpoint files were detected in the committed run evidence.
Do not claim a reusable fine-tuned adapter is available unless checkpoint files exist.

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
| `find_lora_adapter_artifacts.py` | Scan a run directory for adapter files; suggests INFRAGRAPH_LORA_ADAPTER_PATH |
| `export_lora_adapter.py` | Convert vERL actor checkpoint to PEFT format; fails gracefully if format unsupported |
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
| `itsm_ticket_summary` | 6 % | Structured ITSM dict with all five fields |

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

### 5. Real training run with checkpoint persistence (requires vERL + AMD/CUDA GPU)

`SAVE_FREQ` (default 8) controls how often vERL writes actor checkpoints.
`TEST_FREQ` (default 8) controls validation frequency.
`RUN_DIR` overrides the output directory.

```bash
# Install prerequisites
pip install verl vllm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0  # AMD
# or for NVIDIA:
pip install torch --index-url https://download.pytorch.org/whl/cu121

SAVE_FREQ=8 TEST_FREQ=8 \
RUN_DIR=training/verl_grpo/runs/qwen3_4b_grpo_lora_amd_saved \
INFRAGRAPH_RUN_REAL_VERL=1 \
bash training/verl_grpo/train_qwen3_grpo.sh
```

### 6. Scan for adapter artifacts (after a run)

```bash
python training/verl_grpo/find_lora_adapter_artifacts.py \
  --run-dir training/verl_grpo/runs/qwen3_4b_grpo_lora_amd_saved
```

Prints a file table and suggests `INFRAGRAPH_LORA_ADAPTER_PATH` if a
PEFT-compatible directory is found.

### 7. Export vERL checkpoint to PEFT format

If the run directory contains FSDP-sharded actor weights rather than a
standard PEFT adapter:

```bash
python training/verl_grpo/export_lora_adapter.py \
  --run-dir  training/verl_grpo/runs/qwen3_4b_grpo_lora_amd_saved \
  --base-model Qwen/Qwen3-4B \
  --output-dir training/verl_grpo/exported_adapter
```

The script tries two strategies in order:
1. Direct `PeftModel.from_pretrained` if `adapter_config.json` is present.
2. Heuristic FSDP shard merge, extracting `lora_A`/`lora_B` tensors.

It fails with a clear message if neither strategy recognises the format —
it never fabricates adapter files.

### 8. Write training summary (after a run)

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

### Verify checkpoint files exist

```bash
python training/verl_grpo/find_lora_adapter_artifacts.py \
  --run-dir training/verl_grpo/runs/qwen3_4b_grpo_lora_amd_saved
```

### Convert to PEFT format if needed

```bash
python training/verl_grpo/export_lora_adapter.py \
  --run-dir  training/verl_grpo/runs/qwen3_4b_grpo_lora_amd_saved \
  --base-model Qwen/Qwen3-4B \
  --output-dir training/verl_grpo/exported_adapter
```

### Point the app at the adapter

```bash
export INFRAGRAPH_LORA_ADAPTER_PATH=training/verl_grpo/exported_adapter
streamlit run app/streamlit_app.py
```

The AI Resolution Agent in Tab 2 (Topology RCA) and Tab 4 (GNN RCA) will show
"Fine-tuned Adapter: Loaded" and use the aligned model for inference.

Do not set `INFRAGRAPH_LORA_ADAPTER_PATH` unless `adapter_model.safetensors`
and `adapter_config.json` are present in the target directory.
