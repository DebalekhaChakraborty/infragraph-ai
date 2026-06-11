# Qwen3 GRPO Fine-Tuning for InfraGraph AI Remediation

This directory contains the RL training preparation for fine-tuning
Qwen3-4B with LoRA and GRPO using vERL on AMD GPUs.

**Status:** Dataset builder and reward functions are implemented.
Training has not been run — no checkpoint exists yet.

---

## Files

| File | Purpose |
|------|---------|
| `build_rca_rl_dataset.py` | Reads V3 enterprise scenarios, builds GRPO JSONL training dataset |
| `reward_functions.py` | GRPO reward components (root-cause match, grounding, JSON format, …) |
| `train_qwen3_grpo.sh` | Training command structure for vERL (not yet executed) |
| `sample_config.yaml` | vERL GRPO training configuration template |

---

## Training Loop Concept

```
Enterprise scenario
  → alerts.json + enterprise_graph.json + GNN result
  → build_remediation_prompt()
  → Qwen3 rollout (vLLM)
  → batch_reward_fn() scores response
  → GRPO policy gradient update (vERL)
  → LoRA adapter updated
```

## Reward Components

| Function | Weight | Description |
|----------|--------|-------------|
| `root_cause_match_reward` | 0.30 | Does response name the correct root-cause node? |
| `grounded_node_reward` | 0.20 | Fraction of required graph nodes mentioned |
| `no_hallucinated_node_penalty` | 0.15 | Penalty for node IDs not in the graph |
| `validation_before_remediation_reward` | 0.10 | Validation steps present before remediation |
| `action_specificity_reward` | 0.10 | Steps are detailed (>8 words each) |
| `json_format_reward` | 0.10 | Output is valid JSON with all required keys |
| `escalation_if_multi_diagram_reward` | 0.05 | Escalation recommended for multi-domain incidents |

## Quick Start

### 1. Build the RL dataset

```bash
python training/verl_grpo/build_rca_rl_dataset.py \
    --dataset-root ./datasets/infragraph_v3 \
    --gnn-results  ./outputs/enterprise_gnn_rca \
    --out          ./data/rl_training/infragraph_rca_remediation_grpo.jsonl
```

### 2. Start the vLLM server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

### 3. Run GRPO training

```bash
bash training/verl_grpo/train_qwen3_grpo.sh
```

### 4. Point the app at the fine-tuned adapter

```bash
export INFRAGRAPH_LORA_ADAPTER_PATH=./outputs/verl_grpo_checkpoints/qwen3_grpo_rca_remediation/latest
streamlit run app/streamlit_app.py
```
