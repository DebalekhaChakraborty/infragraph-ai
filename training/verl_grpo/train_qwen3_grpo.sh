#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# InfraGraph AI — Qwen3 GRPO fine-tuning with vERL on AMD GPU
#
# This script has NOT been run.  It documents the intended training command
# structure for the vERL workshop.
#
# Prerequisites
# ─────────────────────────────────────────────────────────────────────────────
#   pip install verl
#   pip install vllm
#   pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
#
# Step 1: Build the RL training dataset
# ─────────────────────────────────────────────────────────────────────────────
python training/verl_grpo/build_rca_rl_dataset.py \
    --dataset-root ./datasets/infragraph_v3 \
    --gnn-results  ./outputs/enterprise_gnn_rca \
    --out          ./data/rl_training/infragraph_rca_remediation_grpo.jsonl

# Step 2: Launch local vLLM server for rollout generation
# ─────────────────────────────────────────────────────────────────────────────
# (Run in a separate terminal)
#
# python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen3-4B-Instruct \
#     --host 0.0.0.0 \
#     --port 8000

# Step 3: Run GRPO training with vERL
# ─────────────────────────────────────────────────────────────────────────────
python -m verl.trainer.main_grpo \
    --config training/verl_grpo/sample_config.yaml \
    --data.train_files  ./data/rl_training/infragraph_rca_remediation_grpo.jsonl \
    --data.val_files    ./data/rl_training/infragraph_rca_remediation_grpo.jsonl \
    --actor_rollout_ref.model.path Qwen/Qwen3-4B-Instruct \
    --actor_rollout_ref.actor.lora_rank 16 \
    --actor_rollout_ref.actor.lora_alpha 32 \
    --actor_rollout_ref.actor.target_modules q_proj,k_proj,v_proj,o_proj \
    --trainer.total_epochs 3 \
    --trainer.project_name infragraph_ai \
    --trainer.experiment_name qwen3_grpo_rca_remediation \
    --trainer.default_local_dir ./outputs/verl_grpo_checkpoints

# Step 4: Set adapter path for Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
# export INFRAGRAPH_LORA_ADAPTER_PATH=./outputs/verl_grpo_checkpoints/qwen3_grpo_rca_remediation/latest
# streamlit run app/streamlit_app.py
