#!/usr/bin/env bash
set -euo pipefail

cd /workspace/shared/infragraph-ai

ray stop --force || true
pkill -f vllm || true
pkill -f main_ppo || true
sleep 5

bash scripts/amd_rocm/patch_verl_runtime_for_rocm.sh

RUN_DIR="${RUN_DIR:-training/verl_grpo/runs/qwen3_4b_grpo_lora_amd_saved}"

MODEL_ID=Qwen/Qwen3-4B \
SAVE_FREQ="${SAVE_FREQ:-8}" \
TEST_FREQ="${TEST_FREQ:-8}" \
RUN_DIR="$RUN_DIR" \
INFRAGRAPH_RUN_REAL_VERL=1 \
bash training/verl_grpo/train_qwen3_grpo.sh

python training/verl_grpo/find_lora_adapter_artifacts.py --run-dir "$RUN_DIR" || true

python training/verl_grpo/export_lora_adapter.py \
  --run-dir "$RUN_DIR" \
  --base-model Qwen/Qwen3-4B \
  --output-dir "$RUN_DIR/exported_adapter" || true