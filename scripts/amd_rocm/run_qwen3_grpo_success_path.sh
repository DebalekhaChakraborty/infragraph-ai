#!/usr/bin/env bash
set -euo pipefail

cd /workspace/shared/infragraph-ai

ray stop --force || true
pkill -f vllm || true
pkill -f main_ppo || true
sleep 5

bash scripts/amd_rocm/patch_verl_runtime_for_rocm.sh

MODEL_ID=Qwen/Qwen3-4B \
INFRAGRAPH_RUN_REAL_VERL=1 \
bash training/verl_grpo/train_qwen3_grpo.sh
