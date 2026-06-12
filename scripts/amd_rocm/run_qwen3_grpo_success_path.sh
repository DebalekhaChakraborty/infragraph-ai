#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

cd /workspace/shared/infragraph-ai

# ── Preflight cleanup ─────────────────────────────────────────────────────────
echo "Preflight cleanup: stopping Ray/vLLM/vERL processes ..."
ray stop --force || true
pkill -f vllm || true
pkill -f main_ppo || true
sleep 3
rm -rf /tmp/ray /tmp/vllm* 2>/dev/null || true
echo "  Done."
echo

echo "Applying ROCm runtime patches..."
bash scripts/amd_rocm/patch_verl_runtime_for_rocm.sh

RUN_DIR="${RUN_DIR:-training/verl_grpo/runs/qwen3_4b_grpo_lora_amd_saved}"

echo "Starting Qwen3-4B GRPO/vERL training with checkpoint saving..."
echo "RUN_DIR=$RUN_DIR"

MODEL_ID=Qwen/Qwen3-4B \
SAVE_FREQ="${SAVE_FREQ:-32}" \
TEST_FREQ="${TEST_FREQ:-32}" \
RUN_DIR="$RUN_DIR" \
INFRAGRAPH_RUN_REAL_VERL=1 \
bash training/verl_grpo/train_qwen3_grpo.sh

echo
echo "Scanning for LoRA/checkpoint artifacts..."
"$PYTHON" training/verl_grpo/find_lora_adapter_artifacts.py --run-dir "$RUN_DIR" || true

echo
echo "Trying PEFT LoRA export if vERL checkpoint exists..."
"$PYTHON" training/verl_grpo/export_lora_adapter.py \
  --run-dir "$RUN_DIR" \
  --base-model Qwen/Qwen3-4B \
  --output-dir "$RUN_DIR/exported_adapter" || true

echo
echo "Final adapter scan..."
"$PYTHON" training/verl_grpo/find_lora_adapter_artifacts.py --run-dir "$RUN_DIR" || true

echo
echo "Done."
