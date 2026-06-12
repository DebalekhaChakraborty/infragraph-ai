#!/usr/bin/env bash
set -euo pipefail

# ── Repo root ─────────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"

# ── Configuration ─────────────────────────────────────────────────────────────
RUN_DIR="${RUN_DIR:-/tmp/infragraph_grpo_runs/qwen3_4b_grpo_lora_amd_saved}"
ADAPTER_DIR="${ADAPTER_DIR:-/tmp/infragraph_qwen3_grpo_lora_adapter}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
LORA_NAME="${LORA_NAME:-infragraph}"
SAVE_FREQ="${SAVE_FREQ:-32}"
TEST_FREQ="${TEST_FREQ:-32}"

# ── Keep all heavy state in /tmp ──────────────────────────────────────────────
export TMPDIR="${TMPDIR:-/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/hf_cache}"

echo "============================================================"
echo " InfraGraph GRPO/vERL training + LoRA export pipeline"
echo "============================================================"
echo "  ROOT_DIR    : $ROOT_DIR"
echo "  RUN_DIR     : $RUN_DIR"
echo "  ADAPTER_DIR : $ADAPTER_DIR"
echo "  MODEL_ID    : $MODEL_ID"
echo "  LORA_NAME   : $LORA_NAME"
echo "  SAVE_FREQ   : $SAVE_FREQ  TEST_FREQ : $TEST_FREQ"
echo

# ── Preflight cleanup ─────────────────────────────────────────────────────────
echo "Preflight: stopping Ray/vLLM/vERL processes and clearing /tmp runtime files ..."
ray stop --force || true
pkill -f vllm    || true
pkill -f main_ppo || true
rm -rf /tmp/ray /tmp/vllm* 2>/dev/null || true
mkdir -p "$RUN_DIR"
mkdir -p "$(dirname "$ADAPTER_DIR")"
echo "  Done."
echo

# ── ROCm patches ──────────────────────────────────────────────────────────────
echo "Applying ROCm runtime patches..."
bash scripts/amd_rocm/patch_verl_runtime_for_rocm.sh

# ── Training ──────────────────────────────────────────────────────────────────
echo
echo "Starting Qwen3-4B GRPO/vERL training ..."
echo "  RUN_DIR   = $RUN_DIR"
echo "  SAVE_FREQ = $SAVE_FREQ  TEST_FREQ = $TEST_FREQ"
echo

MODEL_ID="$MODEL_ID" \
SAVE_FREQ="$SAVE_FREQ" \
TEST_FREQ="$TEST_FREQ" \
RUN_DIR="$RUN_DIR" \
INFRAGRAPH_RUN_REAL_VERL=1 \
bash training/verl_grpo/train_qwen3_grpo.sh

# ── Artifact scan ─────────────────────────────────────────────────────────────
echo
echo "Scanning for LoRA/checkpoint artifacts ..."
"$PYTHON" training/verl_grpo/find_lora_adapter_artifacts.py --run-dir "$RUN_DIR" || true

# ── LoRA export — required; script exits non-zero on failure ──────────────────
echo
echo "Exporting PEFT LoRA adapter from vERL/FSDP actor checkpoint ..."
"$PYTHON" training/verl_grpo/export_lora_adapter.py \
  --run-dir      "$RUN_DIR" \
  --base-model   "$MODEL_ID" \
  --output-dir   "$ADAPTER_DIR"

# ── Adapter file verification ─────────────────────────────────────────────────
echo
echo "Verifying exported adapter ..."

for required_file in adapter_config.json adapter_model.safetensors; do
  if [[ ! -f "$ADAPTER_DIR/$required_file" ]]; then
    echo "[ERROR] Expected file not found: $ADAPTER_DIR/$required_file"
    exit 1
  fi
  echo "  OK  $ADAPTER_DIR/$required_file"
done

# ── Tensor count verification ─────────────────────────────────────────────────
"$PYTHON" - <<PY
import sys, json
from pathlib import Path
from safetensors.torch import load_file

adapter = Path("${ADAPTER_DIR}")
cfg     = json.loads((adapter / "adapter_config.json").read_text())
weights = load_file(str(adapter / "adapter_model.safetensors"))
tensor_count = len(weights)

print(f"  adapter_dir  : {adapter}")
print(f"  peft_type    : {cfg.get('peft_type')}")
print(f"  base_model   : {cfg.get('base_model_name_or_path')}")
print(f"  rank / alpha : {cfg.get('r')} / {cfg.get('lora_alpha')}")
print(f"  tensor_count : {tensor_count}")

if tensor_count == 0:
    print("[ERROR] adapter_model.safetensors contains 0 tensors — export failed.")
    sys.exit(1)
PY

# ── vLLM serving command ──────────────────────────────────────────────────────
echo
echo "============================================================"
echo " Adapter verified. Ready for vLLM LoRA serving."
echo "============================================================"
echo
echo "vLLM LoRA serving command:"
echo "  VLLM_USE_TRITON_FLASH_ATTN=0 \\"
echo "  vllm serve $MODEL_ID \\"
echo "    --served-model-name Qwen3-4B \\"
echo "    --enable-lora \\"
echo "    --lora-modules ${LORA_NAME}=${ADAPTER_DIR} \\"
echo "    --host 0.0.0.0 \\"
echo "    --port 8000 \\"
echo "    --gpu-memory-utilization 0.55 \\"
echo "    --max-model-len 2048"
echo

# ── Streamlit env vars ────────────────────────────────────────────────────────
echo "Streamlit env vars (set these after vLLM is serving):"
echo "  export INFRAGRAPH_QWEN_BASE_URL=\"http://127.0.0.1:8000/v1\""
echo "  export INFRAGRAPH_QWEN_MODEL=\"$LORA_NAME\""
echo "  export INFRAGRAPH_LORA_ADAPTER_PATH=\"$ADAPTER_DIR\""
echo "  streamlit run app/streamlit_app.py"
echo

echo "------------------------------------------------------------"
echo "WARNING: Do not commit RUN_DIR or ADAPTER_DIR artifacts to Git."
echo "  RUN_DIR     : $RUN_DIR"
echo "  ADAPTER_DIR : $ADAPTER_DIR"
echo "------------------------------------------------------------"
