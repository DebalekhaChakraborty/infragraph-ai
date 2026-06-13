#!/usr/bin/env bash
# start_qwen_sop_lora_vllm.sh
#
# Start vLLM with the SOP-grounded SFT LoRA adapter for InfraGraph AI.
#
# This is the CURRENT serving script for the SOP-grounded LoRA adapter
# trained with scripts/train_qwen_sop_lora.py.
#
# NOT for the old GRPO/vERL adapter. For GRPO adapter serving, see:
#   docs/evidence/amd_qwen3_grpo_run/s3_lora_adapter_restore.md
#   docs/amd_rocm_qwen_sop_lora_reset.md  (full reset workflow)
#
# Usage:
#   bash scripts/amd_rocm/start_qwen_sop_lora_vllm.sh
#
# Environment overrides:
#   INFRAGRAPH_LORA_ADAPTER_PATH            Adapter dir (default: model_artifacts/qwen_lora/infragraph_sop_grounded)
#   INFRAGRAPH_USE_TMP_ADAPTER_SYMLINK      Set to 0 to disable /tmp symlink (default: 1)
#   INFRAGRAPH_BASE_MODEL                   Base model ID (default: Qwen/Qwen3-4B)
#   INFRAGRAPH_SERVED_MODEL_NAME            served-model-name (default: Qwen3-4B)
#   INFRAGRAPH_VLLM_HOST                    Bind host (default: 0.0.0.0)
#   INFRAGRAPH_VLLM_PORT                    Bind port (default: 8000)
#   INFRAGRAPH_VLLM_GPU_MEMORY_UTILIZATION  GPU memory fraction (default: 0.75)
#   INFRAGRAPH_VLLM_MAX_MODEL_LEN           Max model length (default: 8192)
#   VLLM_USE_TRITON_FLASH_ATTN              Set to 0 for ROCm (default: 0)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# ── Resolve adapter path ──────────────────────────────────────────────────────
DEFAULT_ADAPTER_PATH="$ROOT_DIR/model_artifacts/qwen_lora/infragraph_sop_grounded"
REAL_ADAPTER_PATH="${INFRAGRAPH_LORA_ADAPTER_PATH:-$DEFAULT_ADAPTER_PATH}"

# Resolve to absolute path if given as relative
if [[ ! "$REAL_ADAPTER_PATH" = /* ]]; then
    REAL_ADAPTER_PATH="$ROOT_DIR/$REAL_ADAPTER_PATH"
fi

# ── Validate adapter directory ────────────────────────────────────────────────
if [[ ! -d "$REAL_ADAPTER_PATH" ]]; then
    echo "[ERROR] Adapter directory not found: $REAL_ADAPTER_PATH"
    echo
    echo "Train the SOP-grounded adapter first:"
    echo "  python scripts/expand_sop_grounded_qwen_training_data.py --strict-kb"
    echo "  python scripts/train_qwen_sop_lora.py --epochs 3 --bf16"
    echo
    echo "Or override to an existing adapter directory:"
    echo "  INFRAGRAPH_LORA_ADAPTER_PATH=/path/to/adapter \\"
    echo "    bash scripts/amd_rocm/start_qwen_sop_lora_vllm.sh"
    exit 1
fi

# Canonicalize to absolute path now that dir is confirmed
REAL_ADAPTER_PATH="$(cd "$REAL_ADAPTER_PATH" && pwd)"

# ── Validate adapter_config.json ──────────────────────────────────────────────
if [[ ! -f "$REAL_ADAPTER_PATH/adapter_config.json" ]]; then
    echo "[ERROR] adapter_config.json not found in: $REAL_ADAPTER_PATH"
    echo "        The adapter directory exists but looks incomplete."
    echo "        Re-run training: python scripts/train_qwen_sop_lora.py --epochs 3 --bf16"
    exit 1
fi
echo "  [OK] adapter_config.json"

# ── Validate adapter weights ──────────────────────────────────────────────────
if [[ -f "$REAL_ADAPTER_PATH/adapter_model.safetensors" ]]; then
    WEIGHTS_FILE="adapter_model.safetensors"
elif [[ -f "$REAL_ADAPTER_PATH/adapter_model.bin" ]]; then
    WEIGHTS_FILE="adapter_model.bin"
else
    echo "[ERROR] No adapter weights found in: $REAL_ADAPTER_PATH"
    echo "        Expected adapter_model.safetensors or adapter_model.bin"
    exit 1
fi
echo "  [OK] $WEIGHTS_FILE"
echo

# ── /tmp symlink ──────────────────────────────────────────────────────────────
TMP_SYMLINK="/tmp/infragraph_sop_grounded_lora"
USE_SYMLINK="${INFRAGRAPH_USE_TMP_ADAPTER_SYMLINK:-1}"

if [[ "$USE_SYMLINK" = "1" ]]; then
    # Remove stale symlink or directory
    if [[ -L "$TMP_SYMLINK" ]] || [[ -e "$TMP_SYMLINK" ]]; then
        rm -rf "$TMP_SYMLINK"
    fi
    ln -s "$REAL_ADAPTER_PATH" "$TMP_SYMLINK"
    echo "  [OK] Symlink          : $TMP_SYMLINK -> $REAL_ADAPTER_PATH"
    EFFECTIVE_ADAPTER_PATH="$TMP_SYMLINK"
else
    echo "  Symlink disabled (INFRAGRAPH_USE_TMP_ADAPTER_SYMLINK=0)"
    EFFECTIVE_ADAPTER_PATH="$REAL_ADAPTER_PATH"
fi
echo

# ── Print configuration summary ───────────────────────────────────────────────
echo "============================================================"
echo " InfraGraph AI -- SOP-Grounded LoRA vLLM Server"
echo "============================================================"
echo "  Real adapter path     : $REAL_ADAPTER_PATH"
echo "  Effective path        : $EFFECTIVE_ADAPTER_PATH"
echo "  Base model            : ${INFRAGRAPH_BASE_MODEL:-Qwen/Qwen3-4B}"
echo "  vLLM model alias      : infragraph"
echo "  Served model name     : ${INFRAGRAPH_SERVED_MODEL_NAME:-Qwen3-4B}"
echo "  Max model length      : ${INFRAGRAPH_VLLM_MAX_MODEL_LEN:-8192}"
echo "  GPU memory util       : ${INFRAGRAPH_VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
echo "  Port                  : ${INFRAGRAPH_VLLM_PORT:-8000}"
echo "  Host                  : ${INFRAGRAPH_VLLM_HOST:-0.0.0.0}"
echo "  TRITON_FLASH_ATTN     : ${VLLM_USE_TRITON_FLASH_ATTN:-0}"
echo
echo "After server starts, set these env vars for remediation generation:"
echo "  export INFRAGRAPH_QWEN_BASE_URL=\"http://127.0.0.1:${INFRAGRAPH_VLLM_PORT:-8000}/v1\""
echo "  export INFRAGRAPH_QWEN_MODEL=\"infragraph\""
echo "  export INFRAGRAPH_QWEN_TIMEOUT=\"240\""
echo "  export INFRAGRAPH_QWEN_MAX_TOKENS=\"1400\""
echo "  export INFRAGRAPH_QWEN_TEMPERATURE=\"0.0\""
echo "  export INFRAGRAPH_KB_TOP_K=\"3\""
echo
echo "Then in a second terminal:"
echo "  bash scripts/amd_rocm/generate_qwen_sop_remediation_after_reset.sh"
echo "============================================================"
echo

# ── Start vLLM ────────────────────────────────────────────────────────────────
VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-0}" \
exec vllm serve "${INFRAGRAPH_BASE_MODEL:-Qwen/Qwen3-4B}" \
  --served-model-name "${INFRAGRAPH_SERVED_MODEL_NAME:-Qwen3-4B}" \
  --enable-lora \
  --lora-modules "infragraph=${EFFECTIVE_ADAPTER_PATH}" \
  --host "${INFRAGRAPH_VLLM_HOST:-0.0.0.0}" \
  --port "${INFRAGRAPH_VLLM_PORT:-8000}" \
  --gpu-memory-utilization "${INFRAGRAPH_VLLM_GPU_MEMORY_UTILIZATION:-0.75}" \
  --max-model-len "${INFRAGRAPH_VLLM_MAX_MODEL_LEN:-8192}"
