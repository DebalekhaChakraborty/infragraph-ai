#!/usr/bin/env bash
# generate_qwen_sop_remediation_after_reset.sh
#
# Run the full InfraGraph AI SOP-grounded remediation pipeline after a
# Jupyter/GPU reset, once the vLLM server is already running.
#
# This is the CURRENT generation script for the SOP-grounded LoRA adapter.
# It expects the SOP-grounded vLLM server to already be running (Terminal 1).
#
# Usage:
#   Terminal 1: bash scripts/amd_rocm/start_qwen_sop_lora_vllm.sh
#   Terminal 2: bash scripts/amd_rocm/generate_qwen_sop_remediation_after_reset.sh
#
# Environment overrides:
#   INFRAGRAPH_QWEN_BASE_URL      vLLM base URL   (default: http://127.0.0.1:8000/v1)
#   INFRAGRAPH_QWEN_MODEL         Model alias     (default: infragraph)
#   INFRAGRAPH_QWEN_TIMEOUT       HTTP timeout s  (default: 240)
#   INFRAGRAPH_QWEN_MAX_TOKENS    Max output tok  (default: 1400)
#   INFRAGRAPH_QWEN_TEMPERATURE   Temperature     (default: 0.0)
#   INFRAGRAPH_KB_TOP_K           KB top-k chunks (default: 3)
#   INFRAGRAPH_INCLUDE_RAW        Include raw_model_output field (default: 1)
#   INFRAGRAPH_BUILD_KB_INDEX     Rebuild KB index before generation (default: 1)
#   PYTHON                        Python interpreter (default: python)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"

# ── Export env defaults ───────────────────────────────────────────────────────
export INFRAGRAPH_QWEN_BASE_URL="${INFRAGRAPH_QWEN_BASE_URL:-http://127.0.0.1:8000/v1}"
export INFRAGRAPH_QWEN_MODEL="${INFRAGRAPH_QWEN_MODEL:-infragraph}"
export INFRAGRAPH_QWEN_TIMEOUT="${INFRAGRAPH_QWEN_TIMEOUT:-240}"
export INFRAGRAPH_QWEN_MAX_TOKENS="${INFRAGRAPH_QWEN_MAX_TOKENS:-1400}"
export INFRAGRAPH_QWEN_TEMPERATURE="${INFRAGRAPH_QWEN_TEMPERATURE:-0.0}"

INFRAGRAPH_KB_TOP_K="${INFRAGRAPH_KB_TOP_K:-3}"
INFRAGRAPH_INCLUDE_RAW="${INFRAGRAPH_INCLUDE_RAW:-1}"
INFRAGRAPH_BUILD_KB_INDEX="${INFRAGRAPH_BUILD_KB_INDEX:-1}"

echo "============================================================"
echo " InfraGraph AI -- SOP-Grounded Remediation Generation"
echo "============================================================"
echo "  vLLM base URL  : $INFRAGRAPH_QWEN_BASE_URL"
echo "  Qwen model     : $INFRAGRAPH_QWEN_MODEL"
echo "  Timeout        : ${INFRAGRAPH_QWEN_TIMEOUT}s"
echo "  Max tokens     : $INFRAGRAPH_QWEN_MAX_TOKENS"
echo "  Temperature    : $INFRAGRAPH_QWEN_TEMPERATURE"
echo "  KB top-k       : $INFRAGRAPH_KB_TOP_K"
echo "  Include raw    : $INFRAGRAPH_INCLUDE_RAW"
echo "  Build KB index : $INFRAGRAPH_BUILD_KB_INDEX"
echo

# ── Preflight: check vLLM /models endpoint ────────────────────────────────────
echo "[preflight] Checking vLLM /models endpoint ..."
VLLM_MODELS_URL="${INFRAGRAPH_QWEN_BASE_URL%/}/models"

if ! curl -sf --max-time 10 "$VLLM_MODELS_URL" > /dev/null 2>&1; then
    echo
    echo "[ERROR] vLLM is not reachable at $VLLM_MODELS_URL"
    echo
    echo "Start the SOP-grounded vLLM server first (Terminal 1):"
    echo "  bash scripts/amd_rocm/start_qwen_sop_lora_vllm.sh"
    echo
    echo "Then re-run this script in Terminal 2."
    exit 1
fi

echo "  vLLM is up. Models available:"
curl -sf "$VLLM_MODELS_URL" \
  | "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); [print('    ', m.get('id','?')) for m in d.get('data', [])]" \
  || true
echo

# ── Build KB index ────────────────────────────────────────────────────────────
if [[ "$INFRAGRAPH_BUILD_KB_INDEX" = "1" ]]; then
    echo "------------------------------------------------------------"
    echo "[1/5] Rebuilding SOP/KB vector index..."
    "$PYTHON" scripts/build_kb_index.py --reset
    echo
fi

# ── Validate RCA outputs ──────────────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "[2/5] Validating RCA outputs..."
"$PYTHON" scripts/validate_rca_outputs.py --verbose
echo

# ── Generate SOP-grounded remediation ────────────────────────────────────────
echo "------------------------------------------------------------"
echo "[3/5] Generating SOP-grounded remediation (Qwen LoRA + strict KB)..."

GENERATE_ARGS=(
    --prefer-qwen
    --strict-kb
    --kb-top-k "$INFRAGRAPH_KB_TOP_K"
)
if [[ "$INFRAGRAPH_INCLUDE_RAW" = "1" ]]; then
    GENERATE_ARGS+=(--include-raw)
fi

"$PYTHON" scripts/generate_remediation_demo_assets.py "${GENERATE_ARGS[@]}"
echo

# ── Validate remediation outputs ──────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "[4/5] Validating remediation outputs..."
"$PYTHON" scripts/validate_remediation_outputs.py --verbose
echo

# ── Inspect remediation quality ───────────────────────────────────────────────
echo "------------------------------------------------------------"
echo "[5/5] Remediation quality inspection..."
"$PYTHON" scripts/inspect_remediation_quality.py
echo

# ── Final compact check ───────────────────────────────────────────────────────
echo "============================================================"
echo " Final output check"
echo "============================================================"
"$PYTHON" - <<'PY'
import json
from pathlib import Path

for f in sorted(Path("assets/preloaded/remediation").glob("enterprise_v3_*.json")):
    d = json.loads(f.read_text())
    print(f.name, "source=", d.get("remediation_source"), "model=", d.get("model"), "ok=", d.get("ok"))
PY
echo

echo "============================================================"
echo " Generation complete."
echo "============================================================"
echo
echo "Expected: all four files show"
echo "  source= qwen_vllm  model= infragraph  ok= True"
echo
echo "To launch Streamlit:"
echo "  export INFRAGRAPH_QWEN_BASE_URL=\"${INFRAGRAPH_QWEN_BASE_URL}\""
echo "  export INFRAGRAPH_QWEN_MODEL=\"${INFRAGRAPH_QWEN_MODEL}\""
echo "  export INFRAGRAPH_QWEN_MAX_TOKENS=\"${INFRAGRAPH_QWEN_MAX_TOKENS}\""
echo "  streamlit run app/streamlit_app.py"
