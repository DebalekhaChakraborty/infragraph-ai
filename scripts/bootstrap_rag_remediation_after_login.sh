#!/usr/bin/env bash
# bootstrap_rag_remediation_after_login.sh
#
# Rebuild SOP/KB RAG index and regenerate SOP-grounded remediation
# after Jupyter/GPU relogin.
#
# Usage:
#   INFRAGRAPH_TEMPLATE_ONLY=1 bash scripts/bootstrap_rag_remediation_after_login.sh
#   INFRAGRAPH_TEMPLATE_ONLY=0 bash scripts/bootstrap_rag_remediation_after_login.sh
#
# Environment variables:
#   INFRAGRAPH_TEMPLATE_ONLY=1   Template mode (no vLLM required) — default
#   INFRAGRAPH_TEMPLATE_ONLY=0   Prefer Qwen/vLLM with template fallback
#   INFRAGRAPH_KB_TOP_K          KB evidence chunks per scenario (default: 3)
#   INFRAGRAPH_INCLUDE_RAW       Include raw_model_output field (default: 0)
#   PYTHON                       Python interpreter (default: python)
#
# For the full Qwen LoRA reset flow (SOP-grounded adapter), see instead:
#   docs/amd_rocm_qwen_sop_lora_reset.md
#   bash scripts/amd_rocm/start_qwen_sop_lora_vllm.sh
#   bash scripts/amd_rocm/generate_qwen_sop_remediation_after_reset.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INFRAGRAPH_KB_TOP_K="${INFRAGRAPH_KB_TOP_K:-3}"
INFRAGRAPH_INCLUDE_RAW="${INFRAGRAPH_INCLUDE_RAW:-0}"

echo "===================================================="
echo " InfraGraph AI — RAG + Remediation Bootstrap"
echo "===================================================="
echo "Repo        : $REPO_ROOT"
echo "Python      : $PYTHON"
echo "Mode        : ${INFRAGRAPH_TEMPLATE_ONLY:-1} (1=template-only 0=prefer-qwen)"
echo "KB top-k    : $INFRAGRAPH_KB_TOP_K"
echo "Include raw : $INFRAGRAPH_INCLUDE_RAW"
echo ""

# ── Step 1: Install RAG requirements ─────────────────────────────────────────
echo "[1/5] Installing RAG requirements..."
"$PYTHON" -m pip install -r requirements/requirements-rag.txt
echo ""

# ── Step 2: Rebuild SOP/KB vector index ──────────────────────────────────────
echo "[2/5] Rebuilding SOP/KB vector index..."
"$PYTHON" scripts/build_kb_index.py --reset
echo ""

# ── Step 3: Validate existing RCA outputs ────────────────────────────────────
echo "[3/5] Validating existing RCA outputs..."
"$PYTHON" scripts/validate_rca_outputs.py --verbose
echo ""

# ── Step 4: Generate SOP-grounded remediation outputs ────────────────────────
echo "[4/5] Generating SOP-grounded remediation outputs..."
if [ "${INFRAGRAPH_TEMPLATE_ONLY:-1}" = "1" ]; then
    echo "  Mode        : template-only"
    echo "  KB top-k    : $INFRAGRAPH_KB_TOP_K"
    GENERATE_ARGS=(--build-kb-index --template-only --strict-kb --kb-top-k "$INFRAGRAPH_KB_TOP_K")
    if [ "$INFRAGRAPH_INCLUDE_RAW" = "1" ]; then
        GENERATE_ARGS+=(--include-raw)
        echo "  Include raw : yes"
    fi
    "$PYTHON" scripts/generate_remediation_demo_assets.py "${GENERATE_ARGS[@]}"
else
    echo "  Mode        : prefer-qwen (fallback=template)"
    echo "  KB top-k    : $INFRAGRAPH_KB_TOP_K"
    GENERATE_ARGS=(--build-kb-index --prefer-qwen --strict-kb --kb-top-k "$INFRAGRAPH_KB_TOP_K")
    if [ "$INFRAGRAPH_INCLUDE_RAW" = "1" ]; then
        GENERATE_ARGS+=(--include-raw)
        echo "  Include raw : yes"
    fi
    "$PYTHON" scripts/generate_remediation_demo_assets.py "${GENERATE_ARGS[@]}"
fi
echo ""

# ── Step 5: Validate remediation outputs ─────────────────────────────────────
echo "[5/5] Validating remediation outputs..."
"$PYTHON" scripts/validate_remediation_outputs.py --verbose
echo ""

echo "===================================================="
echo " RAG + remediation bootstrap complete."
echo "===================================================="
