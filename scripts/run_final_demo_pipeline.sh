#!/usr/bin/env bash
# run_final_demo_pipeline.sh
#
# Full InfraGraph AI demo pipeline:
#   1. Generate Enterprise GNN RCA preloaded assets (real GNN required)
#   2. Validate RCA outputs
#   3. Generate remediation outputs (Qwen/vLLM preferred; template fallback)
#   4. Validate remediation outputs
#
# Usage:
#   bash scripts/run_final_demo_pipeline.sh               # prefer Qwen/vLLM
#   INFRAGRAPH_TEMPLATE_ONLY=1 bash scripts/run_final_demo_pipeline.sh  # template only
#
# Environment variables:
#   INFRAGRAPH_TEMPLATE_ONLY=1    Use template mode for remediation (no vLLM required)
#   PYTHON                        Python interpreter (default: python)
#   INFRAGRAPH_QWEN_BASE_URL      vLLM base URL (default: http://localhost:8000/v1)
#   INFRAGRAPH_QWEN_MODEL         vLLM model name (default: infragraph)

set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "===================================================="
echo " InfraGraph AI — Full Demo Pipeline"
echo "===================================================="
echo "Repo     : $REPO_ROOT"
echo "Python   : $PYTHON"
echo "Template : ${INFRAGRAPH_TEMPLATE_ONLY:-0}"
echo ""

# ── Step 1: Generate Enterprise GNN RCA assets ───────────────────────────────
echo "----------------------------------------------------"
echo "[1/4] Generating Enterprise GNN RCA demo assets..."
echo "----------------------------------------------------"
"$PYTHON" scripts/generate_enterprise_rca_demo_assets.py
echo ""

# ── Step 2: Validate RCA outputs ─────────────────────────────────────────────
echo "----------------------------------------------------"
echo "[2/4] Validating RCA outputs..."
echo "----------------------------------------------------"
"$PYTHON" scripts/validate_rca_outputs.py --verbose
echo ""

# ── Step 3: Generate remediation outputs ─────────────────────────────────────
echo "----------------------------------------------------"
echo "[3/4] Generating remediation outputs..."
echo "----------------------------------------------------"
if [ "${INFRAGRAPH_TEMPLATE_ONLY:-0}" = "1" ]; then
    echo "  Mode: template-only (INFRAGRAPH_TEMPLATE_ONLY=1)"
    "$PYTHON" scripts/generate_remediation_demo_assets.py --template-only
else
    echo "  Mode: prefer-qwen (fallback to template if vLLM unavailable)"
    "$PYTHON" scripts/generate_remediation_demo_assets.py --prefer-qwen
fi
echo ""

# ── Step 4: Validate remediation outputs ─────────────────────────────────────
echo "----------------------------------------------------"
echo "[4/4] Validating remediation outputs..."
echo "----------------------------------------------------"
"$PYTHON" scripts/validate_remediation_outputs.py --verbose
echo ""

echo "===================================================="
echo " Pipeline complete."
echo "===================================================="
