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
#   PYTHON                       Python interpreter (default: python)

set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "===================================================="
echo " InfraGraph AI — RAG + Remediation Bootstrap"
echo "===================================================="
echo "Repo   : $REPO_ROOT"
echo "Python : $PYTHON"
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
    echo "  Mode: template-only"
    "$PYTHON" scripts/generate_remediation_demo_assets.py \
        --build-kb-index \
        --template-only \
        --strict-kb
else
    echo "  Mode: prefer-qwen with template fallback"
    "$PYTHON" scripts/generate_remediation_demo_assets.py \
        --build-kb-index \
        --prefer-qwen \
        --strict-kb
fi
echo ""

# ── Step 5: Validate remediation outputs ─────────────────────────────────────
echo "[5/5] Validating remediation outputs..."
"$PYTHON" scripts/validate_remediation_outputs.py --verbose
echo ""

echo "===================================================="
echo " RAG + remediation bootstrap complete."
echo "===================================================="
