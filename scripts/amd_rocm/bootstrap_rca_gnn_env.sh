#!/usr/bin/env bash
# bootstrap_rca_gnn_env.sh — Install InfraGraph RCA + GNN dependencies on AMD ROCm.
#
# Safe to re-run after an ephemeral Jupyter environment restart.
# Does NOT reinstall or overwrite a working ROCm torch installation.
# Exits with a clear message if torch is missing so the user knows
# to run bootstrap_grpo_env.sh (or use the platform-provided torch) first.
#
# Usage:
#   bash scripts/amd_rocm/bootstrap_rca_gnn_env.sh
#
# Override interpreter:
#   PYTHON=python3.11 bash scripts/amd_rocm/bootstrap_rca_gnn_env.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "========================================================"
echo " InfraGraph AI — RCA / GNN environment setup"
echo "========================================================"
echo

# ── 1. Python version ────────────────────────────────────────────────────────
echo "[1/5] Python version"
"$PYTHON" --version
echo

# ── 2. Base requirements ─────────────────────────────────────────────────────
echo "[2/5] Installing base requirements (requirements/requirements.txt)"
"$PYTHON" -m pip install -q -r requirements/requirements.txt
echo "      done."
echo

# ── 3. Training requirements ─────────────────────────────────────────────────
echo "[3/5] Installing training requirements (requirements/requirements-training.txt)"
"$PYTHON" -m pip install -q -r requirements/requirements-training.txt
echo "      done."
echo

# ── 4. Torch check (do not install — must come from AMD wheel or image) ──────
echo "[4/5] Checking torch availability"
if ! "$PYTHON" - <<'PY'
import sys
try:
    import torch
    hip = getattr(torch.version, "hip", None)
    cuda = torch.cuda.is_available()
    print(f"      torch {torch.__version__}  cuda={cuda}  hip={hip}")
except ImportError:
    print("MISSING", file=sys.stderr)
    sys.exit(1)
PY
then
    echo
    echo "[ERROR] torch is missing or broken in this environment."
    echo "        Install ROCm-compatible torch first, then re-run this script."
    echo "        Option A (AMD ROCm full stack):"
    echo "          bash scripts/amd_rocm/bootstrap_grpo_env.sh"
    echo "        Option B (platform image already provides torch — nothing to do):"
    echo "          Confirm with: $PYTHON -c 'import torch; print(torch.__version__)'"
    exit 1
fi
echo

# ── 5. GNN requirements ───────────────────────────────────────────────────────
echo "[5/5] Installing GNN requirements (requirements/requirements-gnn.txt)"
"$PYTHON" -m pip install -q -r requirements/requirements-gnn.txt
echo "      done."
echo

# ── Verify all critical imports ───────────────────────────────────────────────
echo "Verifying imports..."
"$PYTHON" - <<'PY'
import joblib, sklearn, networkx, torch, torch_geometric

hip  = getattr(torch.version, "hip", None)
cuda = torch.cuda.is_available()

print(f"  joblib:          {joblib.__version__}")
print(f"  scikit-learn:    {sklearn.__version__}")
print(f"  networkx:        {networkx.__version__}")
print(f"  torch:           {torch.__version__}  cuda={cuda}  hip={hip}")
print(f"  torch_geometric: {torch_geometric.__version__}")
PY

echo
echo "========================================================"
echo " RCA / GNN environment ready."
echo "========================================================"
