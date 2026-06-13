#!/usr/bin/env bash
# setup_rca_cpu_env.sh — Lightweight CPU setup for InfraGraph RCA.
#
# Installs base and training requirements for running Topology RCA on CPU.
# Does NOT install torch or torch_geometric by default — pass --with-gnn
# to also install the Enterprise GNN stack (requires a CPU-compatible torch).
#
# Usage:
#   bash scripts/setup_rca_cpu_env.sh              # Topology RCA only
#   bash scripts/setup_rca_cpu_env.sh --with-gnn   # + Enterprise GNN RCA
#
# Override interpreter:
#   PYTHON=python3.11 bash scripts/setup_rca_cpu_env.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WITH_GNN=0
for arg in "$@"; do
    if [[ "$arg" == "--with-gnn" ]]; then
        WITH_GNN=1
    fi
done

echo "========================================================"
echo " InfraGraph AI — CPU RCA environment setup"
if [[ $WITH_GNN -eq 1 ]]; then
    echo " (--with-gnn: Enterprise GNN stack included)"
fi
echo "========================================================"
echo

# ── 1. Python version ────────────────────────────────────────────────────────
echo "[1/4] Python version"
"$PYTHON" --version
echo

# ── 2. Base requirements ─────────────────────────────────────────────────────
echo "[2/4] Installing base requirements (requirements/requirements.txt)"
"$PYTHON" -m pip install -q -r requirements/requirements.txt
echo "      done."
echo

# ── 3. Training requirements (non-GPU subset) ────────────────────────────────
echo "[3/4] Installing training requirements (requirements/requirements-training.txt)"
"$PYTHON" -m pip install -q -r requirements/requirements-training.txt
echo "      done."
echo

# ── 4. GNN requirements (optional) ───────────────────────────────────────────
if [[ $WITH_GNN -eq 1 ]]; then
    echo "[4/4] Checking torch before GNN install"
    if ! "$PYTHON" -c "import torch; print('  torch', torch.__version__)" 2>/dev/null; then
        echo "      torch not found — installing CPU torch first"
        "$PYTHON" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    fi
    echo "[4/4] Installing GNN requirements (requirements/requirements-gnn.txt)"
    "$PYTHON" -m pip install -q -r requirements/requirements-gnn.txt
    echo "      done."
else
    echo "[4/4] Skipping GNN requirements (pass --with-gnn to include)"
fi
echo

# ── Verify imports ────────────────────────────────────────────────────────────
echo "Verifying imports..."
"$PYTHON" - <<PY
import joblib, sklearn, networkx
print(f"  joblib:       {joblib.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  networkx:     {networkx.__version__}")
PY

if [[ $WITH_GNN -eq 1 ]]; then
    "$PYTHON" - <<'PY'
import torch, torch_geometric
print(f"  torch:           {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  torch_geometric: {torch_geometric.__version__}")
PY
fi

echo
echo "========================================================"
if [[ $WITH_GNN -eq 1 ]]; then
    echo " RCA + GNN environment ready (CPU)."
else
    echo " Topology RCA environment ready (CPU)."
    echo " Run with --with-gnn to add Enterprise GNN support."
fi
echo "========================================================"
