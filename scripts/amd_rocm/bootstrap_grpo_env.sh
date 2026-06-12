#!/usr/bin/env bash
# bootstrap_grpo_env.sh — Install the AMD ROCm vERL/GRPO training stack.
#
# Checks what is already present before touching anything.
# Safe to re-run on an existing environment — will skip already-installed
# components rather than clobbering a working ROCm torch installation.
#
# Usage:
#   bash scripts/amd_rocm/bootstrap_grpo_env.sh
#
# After this script completes, apply runtime patches:
#   bash scripts/amd_rocm/patch_verl_runtime_for_rocm.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"

echo "========================================================"
echo " InfraGraph AI — AMD ROCm GRPO environment bootstrap"
echo "========================================================"
echo

# ── 1. Check Python ───────────────────────────────────────────────────────────
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "[ERROR] Python not found. Set PYTHON= to the correct interpreter."
  exit 1
fi
echo "[1/6] Python: $($PYTHON --version)"
echo

# ── 2. PyTorch ROCm ───────────────────────────────────────────────────────────
echo "[2/6] Checking PyTorch / ROCm ..."
if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)")
  HIP_VER=$("$PYTHON" -c "import torch; print(getattr(torch.version, 'hip', None))")
  echo "  torch $TORCH_VER  hip=$HIP_VER — already installed, skipping"
else
  echo "  torch not found or CUDA/ROCm unavailable."
  echo "  Installing ROCm 6.0 wheel ..."
  pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
  if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    echo "  torch installed and ROCm available."
  else
    echo "  [WARN] torch installed but torch.cuda.is_available() is False."
    echo "         Training may fail. Check ROCm driver and HSA_OVERRIDE_GFX_VERSION."
  fi
fi
echo

# ── 3. Streamlit + version-pinned base deps ───────────────────────────────────
# These pins are required on the AMD hackathon environment:
#   starlette<0.49.0  — compatibility with the installed uvicorn
#   protobuf<7.0.0    — avoids binary incompatibility with torch/grpc
#   numpy<2.3         — required by several packages that haven't updated yet
echo "[3/6] Installing Streamlit and pinned base deps ..."
pip install streamlit --ignore-installed blinker
pip install "starlette<0.49.0" "protobuf<7.0.0" "numpy<2.3"
echo "  Done."
echo

# ── 4. vERL training utilities ────────────────────────────────────────────────
echo "[4/6] Installing vERL training utilities ..."
pip install --no-cache-dir \
  tensordict torchdata codetiming hydra-core omegaconf \
  ray pandas pyarrow datasets accelerate peft

# Specific version range for transformers that is stable on the AMD hackathon env
pip install --no-cache-dir \
  "transformers>=4.46.0,<4.57.0" \
  "peft>=0.14.0" \
  "accelerate>=1.0.0"
echo "  Done."
echo

# ── 5. vLLM ───────────────────────────────────────────────────────────────────
echo "[5/6] Checking vLLM ..."
if "$PYTHON" -c "import vllm" >/dev/null 2>&1; then
  VLLM_VER=$("$PYTHON" -c "import vllm; print(vllm.__version__)")
  echo "  vLLM $VLLM_VER — already installed, skipping"
else
  echo "  Installing vLLM ..."
  pip install vllm
  echo "  Done."
fi
echo

# ── 6. vERL ───────────────────────────────────────────────────────────────────
echo "[6/6] Checking vERL ..."
if "$PYTHON" -c "import verl" >/dev/null 2>&1; then
  echo "  vERL — already installed"
else
  echo "  Installing vERL from source (--no-deps to avoid clobbering torch/vLLM) ..."
  pip install --no-cache-dir --no-deps "git+https://github.com/volcengine/verl.git"
  if ! "$PYTHON" -c "import verl" >/dev/null 2>&1; then
    echo "  [ERROR] vERL install failed."
    exit 1
  fi
  echo "  vERL installed."
fi

if "$PYTHON" -c "import verl.trainer.main_ppo" >/dev/null 2>&1; then
  echo "  verl.trainer.main_ppo: available"
else
  echo "  [WARN] verl.trainer.main_ppo not found in installed vERL."
  echo "         Try: pip install --force-reinstall --no-deps git+https://github.com/volcengine/verl.git"
fi
echo

# ── Summary ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo " Environment summary"
echo "========================================================"
"$PYTHON" - <<'EOF'
import sys
print(f"  Python:   {sys.version.split()[0]}")
try:
    import torch
    hip = getattr(torch.version, "hip", None)
    print(f"  torch:    {torch.__version__}  hip={hip}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device:   {torch.cuda.get_device_name(0)}")
except ImportError:
    print("  torch:    NOT installed")
try:
    import vllm
    print(f"  vLLM:     {vllm.__version__}")
except ImportError:
    print("  vLLM:     NOT installed")
try:
    import verl
    print("  vERL:     installed")
except ImportError:
    print("  vERL:     NOT installed")
try:
    import datasets
    print(f"  datasets: {datasets.__version__}")
except ImportError:
    print("  datasets: NOT installed")
try:
    import transformers
    print(f"  transformers: {transformers.__version__}")
except ImportError:
    print("  transformers: NOT installed")
EOF
echo
echo "Bootstrap complete."
echo
echo "Next step — apply runtime patches:"
echo "  bash scripts/amd_rocm/patch_verl_runtime_for_rocm.sh"
echo
echo "Then run GRPO training (dry-run by default):"
echo "  bash training/verl_grpo/train_qwen3_grpo.sh"
echo
echo "To launch a real training run:"
echo "  INFRAGRAPH_RUN_REAL_VERL=1 bash training/verl_grpo/train_qwen3_grpo.sh"
