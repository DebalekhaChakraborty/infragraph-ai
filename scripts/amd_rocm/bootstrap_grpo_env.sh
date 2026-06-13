#!/usr/bin/env bash
# bootstrap_grpo_env.sh — Install the AMD ROCm vERL/GRPO training stack.
#
# PURPOSE: Environment setup for the old GRPO/vERL reinforcement learning
# training path only. This installs torch (ROCm), vERL, vLLM, and related
# training utilities.
#
# This script does NOT start vLLM, does NOT run remediation generation, and
# is NOT required for serving the SOP-grounded SFT LoRA adapter.
#
# For the current SOP-grounded LoRA serving and reset flow, see instead:
#   bash scripts/amd_rocm/start_qwen_sop_lora_vllm.sh       (Terminal 1)
#   bash scripts/amd_rocm/generate_qwen_sop_remediation_after_reset.sh  (Terminal 2)
#   docs/amd_rocm_qwen_sop_lora_reset.md
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
echo "[1/7] Python: $($PYTHON --version)"
echo

# ── 2. PyTorch ROCm ───────────────────────────────────────────────────────────
echo "[2/7] Checking PyTorch / ROCm ..."
if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)")
  HIP_VER=$("$PYTHON" -c "import torch; print(getattr(torch.version, 'hip', None))")
  echo "  torch $TORCH_VER  hip=$HIP_VER — already installed, skipping"
else
  echo "  torch not found or CUDA/ROCm unavailable."
  echo "  Installing ROCm 6.0 wheel ..."
  "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
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
#   numpy==2.2.6      — vLLM's numba dependency requires NumPy <= 2.2
echo "[3/7] Installing Streamlit and pinned base deps ..."
"$PYTHON" -m pip install streamlit --ignore-installed blinker
"$PYTHON" -m pip install "starlette<0.49.0" "protobuf<7.0.0" "numpy==2.2.6"
echo "  Done."
echo

# ── 4. vERL training utilities ────────────────────────────────────────────────
echo "[4/7] Installing vERL training utilities ..."
"$PYTHON" -m pip install --no-cache-dir \
  tensordict torchdata codetiming hydra-core omegaconf \
  ray pandas pyarrow datasets accelerate peft

# Specific version range for transformers that is stable on the AMD hackathon env
"$PYTHON" -m pip install --no-cache-dir \
  "transformers>=4.46.0,<4.57.0" \
  "peft>=0.14.0" \
  "accelerate>=1.0.0"
echo "  Done."
echo

# ── 5. vLLM ───────────────────────────────────────────────────────────────────
echo "[5/7] Checking vLLM ..."
if "$PYTHON" -c "import vllm" >/dev/null 2>&1; then
  VLLM_VER=$("$PYTHON" -c "import vllm; print(vllm.__version__)")
  echo "  vLLM $VLLM_VER — already installed, skipping"
else
  echo "  Installing vLLM ..."
  "$PYTHON" -m pip install vllm
  echo "  Done."
fi
echo

# ── 6. vERL ───────────────────────────────────────────────────────────────────
echo "[6/7] Checking vERL ..."
if "$PYTHON" -c "import verl" >/dev/null 2>&1; then
  echo "  vERL — already installed"
else
  echo "  Installing vERL from source (--no-deps to avoid clobbering torch/vLLM) ..."
  "$PYTHON" -m pip install --no-cache-dir --no-deps "git+https://github.com/volcengine/verl.git"
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
  echo "         Try: $PYTHON -m pip install --force-reinstall --no-deps git+https://github.com/volcengine/verl.git"
fi
echo

# ── 7. Force-pin NumPy 2.2.6 ─────────────────────────────────────────────────
# vLLM's numba dependency requires NumPy <= 2.2.  Any of the packages installed
# above may have pulled in a newer numpy.  Force-reinstall last to guarantee
# the pin regardless of what the other packages requested.
echo "[7/7] Force-pinning numpy==2.2.6 (numba/vLLM requirement) ..."
"$PYTHON" -m pip install --no-cache-dir --force-reinstall "numpy==2.2.6"
echo "  Done."
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
    import numpy
    print(f"  numpy:    {numpy.__version__}")
except ImportError:
    print("  numpy:    NOT installed")
try:
    import numba
    print(f"  numba:    {numba.__version__}")
except ImportError:
    print("  numba:    NOT installed (ok if vLLM does not use it on this platform)")
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
echo "NOTE: This environment is for GRPO/vERL training only."
echo "      For SOP-grounded LoRA serving (normal reset flow), see:"
echo "        docs/amd_rocm_qwen_sop_lora_reset.md"
echo
echo "Next step — apply runtime patches (GRPO/vERL only):"
echo "  bash scripts/amd_rocm/patch_verl_runtime_for_rocm.sh"
echo
echo "Then run GRPO training (dry-run by default):"
echo "  bash training/verl_grpo/train_qwen3_grpo.sh"
echo
echo "To launch a real training run:"
echo "  INFRAGRAPH_RUN_REAL_VERL=1 bash training/verl_grpo/train_qwen3_grpo.sh"
