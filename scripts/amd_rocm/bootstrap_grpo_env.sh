#!/usr/bin/env bash
set -euo pipefail

echo "Installing minimal AMD hackathon dependencies..."

pip install streamlit --ignore-installed blinker
pip install "starlette<0.49.0" "protobuf<7.0.0" "numpy<2.3"

# Install vERL carefully. Do not reinstall torch/vLLM from scratch unless needed.
python - <<'PY'
try:
    import verl
    print("vERL already installed")
except Exception:
    raise SystemExit(1)
PY
if [ $? -ne 0 ]; then
  pip install --no-cache-dir --no-deps "git+https://github.com/volcengine/verl.git"
fi

pip install --no-cache-dir \
  tensordict torchdata codetiming hydra-core omegaconf ray pandas pyarrow datasets accelerate peft

pip install --no-cache-dir \
  "transformers>=4.46.0,<4.57.0" \
  "peft>=0.14.0" \
  "accelerate>=1.0.0"

echo "Environment install/check complete."
