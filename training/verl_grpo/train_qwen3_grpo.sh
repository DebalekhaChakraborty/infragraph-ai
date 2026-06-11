#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="training/verl_grpo/sample_config.yaml"
RUN_DIR="training/verl_grpo/runs"
TRAIN_DATA="training/verl_grpo/data/rca_remediation_rl_train.jsonl"
EVAL_DATA="training/verl_grpo/data/rca_remediation_rl_eval.jsonl"

mkdir -p "$RUN_DIR"

echo "InfraGraph AI Qwen3 GRPO/vERL scaffold"
echo "Config: $CONFIG"
echo "Run dir: $RUN_DIR"

python training/verl_grpo/build_rca_rl_dataset.py

if ! python -c "import verl" >/dev/null 2>&1; then
  echo
  echo "vERL is not installed. This scaffold did not start training."
  echo "Install guidance:"
  echo "  pip install verl vllm"
  echo "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.0"
  echo
  echo "Training command that would run:"
  echo "python -m verl.trainer.main_grpo --config $CONFIG --data.train_files $TRAIN_DATA --data.val_files $EVAL_DATA --trainer.default_local_dir $RUN_DIR"
  exit 0
fi

echo "vERL detected. Launching GRPO training scaffold."
python -m verl.trainer.main_grpo \
  --config "$CONFIG" \
  --data.train_files "$TRAIN_DATA" \
  --data.val_files "$EVAL_DATA" \
  --trainer.default_local_dir "$RUN_DIR"
