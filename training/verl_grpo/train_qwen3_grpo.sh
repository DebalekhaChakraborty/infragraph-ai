#!/usr/bin/env bash
# train_qwen3_grpo.sh — InfraGraph AI Qwen3-4B LoRA + GRPO/vERL training script
#
# Default: dry-run (prints the exact command).
# To launch a real training run:
#   INFRAGRAPH_RUN_REAL_VERL=1 bash training/verl_grpo/train_qwen3_grpo.sh
#
# vERL uses verl.trainer.main_ppo with algorithm.adv_estimator=grpo.
# It does NOT use verl.trainer.main_grpo (that module does not exist).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_DIR="training/verl_grpo"
DATA_DIR="$SCRIPT_DIR/data"
TRAIN_JSONL="$DATA_DIR/rca_remediation_rl_train.jsonl"
EVAL_JSONL="$DATA_DIR/rca_remediation_rl_eval.jsonl"
TRAIN_PARQ="$DATA_DIR/verl_train.parquet"
EVAL_PARQ="$DATA_DIR/verl_eval.parquet"
RUN_DIR="$SCRIPT_DIR/runs/qwen3_4b_grpo_lora_amd"
REWARD_MODULE="$SCRIPT_DIR/verl_reward.py"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"

echo "========================================================"
echo " InfraGraph AI — Qwen3-4B LoRA + GRPO/vERL"
echo "========================================================"
echo

# ── Step 1: Build JSONL alignment dataset ────────────────────────────────────
echo "[1/4] Building RCA alignment dataset ..."
python "$SCRIPT_DIR/build_rca_rl_dataset.py"
echo

# ── Step 2: Convert JSONL to vERL parquet ────────────────────────────────────
echo "[2/4] Converting to vERL parquet format ..."
python "$SCRIPT_DIR/prepare_verl_dataset.py" \
  --train-jsonl "$TRAIN_JSONL" \
  --eval-jsonl  "$EVAL_JSONL"  \
  --train-parquet "$TRAIN_PARQ" \
  --eval-parquet  "$EVAL_PARQ"
echo

# ── Step 3: Check vERL installation ──────────────────────────────────────────
echo "[3/4] Checking vERL installation ..."

VERL_AVAILABLE=0
if python -c "import verl" >/dev/null 2>&1; then
  VERL_AVAILABLE=1
  echo "  vERL: installed"
else
  echo "  vERL: NOT installed"
fi

TRAINER_AVAILABLE=0
if [ "$VERL_AVAILABLE" -eq 1 ]; then
  if python -c "import verl.trainer.main_ppo" >/dev/null 2>&1; then
    TRAINER_AVAILABLE=1
    echo "  verl.trainer.main_ppo: available"
  else
    echo "  verl.trainer.main_ppo: NOT found"
    echo "  Available trainer modules:"
    python - <<'EOF'
import importlib, pkgutil, sys
try:
    import verl.trainer as _t
    pkg_path = getattr(_t, "__path__", [])
    for info in pkgutil.walk_packages(pkg_path, prefix="verl.trainer."):
        print("    " + info.name)
except Exception as e:
    print("    (could not inspect verl.trainer:", e, ")")
EOF
  fi
fi

echo

# ── Training command ──────────────────────────────────────────────────────────
TRAIN_CMD="python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$TRAIN_PARQ \
  data.val_files=$EVAL_PARQ \
  data.train_batch_size=2 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  custom_reward_function.path=$REWARD_MODULE \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path=$MODEL_ID \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_shm=False \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=3e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=2048 \
  trainer.n_gpus_per_node=1 \
  trainer.total_epochs=1 \
  trainer.logger=[console] \
  trainer.default_local_dir=$RUN_DIR"

# ── Step 4: Run or dry-run ────────────────────────────────────────────────────
RUN_REAL="${INFRAGRAPH_RUN_REAL_VERL:-0}"

echo "[4/4] Training ..."
if [ "$RUN_REAL" != "1" ]; then
  echo
  echo "  DRY-RUN MODE (set INFRAGRAPH_RUN_REAL_VERL=1 to launch real training)"
  echo
  if [ "$VERL_AVAILABLE" -eq 0 ]; then
    echo "  vERL is not installed. Install guidance:"
    echo "    pip install verl vllm"
    echo "    # For AMD ROCm:"
    echo "    pip install torch --index-url https://download.pytorch.org/whl/rocm6.0"
  fi
  if [ "$TRAINER_AVAILABLE" -eq 0 ] && [ "$VERL_AVAILABLE" -eq 1 ]; then
    echo "  WARNING: verl.trainer.main_ppo was not found in the installed vERL."
    echo "  Check your vERL version or install from source:"
    echo "    pip install git+https://github.com/volcengine/verl.git"
  fi
  echo
  echo "  Training command that would run:"
  echo "  -------------------------------------------------------"
  echo "  $TRAIN_CMD"
  echo "  -------------------------------------------------------"
  echo
  # Write dry-run summary
  python "$SCRIPT_DIR/write_training_summary.py" \
    --run-dir "$RUN_DIR" \
    --dry-run
  echo
  echo "  Parquet files ready for a real vERL run:"
  echo "    $TRAIN_PARQ"
  echo "    $EVAL_PARQ"
  echo
  echo "  Status: scaffold complete — no training was run."
  exit 0
fi

# Real run
if [ "$VERL_AVAILABLE" -eq 0 ]; then
  echo "[ERROR] INFRAGRAPH_RUN_REAL_VERL=1 was set but vERL is not installed."
  echo "        Install: pip install verl vllm"
  exit 1
fi
if [ "$TRAINER_AVAILABLE" -eq 0 ]; then
  echo "[ERROR] verl.trainer.main_ppo is not available in the installed vERL."
  echo "        Install from source: pip install git+https://github.com/volcengine/verl.git"
  exit 1
fi

mkdir -p "$RUN_DIR"
echo "  Launching: $TRAIN_CMD"
echo
eval "$TRAIN_CMD"

echo
echo "Training complete. Writing summary ..."
python "$SCRIPT_DIR/write_training_summary.py" --run-dir "$RUN_DIR"
echo "Done."
