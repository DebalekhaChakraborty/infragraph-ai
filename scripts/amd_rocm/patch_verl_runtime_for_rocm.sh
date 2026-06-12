#!/usr/bin/env bash
# patch_verl_runtime_for_rocm.sh — Verify and document the ROCm-specific
# runtime fixes required for InfraGraph AI vERL GRPO training.
#
# What this script does:
#   1. Verifies that prepare_verl_dataset.py writes extra_info as a dict
#      (not a JSON string), which is required by vERL's rl_dataset.py.
#   2. Verifies that train_qwen3_grpo.sh includes all required ROCm overrides.
#   3. Prints instructions for any remaining manual steps.
#
# Usage:
#   bash scripts/amd_rocm/patch_verl_runtime_for_rocm.sh
#
# Run this after bootstrap_grpo_env.sh and before training.
set -euo pipefail

PYTHON="${PYTHON:-python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "========================================================"
echo " InfraGraph AI — ROCm runtime patch verification"
echo "========================================================"
echo

PASS=0
FAIL=0

check() {
  local label="$1"
  local result="$2"   # "ok" or "fail"
  local detail="$3"
  if [ "$result" = "ok" ]; then
    echo "  [OK]   $label"
    PASS=$((PASS + 1))
  else
    echo "  [FAIL] $label"
    echo "         $detail"
    FAIL=$((FAIL + 1))
  fi
}

# ── Check 1: extra_info written as dict, not JSON string ─────────────────────
echo "[1/4] Checking prepare_verl_dataset.py — extra_info serialisation ..."
if grep -q "datasets.Dataset\|from datasets import Dataset" \
     training/verl_grpo/prepare_verl_dataset.py 2>/dev/null; then
  check "prepare_verl_dataset uses datasets.Dataset (struct columns, not strings)" "ok" ""
else
  check "prepare_verl_dataset uses datasets.Dataset" "fail" \
    "File uses pandas/pyarrow string serialisation. vERL's rl_dataset.py calls
         row_dict.get('extra_info', {}).get('index', 0) and expects a dict.
         Fix: update _write_parquet() in prepare_verl_dataset.py to use
         datasets.Dataset.from_list(rows).to_parquet(path)."
fi
echo

# ── Check 2: free_cache_engine=False (disables --enable_sleep_mode) ──────────
echo "[2/4] Checking train_qwen3_grpo.sh — vLLM sleep-mode override ..."
if grep -q "free_cache_engine=False" training/verl_grpo/train_qwen3_grpo.sh 2>/dev/null; then
  check "free_cache_engine=False present (prevents --enable_sleep_mode on ROCm)" "ok" ""
else
  check "free_cache_engine=False" "fail" \
    "vLLM passes --enable_sleep_mode which is unsupported on ROCm.
         Add to TRAIN_CMD in train_qwen3_grpo.sh:
           actor_rollout_ref.rollout.free_cache_engine=False"
fi
echo

# ── Check 3: enforce_eager=True (disables CUDA graph capture) ─────────────────
echo "[3/4] Checking train_qwen3_grpo.sh — ROCm eager mode ..."
if grep -q "enforce_eager=True" training/verl_grpo/train_qwen3_grpo.sh 2>/dev/null; then
  check "enforce_eager=True present (skips CUDA graph capture, safe on ROCm)" "ok" ""
else
  check "enforce_eager=True" "fail" \
    "CUDA graph capture can fail on ROCm. Add to TRAIN_CMD:
           actor_rollout_ref.rollout.enforce_eager=True"
fi
echo

# ── Check 4: log_prob_micro_batch_size_per_gpu set for both ref and rollout ───
echo "[4/4] Checking train_qwen3_grpo.sh — log_prob_micro_batch_size_per_gpu ..."
REF_OK=0
ROLLOUT_OK=0
grep -q "ref.log_prob_micro_batch_size_per_gpu=1" \
  training/verl_grpo/train_qwen3_grpo.sh 2>/dev/null && REF_OK=1
grep -q "rollout.log_prob_micro_batch_size_per_gpu=1" \
  training/verl_grpo/train_qwen3_grpo.sh 2>/dev/null && ROLLOUT_OK=1

if [ "$REF_OK" -eq 1 ] && [ "$ROLLOUT_OK" -eq 1 ]; then
  check "log_prob_micro_batch_size_per_gpu set for ref and rollout" "ok" ""
else
  MSG=""
  [ "$REF_OK" -eq 0 ] && MSG="${MSG}actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 missing. "
  [ "$ROLLOUT_OK" -eq 0 ] && MSG="${MSG}actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 missing."
  check "log_prob_micro_batch_size_per_gpu" "fail" \
    "vERL requires both ref and rollout micro-batch sizes to be set explicitly.
         $MSG"
fi
echo

# ── Summary ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo " Results: $PASS passed, $FAIL failed"
echo "========================================================"
echo
if [ "$FAIL" -eq 0 ]; then
  echo "All checks passed. The environment is ready for AMD ROCm GRPO training."
  echo
  echo "Dry-run (no GPU needed):"
  echo "  bash training/verl_grpo/train_qwen3_grpo.sh"
  echo
  echo "Real training run:"
  echo "  INFRAGRAPH_RUN_REAL_VERL=1 bash training/verl_grpo/train_qwen3_grpo.sh"
  echo
  echo "Override model (e.g. smaller variant for memory testing):"
  echo "  MODEL_ID=Qwen/Qwen3-1.7B INFRAGRAPH_RUN_REAL_VERL=1 \\"
  echo "    bash training/verl_grpo/train_qwen3_grpo.sh"
else
  echo "Fix the $FAIL issue(s) above before running training."
  exit 1
fi
