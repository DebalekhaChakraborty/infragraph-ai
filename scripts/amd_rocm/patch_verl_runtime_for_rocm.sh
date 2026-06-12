#!/usr/bin/env bash
# patch_verl_runtime_for_rocm.sh — Apply and verify ROCm-specific runtime fixes
# for InfraGraph AI vERL GRPO training.
#
# Two modes of protection:
#   1. Config-level overrides (in train_qwen3_grpo.sh) — verified here.
#   2. Source-level patches to the installed vERL package — applied here.
#      Backups are written before any file is modified.
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
echo " InfraGraph AI — ROCm runtime patch + verification"
echo "========================================================"
echo

PASS=0
FAIL=0

check() {
  local label="$1" result="$2" detail="$3"
  if [ "$result" = "ok" ]; then
    echo "  [OK]   $label"
    PASS=$((PASS + 1))
  else
    echo "  [FAIL] $label"
    [ -n "$detail" ] && echo "         $detail"
    FAIL=$((FAIL + 1))
  fi
}

# ── Part 1: Config-level verification ────────────────────────────────────────
echo "--- Config checks (train_qwen3_grpo.sh) ---"
echo

grep -q "free_cache_engine=False" training/verl_grpo/train_qwen3_grpo.sh 2>/dev/null \
  && check "free_cache_engine=False (prevents --enable_sleep_mode on ROCm)" "ok" "" \
  || check "free_cache_engine=False" "fail" \
     "Add to TRAIN_CMD: actor_rollout_ref.rollout.free_cache_engine=False"

grep -q "enforce_eager=True" training/verl_grpo/train_qwen3_grpo.sh 2>/dev/null \
  && check "enforce_eager=True (skips CUDA graph capture on ROCm)" "ok" "" \
  || check "enforce_eager=True" "fail" \
     "Add to TRAIN_CMD: actor_rollout_ref.rollout.enforce_eager=True"

REF_OK=0; ROLLOUT_OK=0
grep -q "ref.log_prob_micro_batch_size_per_gpu=1" \
  training/verl_grpo/train_qwen3_grpo.sh 2>/dev/null && REF_OK=1
grep -q "rollout.log_prob_micro_batch_size_per_gpu=1" \
  training/verl_grpo/train_qwen3_grpo.sh 2>/dev/null && ROLLOUT_OK=1
if [ "$REF_OK" -eq 1 ] && [ "$ROLLOUT_OK" -eq 1 ]; then
  check "log_prob_micro_batch_size_per_gpu set for ref + rollout" "ok" ""
else
  MSG=""
  [ "$REF_OK"     -eq 0 ] && MSG="${MSG}ref.log_prob_micro_batch_size_per_gpu=1 missing. "
  [ "$ROLLOUT_OK" -eq 0 ] && MSG="${MSG}rollout.log_prob_micro_batch_size_per_gpu=1 missing."
  check "log_prob_micro_batch_size_per_gpu" "fail" "$MSG"
fi

grep -q "datasets.Dataset\|from datasets import Dataset" \
  training/verl_grpo/prepare_verl_dataset.py 2>/dev/null \
  && check "prepare_verl_dataset writes struct columns via datasets.Dataset" "ok" "" \
  || check "prepare_verl_dataset struct columns" "fail" \
     "extra_info must be a dict, not a JSON string. Update _write_parquet() to use
         datasets.Dataset.from_list(rows).to_parquet(path)."

echo

# ── Part 2: Source-level patches to installed vERL ───────────────────────────
echo "--- Source patches (installed vERL package) ---"
echo

"$PYTHON" - <<'PY'
from pathlib import Path

# ── Patch 1: disable sleep mode in vLLM async server ─────────────────────────
SERVER = Path("/usr/local/lib/python3.12/dist-packages/verl/workers/rollout/vllm_rollout/vllm_async_server.py")

if not SERVER.exists():
    print(f"  [SKIP] vllm_async_server.py not found at expected path — skipping sleep-mode patch")
    print(f"         (config override free_cache_engine=False is still active)")
else:
    text = SERVER.read_text()
    if "InfraGraph ROCm patch: unsupported on current platform" in text:
        print(f"  [OK]   vllm_async_server.py — sleep-mode patch already applied")
    else:
        bak = SERVER.with_suffix(".py.bak_infragraph_rocm")
        bak.write_text(text)
        text = text.replace(
            'logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")',
            'logger.info("enable_sleep_mode: False  # forced for ROCm")',
        )
        text = text.replace(
            '"enable_sleep_mode": self.config.enable_sleep_mode,',
            '"enable_sleep_mode": False,  # InfraGraph ROCm patch: unsupported on current platform',
        )
        SERVER.write_text(text)
        print(f"  [OK]   Patched vllm_async_server.py  (backup: {bak.name})")

# ── Patch 2: tolerate string extra_info/reward_model in rl_dataset.py ─────────
DATASET = Path("/usr/local/lib/python3.12/dist-packages/verl/utils/dataset/rl_dataset.py")

if not DATASET.exists():
    print(f"  [SKIP] rl_dataset.py not found at expected path — skipping extra_info patch")
    print(f"         (prepare_verl_dataset.py already writes dicts via datasets.Dataset)")
else:
    text = DATASET.read_text()
    if "InfraGraph patch: tolerate JSON-string extra_info" in text:
        print(f"  [OK]   rl_dataset.py — extra_info patch already applied")
    else:
        OLD = '        index = row_dict.get("extra_info", {}).get("index", 0)\n'
        NEW = '''        # InfraGraph patch: tolerate JSON-string extra_info / reward_model.
        extra_info = row_dict.get("extra_info", {})
        if isinstance(extra_info, str):
            import json as _json
            try:
                extra_info = _json.loads(extra_info)
            except Exception:
                extra_info = {"raw_extra_info": extra_info}
        if not isinstance(extra_info, dict):
            extra_info = {"raw_extra_info": str(extra_info)}
        row_dict["extra_info"] = extra_info

        reward_model = row_dict.get("reward_model", {})
        if isinstance(reward_model, str):
            import json as _json
            try:
                reward_model = _json.loads(reward_model)
            except Exception:
                reward_model = {"raw_reward_model": reward_model}
        row_dict["reward_model"] = reward_model

        index = extra_info.get("index", 0)
'''
        if OLD in text:
            bak = DATASET.with_suffix(".py.bak_infragraph_extra_info")
            bak.write_text(text)
            DATASET.write_text(text.replace(OLD, NEW, 1))
            print(f"  [OK]   Patched rl_dataset.py  (backup: {bak.name})")
        else:
            print(f"  [WARN] rl_dataset.py — anchor line not found; patch skipped.")
            print(f"         Anchor:  {OLD.strip()!r}")
            print(f"         The datasets.Dataset parquet fix in prepare_verl_dataset.py")
            print(f"         should be sufficient; this source patch is a belt-and-suspenders backup.")
PY

echo

# ── Summary ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo " Config checks: $PASS passed, $FAIL failed"
echo "========================================================"
echo
if [ "$FAIL" -eq 0 ]; then
  echo "All config checks passed. Ready for AMD ROCm GRPO training."
  echo
  echo "Dry-run (no GPU needed):"
  echo "  bash training/verl_grpo/train_qwen3_grpo.sh"
  echo
  echo "Real training run:"
  echo "  INFRAGRAPH_RUN_REAL_VERL=1 bash training/verl_grpo/train_qwen3_grpo.sh"
  echo
  echo "Override model:"
  echo "  MODEL_ID=Qwen/Qwen3-1.7B INFRAGRAPH_RUN_REAL_VERL=1 \\"
  echo "    bash training/verl_grpo/train_qwen3_grpo.sh"
else
  echo "Fix the $FAIL config issue(s) above before running training."
  exit 1
fi
