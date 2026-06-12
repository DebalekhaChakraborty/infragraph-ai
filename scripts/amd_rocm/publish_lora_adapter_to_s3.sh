#!/usr/bin/env bash
# publish_lora_adapter_to_s3.sh
#
# Uploads the exported InfraGraph GRPO LoRA adapter to S3.
#
# Adapter binaries are NOT committed to Git — they live in S3.
# Git stores only code, evidence documents, and restore instructions.
#
# Run this after scripts/amd_rocm/run_qwen3_grpo_success_path.sh completes
# successfully and the adapter has been verified locally.
#
# Usage:
#   scripts/amd_rocm/publish_lora_adapter_to_s3.sh
#
#   Override defaults:
#   ADAPTER_DIR=/path/to/adapter S3_URI=s3://bucket/path/ \
#       scripts/amd_rocm/publish_lora_adapter_to_s3.sh

set -euo pipefail

ADAPTER_DIR="${ADAPTER_DIR:-/tmp/infragraph_qwen3_grpo_lora_adapter}"
S3_URI="${S3_URI:-s3://my-hackathons/infragraph-ai/model_artifacts/qwen3_grpo_lora_adapter/}"

echo "============================================================"
echo " InfraGraph LoRA Adapter — S3 Publish"
echo "============================================================"
echo "  ADAPTER_DIR : $ADAPTER_DIR"
echo "  S3_URI      : $S3_URI"
echo

# ── Validate AWS CLI ──────────────────────────────────────────────────────────
if ! command -v aws &>/dev/null; then
    echo "[ERROR] AWS CLI not found."
    echo "        Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# ── Validate local adapter directory ─────────────────────────────────────────
if [[ ! -d "$ADAPTER_DIR" ]]; then
    echo "[ERROR] Adapter directory not found: $ADAPTER_DIR"
    echo "        Run scripts/amd_rocm/run_qwen3_grpo_success_path.sh first."
    exit 1
fi

# ── Validate required files ───────────────────────────────────────────────────
all_present=true
for required_file in adapter_config.json adapter_model.safetensors README.md; do
    if [[ ! -f "$ADAPTER_DIR/$required_file" ]]; then
        echo "[ERROR] Required file missing: $ADAPTER_DIR/$required_file"
        all_present=false
    else
        echo "  OK  $ADAPTER_DIR/$required_file"
    fi
done

if [[ "$all_present" != "true" ]]; then
    echo
    echo "[ERROR] One or more required adapter files are missing. Aborting upload."
    exit 1
fi

echo

# ── Upload ────────────────────────────────────────────────────────────────────
echo "Uploading to $S3_URI ..."
aws s3 sync "$ADAPTER_DIR/" "$S3_URI" --delete --no-progress
echo "  Upload complete."
echo

# ── Verify S3 contents ────────────────────────────────────────────────────────
echo "S3 contents after upload:"
aws s3 ls "$S3_URI"
echo

# ── Restore command ───────────────────────────────────────────────────────────
echo "============================================================"
echo " To restore this adapter on a new machine:"
echo "============================================================"
echo "  aws s3 sync s3://my-hackathons/infragraph-ai/model_artifacts/qwen3_grpo_lora_adapter/ \\"
echo "    /tmp/infragraph_qwen3_grpo_lora_adapter/ --no-progress"
echo
echo "  See docs/evidence/amd_qwen3_grpo_run/s3_lora_adapter_restore.md"
echo "  for the full restore + vLLM serving + Streamlit env var instructions."
echo "============================================================"
