#!/usr/bin/env bash
# upload_full_codebase_to_s3.sh
#
# Uploads the entire infragraph-ai codebase to S3, preserving directory
# structure under s3://my-hackathons/infragraph-ai/.
#
# Safe by design:
#   - No --delete flag: nothing is removed from S3 or from the local repo.
#   - Only adds or updates files that are new or changed.
#
# Excluded (not uploaded):
#   __pycache__, *.pyc, *.pyo, *.pyd
#   .ipynb_checkpoints, .pytest_cache, .mypy_cache, .ruff_cache
#   .venv, venv, env
#
# Usage:
#   bash upload_full_codebase_to_s3.sh
#
# Run from either the parent directory (one level above infragraph-ai/)
# or from inside the infragraph-ai/ repo root.

set -euo pipefail

BUCKET="my-hackathons"
S3_PREFIX="infragraph-ai"

# ── Detect repo root ──────────────────────────────────────────────────────────
if [ -d "./infragraph-ai/src" ] || [ -d "./infragraph-ai/scripts" ]; then
  # Running from the parent directory
  LOCAL_DIR="./infragraph-ai"
elif [ -d "./src" ] && [ -d "./scripts" ]; then
  # Running from inside the repo
  LOCAL_DIR="."
else
  echo "[ERROR] Cannot find the infragraph-ai repo."
  echo "  Run this script from the parent directory of infragraph-ai/"
  echo "  or from inside the infragraph-ai/ repo root."
  exit 1
fi

S3_URI="s3://${BUCKET}/${S3_PREFIX}"

echo "===================================================="
echo " InfraGraph AI -- Upload full codebase to S3"
echo "===================================================="
echo "Local source : ${LOCAL_DIR}"
echo "S3 target    : ${S3_URI}"
echo
echo "Excluded     : __pycache__  *.pyc  *.pyo  *.pyd"
echo "               .ipynb_checkpoints  .pytest_cache"
echo "               .mypy_cache  .ruff_cache"
echo "               .venv  venv  env"
echo
echo "Note: uploads without --delete."
echo "      No files are removed from the local repo or from S3."
echo

# ── Check AWS CLI ─────────────────────────────────────────────────────────────
if ! command -v aws >/dev/null 2>&1; then
  echo "[ERROR] AWS CLI is not installed or not in PATH."
  echo "        Install it from https://aws.amazon.com/cli/"
  exit 1
fi

# ── Check AWS identity ────────────────────────────────────────────────────────
echo "[1/3] Checking AWS identity..."
aws sts get-caller-identity >/dev/null
echo "AWS credentials OK."
echo

# ── Show local size ───────────────────────────────────────────────────────────
echo "[2/3] Local codebase size (approximate — includes excluded dirs):"
du -sh "${LOCAL_DIR}" || true
echo

# ── Upload ────────────────────────────────────────────────────────────────────
echo "[3/3] Uploading to S3..."
echo

aws s3 sync "${LOCAL_DIR}" "${S3_URI}" \
  --only-show-errors \
  --exclude "*__pycache__*"        \
  --exclude "*.pyc"                \
  --exclude "*.pyo"                \
  --exclude "*.pyd"                \
  --exclude ".ipynb_checkpoints/*" \
  --exclude "*/.ipynb_checkpoints/*" \
  --exclude ".pytest_cache/*"      \
  --exclude "*/.pytest_cache/*"    \
  --exclude ".mypy_cache/*"        \
  --exclude "*/.mypy_cache/*"      \
  --exclude ".ruff_cache/*"        \
  --exclude "*/.ruff_cache/*"      \
  --exclude ".venv/*"              \
  --exclude "*/.venv/*"            \
  --exclude "venv/*"               \
  --exclude "*/venv/*"             \
  --exclude "env/*"                \
  --exclude "*/env/*"

echo
echo "===================================================="
echo "Upload complete."
echo "S3 location : ${S3_URI}"
echo "===================================================="
