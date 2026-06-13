#!/usr/bin/env bash
set -euo pipefail

BUCKET="my-hackathons"
S3_PREFIX="infragraph-ai/model_artifacts"

# Detect local model_artifacts path
if [ -d "./infragraph-ai/model_artifacts" ]; then
  LOCAL_DIR="./infragraph-ai/model_artifacts"
elif [ -d "./model_artifacts" ]; then
  LOCAL_DIR="./model_artifacts"
else
  echo "[ERROR] model_artifacts folder not found."
  echo "Expected one of:"
  echo "  ./infragraph-ai/model_artifacts"
  echo "  ./model_artifacts"
  exit 1
fi

S3_URI="s3://${BUCKET}/${S3_PREFIX}"

echo "===================================================="
echo " InfraGraph AI — Upload model_artifacts to S3"
echo "===================================================="
echo "Local folder : ${LOCAL_DIR}"
echo "S3 target    : ${S3_URI}"
echo

# Check AWS CLI
if ! command -v aws >/dev/null 2>&1; then
  echo "[ERROR] AWS CLI is not installed or not in PATH."
  echo "Install/configure AWS CLI first."
  exit 1
fi

# Check AWS identity
echo "[1/3] Checking AWS identity..."
aws sts get-caller-identity >/dev/null
echo "AWS credentials OK."
echo

# Show local folder size
echo "[2/3] Local artifact size:"
du -sh "${LOCAL_DIR}" || true
echo

# Upload
echo "[3/3] Uploading to S3..."
aws s3 sync "${LOCAL_DIR}" "${S3_URI}" \
  --only-show-errors

echo
echo "Upload completed successfully."
echo "S3 location: ${S3_URI}"
