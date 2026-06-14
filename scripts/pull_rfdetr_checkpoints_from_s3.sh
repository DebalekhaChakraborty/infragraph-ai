#!/usr/bin/env bash
set -euo pipefail

BUCKET_PREFIX="${1:-s3://my-hackathon/infragraph-ai/model_artifacts/rfdetr_v3}"
REPO_ROOT="${REPO_ROOT:-/workspace/shared/infragraph-ai}"
LOCAL_DIR="${REPO_ROOT}/model_artifacts/rfdetr_v3/model"

mkdir -p "${LOCAL_DIR}"

echo "Checking AWS identity..."
aws sts get-caller-identity >/dev/null

echo "Downloading RF-DETR checkpoints from:"
echo "  ${BUCKET_PREFIX}"
echo "to:"
echo "  ${LOCAL_DIR}"
echo

for f in checkpoint_best_ema.pth checkpoint_best_regular.pth checkpoint_best_total.pth; do
  echo "Downloading ${f}..."
  aws s3 cp "${BUCKET_PREFIX}/${f}" "${LOCAL_DIR}/${f}"
done

echo
echo "Downloaded files:"
ls -lh "${LOCAL_DIR}"

echo
echo "Recommended checkpoint:"
echo "${LOCAL_DIR}/checkpoint_best_total.pth"

echo
echo "Export this before starting RF-DETR/Streamlit:"
echo "export INFRAGRAPH_RFDETR_CHECKPOINT=\"${LOCAL_DIR}/checkpoint_best_total.pth\""
