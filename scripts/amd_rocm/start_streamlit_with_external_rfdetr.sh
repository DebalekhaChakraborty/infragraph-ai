#!/usr/bin/env bash
set -euo pipefail

# Run this from the detector/base shell before activating the Streamlit venv.
# It captures the current shell's Python for RF-DETR subprocess inference, then
# starts Streamlit from the application venv.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export INFRAGRAPH_RFDETR_PYTHON="$(command -v python)"

STREAMLIT_VENV="${INFRAGRAPH_STREAMLIT_VENV:-/workspace/shared/venvs/infragraph-app}"
STREAMLIT_HOST="${INFRAGRAPH_STREAMLIT_HOST:-0.0.0.0}"
STREAMLIT_PORT="${INFRAGRAPH_STREAMLIT_PORT:-8501}"

echo "RF-DETR detector Python: ${INFRAGRAPH_RFDETR_PYTHON}"
echo "Streamlit venv: ${STREAMLIT_VENV}"

if [[ ! -f "${STREAMLIT_VENV}/bin/activate" ]]; then
  echo "Streamlit venv not found: ${STREAMLIT_VENV}/bin/activate" >&2
  echo "Set INFRAGRAPH_STREAMLIT_VENV to the application venv path." >&2
  exit 1
fi

cd "${REPO_ROOT}"
# shellcheck disable=SC1090
source "${STREAMLIT_VENV}/bin/activate"

python -m streamlit run app/streamlit_app.py \
  --server.address="${STREAMLIT_HOST}" \
  --server.port="${STREAMLIT_PORT}"

