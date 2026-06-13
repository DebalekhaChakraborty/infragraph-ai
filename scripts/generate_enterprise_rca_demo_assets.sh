#!/usr/bin/env bash
# generate_enterprise_rca_demo_assets.sh
#
# Build clean, demo-safe event correlation + Enterprise GNN RCA preloaded
# outputs for the four primary demo scenarios.
#
# Outputs written to:
#   assets/preloaded/event_correlation/enterprise_v3_<id>.json
#   assets/preloaded/enterprise_gnn_rca/enterprise_v3_<id>.json
#
# Usage:
#   bash scripts/generate_enterprise_rca_demo_assets.sh
#
# Override Python interpreter:
#   PYTHON=python3 bash scripts/generate_enterprise_rca_demo_assets.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCENARIOS=(
    "enterprise_v3_0000"
    "enterprise_v3_0072"
    "enterprise_v3_0073"
    "enterprise_v3_0074"
)

echo "===================================================="
echo " InfraGraph AI — Generate Enterprise RCA Demo Assets"
echo "===================================================="
echo "Scenarios : ${SCENARIOS[*]}"
echo

for SCENARIO_ID in "${SCENARIOS[@]}"; do
    CASE_ID="ent_${SCENARIO_ID}"
    CLUSTER_FILE="assets/preloaded/event_correlation/${SCENARIO_ID}.json"

    echo "----------------------------------------------------"
    echo "Scenario : ${SCENARIO_ID}"
    echo "Case     : ${CASE_ID}"
    echo

    echo "  [1/2] Building event correlation clusters..."
    "$PYTHON" scripts/build_event_correlation_clusters.py \
        --case-id "${CASE_ID}"

    echo "  [2/2] Predicting Enterprise GNN RCA with cluster evidence..."
    "$PYTHON" scripts/predict_enterprise_gnn_rca.py \
        --scenario-id "${SCENARIO_ID}" \
        --cluster-file "${CLUSTER_FILE}"

    echo "  Done: ${SCENARIO_ID}"
    echo
done

echo "===================================================="
echo "Validating all preloaded outputs..."
"$PYTHON" scripts/validate_rca_outputs.py --verbose
echo "===================================================="
echo " All demo assets generated and validated."
echo "===================================================="
