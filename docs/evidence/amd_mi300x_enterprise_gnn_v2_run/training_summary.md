# AMD MI300X Enterprise GNN RCA V2 Run Evidence

- Environment: AMD hackathon Jupyter / ROCm
- GPU: AMD Instinct MI300X
- Model: EnterpriseRcaTemporalRelGNN
- Architecture: RelationAwareTemporalGraphSAGE
- Dataset: 80 generated enterprise RCA scenarios
- Train/val/test: 64 / 8 / 8
- Feature dimension: 54
- Edge type present: true
- Epochs: 80
- Hidden dimension: 64
- Layers: 2
- Output artifact: `model_artifacts/enterprise_gnn_rca_v2/enterprise_gnn_v2_rca.pt`
- Inference scenario: `enterprise_v3_0079`
- Predicted root: `DC-FW-01`
- Ground truth: `DC-FW-01`
- Cross-diagram edges: `8`
- Output: `outputs/enterprise_gnn_rca_v2/enterprise_v3_0079_enterprise_gnn_v2_rca_result.json`

Honest note: These metrics are from generated/synthetic enterprise RCA benchmarks with known ground truth, not production customer telemetry.
