# InfraGraph AI — Final Submission Metrics
Generated: 2026-06-16T00:26:46.827872+00:00

## Slide-4 Summary Table
| Category | Evidence |
|----------|----------|
| Diagram model | RF-DETR-supported detector + verified fallback + vision connector extraction |
| RCA model | EnterpriseRcaTemporalRelGNN / Temporal Relation-Aware GraphSAGE |
| RCA dataset | 80 generated enterprise RCA scenarios |
| Split | 64 train / 8 val / 8 test |
| GNN feature dim | 54 |
| GNN V2 best epoch | 5 |
| GNN V2 test top-1 | 1.0 (synthetic/generated enterprise benchmark) |
| GNN V2 inference latency | 39982.8 ms avg (33129.5–48591.4 ms range) |
| Qwen model | Qwen/Qwen3-4B |
| Alignment | LoRA rank 16 + GRPO/vERL (32/32 steps) |
| Qwen tokens | 655 (api_reported) |
| Qwen latency | 3312 ms |
| AMD GPU evidence | Live telemetry captured via `amd-smi` |
| Detector metrics | precision=0.0000, recall=0.0000, F1=N/A (prototype benchmark) |

---

## A. GNN V2 Training Evidence
- Source: `/workspace/shared/infragraph-ai/model_artifacts/enterprise_gnn_rca_v2/training_report.json`
- Status: `ok`
- Model type: EnterpriseRcaTemporalRelGNN
- Architecture: RelationAwareTemporalGraphSAGE
- Graphs: 80 (train=64 / val=8 / test=8)
- Epochs: 80, best epoch: 5
- Best val MRR: 1.0
- Test metrics: {"top1": 1.0, "top3": 1.0, "mrr": 1.0}
- uses_edge_type: True
- uses_temporal_features: True

## B. GNN V2 Inference Latency
- Status: `ok`
- Runs: 3/3
- Min: 33129.5 ms
- Avg: 39982.8 ms
- Max: 48591.4 ms
- Command: `/usr/bin/python /workspace/shared/infragraph-ai/scripts/run_enterprise_gnn_v2_inference.py --scenario-id enterprise_v3_0079 --split test`

## C. GNN Training Benchmark
- Status: `not_run`
- Note: Pass --run-training-benchmark to enable.

## D. Qwen/vLLM Latency
- Live latency: available
- Model: infragraph
- Endpoint: http://127.0.0.1:8000/v1/chat/completions
- Latency: 3312 ms
- Tokens: 655 (api_reported)

## E. AMD GPU Telemetry
- Available: True
- Command used: amd-smi
- Timestamp: 2026-06-16T00:28:50.459844+00:00
- Output snippet:
```
+------------------------------------------------------------------------------+
| AMD-SMI 26.0.0+37d158ab      amdgpu version: 6.16.13  ROCm version: 7.0.0    |
| Platform: Linux Baremetal                                                    |
|-------------------------------------+----------------------------------------|
| BDF                        GPU-Name | Mem-Uti   Temp   UEC       Power-Usage |
| GPU  HIP-ID  OAM-ID  Partition-Mode | GFX-Uti    Fan               Mem-Usage |
|=====================================+========================================|
| 0000:1b:00.0 ...Instinct MI300X OAM | N/A        N/A   0           N/A/750 W |
|   0       0       1        SPX/NPS1 | N/A        N/A        149820/196592 MB |
+-------------------------------------+--------------------------------
```

## F. RF-DETR Evidence
- Status: `eval_report_present`
- Source: `/workspace/shared/infragraph-ai/reports/rfdetr_v3_eval/rfdetr_v3_eval_report.json`
- Report status: `completed`
- Split: val
- Processed images: 30
- Precision: 0.0000
- Recall: 0.0000
- F1: N/A
- mAP@0.5: 0.0000
