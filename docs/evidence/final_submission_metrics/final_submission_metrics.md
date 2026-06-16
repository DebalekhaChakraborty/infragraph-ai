# InfraGraph AI — Final Submission Metrics
Generated: 2026-06-16T00:03:51.626051+00:00

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
| GNN V2 inference latency | 25093.9 ms avg (24932.7–25306.3 ms range) |
| Qwen model | Qwen/Qwen3-4B |
| Alignment | LoRA rank 16 + GRPO/vERL (32/32 steps) |
| Qwen tokens | not measured live |
| Qwen latency | unavailable |
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
- Min: 24932.7 ms
- Avg: 25093.9 ms
- Max: 25306.3 ms
- Command: `/usr/bin/python /workspace/shared/infragraph-ai/scripts/run_enterprise_gnn_v2_inference.py --scenario-id enterprise_v3_0079 --split test`

## C. GNN Training Benchmark
- Status: `not_run`
- Note: Pass --run-training-benchmark to enable.

## D. Qwen/vLLM Latency
- Live latency: unavailable
- Model: Qwen/Qwen3-4B
- Note: INFRAGRAPH_QWEN_BASE_URL not set. Committed GRPO training evidence included below.
- Committed evidence: `/workspace/shared/infragraph-ai/training/verl_grpo/runs/qwen3_4b_grpo_lora_amd/completion_evidence.md`

## E. AMD GPU Telemetry
- Available: True
- Command used: amd-smi
- Timestamp: 2026-06-16T00:05:07.102928+00:00
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
