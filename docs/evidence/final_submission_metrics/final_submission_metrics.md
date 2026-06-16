# InfraGraph AI — Final Submission Metrics
Generated: 2026-06-16T01:18:15.010586+00:00

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
| GNN V2 inference latency | not measured |
| Qwen model | Qwen/Qwen3-4B |
| Alignment | LoRA rank 16 + GRPO/vERL (32/32 steps) |
| Qwen tokens | not measured live |
| Qwen latency | unavailable |
| AMD GPU evidence | MI300X / ROCm — GPU 100% utilization, VRAM ~42%, Power ~278W (training evidence) |
| Detector metrics | not claimed — eval report present but status='evaluation_failed', processed images=0 |

---

## A. GNN V2 Training Evidence
- Source: `D:\My Folders\Hackathons\infragraph-ai\model_artifacts\enterprise_gnn_rca_v2\training_report.json`
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
- Status: `inference_error`
- Errors: ['Run 1: exit code 1: ', 'Run 2: exit code 1: ', 'Run 3: exit code 1: ']

## C. GNN Training Benchmark
- Status: `not_run`
- Note: Pass --run-training-benchmark to enable.

## D. Qwen/vLLM Latency
- Live latency: unavailable
- Model: Qwen/Qwen3-4B
- Note: INFRAGRAPH_QWEN_BASE_URL not set. Committed GRPO training evidence included below.
- Committed evidence: `D:\My Folders\Hackathons\infragraph-ai\training\verl_grpo\runs\qwen3_4b_grpo_lora_amd\completion_evidence.md`

## E. AMD GPU Telemetry
- Available: False
- Command used: None
- Timestamp: 2026-06-16T01:18:17.657992+00:00
- Note: AMD telemetry command unavailable in this environment; see committed AMD evidence files.

## F. RF-DETR Evidence
- Status: `eval_report_present`
- Source: `D:\My Folders\Hackathons\infragraph-ai\reports\rfdetr_v3_eval\rfdetr_v3_eval_report.json`
- Report status: `evaluation_failed`
- Split: val
- Processed images: 0
- **Claim guidance:** No usable metrics — strict accuracy not claimed. Run eval on AMD/ROCm env with rfdetr installed.
- **Detector accuracy is not claimed from this report** — inference unavailable or no images were successfully processed.
- First inference errors:
  - `enterprise_v3_0064__branch_topology.png`: Non-zero exit code 4: 
  - `enterprise_v3_0064__datacenter_topology.png`: Non-zero exit code 4: 
  - `enterprise_v3_0064__wan_topology.png`: Non-zero exit code 4: 
  - `enterprise_v3_0065__app_db_topology.png`: Non-zero exit code 4: 
  - `enterprise_v3_0065__branch_topology.png`: Non-zero exit code 4: 
