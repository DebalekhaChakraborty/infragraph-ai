# InfraGraph AI — Product-Safe Claims

This document lists conservative, reusable claims for public documentation, demos, and portfolio writeups.

## 1. RF-DETR Diagram Intelligence

**Safe wording:**
RF-DETR-supported diagram intelligence with trained checkpoints, verified annotation fallback, and vision connector extraction.

**Avoid unless a completed evaluation report exists:**
RF-DETR mAP/accuracy/precision/recall equals a specific value.

**Evidence:**
- Checkpoint: `model_artifacts/rfdetr_v3/checkpoint_best_total.pth`
- Inference script: `scripts/run_rfdetr_inference.py`
- Runtime evidence: `reports/rfdetr_runtime_evidence/rfdetr_runtime_evidence.md`
- Evaluation script: `scripts/evaluate_rfdetr_v3_detector.py`
- Diagnostic eval report, when generated: `reports/rfdetr_v3_eval/rfdetr_v3_eval_report.md`

Use RF-DETR runtime evidence and annotated outputs to demonstrate detector capability. Do not quote mAP, precision, or recall unless a successful eval report with non-zero matched detections is available.

## 2. Enterprise Graph/GNN RCA

**Safe wording:**
RCA is not guessed by the LLM. RCA is performed using graph algorithms, engineered 54-dimensional temporal/topology/alert features, Enterprise GNN RCA V1 GraphSAGE, and Enterprise GNN RCA V2 Temporal Relation-Aware GraphSAGE.

**Architecture:**
- V1: 2-layer GraphSAGE, 54-dimensional features.
- V2: `EnterpriseRcaTemporalRelGNN`, a temporal-aware relation-aware GraphSAGE model with edge type and temporal features.
- V2 is not a fully dynamic temporal heterogeneous graph transformer.
- Benchmark dataset: 80 generated enterprise scenarios, 64/8/8 split.

**Evidence:**
- Training report: `model_artifacts/enterprise_gnn_rca_v2/training_report.json`
- Performance metrics: `docs/evidence/performance_metrics/performance_metrics.md`

## 3. Qwen/vLLM Downstream Remediation

**Safe wording:**
Qwen/vLLM is used after RCA for RAG/runbook-grounded remediation drafting, validation steps, rollback planning, escalation guidance, and ITSM-ready summaries. Remediation remains governance-reviewed and human approval-gated.

**Architecture:**
- Qwen/Qwen3-4B adapter work uses LoRA rank 16 + GRPO/vERL alignment artifacts.
- Qwen/vLLM is downstream of graph/GNN RCA.
- The LLM does not perform root cause identification.
- Retrieval is grounded against local network runbook/SOP knowledge.

**Evidence:**
- Reward evaluation: `training/verl_grpo/reward_eval_report.json`
- Adapter: `model_artifacts/qwen3_grpo_lora_adapter/`
- Performance metrics: `docs/evidence/performance_metrics/performance_metrics.md`

## 4. Performance Metrics

See: [`docs/evidence/performance_metrics/performance_metrics.md`](evidence/performance_metrics/performance_metrics.md)

Key guardrails:

- GNN V2 metrics are on a synthetic/generated enterprise benchmark, not production data.
- RF-DETR detector accuracy is not claimed unless `reports/rfdetr_v3_eval/rfdetr_v3_eval_report.md` is present and values are computed.
- LLM/Qwen/vLLM is downstream remediation AI, not the RCA engine.
- Hardware-specific telemetry, when present, is optional execution evidence and not a requirement for the default app demo.
