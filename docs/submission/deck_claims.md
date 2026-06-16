# InfraGraph AI — Deck-Safe Claims

## 1. RF-DETR Diagram Intelligence

**Safe wording:**
RF-DETR-supported diagram intelligence with trained checkpoints, verified annotation fallback, and vision connector extraction.

**Avoid unless eval report exists:**
RF-DETR mAP/accuracy/precision/recall = X.

**Evidence:**
- Checkpoint: `model_artifacts/rfdetr_v3/checkpoint_best_total.pth`
- Inference script: `scripts/run_rfdetr_inference.py`
- **Runtime evidence** (recommended for judges): `reports/rfdetr_runtime_evidence/rfdetr_runtime_evidence.md` — annotated outputs and detection counts per image; generate with `python scripts/generate_rfdetr_runtime_evidence.py`
- Evaluation script (diagnostic): `scripts/evaluate_rfdetr_v3_detector.py`
- Eval report (diagnostic, when generated): `reports/rfdetr_v3_eval/rfdetr_v3_eval_report.md`

Use RF-DETR runtime evidence and annotated outputs to demonstrate live detector capability. Do not quote mAP/precision/recall unless a successful eval report with non-zero matched detections is available.

## 2. Enterprise Graph/GNN RCA

**Safe wording:**
RCA is not guessed by the LLM. RCA is performed using graph algorithms, engineered 54-dimensional temporal/topology/alert features, Enterprise GNN RCA V1 GraphSAGE, and Enterprise GNN RCA V2 Temporal Relation-Aware GraphSAGE.

**Architecture:**
- V1: 2-layer GraphSAGE, 54-dim features
- V2: EnterpriseRcaTemporalRelGNN — Temporal Relation-Aware GraphSAGE with edge type and temporal features. **Not** a fully dynamic temporal heterogeneous graph transformer.
- Dataset: 80 generated enterprise scenarios, 64/8/8 split

**Evidence:**
- Training report: `model_artifacts/enterprise_gnn_rca_v2/training_report.json`
- Submission metrics: `docs/evidence/final_submission_metrics/final_submission_metrics.md`

## 3. Qwen/vLLM Downstream Remediation

**Safe wording:**
Qwen/vLLM is used after RCA for RAG/runbook-grounded remediation drafting, validation steps, rollback planning, escalation guidance, and ITSM-ready summaries. Remediation remains governance-reviewed and human approval-gated.

**Architecture:**
- Qwen/Qwen3-4B fine-tuned with LoRA rank 16 + GRPO/vERL alignment
- Downstream of GNN RCA — LLM does not perform root cause identification
- RAG-grounded against network runbook knowledge base

**Evidence:**
- GRPO training: `training/verl_grpo/runs/qwen3_4b_grpo_lora_amd/completion_evidence.md`
- Adapter: `model_artifacts/qwen3_grpo_lora_adapter/`
- Submission metrics: `docs/evidence/final_submission_metrics/final_submission_metrics.md`

## 4. Slide-4 Metrics Evidence

See: [`docs/evidence/final_submission_metrics/final_submission_metrics.md`](../evidence/final_submission_metrics/final_submission_metrics.md)

Key guardrails:
- GNN V2 metrics are on a synthetic/generated enterprise benchmark, not production data.
- RF-DETR detector accuracy is not claimed unless `reports/rfdetr_v3_eval/rfdetr_v3_eval_report.md` is present and values are computed.
- LLM/Qwen/vLLM is downstream remediation AI, not the RCA engine.
