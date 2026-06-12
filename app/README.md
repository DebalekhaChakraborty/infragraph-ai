# InfraGraph AI Command Center — Streamlit App

Premium presentation UI for the InfraGraph AI hackathon presentation.

## How to run

```bash
# From the repo root (infragraph-ai/)
pip install streamlit pandas requests plotly

streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501

## What's inside

| Tab | Content |
|-----|---------|
| ⚡ Live Incident | P1 incident card · live alert stream · topology image · AI conclusion · recommended actions |
| 🔍 Diagram Intelligence | Original PNG vs YOLO detection output · device class breakdown · top-10 detections |
| 🕸 Topology Memory | Topology graph · graph summary · alerting/impacted node chips · impact propagation paths |
| 🤖 RCA Model Arena | 3-model comparison cards (Heuristic/MLP/GNN) · candidate ranking tables · training transparency expander |
| 🌊 GNN Propagation | 5-step message-passing slider · per-step score bars · topology image · training curves |
| 📄 AI Report | Qwen-generated incident report rendered as Markdown · download button |
| 💬 Ask InfraGraph | Quick-question buttons · free-text chat · optional live Qwen endpoint |

## Expected artifacts

All paths are relative to the repo root. The app shows graceful warning cards for any missing file.

```
datasets/infragraph_v2/images/test/diagram_0373.png
outputs/v2_test_predictions_cpu/diagram_0373.jpg
outputs/topology_demo/diagram_0373_topology.png
outputs/topology_demo/diagram_0373_rca_result.json
outputs/topology_demo/diagram_0373_graph_summary.json
outputs/topology_demo/diagram_0373_detected_nodes.json
outputs/gnn_rca/diagram_0373_gnn_rca_result.json
outputs/gnn_rca/gnn_training_curve.png
outputs/mlp_rca/diagram_0373_mlp_rca_result.json
outputs/mlp_rca/mlp_training_curve.png
outputs/qwen_explanation/diagram_0373_explanation.md
```

Generate missing artifacts:

```bash
# Heuristic RCA + topology
python scripts/build_topology_rca_demo.py --diagram-id diagram_0373

# MLP node-ranker
python scripts/train_mlp_rca.py

# GNN
python scripts/train_gnn_rca.py

# AI explanation (mock — no LLM needed)
python scripts/generate_qwen_rca_explanation.py --diagram-id diagram_0373 --mode mock
```

## Live Qwen endpoint (optional)

Set environment variables before running Streamlit to enable real LLM responses in the chat tab:

```bash
export QWEN_BASE_URL=http://localhost:8000/v1
export QWEN_MODEL=infragraph

streamlit run app/streamlit_app.py
```

Without these variables the chat uses deterministic pre-built answers.

## Video recording flow

1. Start `streamlit run app/streamlit_app.py`
2. Walk through tabs in order: Live Incident → Diagram Intelligence → Topology → RCA Arena → GNN Propagation → AI Report → Chat
3. In the GNN Propagation tab, drag the slider from Step 1 to Step 5 slowly
4. In the RCA Arena tab, expand the "Model Transparency" expander
5. In the Chat tab, click each quick-question button, then type a custom question
6. Optional: set `QWEN_BASE_URL` before recording to show live LLM responses

