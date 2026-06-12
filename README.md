# InfraGraph AI

Synthetic network-diagram dataset generator and AI pipeline for **automated topology extraction**, **root-cause analysis (RCA)**, and **AI-driven remediation** of network incidents.

---

## Installation

Install only what you need.

| Tier | Command | Use case |
|------|---------|----------|
| **App / demo** | `pip install -r requirements.txt` | Streamlit cockpit, graph RCA, topology — no GPU required |
| **Vision / detector** | + `pip install -r requirements-vision.txt` | YOLO / RF-DETR diagram detector, OCR |
| **Vector memory** | + `pip install -r requirements-rag.txt` | ChromaDB + sentence-transformers for Graph Copilot |
| **Dataset / reward prep** | + `pip install -r requirements-training.txt` | JSONL→parquet, reward evaluation, LoRA utilities |
| **AMD GRPO training** | `bash scripts/amd_rocm/bootstrap_grpo_env.sh` | ROCm torch + vLLM + vERL (AMD GPU only) |

> `pip install -r requirements.txt` alone **cannot** reproduce the AMD ROCm GRPO
> training run. It does not install vERL, vLLM, ROCm torch, or the training stack.

### AMD ROCm GRPO training setup

```bash
# 1. Install ROCm torch, vLLM, vERL (checks existing install before touching anything)
bash scripts/amd_rocm/bootstrap_grpo_env.sh

# 2. Verify all ROCm runtime patches are in place
bash scripts/amd_rocm/patch_verl_runtime_for_rocm.sh

# 3. Dry-run (no GPU needed — prints the full training command)
bash training/verl_grpo/train_qwen3_grpo.sh

# 4. Real training run
INFRAGRAPH_RUN_REAL_VERL=1 bash training/verl_grpo/train_qwen3_grpo.sh
```

See [training/verl_grpo/README.md](training/verl_grpo/README.md) for the full
GRPO pipeline, reward functions, and honest status levels.

---

## AI Remediation Agent: Qwen3 + vLLM + vERL/GRPO

InfraGraph AI uses a three-stage intelligence pipeline:

1. **Diagram Intelligence** — RF-DETR extracts topology graph memory from network diagrams.
2. **Topology RCA + Enterprise GNN RCA** — single-diagram graph reasoning first, then cross-diagram GNN ranking across scenario graphs.
3. **Qwen3 Remediation Agent** — served via vLLM and designed for LoRA/GRPO fine-tuning with vERL, generates grounded resolution plans from graph memory, alert timeline, RCA path, and GNN ranking.

The GRPO reward functions align Qwen3 outputs to be:
- **Graph-grounded** — only referencing nodes and IPs present in the enterprise graph.
- **Safe** — validation steps precede remediation, rollback notes always included.
- **Operator-ready** — specific, actionable steps with escalation guidance.

### Start the vLLM server (local AMD/CUDA GPU)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 8000
```

### Build the RL training dataset

```bash
python training/verl_grpo/build_rca_rl_dataset.py \
    --dataset-root ./datasets/infragraph_v3 \
    --gnn-results  ./demo_assets/enterprise_gnn_rca \
    --out          ./training/verl_grpo/data
```

### Run GRPO fine-tuning with vERL

```bash
bash training/verl_grpo/train_qwen3_grpo.sh
```

### Point the app at the base model (current status)

A real GRPO/vERL training run completed on AMD ROCm — see
[training/verl_grpo/README.md](training/verl_grpo/README.md) for the honest
status. The live demo uses base Qwen/Qwen3-4B unless a PEFT adapter is
exported and served with `--enable-lora`.

```bash
export INFRAGRAPH_QWEN_BASE_URL=http://localhost:8000/v1
export INFRAGRAPH_QWEN_MODEL=Qwen/Qwen3-4B
streamlit run app/streamlit_app.py
```

If a PEFT adapter is available (see `training/verl_grpo/export_lora_adapter.py`):

```bash
export INFRAGRAPH_LORA_ADAPTER_PATH=training/verl_grpo/exported_adapter
export INFRAGRAPH_QWEN_BASE_URL=http://localhost:8000/v1
export INFRAGRAPH_QWEN_MODEL=Qwen/Qwen3-4B
streamlit run app/streamlit_app.py
```

---

---

## AI Remediation + Qwen Alignment Pipeline

InfraGraph AI now presents a full training + inference story:

- **Topology RCA** handles single-diagram dependency reasoning and local remediation.
- **Enterprise GNN RCA** handles cross-diagram graph reasoning and root-cause ranking when a matching GNN result exists.
- **Qwen3/vLLM remediation** generates graph-grounded resolution plans from RCA context, alert timelines, GNN ranking, and retrieved vector evidence.
- **Vector memory** uses ChromaDB to retrieve evidence IDs for Graph Copilot and remediation prompts.
- **LoRA + GRPO/vERL scaffold** under `training/verl_grpo/` turns RCA/remediation records into sample alignment data with deterministic reward functions for AMD GPU fine-tuning.

See [docs/ai_training_and_remediation_story.md](docs/ai_training_and_remediation_story.md) and [training/verl_grpo/README.md](training/verl_grpo/README.md).

---

## Presentation Flow

The Streamlit cockpit walks through a real-time ingestion journey:

| Step | Tab | What happens |
|------|-----|-------------|
| 1. **Run Diagram Intelligence** | Diagram Intelligence | Loads the selected sample, resolves the detection source, writes `runtime_state/live_ingestion/` evidence |
| 2a. **Generate Topology Alert Stream** | Topology RCA | Builds a realistic alert timeline for the selected diagram topology (T+00m → T+20m) |
| 2b. **Find Topology Root Cause** | Topology RCA | Runs BFS graph-traversal RCA; shows root cause, reasoning, traversal slider, and graph overlay |
| 3. **Absorb into Enterprise Brain** | Enterprise Graph Brain | Absorbs the local graph into the enterprise scenario graph; shows before → after PyVis comparison and Global InfraGraph Galaxy |
| 4a. **Generate Cross-Diagram Alert Stream** | Enterprise GNN RCA | Builds a cross-diagram alert timeline across all scenario diagrams, ordered by dependency |
| 4b. **Run Enterprise RCA** | Enterprise GNN RCA | Uses Enterprise GNN result if available; otherwise uses scenario-grounded evidence. Never fakes GNN output. |
| 5. **Ask Graph Copilot** | Graph Copilot | Answers are grounded in the loaded graph evidence (live Qwen/vLLM when `INFRAGRAPH_QWEN_BASE_URL` or `QWEN_BASE_URL` is set) |

### Detection source labels

| Label | When shown |
|-------|-----------|
| `Live RF-DETR Detector` | RF-DETR checkpoint found and inference succeeds |
| `RF-DETR Trained Prediction` | Static prediction image found in `outputs/rfdetr_v3_predictions/` |
| `Verified Annotation Overlay` | No trained prediction — renders bounding boxes from ground-truth annotation |

---

## Asset Structure

The `assets/` layer provides a clean product-facing identity for datasets, decoupling the UI from raw dataset folder structure.

```
assets/
├── gallery/
│   └── manifest.json        # DG-0001 … DG-0250 — known diagrams in graph memory
└── onboarding/
    ├── manifest.json        # ONB-001 … ONB-020 — curated samples for live ingestion
    ├── ONB-001/
    │   ├── original.png
    │   ├── annotation.json
    │   ├── local_graph.json
    │   ├── enterprise_graph.json
    │   ├── stitch_map.json
    │   └── alerts.json
    └── ONB-002/ …
```

Build the asset layer from the raw datasets:

```bash
python scripts/build_presentation_assets.py \
    --max-onboarding-samples 20 \
    --max-gallery-items 250
```

The gallery manifest lists up to 250 records (V3 > V2 > V1 priority). Each record carries a `gallery_id` (e.g. `DG-0001`), display name, source metadata, and resolved paths. The onboarding manifest lists 20 curated samples (4 per diagram type, test > val > train) with files copied into `assets/onboarding/ONB-XXX/`.

### Directory structure

| Directory | Purpose | Notes |
|-----------|---------|-------|
| `runtime_state/` | Live/generated runtime state | Not committed. Contains ingestion runs, absorption runs, incident JSON, vector memory, global graph memory. |
| `demo_assets/` | Curated demo artifacts used by the Streamlit app | Committed where small. GNN results, hero scenario selection, Qwen explanations, RCA model outputs. |
| `model_artifacts/` | Detector and model checkpoints | Ignored in git (large binaries). RF-DETR V3 weights, trained GNN models. |
| `reports/` | Evaluation reports and annotation QA | `val_eval/`, `v3_annotation_qa/`, `hydra_runs/` (ignored). |
| `outputs/` | **Legacy only** — do not use for new writes | Kept for backward compatibility. The app falls back to `outputs/<subpath>` if the new canonical path does not exist. Run `python scripts/migrate_outputs_structure.py --apply` to move existing files. |

#### Runtime output folders

| Folder | Contents |
|--------|---------|
| `runtime_state/live_ingestion/<scenario>__<diagram>/` | `original.png`, `detected_nodes.json`, `detected_edges.json`, `node_table.csv`, `edge_table.csv`, `graph_memory_packet.json` |
| `runtime_state/live_absorption/<scenario>__<diagram>/` | `enterprise_before.json`, `enterprise_after.json`, `absorption_summary.json`, `alerts.json` |
| `runtime_state/incident_runs/<hash>/` | `local_incident.json`, `enterprise_incident.json` — persisted incident simulation runs |

#### Migrate legacy outputs/ to new structure

```bash
# Preview (no changes made):
python scripts/migrate_outputs_structure.py --dry-run

# Apply:
python scripts/migrate_outputs_structure.py --apply
```

---

## Incident Simulation Layer

`src/incident_simulation/` provides deterministic, topology-aware incident builders for both the Topology RCA and Enterprise GNN RCA workspaces.

### Topology RCA simulation (`local_incidents.py`)

- Simulates alerts **within a single diagram**.
- Detects topology type from the diagram ID (branch, WAN, datacenter, app/DB, shared services).
- Uses node type priority to select the first-observed endpoint and the root-cause node.
- Builds a BFS path from first-observed node to root cause.
- Generates 3–5 `AlertTimelineEvent` records along the path (T+00m → T+20m).
- Deterministic: the same graph always produces the same alert stream and root cause.
- RCA source label: **"Scenario-guided graph RCA"** (never claimed as a trained model output).

### Enterprise / GNN RCA simulation (`enterprise_incidents.py`)

- Simulates **cross-diagram alert propagation** within a V3 scenario enterprise graph.
- Priority chain:
  1. `alerts.json` ground truth (real alert records per node and diagram).
  2. Enterprise GNN RCA result (`predicted_root_cause`, `top_candidates`) if available.
  3. Diagram-level template messages otherwise.
- Diagram ordering: symptom diagrams first (branch, WAN), root-cause diagrams last (shared services, datacenter).
- RCA source label: **"Enterprise GNN RCA"** only when a trained inference result exists for the selected scenario. Otherwise **"Scenario-grounded RCA simulation"**.
- Never fakes GNN output.

### Graph Memory vs. RCA inference

| Concept | Where shown | Purpose |
|---------|-------------|---------|
| Local Graph | Diagram Intelligence, Topology RCA | Single-diagram topology — one diagram's nodes and edges |
| Scenario Enterprise Graph | Enterprise GNN RCA — Interactive graph | Multi-diagram scenario stitched together — RCA inference target |
| Global InfraGraph Galaxy | Enterprise Graph Brain | All scenarios combined — graph-memory exploration only, not RCA inference |

---

## Current milestone

| # | Milestone | Status |
|---|-----------|--------|
| 1 | V1 synthetic dataset generated under `datasets/infragraph_v1` | Done |
| 2 | YOLO V1 detector trained | Done |
| 3 | Trained weights under `training_runs/infragraph_yolo_v1/weights` | Done |
| 4 | V2 dataset (400 diagrams) with graph + alert scenarios | Done |
| 5 | Heuristic RCA (`build_topology_rca_pipeline.py`) | Done |
| 6 | MLP node-ranker RCA (`train_mlp_rca.py`) — learned non-graph baseline | Done |
| 7 | GNN root-cause ranking (`train_gnn_rca.py`) — 100% test top-1 | Done |
| 8 | Qwen/vLLM explanation layer (`generate_qwen_rca_explanation.py`) | Done |
| 9 | Run detector prediction on full test set | Next |

---

## What it does

| Stage | Description |
|-------|-------------|
| **Generate** | Synthesise enterprise network diagrams (PNG) with paired YOLO labels, topology graphs (JSON), and alert/RCA scenarios |
| **Detect** | Fine-tune YOLOv8 to locate network devices (router, switch, firewall, server, database, load_balancer, cloud_or_wan) |
| **Extract** | Detect inter-device connections via line detection + OCR to rebuild the topology graph |
| **Analyse** | Heuristic graph scoring → learned MLP node-ranker → topology-aware GNN RCA — three-stage root-cause pipeline |
| **Explain** | Qwen / open LLM generates a plain-language incident explanation from the RCA output |

---

## Quick start

```bash
# 1. Install base dependencies (app / non-GPU)
pip install -r requirements.txt
# For training data prep: pip install -r requirements-training.txt
# For AMD ROCm GRPO training: bash scripts/amd_rocm/bootstrap_grpo_env.sh

# 2. Generate a 300-image dataset
python data_generator/generate_infragraph_dataset.py \
    --num 300 --out ./datasets/infragraph_v1 \
    --seed 42 --annotated-preview --clean

# 3. Run inference with the trained model
#    device=cpu: AMD ROCm TorchVision NMS workaround
yolo detect predict \
    model=./training_runs/infragraph_yolo_v1/weights/best.pt \
    source=./datasets/infragraph_v1/images/test \
    imgsz=960 conf=0.25 device=cpu save=True \
    project=./outputs name=v1_test_predictions_cpu

# 4. Heuristic RCA (Stage 2)
python scripts/build_topology_rca_pipeline.py --diagram-id diagram_0373

# 5. MLP node-ranker RCA (learned non-graph baseline)
python scripts/train_mlp_rca.py

# 6. GNN root cause ranking (topology-aware learned model)
python scripts/train_gnn_rca.py

# 7. Qwen explanation layer — mock mode (no LLM)
python scripts/generate_qwen_rca_explanation.py --diagram-id diagram_0373 --mode mock

# 6b. Qwen explanation layer — vLLM mode (AMD Jupyter)
python scripts/generate_qwen_rca_explanation.py \
    --diagram-id diagram_0373 --mode vllm \
    --model Qwen/Qwen3-4B --base-url http://localhost:8000/v1

# 8. Launch the Streamlit cockpit
streamlit run app/streamlit_app.py
```

### Running the Streamlit Cockpit in AMD Jupyter

See: [docs/run_streamlit_in_jupyter.md](docs/run_streamlit_in_jupyter.md)

### Running with live Qwen/vLLM on AMD

| Service | Port | Role |
|---------|------|------|
| vLLM (Qwen) | 8000 | OpenAI-compatible inference endpoint |
| Streamlit | 8501 | InfraGraph AI cockpit UI |

`INFRAGRAPH_QWEN_BASE_URL` and `INFRAGRAPH_QWEN_MODEL` are preferred for live inference. Legacy `QWEN_BASE_URL` and `QWEN_MODEL` are still supported:

```bash
# Start vLLM on AMD
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-7B-Instruct --port 8000 --host 0.0.0.0

# Start Streamlit pointing at local vLLM
QWEN_BASE_URL=http://localhost:8000/v1 \
QWEN_MODEL=Qwen/Qwen2-7B-Instruct \
python -m streamlit run app/streamlit_app.py \
  --server.port 8501 --server.address 0.0.0.0 \
  --server.headless true --server.enableCORS false \
  --server.enableXsrfProtection false
```

When `QWEN_BASE_URL` is a localtunnel URL the cockpit automatically sends the `Bypass-Tunnel-Reminder: true` header.

See: [docs/run_qwen_vllm_amd.md](docs/run_qwen_vllm_amd.md)

### Vector Memory Layer

ChromaDB indexes graph-memory evidence for semantic retrieval. It complements
the topology graph and GNN; it does not replace them.

| Layer | Role |
|-------|------|
| Graph JSON / graph memory packet | Structured topology truth |
| Vector DB | Semantic retrieval over graph evidence, timelines, RCA, and resolution plans |
| GNN | Root-cause ranking over graph structure |
| Qwen/vLLM | Remediation and reasoning generation |

Graph Copilot retrieves Chroma chunks before answering. AI Resolution plans can
include retrieved graph evidence in the Qwen context.

```bash
pip install chromadb sentence-transformers

python scripts/build_vector_memory.py \
    --repo-root . \
    --persist-dir ./runtime_state/vector_memory/chroma \
    --collection infragraph_memory
```

Vector memory files are local runtime artifacts under `runtime_state/vector_memory/`
and are intentionally ignored by git.

### RCA model architecture

| Model | Uses graph? | Learned? | Script |
|-------|-------------|----------|--------|
| Heuristic scorer | Yes (rules) | No | `build_topology_rca_pipeline.py` |
| **MLP node-ranker** | No | Yes | `train_mlp_rca.py` |
| **GNN** | Yes (message passing) | Yes | `train_gnn_rca.py` |

The MLP RCA model is a learned non-graph baseline. It scores each node
independently from engineered features without any graph message passing.
The GNN RCA model is the topology-aware learned model that additionally
aggregates neighbour information across the network graph.

Results on infragraph_v2 test set (28 graphs):

| Model | Top-1 | Top-3 | MRR | Convergence |
|-------|------:|------:|----:|-------------|
| Heuristic | ~60% | ~90% | ~0.75 | N/A |
| MLP (no graph) | **100%** | **100%** | **1.000** | epoch 56 |
| GNN (graph MP) | **100%** | **100%** | **1.000** | epoch 6 |

The GNN converges ~9x faster than the MLP because graph message-passing
propagates the root-cause signal through topology neighbours.

```bash
# MLP node-ranker (learned non-graph baseline)
python scripts/train_mlp_rca.py \
    --dataset-root datasets/infragraph_v2 \
    --out demo_assets/mlp_rca \
    --presentation-diagram diagram_0373

# GNN (topology-aware learned model)
python scripts/train_gnn_rca.py \
    --dataset-root datasets/infragraph_v2 \
    --out demo_assets/gnn_rca \
    --presentation-diagram diagram_0373
```

See `docs/mlp_rca.md` and `docs/gnn_rca.md` for full details.

---

### Enterprise GNN RCA

Trains a 3-layer GCN on stitched multi-diagram enterprise topology graphs.
Each scenario graph (branch + WAN + datacenter + app/db + shared services stitched together)
is one training sample.  The model ranks all nodes across the scenario to identify
the root-cause node that triggered cross-diagram alert propagation.

**How it works:**
- The GNN trains across many V3 scenario enterprise graphs (one graph = one scenario).
- Each scenario graph is a separate training sample with its own label (root-cause node).
- At inference time the model ranks nodes inside the **selected scenario graph** to identify the root cause.
- The Global InfraGraph Galaxy is a separate graph-memory index used for exploration and storytelling — it is not the GNN inference graph.

```bash
# Preferred: V3 multi-diagram enterprise scenarios
python scripts/train_enterprise_gnn_rca.py \
    --dataset-root ./datasets/infragraph_v3 \
    --out ./demo_assets/enterprise_gnn_rca \
    --epochs 80 \
    --presentation-scenario enterprise_v3_0000 \
    --presentation-split test
```

> **Note:** The script also accepts `--dataset-root ./datasets/infragraph_v1/enterprise_graph`
> for backward compatibility with V1 single-graph datasets.

| Output | Description |
|--------|-------------|
| `demo_assets/enterprise_gnn_rca/enterprise_gnn_model.pt` | Trained GCN checkpoint |
| `demo_assets/enterprise_gnn_rca/enterprise_gnn_metrics.json` | Top-1, Top-3, MRR across splits |
| `demo_assets/enterprise_gnn_rca/enterprise_gnn_training_curve.png` | Loss and ranking curves |
| `demo_assets/enterprise_gnn_rca/<scenario_id>_enterprise_gnn_rca_result.json` | Per-scenario inference result |
| `demo_assets/enterprise_gnn_rca/<scenario_id>_enterprise_gnn_prediction.png` | Graph visualisation |

The app UI shows **"Enterprise GNN RCA"** only when a result JSON exists for the
**exact selected scenario**.  If no matching result is found, the UI shows
**"Scenario-grounded RCA simulation"** using the scenario's `alerts.json` ground truth.

### Global InfraGraph Galaxy

Builds a combined graph-memory index across all V3 scenarios for exploration and storytelling.

```bash
python scripts/build_global_infragraph_galaxy.py \
    --dataset-root ./datasets/infragraph_v3 \
    --out ./runtime_state/global_graph_memory
```

| Output | Description |
|--------|-------------|
| `runtime_state/global_graph_memory/infragraph_global_graph.json` | Full global node/edge list |
| `runtime_state/global_graph_memory/nodes.csv` | One row per node (global_node_id = scenario::node_id) |
| `runtime_state/global_graph_memory/edges.csv` | One row per edge |
| `runtime_state/global_graph_memory/scenario_index.json` | Per-scenario metadata |
| `runtime_state/global_graph_memory/summary.json` | Aggregate counts |

See: [docs/enterprise_gnn_rca.md](docs/enterprise_gnn_rca.md)

---

### Enterprise Graph Dataset

This dataset shows multi-diagram graph stitching, where each architecture diagram
becomes a local graph and multiple local graphs are stitched into **enterprise graph memory**
for cross-diagram RCA.

| Component | Description |
|-----------|-------------|
| Local graph | One infrastructure diagram → one NetworkX sub-graph |
| Stitch map | Declares cross-diagram edges and shared entities |
| Enterprise graph | All local graphs merged into one unified topology |
| Alert scenario | Root cause in one diagram, symptoms in another |

Each enterprise scenario contains 3–5 local diagrams stitched into one galaxy-scale graph.
The GNN learns to trace root causes across diagram boundaries using `cross_diagram_edges`.

```bash
python scripts/generate_enterprise_scenarios.py \
    --num 120 --out ./datasets/infragraph_v1/enterprise_graph \
    --seed 2026 --clean
```

See: [docs/enterprise_graph_dataset.md](docs/enterprise_graph_dataset.md)

---

### Diagram Intelligence V3 + RF-DETR

V3 starts the scenario-native diagram intelligence track. Each scenario is one
enterprise environment with 3-5 related topology diagrams. The individual diagram
images are used for detector training, OCR/connector validation, local graph
creation, and the stitched enterprise "galaxy" graph used later by enterprise GNN
RCA.

```bash
python scripts/generate_infragraph_v3_dataset.py \
    --num-scenarios 100 \
    --out ./datasets/infragraph_v3 \
    --seed 2026 \
    --clean

python scripts/prepare_rfdetr_dataset.py \
    --dataset-root ./datasets/infragraph_v3 \
    --out ./datasets/infragraph_v3/rfdetr

python scripts/train_rfdetr_diagram_detector.py \
    --dataset-root ./datasets/infragraph_v3/rfdetr \
    --out ./model_artifacts/rfdetr_v3 \
    --epochs 25
```

| Output | Description |
|--------|-------------|
| `datasets/infragraph_v3/scenarios/` | Scenario-native source diagrams, annotations, local graphs, stitch maps, enterprise graphs, alerts, and previews |
| `datasets/infragraph_v3/rfdetr/` | COCO-style RF-DETR export with metadata linking each image back to its scenario graphs |
| `datasets/infragraph_v3/yolo/dataset.yaml` | YOLO-compatible export from the same annotations |
| `model_artifacts/rfdetr_v3/` | RF-DETR model outputs when the external RF-DETR package is installed |

RF-DETR is the advanced V3 detector path. YOLO remains the stable baseline for
comparison. The stitched enterprise graph is generated from the same local graphs
derived from the scenario diagrams, so future enterprise GNN RCA can analyze
alerts across diagram boundaries.

#### V3 annotation quality check

Run annotation QA before detector training:

```bash
python scripts/qa_infragraph_v3_annotations.py \
    --dataset-root ./datasets/infragraph_v3 \
    --out ./reports/v3_annotation_qa
```

If QA reports `DISPLAY_ONLY_FIX`, the clean Verified Annotation Overlay is enough.
If QA reports `ANNOTATION_REGENERATION_RECOMMENDED`, fix the generator and
regenerate V3 before retraining. Do not retrain the detector until annotation QA passes.

Current V3 QA returned `DISPLAY_ONLY_FIX`, so the source annotations are
acceptable and the production overlay uses clean mode by default. Technical
connector overlays are available only for developer diagnostics.

Verified Annotation Overlay is a graph-ready ground-truth metadata view: it
shows node identity and device type without confidence scores. Detector Output
is the model inference view: live RF-DETR, trained RF-DETR, and YOLO prediction
images may show predicted class and confidence.

See: [docs/diagram_intelligence_v3_dataset.md](docs/diagram_intelligence_v3_dataset.md)
and [docs/rfdetr_v3_detector.md](docs/rfdetr_v3_detector.md)

---

### Diagram Onboarding

Onboard any topology diagram into graph memory with a single command:

```bash
python scripts/onboard_diagram.py \
    --image <path-to-diagram-image> \
    --diagram-id sample_onboard_0373 \
    --out demo_assets/onboarded_diagrams
```

Outputs: `original.png`, `detected.png`, `detected_nodes.json`, `local_graph.json`,
`graph_preview.png`, `onboarding_summary.json`, and an updated `graph_memory/index.json`.

The **Diagram Intelligence** workspace in the Streamlit cockpit exposes the same flow
via the **Onboard New Diagram** tab using manifest-selected samples.

See: [docs/diagram_onboarding.md](docs/diagram_onboarding.md)

---

## Folder structure

```
infragraph-ai/
├── app/                          # Streamlit cockpit
├── assets/                       # Product-facing gallery/onboarding manifests
├── datasets/
│   ├── infragraph_v1/            # V1 baseline topology dataset + enterprise_graph/
│   ├── infragraph_v2/            # V2 improved topology dataset
│   └── infragraph_v3/            # V3 scenario-native dataset (committed)
├── docs/                         # Architecture and pipeline docs
├── notebooks/
├── requirements/                 # requirements-*.txt per environment tier
├── scripts/                      # All CLI pipeline scripts
├── src/                          # Core library (topology, rca, ai_remediation, …)
├── training/
│   └── verl_grpo/                # GRPO/vERL fine-tuning pipeline
│
├── runtime_state/                # Live runtime outputs — NOT committed
│   ├── live_ingestion/           # Per-diagram ingestion evidence
│   ├── live_absorption/          # Enterprise graph absorption runs
│   ├── incident_runs/            # Persisted incident simulation JSON
│   ├── vector_memory/            # ChromaDB (gitignored)
│   └── global_graph_memory/      # Global InfraGraph Galaxy index
│
├── demo_assets/                  # Curated artifacts used by the Streamlit cockpit
│   ├── enterprise_gnn_rca/       # Trained GNN metrics + per-scenario RCA results
│   ├── gnn_rca/                  # V2 GNN model + metrics
│   ├── mlp_rca/                  # V2 MLP model + metrics
│   ├── demo_hero/                # Hero scenario selection
│   ├── onboarded_diagrams/       # Onboarded diagram output packages
│   └── qwen_explanation/         # Qwen explanation reports
│
├── model_artifacts/              # Detector and model checkpoints — gitignored
│   ├── rfdetr_v3/                # RF-DETR V3 checkpoint + outputs
│   └── rfdetr_v3_smoke/          # Smoke-test run checkpoint
│
├── reports/                      # Evaluation reports and annotation QA
│   ├── val_eval/                 # Validation curves and confusion matrix
│   ├── v3_annotation_qa/         # V3 annotation QA results
│   └── hydra_runs/               # Hydra/dated run dirs — gitignored
│
└── outputs/                      # Legacy only — kept for backward compatibility
    └── .gitkeep
```

### Dataset evolution

```
datasets/
├── infragraph_v1/         baseline topology image dataset
│   └── enterprise_graph/  legacy enterprise RCA baseline owned by V1
├── infragraph_v2/         improved topology image dataset and RCA experiments
└── infragraph_v3/         scenario-native diagram intelligence + enterprise RCA dataset
```

InfraGraph V1 contains the baseline topology dataset and the original enterprise
graph RCA baseline under `infragraph_v1/enterprise_graph`. InfraGraph V3 is
scenario-native and keeps diagrams, annotations, local graphs, stitch maps,
enterprise graphs, and RCA ground truth together inside each scenario.

### Presentation Flow

1. Diagram Gallery shows known diagrams available in graph memory.
2. Onboard New Diagram selects curated samples and runs live diagram intelligence.
3. A graph memory packet is created.
4. The diagram is absorbed into Enterprise Graph Brain.
5. Enterprise RCA runs on the updated graph.
6. Graph Copilot answers using graph evidence.

---

## Artifact layout policy

| Write here | For |
|------------|-----|
| `runtime_state/` | Any file generated at runtime (ingestion, absorption, incidents, vector DB, global graph) |
| `demo_assets/` | Curated, repeatable artifacts committed to the repo (GNN results, hero scenarios, Qwen explanations) |
| `model_artifacts/` | Detector and model checkpoints (gitignored; large binaries) |
| `reports/` | Evaluation reports and annotation QA — commit summary files, gitignore large artefacts |
| `outputs/` | **Do not write here.** Legacy only. Existing files are accessible via `src/paths.py` fallback. |

The `src/paths.py` helpers (`runtime_path()`, `demo_asset_path()`, `model_artifact_path()`,
`report_path()`) return the canonical new path for writes and fall back to `outputs/<subpath>`
for reads if the new path does not yet exist — so the cockpit runs on either layout.

---

## Qwen3-4B GRPO/vERL training status

The project ships a full GRPO/vERL training pipeline under `training/verl_grpo/` for
fine-tuning Qwen/Qwen3-4B on the InfraGraph RCA/remediation alignment dataset with
deterministic reward functions on AMD ROCm.

### Current honest status

The project includes evidence of a completed real GRPO/vERL training run on AMD ROCm
with Qwen/Qwen3-4B and LoRA configuration. The run reached `global_step_32` and
persisted a vERL/FSDP actor checkpoint. However, the current public evidence does not
include a standalone PEFT LoRA adapter folder containing `adapter_model.safetensors`
and `adapter_config.json`.

For this reason, the live demo should be presented as:
- base Qwen3-4B served through vLLM for explanation generation;
- plus completed GRPO/vERL training evidence for the InfraGraph RCA/remediation alignment workflow.

Do not claim the fine-tuned LoRA is actively loaded in the live vLLM demo until a PEFT
adapter is exported and served with `--enable-lora`.

See [training/verl_grpo/README.md](training/verl_grpo/README.md) for the full training
pipeline, checkpoint structure, and adapter export instructions.

---

## Dataset generator CLI

```
python data_generator/generate_infragraph_dataset.py \
    [--num N]                   # diagrams to generate (default 20)
    [--out PATH]                # output directory (default ./infragraph_dataset)
    [--seed INT]                # random seed (default 42)
    [--difficulty easy|medium|hard|mixed]   # curriculum difficulty (default mixed)
    [--augment-document-noise]  # apply scan/print noise to hard diagrams
    [--yolo-path-mode relative|absolute]
    [--annotated-preview]       # save previews/bbox_contact_sheet.png
    [--clean]                   # wipe output subfolders before generating
```

## Device classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `router` | WAN/edge/core routers |
| 1 | `switch` | Access/distribution/core switches |
| 2 | `firewall` | Perimeter and internal firewalls |
| 3 | `server` | App, web, API, management servers |
| 4 | `database` | SQL/NoSQL database nodes |
| 5 | `load_balancer` | Hardware/software load balancers |
| 6 | `cloud_or_wan` | Cloud VPCs, WAN circuits, ISP nodes |

---

## License

MIT
