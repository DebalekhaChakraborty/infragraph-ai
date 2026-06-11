# InfraGraph AI

Synthetic network-diagram dataset generator and AI pipeline for **automated topology extraction** and **root-cause analysis (RCA)** of network incidents.

---

## Presentation Flow

The Streamlit cockpit walks through a real-time ingestion journey:

| Step | Tab | What happens |
|------|-----|-------------|
| 1. **Run Diagram Intelligence** | Diagram Intelligence | Loads the selected sample, resolves the detection source, writes `outputs/live_ingestion/` evidence |
| 2. **Simulate Local RCA** | Local RCA | Runs deterministic BFS on the local graph to find root cause within the selected diagram |
| 3. **Absorb into Enterprise Brain** | Enterprise Graph Brain | Absorbs the local graph into the enterprise galaxy graph, writes `outputs/live_absorption/`, shows before → after comparison |
| 4. **Simulate Enterprise Alert** | Enterprise Graph Brain | Runs cross-diagram RCA (trained GNN result or scenario-grounded simulation); renders overlay on interactive PyVis graph |
| 5. **Ask Graph Copilot** | Graph Copilot | Answers are grounded in the loaded graph evidence (live Qwen/vLLM when `QWEN_BASE_URL` is set) |

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

### Runtime output folders

| Folder | Contents |
|--------|---------|
| `outputs/live_ingestion/<scenario>__<diagram>/` | `original.png`, `detected_nodes.json`, `detected_edges.json`, `node_table.csv`, `edge_table.csv`, `graph_memory_packet.json` |
| `outputs/live_absorption/<scenario>__<diagram>/` | `enterprise_before.json`, `enterprise_after.json`, `absorption_summary.json`, `alerts.json` |

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
# 1. Install dependencies
pip install -r requirements.txt

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

`QWEN_BASE_URL` controls whether the cockpit uses live inference or deterministic graph-evidence answers:

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
    --out outputs/mlp_rca \
    --presentation-diagram diagram_0373

# GNN (topology-aware learned model)
python scripts/train_gnn_rca.py \
    --dataset-root datasets/infragraph_v2 \
    --out outputs/gnn_rca \
    --presentation-diagram diagram_0373
```

See `docs/mlp_rca.md` and `docs/gnn_rca.md` for full details.

---

### Enterprise GNN RCA

Trains a GCN on stitched multi-diagram enterprise topology graphs and ranks all nodes
across the unified enterprise graph to identify the root-cause node.

```bash
python scripts/train_enterprise_gnn_rca.py \
    --dataset-root ./datasets/infragraph_v1/enterprise_graph \
    --out ./outputs/enterprise_gnn_rca \
    --epochs 80 \
    --presentation-scenario enterprise_0000 \
    --presentation-split test
```

| Output | Description |
|--------|-------------|
| `outputs/enterprise_gnn_rca/enterprise_gnn_model.pt` | Trained GCN checkpoint |
| `outputs/enterprise_gnn_rca/enterprise_gnn_metrics.json` | Top-1, Top-3, MRR across splits |
| `outputs/enterprise_gnn_rca/enterprise_gnn_training_curve.png` | Loss and ranking curves |
| `outputs/enterprise_gnn_rca/<scenario_id>_enterprise_gnn_rca_result.json` | Sample scenario output |
| `outputs/enterprise_gnn_rca/<scenario_id>_enterprise_gnn_prediction.png` | Graph visualisation |

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
    --out ./outputs/rfdetr_v3 \
    --epochs 25
```

| Output | Description |
|--------|-------------|
| `datasets/infragraph_v3/scenarios/` | Scenario-native source diagrams, annotations, local graphs, stitch maps, enterprise graphs, alerts, and previews |
| `datasets/infragraph_v3/rfdetr/` | COCO-style RF-DETR export with metadata linking each image back to its scenario graphs |
| `datasets/infragraph_v3/yolo/dataset.yaml` | YOLO-compatible export from the same annotations |
| `outputs/rfdetr_v3/` | RF-DETR model outputs when the external RF-DETR package is installed |

RF-DETR is the advanced V3 detector path. YOLO remains the stable baseline for
comparison. The stitched enterprise graph is generated from the same local graphs
derived from the scenario diagrams, so future enterprise GNN RCA can analyze
alerts across diagram boundaries.

#### V3 annotation quality check

Run annotation QA before detector training:

```bash
python scripts/qa_infragraph_v3_annotations.py \
    --dataset-root ./datasets/infragraph_v3 \
    --out ./outputs/v3_annotation_qa
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
    --image datasets/infragraph_v2/images/test/diagram_0373.png \
    --diagram-id sample_onboard_0373 \
    --out outputs/onboarded_diagrams
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
├── data_generator/
│   └── generate_infragraph_dataset.py   # Synthetic dataset generator
│
├── datasets/
│   └── infragraph_v1/                   # V1 dataset (images, labels, graphs, alerts)
│       ├── images/{train,val,test}/
│       ├── labels/{train,val,test}/
│       ├── graphs/{train,val,test}/
│       ├── alerts/{train,val,test}/
│       ├── previews/                    # Contact sheets
│       ├── dataset.yaml
│       └── classes.txt
│
├── training_runs/
│   └── infragraph_yolo_v1/              # YOLOv8 training artifacts
│       ├── weights/
│       │   ├── best.pt                  # Best checkpoint (canonical model path)
│       │   └── last.pt                  # Final checkpoint
│       ├── results.csv                  # Training metrics per epoch
│       └── train_batch*.jpg             # Training batch visualisations
│
├── outputs/
│   ├── v1_test_predictions_cpu/         # Detector output on test set
│   ├── val_eval/                        # Validation curves and confusion matrix
│   ├── topology_rca/                   # Stage 2: heuristic RCA outputs
│   ├── mlp_rca/                         # Stage 3a: MLP model and metrics
│   ├── gnn_rca/                         # Stage 3b: GNN model and metrics
│   └── qwen_explanation/                # Stage 4: LLM explanation reports
│
├── scripts/
│   ├── verify_repo_state.py             # Repo integrity checker
│   ├── build_topology_rca_pipeline.py       # Stage 2: heuristic RCA + topology vis
│   ├── train_mlp_rca.py                 # Stage 3a: MLP node-ranker (learned baseline)
│   ├── train_gnn_rca.py                 # Stage 3b: GNN root cause ranking
│   └── generate_qwen_rca_explanation.py # Stage 4: LLM explanation (mock/vLLM)
│
├── notebooks/
│   ├── 01_generate_dataset.ipynb        # Dataset generation walkthrough
│   ├── 02_train_yolo_amd.ipynb          # YOLOv8 training (AMD / ROCm / CUDA)
│   ├── 03_evaluate_detector.ipynb       # mAP, confusion matrix, per-class PR curves
│   ├── 04_extract_topology_graph.ipynb  # Line detection + OCR → graph
│   └── 05_rca_graph_walkthrough.ipynb          # Graph RCA & GNN RCA walkthrough
│
├── src/
│   ├── topology/
│   │   ├── line_detection.py            # Hough-line connector detection
│   │   ├── graph_builder.py             # Assemble NetworkX topology graph
│   │   └── ocr_extractor.py             # Tesseract OCR label extraction
│   ├── rca/
│   │   ├── graph_rca.py                 # Rule-based graph traversal RCA
│   │   ├── alert_simulator.py           # Synthetic alert sequence generator
│   │   └── gnn_rca.py                   # GNN root-cause classifier
│   └── utils/
│       ├── visualization.py             # Diagram + graph overlay helpers
│       └── yolo_utils.py                # YOLO label I/O utilities
│
├── configs/
│   ├── dataset_config.yaml              # Dataset paths & class definitions
│   └── train_config.yaml                # YOLO training hyperparameters
│
├── samples/
│   ├── sample_diagrams/                 # Example generated PNGs
│   └── sample_outputs/                  # Example detector / RCA outputs
│
├── yolov8n.pt                           # YOLOv8n base weights
└── requirements.txt
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



