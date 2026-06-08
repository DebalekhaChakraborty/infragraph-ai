# InfraGraph AI

Synthetic network-diagram dataset generator and AI pipeline for **automated topology extraction** and **root-cause analysis (RCA)** of network incidents.

---

## Current milestone

| # | Milestone | Status |
|---|-----------|--------|
| 1 | V1 synthetic dataset generated under `datasets/infragraph_v1` | Done |
| 2 | YOLO V1 detector trained | Done |
| 3 | Trained weights under `training_runs/infragraph_yolo_v1/weights` | Done |
| 4 | V2 dataset (400 diagrams) with graph + alert scenarios | Done |
| 5 | Heuristic RCA demo (`build_topology_rca_demo.py`) | Done |
| 6 | GNN root-cause ranking (`train_gnn_rca.py`) — 100% test top-1 | Done |
| 7 | Run detector prediction on test set | Next |
| 8 | Add Qwen explanation layer | Next |

---

## What it does

| Stage | Description |
|-------|-------------|
| **Generate** | Synthesise enterprise network diagrams (PNG) with paired YOLO labels, topology graphs (JSON), and alert/RCA scenarios |
| **Detect** | Fine-tune YOLOv8 to locate network devices (router, switch, firewall, server, database, load_balancer, cloud_or_wan) |
| **Extract** | Detect inter-device connections via line detection + OCR to rebuild the topology graph |
| **Analyse** | Run graph-based and GNN-powered RCA to identify root-cause nodes from alert sequences |
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

# 4. Heuristic RCA demo (Stage 2)
python scripts/build_topology_rca_demo.py --diagram-id diagram_0373

# 5. GNN root cause ranking (Stage 3)
python scripts/train_gnn_rca.py

# 6. Open notebooks in order
jupyter lab
```

### Stage 3: GNN-based RCA

```bash
# Train GNN on infragraph_v2 (400 alert scenarios, 80 epochs)
python scripts/train_gnn_rca.py \
    --dataset-root datasets/infragraph_v2 \
    --epochs 80 \
    --out outputs/gnn_rca \
    --demo-diagram diagram_0373

# Falls back to pure-numpy GCN if torch is not installed
python scripts/train_gnn_rca.py
```

Results on infragraph_v2 (test set, 28 graphs):

| Metric | Heuristic (Stage 2) | GNN (Stage 3) |
|--------|--------------------:|---------------:|
| Top-1 accuracy | ~60% | **100%** |
| Top-3 accuracy | ~90% | **100%** |
| MRR | ~0.75 | **1.000** |

The GNN learns propagation direction from graph structure and temporal alert
features, correctly identifying `FW-01` as root cause for diagram_0373 where
the heuristic chose `SW-CORE`.  See `docs/gnn_rca.md` for full details.

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
│   ├── topology_demo/                   # Stage 2: heuristic RCA outputs
│   └── gnn_rca/                         # Stage 3: GNN model, metrics, demo
│
├── scripts/
│   ├── verify_repo_state.py             # Repo integrity checker
│   ├── build_topology_rca_demo.py       # Stage 2: heuristic RCA + topology vis
│   └── train_gnn_rca.py                 # Stage 3: GNN root cause ranking
│
├── notebooks/
│   ├── 01_generate_dataset.ipynb        # Dataset generation walkthrough
│   ├── 02_train_yolo_amd.ipynb          # YOLOv8 training (AMD / ROCm / CUDA)
│   ├── 03_evaluate_detector.ipynb       # mAP, confusion matrix, per-class PR curves
│   ├── 04_extract_topology_graph.ipynb  # Line detection + OCR → graph
│   └── 05_rca_graph_demo.ipynb          # Graph RCA & GNN RCA demo
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
