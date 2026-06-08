# InfraGraph AI

Synthetic network-diagram dataset generator and AI pipeline for **automated topology extraction** and **root-cause analysis (RCA)** of network incidents.

---

## What it does

| Stage | Description |
|-------|-------------|
| **Generate** | Synthesise enterprise network diagrams (PNG) with paired YOLO labels, topology graphs (JSON), and alert/RCA scenarios |
| **Detect** | Fine-tune YOLOv8 to locate network devices (router, switch, firewall, server, database, load_balancer, cloud_or_wan) |
| **Extract** | Detect inter-device connections via line detection + OCR to rebuild the topology graph |
| **Analyse** | Run graph-based and GNN-powered RCA to identify root-cause nodes from alert sequences |

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
yolo detect predict \
    model=./runs/detect/yolo_runs/infragraph_yolo_v1/weights/best.pt \
    source=./datasets/infragraph_v1/images/test \
    imgsz=960 \
    conf=0.25 \
    device=cpu \
    save=True \
    project=./outputs \
    name=v1_test_predictions_cpu

# 4. Open notebooks in order
jupyter lab
```

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
├── runs/
│   └── detect/
│       ├── yolo_runs/
│       │   └── infragraph_yolo_v1/      # Trained YOLOv8 run
│       │       ├── weights/
│       │       │   ├── best.pt          # Best checkpoint
│       │       │   └── last.pt          # Final checkpoint
│       │       └── results.csv
│       └── val/                         # Validation curves and confusion matrix
│
├── outputs/                             # Inference outputs
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
