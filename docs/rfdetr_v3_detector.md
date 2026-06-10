# RF-DETR V3 Detector

RF-DETR is the advanced detector path for Diagram Intelligence V3. YOLO remains
the stable baseline and alternate path comparison path.

## Why RF-DETR For V3

V3 diagrams include more than isolated device icons. They include topology
documents with zones, labels, IPs, shared entities, and connector context. RF-DETR
is a DETR-style detector and is a good fit to evaluate against the YOLO baseline
on richer document-like diagram scenes.

The V3 detector still trains on object boxes only. The important difference is
traceability: every detector image remains linked to its scenario annotation,
local graph, stitched enterprise graph, and alert labels through
`rfdetr/diagram_metadata.json`.

## Dataset Preparation

Generate V3 scenarios first:

```bash
python scripts/generate_infragraph_v3_dataset.py \
  --num-scenarios 100 \
  --out ./datasets/infragraph_v3 \
  --seed 2026 \
  --clean
```

Build the COCO export for RF-DETR:

```bash
python scripts/prepare_rfdetr_dataset.py \
  --dataset-root ./datasets/infragraph_v3 \
  --out ./datasets/infragraph_v3/rfdetr
```

Expected RF-DETR structure:

```text
datasets/infragraph_v3/rfdetr/
  images/train/
  images/val/
  images/test/
  annotations/instances_train.json
  annotations/instances_val.json
  annotations/instances_test.json
  diagram_metadata.json
```

Categories:

1. `router`
2. `switch`
3. `firewall`
4. `server`
5. `database`
6. `load_balancer`
7. `cloud_or_wan`
8. `service`

## Training

Default command:

```bash
python scripts/train_rfdetr_diagram_detector.py \
  --dataset-root ./datasets/infragraph_v3/rfdetr \
  --out ./outputs/rfdetr_v3 \
  --epochs 25
```

AMD/Jupyter-friendly options:

```bash
python scripts/train_rfdetr_diagram_detector.py \
  --dataset-root ./datasets/infragraph_v3/rfdetr \
  --out ./outputs/rfdetr_v3 \
  --epochs 25 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --num-workers 2
```

The wrapper tries to import the RF-DETR package. If it is not installed, it prints
install instructions, writes `outputs/rfdetr_v3/training_summary.json`, and exits.
It does not silently train YOLO or another detector.

## Expected Outputs

Outputs are written under:

```text
outputs/rfdetr_v3/
  model/
  training_summary.json
  metrics.json
  sample_predictions/
  rfdetr_v3_detector_notes.md
```

`metrics.json` is copied into the output root if the installed RF-DETR package
produces one in a recognized location.

## YOLO Baseline

The V3 generator also produces:

```text
datasets/infragraph_v3/yolo/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
  dataset.yaml
```

This keeps the comparison clear:

- YOLO: stable baseline detector.
- RF-DETR: advanced V3 detector.

## Future Options

DINOv3 and Grounding DINO are future options for richer document and open-vocabulary
diagram understanding. They are not used by the current wrapper.

