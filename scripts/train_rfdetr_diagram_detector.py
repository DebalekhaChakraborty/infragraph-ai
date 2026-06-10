#!/usr/bin/env python3
"""
Train an RF-DETR detector on the V3 scenario-native diagram dataset.

Default:
    python scripts/train_rfdetr_diagram_detector.py \
        --dataset-root ./datasets/infragraph_v3/rfdetr \
        --out ./outputs/rfdetr_v3 \
        --epochs 25

The wrapper intentionally does not use YOLO or any other detector.
If RF-DETR is unavailable or its local API differs, it exits with explicit
instructions instead of faking a training run.
"""

import argparse
import datetime as _dt
import importlib
import inspect
import json
import shutil
from pathlib import Path


REQUIRED_ANNOTATIONS = [
    "instances_train.json",
    "instances_val.json",
    "instances_test.json",
]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_notes(out_root, status, message):
    notes = out_root / "rfdetr_v3_detector_notes.md"
    notes.write_text(
        "\n".join([
            "# RF-DETR V3 Detector Notes",
            "",
            f"Status: {status}",
            "",
            message,
            "",
            "Dataset contract:",
            "- COCO annotations under `annotations/instances_{split}.json`.",
            "- Images under `images/{split}`.",
            "- `diagram_metadata.json` links each image back to scenario graphs and alerts.",
            "",
            "Install options:",
            "- `pip install rfdetr`",
            "- or follow the upstream RF-DETR installation instructions for your AMD/Jupyter environment.",
            "",
            "This script does not train YOLO as an alternate model.",
        ]),
        encoding="utf-8",
    )


def validate_dataset(dataset_root):
    missing = []
    for rel in REQUIRED_ANNOTATIONS:
        p = dataset_root / "annotations" / rel
        if not p.exists():
            missing.append(str(p))
    for split in ["train", "val", "test"]:
        p = dataset_root / "images" / split
        if not p.exists():
            missing.append(str(p))
    return missing


def import_rfdetr():
    try:
        module = importlib.import_module("rfdetr")
    except ImportError as exc:
        return None, None, exc

    for attr in ["RFDETRBase", "RFDETRMedium", "RFDETRLarge"]:
        model_cls = getattr(module, attr, None)
        if model_cls is not None:
            return module, model_cls, None

    try:
        models = importlib.import_module("rfdetr.models")
        for attr in ["RFDETRBase", "RFDETRMedium", "RFDETRLarge"]:
            model_cls = getattr(models, attr, None)
            if model_cls is not None:
                return module, model_cls, None
    except ImportError:
        pass

    return module, None, None


def filtered_kwargs(callable_obj, candidates):
    sig = inspect.signature(callable_obj)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return candidates
    return {k: v for k, v in candidates.items() if k in params}


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR on Diagram Intelligence V3")
    parser.add_argument("--dataset-root", default="./datasets/infragraph_v3/rfdetr")
    parser.add_argument("--out", default="./outputs/rfdetr_v3")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.out).resolve()
    model_dir = out_root / "model"
    sample_dir = out_root / "sample_predictions"
    out_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "training_summary.json"
    started_at = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")

    missing = validate_dataset(dataset_root)
    if missing:
        message = "RF-DETR dataset is incomplete. Run `scripts/prepare_rfdetr_dataset.py` first."
        print(f"[FAIL] {message}")
        for item in missing:
            print(f"  missing: {item}")
        write_notes(out_root, "dataset_missing", message)
        write_json(summary_path, {
            "status": "dataset_missing",
            "started_at": started_at,
            "dataset_root": str(dataset_root),
            "missing": missing,
        })
        return 1

    module, model_cls, import_error = import_rfdetr()
    if import_error is not None:
        message = (
            "RF-DETR is not installed. Install it in the active environment, then rerun this script.\n"
            "Suggested commands:\n"
            "  pip install rfdetr\n"
            "  # or follow the upstream RF-DETR AMD/Jupyter setup instructions"
        )
        print(message)
        write_notes(out_root, "missing_dependency", message)
        write_json(summary_path, {
            "status": "missing_dependency",
            "started_at": started_at,
            "dataset_root": str(dataset_root),
            "out": str(out_root),
            "install_hint": "pip install rfdetr",
        })
        return 1

    if model_cls is None:
        message = (
            "The `rfdetr` package imported, but no known model class was found. "
            "Update `import_rfdetr()` with the installed package API before training."
        )
        print(f"[FAIL] {message}")
        write_notes(out_root, "api_unresolved", message)
        write_json(summary_path, {
            "status": "api_unresolved",
            "started_at": started_at,
            "dataset_root": str(dataset_root),
            "rfdetr_module": getattr(module, "__file__", ""),
        })
        return 1

    print(f"Using RF-DETR class: {model_cls.__module__}.{model_cls.__name__}")
    model = model_cls()
    if not hasattr(model, "train"):
        message = "The resolved RF-DETR model object has no `train` method."
        print(f"[FAIL] {message}")
        write_notes(out_root, "api_unresolved", message)
        write_json(summary_path, {"status": "api_unresolved", "started_at": started_at})
        return 1

    train_candidates = {
        "dataset_dir": str(dataset_root),
        "data_dir": str(dataset_root),
        "coco_dir": str(dataset_root),
        "output_dir": str(model_dir),
        "project": str(out_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "num_workers": args.num_workers,
    }
    if args.device != "auto":
        train_candidates["device"] = args.device

    train_kwargs = filtered_kwargs(model.train, train_candidates)
    if not train_kwargs:
        message = (
            "Could not map this script's arguments to the installed RF-DETR train API. "
            "Inspect the package train signature and update `train_candidates`."
        )
        print(f"[FAIL] {message}")
        write_notes(out_root, "api_unresolved", message)
        write_json(summary_path, {
            "status": "api_unresolved",
            "started_at": started_at,
            "train_signature": str(inspect.signature(model.train)),
        })
        return 1

    print("Starting RF-DETR training...")
    print(f"  dataset_root={dataset_root}")
    print(f"  out={out_root}")
    print(f"  epochs={args.epochs}")

    result = model.train(**train_kwargs)

    metrics_candidates = [
        out_root / "metrics.json",
        model_dir / "metrics.json",
        model_dir / "results.json",
    ]
    metrics_path = next((p for p in metrics_candidates if p.exists()), None)
    if metrics_path and metrics_path != out_root / "metrics.json":
        shutil.copy2(metrics_path, out_root / "metrics.json")

    write_notes(out_root, "complete", "RF-DETR training completed through the installed package API.")
    write_json(summary_path, {
        "status": "complete",
        "started_at": started_at,
        "finished_at": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_root": str(dataset_root),
        "out": str(out_root),
        "model_dir": str(model_dir),
        "epochs": args.epochs,
        "train_kwargs": train_kwargs,
        "result_repr": repr(result),
        "metrics_json": str(out_root / "metrics.json") if (out_root / "metrics.json").exists() else "",
    })
    print(f"Done. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

