from pathlib import Path
import torch
import torchvision
from ultralytics import YOLO

# -------------------------------
# ROCm / AMD workaround:
# torchvision.ops.nms is failing on the CUDA/ROCm backend.
# Patch it so NMS runs on CPU, then returns indices to original device.
# -------------------------------
_original_nms = torchvision.ops.nms

def cpu_safe_nms(boxes, scores, iou_threshold):
    original_device = boxes.device
    keep = _original_nms(boxes.detach().cpu(), scores.detach().cpu(), iou_threshold)
    return keep.to(original_device)

torchvision.ops.nms = cpu_safe_nms

# -------------------------------
# Resolve model path
# -------------------------------
candidate_model_paths = [
    "./training_runs/infragraph_yolo_v1/weights/best.pt",
    "./runs/detect/yolo_runs/infragraph_yolo_v1/weights/best.pt",
]

model_path = None
for p in candidate_model_paths:
    if Path(p).exists():
        model_path = p
        break

if model_path is None:
    raise FileNotFoundError(
        "Could not find V1 best.pt. Checked:\n" + "\n".join(candidate_model_paths)
    )

data_path = "./datasets/infragraph_v2/dataset.yaml"
if not Path(data_path).exists():
    raise FileNotFoundError(f"Missing V2 dataset YAML: {data_path}")

print(f"Using base model: {model_path}")
print(f"Using dataset: {data_path}")
print("Applied CPU-safe torchvision.ops.nms patch.")

model = YOLO(model_path)

model.train(
    data=data_path,
    epochs=40,
    imgsz=960,
    batch=16,
    device=0,
    amp=False,
    val=True,       # Now okay because NMS is patched to CPU
    plots=True,
    workers=4,
    project="./training_runs",
    name="infragraph_yolo_v2",
    exist_ok=True,
)
