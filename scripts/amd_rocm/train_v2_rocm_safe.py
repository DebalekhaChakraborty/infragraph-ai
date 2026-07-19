from ultralytics import YOLO
from pathlib import Path

model_path = "./training_runs/infragraph_yolo_v1/weights/best.pt"
data_path = "./datasets/infragraph_v2/dataset.yaml"

assert Path(model_path).exists(), f"Missing model: {model_path}"
assert Path(data_path).exists(), f"Missing dataset yaml: {data_path}"

model = YOLO(model_path)

model.train(
    data=data_path,
    epochs=40,
    imgsz=960,
    batch=16,
    device=0,
    amp=False,
    val=False,
    plots=False,
    workers=4,
    project="./training_runs",
    name="infragraph_yolo_v2",
    exist_ok=True,
)
