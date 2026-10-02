from glob import glob

from ultralytics import YOLO

model_paths = []

model_paths.extend(glob("*.pt"))

for path in model_paths:
    model = YOLO(path)
    model.export(format="onnx")
