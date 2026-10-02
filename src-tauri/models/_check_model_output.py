import onnx

model = onnx.load("yolo26n.onnx")
model_seg = onnx.load("YOLO26n-seg-motion-600e-18062026.onnx")

for output in model.graph.output:
    print(output)

for output in model_seg.graph.output:
    print(output)
