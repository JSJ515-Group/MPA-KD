from ultralytics import YOLO
import onnx

# Load a model
model = YOLO(r'E:\BaiduNetdiskDownload\change-ASFF-Rep\yolo11n.pt')  # load an official model
model = YOLO(r'E:\BaiduNetdiskDownload\change-ASFF-Rep\runs\train\exp121\weights\best.pt')  # load a custom trained model

# Export the model
model.export(format="onnx")