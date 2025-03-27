from ultralytics import YOLO

model = YOLO("LMSCD-Net.yaml")

model.train(data="g384.yaml", epochs=450, imgsz=640, patience=0, optimizer="SGD")
# model.info()