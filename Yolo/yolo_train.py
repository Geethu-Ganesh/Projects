from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='drone_dataset/data.yaml',epochs=2)