from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/last_web_model')

results = model('test.png')