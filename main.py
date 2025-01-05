import os
import torch
from ultralytics import YOLO
import packaging
print(packaging.__version__)

print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())

config_path = 'data/data.yaml'

# Load a model
model = YOLO("yolo11n.pt")

# Use the model
model.train(data=config_path, epochs=200, batch=32, device='mps')
