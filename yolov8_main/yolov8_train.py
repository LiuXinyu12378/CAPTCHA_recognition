from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
model.train(data='captcha.yaml', epochs=10, imgsz=320, device='cpu')



