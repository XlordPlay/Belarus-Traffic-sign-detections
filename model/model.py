from ultralytics import YOLO
import torch
from torch.cuda.amp import autocast, GradScaler
import path

# Задаем путь к модели и данным
model = YOLO("yolov8n.pt")  # Используем YOLOv8 Nano для скорости

# Подготавливаем устройство (GPU или CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Переносим модель на доступное устройство (GPU или CPU)
model.to(device)

# Для mixed precision, используем GradScaler и autocast из PyTorch
scaler = GradScaler()

# Начинаем тренировку модели
model.train(data=f"data/yolo_dataset/data.yaml", epochs=50, batch=16, imgsz=1280, name='yolov8n_only_detection')

# Сохраняем обученную модель
model.export(format="onnx")

# Ожидаем завершения процесса
print("Обучение завершено и модель сохранена.")