from IPython.display import Image, display
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")

    train_results = model.train(data = "data.yaml", epochs = 100, imgsz = 640, device = "cuda", batch = 32)

    metrics = model.val()

