from ultralytics import YOLO
from PIL import Image
import cv2


if __name__ == '__main__':
    model = YOLO("finalv4.pt")

    model.predict(source = 0, conf = 0.3, show = True, save = True)
    
