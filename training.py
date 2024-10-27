from IPython.display import Image, display
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO

if __name__ == '__main__':
    # lets keep 11s for now.
    # 11m is a lot slower.
    model = YOLO("yolo11s.pt")
    
    # scale keyword migth improve detection quality for far/near objects.
    # cache keyword is caching images in disk
    # amp keyword is for mixed precision training
    # patience keyword is for early stopping, the model will stop training if the validation loss does not improve for 10 epochs.
    # adam optimizer: free model tuning (probably good)
    train_results = model.train(data = "data.yaml", epochs = 100, imgsz = 640, device = "cuda", batch = 24, cache = 'disk', amp = True, patience = 10, optimizer = 'auto')

    # model.tune(data="data.yaml", epochs=30, iterations = 100, optimizer="AdamW", plots=False, save=False, val=False)
    # metrics = model.val()
    # print(metrics.box.map)