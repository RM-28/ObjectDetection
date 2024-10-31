from IPython.display import Image, display
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import ray
import torch
import copy
import os


if __name__ == '__main__':
    model = YOLO("yolo11s.pt")
    
    # cache keyword is caching images in disk
    # amp keyword is for mixed precision training
    # patience keyword is for early stopping, the model will stop training if the validation loss does not improve for 10 epochs.
    # adam optimizer: changes learning rate dynamically
    train_results = model.train(data = "data.yaml", epochs = 200, imgsz = 640, device = "cuda", batch = 24, cache = 'disk', amp = True, patience = 10, optimizer = 'AdamW')
    
    
    
    
    
    
    hyp = dict()
    hyp['lr0']= 0.0001 # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    hyp['lrf']= 0.02 # final learning rate (lr0 * lrf)
    hyp['momentum']= 0.937 # SGD momentum/Adam beta1
    hyp['weight_decay']= 0.0005 # optimizer weight decay 5e-4
    hyp['warmup_epochs']= 3.0 # warmup epochs (fractions ok)
    hyp['warmup_momentum']= 0.8 # warmup initial momentum
    hyp['warmup_bias_lr']= 0.1 # warmup initial bias lr
    hyp['box']= 7.5 # box loss gain
    hyp['cls']= 0.5 # cls loss gain (scale with pixels)
    hyp['dfl']= 1.5 # dfl loss gain
    hyp['pose']= 12.0 # pose loss gain
    hyp['kobj']= 1.0 # keypoint obj loss gain
    hyp['label_smoothing']= 0.0 # label smoothing (fraction)
    hyp['nbs']= 64 # nominal batch size
    hyp['hsv_h']= 0.015 # image HSV-Hue augmentation (fraction)
    hyp['hsv_s']= 0.7 # image HSV-Saturation augmentation (fraction)
    hyp['hsv_v']= 0.4 # image HSV-Value augmentation (fraction)
    hyp['degrees']= 0.0 # image rotation (+/- deg)
    hyp['translate']= 0.1 # image translation (+/- fraction)
    hyp['scale']= 0.5 # image scale (+/- gain)
    hyp['shear']= 0.0 # image shear (+/- deg)
    hyp['perspective']= 0.0 # image perspective (+/- fraction), range 0-0.001
    hyp['flipud']= 0.5 # image flip up-down (probability)
    hyp['fliplr']= 0.5 # image flip left-right (probability)
    hyp['mosaic']= 1.0 # image mosaic (probability)
    hyp['mixup']= 0.0 # image mixup (probability)
    hyp['copy_paste']= 0.1 # segment copy-paste (probability)
    
    
    # scale keyword migth improve detection quality for far/near objects.
    
    
   
    
 
