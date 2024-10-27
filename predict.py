from ultralytics import YOLO
from PIL import Image

if __name__ == '__main__':
    model = YOLO("best.pt")

    results = model.tune(data="data.yaml", epochs=100, workers = 4, iterations = 100, optimizer="AdamW", plots=False, save=False, val=False)
    # path = Image.open("888.png")
    
    # newImage = path.resize((640, 640))
    # newImage.save("finalImage.png")
    
    # model.predict(source = "finalImage.png",show = True, save = True, conf = 0.5, line_thickness = 2)
    
    # metrics = model.val()
    # print(metrics.box.map)