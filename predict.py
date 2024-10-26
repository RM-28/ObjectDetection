from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("predictModel.pt")

    model.predict(source = "sofa2.png",show = True, save = True, conf = 0.5, line_thickness = 2)
    
    # metrics = model.val()
    # print(metrics.box.map)