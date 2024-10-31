# from ultralytics.data.annotator import auto_annotate

# auto_annotate(data="annotate.jpg", det_model="yolo11s.pt", sam_model="sam2.1_s.pt", output_dir = "newannotated")

from roboflow import Roboflow
rf = Roboflow(api_key="GpMbbgCoy0jmmJRQRirS")
project = rf.workspace("objectdetection-wcdta").project("door-detection-xxakd-ixmpg")
version = project.version(2)
dataset = version.download("yolov11")