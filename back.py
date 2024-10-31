from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import torch
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
model = YOLO("finalv3.pt")


# Load YOLO model (use YOLOv11 if available)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
def prediction(image):
    model.predict(source=image, stream=True)


def decode_image(image_data):
    """ Decode base64 image from string """
    image_data = base64.b64decode(image_data)
    np_image = np.frombuffer(image_data, dtype=np.uint8)
    return cv2.imdecode(np_image, cv2.IMREAD_COLOR)

def encode_image(image):
    """ Encode image back to base64 format for transmission """
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@socketio.on('frame')
def handle_frame(data):
    # Get base64 image data from the frontend
    frame_data = data['image']
    # Decode the frame from base64
    frame = decode_image(frame_data)
    

    # Perform YOLOv11 inference
    results = model(frame, verbose=False, stream_buffer = True)
    annotated_frame = results[0].plot()
    #results.numpy()
    #results.render()  # Annotates results onto the image
    #results = np.array(results)
    #results = cv2.UMat(results)
  
    
    # Encode the annotated frame back to base64
    annotated_frame = encode_image(annotated_frame)

    # Send the processed frame back to the frontend
    socketio.emit('processed_frame', {'image': annotated_frame})
    #print("req")
if __name__ == '__main__':
    # Use eventlet for asynchronous handling
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)
