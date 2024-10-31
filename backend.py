from enum import Enum
from ultralytics import YOLO

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn

app = FastAPI()
model = YOLO("finalv3.pt")

async def detect(file: UploadFile = File()):
    image = await file.read()
    detections = model.predict(image)
    return {"detections": detections}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)