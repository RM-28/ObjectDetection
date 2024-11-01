from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
import io
from PIL import Image
from ultralytics import YOLO
import cv2 

app = FastAPI()

model = YOLO("finalv4.pt")


# Route to render the HTML form
@app.get("/", response_class=HTMLResponse)
async def main():
    html_content = """
    <html>
        <head>
            <title>Image Upload</title>
        </head>
        <body>
            <h1>Upload an Image</h1>
            <form action="/predicted_image/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Route to handle image upload
@app.post("/predicted_image/")
async def upload_file(file: UploadFile):

    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))
    results = model(img)
    results = results[0].plot()
    results = cv2.cvtColor(results, cv2.COLOR_BGR2RGB)
    return_img = Image.fromarray(results)
    buf = io.BytesIO()
    return_img.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")

# To serve the uploaded files, we can add an additional route
@app.get("/uploads/")
async def get_file():
    print("upload")
 
