from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware import cors
import uvicorn
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2 as cv

from .detector import detect_info, cls_image
from .ocr_text import ocr

BASE_DIR = Path(__file__).resolve(strict=True).parent
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5500/test.html",
    "http://localhost:63342/"
]

app.add_middleware(
    cors.CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
async def hello_world():
    return 'Hello World!'


@app.post('/get-info')
async def get_info(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))
    img_arr = np.array(img)

    list_images, list_names, source_img = detect_info(img_arr)

    avatar_img, qr_img, text_imgs = cls_image(list_images, list_names)

    text_imgs = [cv.cvtColor(x, cv.COLOR_RGB2GRAY) for x in text_imgs]
    texts = ocr(text_imgs)
    return {
        "card": image_to_bytes(source_img),
        "avatar": image_to_bytes(avatar_img),
        "qr_code": image_to_bytes(qr_img),
        "text_extract": [{"image": image_to_bytes(x), "text": y} for x, y in zip(text_imgs, texts)]
    }


def image_to_bytes(img):
    if img is None:
        return None
    img_bytes, buffer = cv.imencode('.jpg', img=img)
    return base64.b64encode(buffer).decode('utf-8')


if __name__ == '__main__':
    uvicorn.run(app, host='127.192.0.1', port=3000)
