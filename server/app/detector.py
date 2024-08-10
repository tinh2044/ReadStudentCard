import cv2 as cv
from ultralytics import YOLO
from pathlib import Path
import numpy as np


BASE_DIR = Path(__file__).resolve(strict=True).parent

yolo_model = YOLO(f'{BASE_DIR.parent}/model/detect_info.pt')
class_names = ['avatar', 'qr_code', 'text']


def detect_info(img):
    img_arr = cv.resize(img, (640, 640))
    img_arr = cv.cvtColor(img_arr, cv.COLOR_BGR2RGB)
    source_img = img_arr.copy()

    result = yolo_model.predict(img_arr, conf=0.6)

    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    list_element = []
    list_names = []
    for box, idx, conf in zip(result[0].boxes.xyxy, result[0].boxes.cls, result[0].boxes.conf):
        x1, y1, x2, y2 = [int(x) for x in box]
        color = colors[int(idx)]
        name = class_names[int(idx)]
        cv.rectangle(source_img, (x1, y1), (x2, y2), color, thickness=2)
        title = name + " " + str(float(conf))[:4]
        list_names.append(name)
        cv.putText(source_img, title, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA, False)

        bbox = np.array(img_arr[y1:y2, x1:x2, :])
        list_element.append(bbox)
    return list_element, list_names, source_img


def cls_image(list_image, list_names):
    avatar_img = None
    qr_img = None
    text_imgs = []
    for i, name in enumerate(list_names):
        if name == class_names[0]:
            avatar_img = list_image[i].copy()
        elif name == class_names[1]:
            qr_img = list_image[i].copy()
        else:
            text_imgs.append(cv.resize(list_image[i], (512, 64)))

    return avatar_img, qr_img, text_imgs
