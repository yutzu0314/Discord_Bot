# detector.py
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import json
import shutil

with open("setting.json", "r", encoding="utf-8") as f:
    jdata = json.load(f)

model = YOLO(jdata["yolo_model"])

async def detect_video_live(video_path: str, on_detected, interval: int = 10):
    """
    每 interval 幀進行一次偵測，並在偵測到物件時呼叫 on_detected(img_path)
    """
    cap = cv2.VideoCapture(video_path)
    tmp_dir = "temp_shots"
    os.makedirs(tmp_dir, exist_ok=True)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % interval != 0:
            continue

        results = model.predict(frame, save=False, verbose=False)
        boxes = results[0].boxes

        if boxes and len(boxes.cls) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_path = os.path.join(tmp_dir, f"violation_{timestamp}.jpg")
            results[0].save(filename=img_path)

            await on_detected(img_path)

    cap.release()
    shutil.rmtree(tmp_dir, ignore_errors=True)
