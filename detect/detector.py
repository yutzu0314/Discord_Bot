from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import json
import tempfile

with open("setting.json", "r", encoding="utf-8") as f:
    jdata = json.load(f)

model = YOLO(jdata["yolo_model"])

async def detect_video_live(video_path: str, on_error=None, interval: int = 10):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        if on_error:
            await on_error(f"❌ 串流開啟失敗：{video_path}")
        return

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 每幾偵進行偵測
        frame_idx += 1
        if frame_idx % interval != 0:
            continue

        results = model.predict(frame, save=False, verbose=False)
        boxes = results[0].boxes

        if boxes and len(boxes.cls) > 0:
            # 建立臨時圖片檔案
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                img_path = tmp_file.name
                results[0].save(filename=img_path)
                yield img_path  # 🔄 回傳給上層處理者

    cap.release()
