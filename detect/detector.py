from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import json
import tempfile
import asyncio

with open("setting.json", "r", encoding="utf-8") as f:
    jdata = json.load(f)

model = YOLO(jdata["yolo_model"])

async def detect_video_live(video_path: str, on_error=None, interval: int = 10):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        if on_error:
            await on_error(f"串流開啟失敗：{video_path}")
        return

    frame_idx = 0

    try:
        while cap.isOpened():
            # 支援中途取消
            await asyncio.sleep(0)  # 給事件循環機會去偵測 cancel

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % interval != 0:
                continue

            results = model.predict(frame, save=False, verbose=False)
            boxes = results[0].boxes

            # 過濾出分數大於 0.5 的框
            high_conf_indices = (boxes.conf > 0.5).nonzero().flatten()
            if len(high_conf_indices) > 0:
                # 提取類別名稱
                class_ids = boxes.cls[high_conf_indices].int().tolist()
                class_names = list({model.names[class_id] for class_id in class_ids})

                # 建立臨時圖片檔案
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    img_path = tmp_file.name
                    results[0].save(filename=img_path)
                    yield img_path, class_names  # 🔄 回傳給上層處理者

    except asyncio.CancelledError:
        print("🔴 偵測任務被強制取消")
        # 清理資源，然後重新拋出讓上層知道
        cap.release()
        raise

    except Exception as e:
        if on_error:
            await on_error(f"偵測發生錯誤：{str(e)}")

    cap.release()
