from ultralytics import YOLO
from datetime import datetime
import cv2
import os
import json
import tempfile
import asyncio
import torch

with open("setting.json", "r", encoding="utf-8") as f:
    jdata = json.load(f)

# ---------- 裝置選擇：優先嘗試 CUDA，用不了就退回 CPU ----------
DEVICE = "cpu"
try:
    if torch.cuda.is_available():
        # 先載入模型，再試著搬到 CUDA
        _tmp_model = YOLO(jdata["yolo_model"])
        _tmp_model.to("cuda")  # 在這一步，如果是你現在這台，會丟 operation not supported
        DEVICE = "cuda"
        model = _tmp_model
        print("✅ YOLO 模型載入完成，使用 CUDA")
    else:
        model = YOLO(jdata["yolo_model"])
        model.to("cpu")
        print("⚠️ 沒有可用 CUDA，改用 CPU")
except Exception as e:
    # 任何 CUDA 錯誤都改用 CPU
    print(f"⚠️ CUDA 不可用，改用 CPU：{e}")
    model = YOLO(jdata["yolo_model"])
    model.to("cpu")
    DEVICE = "cpu"


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

            results = model.predict(frame, save=False, verbose=False, device=DEVICE,)
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
