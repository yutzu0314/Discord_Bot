from ultralytics import YOLO
from datetime import datetime
import cv2
import os
import json
import tempfile
import asyncio
import torch
from norfair import Detection, Tracker
import numpy as np

with open("setting.json", "r", encoding="utf-8") as f:
    jdata = json.load(f)

# ---------- 裝置選擇：優先嘗試 CUDA，用不了就退回 CPU ----------
DEVICE = "cpu"

def load_yolo_model(model_path: str, model_name: str = "YOLO"):
    global DEVICE

    try:
        if torch.cuda.is_available():
            tmp_model = YOLO(model_path)
            tmp_model.to("cuda")
            DEVICE = "cuda"
            print(f"✅ {model_name} 載入完成，使用 CUDA：{model_path}")
            return tmp_model
        else:
            tmp_model = YOLO(model_path)
            tmp_model.to("cpu")
            print(f"⚠️ 沒有可用 CUDA，{model_name} 改用 CPU：{model_path}")
            return tmp_model
    except Exception as e:
        print(f"⚠️ {model_name} 無法使用 CUDA，改用 CPU：{e}")
        tmp_model = YOLO(model_path)
        tmp_model.to("cpu")
        DEVICE = "cpu"
        print(f"✅ {model_name} 載入完成，使用 CPU：{model_path}")
        return tmp_model


# 一般違規偵測模型
model = load_yolo_model(jdata["yolo_model"], "違規偵測模型")

# 車禍偵測模型
# ACCIDENT_MODEL_PATH = jdata.get("accident_yolo_model", jdata["yolo_model"])
# accident_model = load_yolo_model(ACCIDENT_MODEL_PATH, "車禍偵測模型")
accident_model = jdata["yolo_model"]


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

async def detect_accident_live(video_path: str, on_error=None, interval: int = 1):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        if on_error:
            await on_error(f"串流開啟失敗：{video_path}")
        return

    # =========================
    # Norfair 距離函式
    # =========================
    def distance(a, b):
        if hasattr(a, "points"):
            p1 = np.array(a.points[0])
        else:
            p1 = np.array(a.estimate[0])

        if hasattr(b, "points"):
            p2 = np.array(b.points[0])
        else:
            p2 = np.array(b.estimate[0])

        return np.linalg.norm(p1 - p2)

    tracker = Tracker(
        distance_function=distance,
        distance_threshold=30,
    )

    # =========================
    # 參數
    # =========================
    FPS = 30
    SPEED_HIGH = 60
    SPEED_LOW = 5
    MAX_SPEED = 200

    MOVE_MIN = 6
    DIR_VAR_TH = 0.05

    HISTORY_LEN = 5
    NEAR_FRAMES = 3

    SIZE_NEAR = 40
    SIZE_MID = 25

    MIN_TRACK_AGE = 8

    MID_SCORE_TH = 3
    MID_SCORE_WINDOW = int(1.2 * FPS)
    MID_HOLD_FRAMES = 3

    GRID_COLS = 8
    GRID_ROWS = 6
    HOT_WINDOW = int(1.5 * FPS)
    HOT_SUDDEN_STOP_TH = 3
    HOT_COMBO_STOP = 2
    HOT_DIRVAR_TH = 1
    HOT_HOLD = int(2.0 * FPS)

    EVENT_COOLDOWN = int(2.0 * FPS)

    # =========================
    # 狀態
    # =========================
    prev_positions = {}
    track_history = {}
    prev_speed = {}
    dir_history = {}
    near_counter = {}
    last_seen = {}

    track_age = {}
    mid_score_hist = {}
    mid_hold = {}

    hot_events = {}
    hot_active = {}

    last_accident_frame = {}
    last_hotspot_frame = {}

    frame_idx = 0

    def calc_avg_speed(history, fps):
        if len(history) < 2:
            return 0

        dist = 0.0
        for i in range(1, len(history)):
            dist += np.linalg.norm(history[i][1] - history[i - 1][1])

        return min((dist / (len(history) - 1)) * fps, MAX_SPEED)

    def get_layer(size):
        if size >= SIZE_NEAR:
            return "near"
        elif size >= SIZE_MID:
            return "mid"
        else:
            return "far"

    def grid_cell(cx, cy, w, h):
        cx = max(0, min(w - 1, int(cx)))
        cy = max(0, min(h - 1, int(cy)))
        col = int(cx / (w / GRID_COLS))
        row = int(cy / (h / GRID_ROWS))
        col = max(0, min(GRID_COLS - 1, col))
        row = max(0, min(GRID_ROWS - 1, row))
        return (row, col)

    def hot_add_event(cell, frame_idx, event_type):
        hot_events.setdefault(cell, []).append((frame_idx, event_type))
        cutoff = frame_idx - HOT_WINDOW
        hot_events[cell] = [(f, t) for (f, t) in hot_events[cell] if f >= cutoff]

    def hot_check_and_activate(cell, frame_idx):
        events = hot_events.get(cell, [])
        stop_cnt = sum(1 for _, t in events if t == "stop")
        dir_cnt = sum(1 for _, t in events if t == "dir")

        if stop_cnt >= HOT_SUDDEN_STOP_TH or (
            stop_cnt >= HOT_COMBO_STOP and dir_cnt >= HOT_DIRVAR_TH
        ):
            hot_active[cell] = frame_idx + HOT_HOLD
            return True, stop_cnt, dir_cnt
        return False, stop_cnt, dir_cnt

    def is_hot_active(cell, frame_idx):
        until = hot_active.get(cell, -1)
        return frame_idx <= until

    def mid_add_score(obj_id, frame_idx, delta):
        if delta == 0:
            return
        mid_score_hist.setdefault(obj_id, []).append((frame_idx, delta))
        cutoff = frame_idx - MID_SCORE_WINDOW
        mid_score_hist[obj_id] = [(f, d) for (f, d) in mid_score_hist[obj_id] if f >= cutoff]

    def mid_sum_score(obj_id):
        return sum(d for _, d in mid_score_hist.get(obj_id, []))

    try:
        while cap.isOpened():
            await asyncio.sleep(0)

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % interval != 0:
                continue

            h, w = frame.shape[:2]
            results = accident_model.predict(frame, save=False, verbose=False, device=DEVICE)
            boxes = results[0].boxes

            detections = []
            for box in boxes:
                cls = int(box.cls)
                conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)

                # 只追蹤 car，且信心值過濾
                if cls == 0 and conf > 0.4:  # 第 0 類:車
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)
                    size = float(max(x2 - x1, y2 - y1))

                    detections.append(
                        Detection(
                            points=np.array([[cx, cy]]),
                            data={"size": size}
                        )
                    )

            tracked_objects = tracker.update(detections)

            accident_triggered = False
            accident_labels = set()

            for obj in tracked_objects:
                obj_id = obj.id
                cx, cy = obj.estimate[0]
                current = np.array([cx, cy], dtype=np.float32)
                size = float(obj.last_detection.data.get("size", 0))

                last_seen[obj_id] = frame_idx
                track_age[obj_id] = track_age.get(obj_id, 0) + 1

                layer = get_layer(size)
                cell = grid_cell(cx, cy, w, h)

                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                cv2.putText(
                    frame, f"ID {obj_id} [{layer}]", (int(cx), int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                track_history.setdefault(obj_id, []).append((frame_idx, current))
                if len(track_history[obj_id]) > HISTORY_LEN:
                    track_history[obj_id].pop(0)

                speed = calc_avg_speed(track_history[obj_id], FPS)
                cv2.putText(
                    frame, f"v={int(speed)}", (int(cx), int(cy) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
                )

                dir_var_hit = False
                if obj_id in prev_positions:
                    move = current - prev_positions[obj_id]
                    move_norm = float(np.linalg.norm(move))
                    if move_norm > MOVE_MIN:
                        move_dir = move / move_norm
                        dir_history.setdefault(obj_id, []).append(move_dir)
                        if len(dir_history[obj_id]) > 5:
                            dir_history[obj_id].pop(0)

                prev_positions[obj_id] = current

                accident = False
                reasons = []

                if obj_id in dir_history and len(dir_history[obj_id]) >= 3:
                    dirs = np.array(dir_history[obj_id])
                    var = float(np.var(dirs, axis=0).sum())
                    if var > DIR_VAR_TH and speed > SPEED_LOW:
                        dir_var_hit = True

                sudden_stop_hit = False
                if obj_id in prev_speed:
                    if prev_speed[obj_id] > SPEED_HIGH and speed < SPEED_LOW:
                        sudden_stop_hit = True

                if layer == "far":
                    if sudden_stop_hit and track_age[obj_id] >= MIN_TRACK_AGE:
                        hot_add_event(cell, frame_idx, "stop")
                    if dir_var_hit and track_age[obj_id] >= MIN_TRACK_AGE:
                        hot_add_event(cell, frame_idx, "dir")

                elif layer == "mid":
                    if track_age[obj_id] >= MIN_TRACK_AGE:
                        delta = 0
                        if sudden_stop_hit:
                            delta += 2
                        if dir_var_hit:
                            delta += 1

                        dv = float(prev_speed.get(obj_id, speed) - speed)
                        if dv > 20:
                            delta += 1

                        mid_add_score(obj_id, frame_idx, delta)
                        score = mid_sum_score(obj_id)

                        if score >= MID_SCORE_TH:
                            mid_hold[obj_id] = mid_hold.get(obj_id, 0) + 1
                        else:
                            mid_hold[obj_id] = 0

                        if mid_hold.get(obj_id, 0) >= MID_HOLD_FRAMES:
                            accident = True
                            reasons.append(f"mid_score={score}")
                            if sudden_stop_hit:
                                reasons.append("sudden_stop")
                            if dir_var_hit:
                                reasons.append("dir_var")

                else:  # near
                    if track_age[obj_id] >= MIN_TRACK_AGE:
                        if dir_var_hit:
                            accident = True
                            reasons.append("dir_var")

                        if sudden_stop_hit:
                            accident = True
                            reasons.append("sudden_stop")

                        for other in tracked_objects:
                            if other.id == obj_id:
                                continue

                            other_id = other.id
                            if track_age.get(other_id, 0) < MIN_TRACK_AGE:
                                continue

                            pair = tuple(sorted([obj_id, other_id]))
                            ox, oy = other.estimate[0]
                            dist = float(np.linalg.norm(current - np.array([ox, oy], dtype=np.float32)))

                            s1 = size
                            s2 = float(other.last_detection.data.get("size", 0))
                            adaptive_dist = 0.6 * min(s1, s2)

                            if adaptive_dist < 10:
                                near_counter[pair] = 0
                                continue

                            if dist < adaptive_dist and speed < SPEED_LOW:
                                near_counter[pair] = near_counter.get(pair, 0) + 1
                            else:
                                near_counter[pair] = 0

                            if near_counter[pair] >= NEAR_FRAMES:
                                accident = True
                                reasons.append("collision")
                                break

                prev_speed[obj_id] = speed

                if accident:
                    last_frame = last_accident_frame.get(obj_id, -999999)
                    if frame_idx - last_frame >= EVENT_COOLDOWN:
                        last_accident_frame[obj_id] = frame_idx
                        accident_triggered = True
                        accident_labels.add("accident")

                    w_box = int(size)
                    h_box = int(size * 0.6)
                    x1 = int(cx - w_box / 2)
                    y1 = int(cy - h_box / 2)
                    x2 = int(cx + w_box / 2)
                    y2 = int(cy + h_box / 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "ACCIDENT:" + ",".join(reasons),
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

            cell_w = w / GRID_COLS
            cell_h = h / GRID_ROWS

            for cell in list(hot_events.keys()):
                activated, stop_cnt, dir_cnt = hot_check_and_activate(cell, frame_idx)
                active = is_hot_active(cell, frame_idx)

                if not active and not activated:
                    continue

                row, col = cell
                x1 = int(col * cell_w)
                y1 = int(row * cell_h)
                x2 = int((col + 1) * cell_w)
                y2 = int((row + 1) * cell_h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"HOTSPOT stop={stop_cnt} dir={dir_cnt}",
                    (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                )

                if activated:
                    last_frame = last_hotspot_frame.get(cell, -999999)
                    if frame_idx - last_frame >= EVENT_COOLDOWN:
                        last_hotspot_frame[cell] = frame_idx
                        accident_triggered = True
                        accident_labels.add("hotspot")

            if accident_triggered:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    img_path = tmp_file.name
                    cv2.imwrite(img_path, frame)
                    yield img_path, list(accident_labels)

    except asyncio.CancelledError:
        print("🔴 車禍偵測任務被強制取消")
        cap.release()
        raise

    except Exception as e:
        if on_error:
            await on_error(f"車禍偵測發生錯誤：{str(e)}")

    cap.release()