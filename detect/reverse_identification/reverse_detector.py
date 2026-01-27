import cv2
import numpy as np
import asyncio
import os
import time
from ultralytics import YOLO
from norfair import Detection, Tracker
import tempfile
import torch

# -----------------------------
# Norfair distance
# -----------------------------
def distance(a, b):
    p1 = np.array(a.points[0] if hasattr(a, "points") else a.estimate[0])
    p2 = np.array(b.points[0] if hasattr(b, "points") else b.estimate[0])
    return np.linalg.norm(p1 - p2)


def point_in_poly(x, y, poly):
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0


# -----------------------------
# 主入口（給 Discord 用）
# -----------------------------
async def detect_reverse_live(
    video_path: str,
    on_error,
    interval: int = 5,
    profile: str = "more",
    model_path: str = "yolov8n.pt",
    config: dict | None = None,
):
    try:
        # -------- config --------
        min_conf = config.get("min_conf", 0.3)
        move_min_pixels = config.get("move_min_pixels", 3)
        roads_cfg = config.get("roads", [])

        roads = []
        road_dir_by_name = {}

        for r in roads_cfg:
            poly = np.array(r["poly"])
            direction = np.array(r["dir"], dtype=float)
            direction /= np.linalg.norm(direction)
            roads.append({"name": r["name"], "poly": poly, "dir": direction})
            road_dir_by_name[r["name"]] = direction

        # -------- model --------
        model = YOLO("yolov8n.pt")
        model.to("cpu")

        tracker = Tracker(
            distance_function=distance,
            distance_threshold=30,
            hit_counter_max=8,
            initialization_delay=2,
            pointwise_hit_counter_max=3,
            detection_threshold=min_conf,
        )

        cap = cv2.VideoCapture(video_path)

        prev_positions = {}
        reported_ids = set()
        track_road = {}   # ⭐ 關鍵：記錄 id → road name

        last_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                await on_error("讀不到串流畫面")
                break

            now = time.time()
            if now - last_time < interval:
                await asyncio.sleep(0.01)
                continue
            last_time = now

            results = model(frame)[0]
            detections = []

            for box in results.boxes:
                cls = int(box.cls)
                conf = float(box.conf)

                if cls != 2 or conf < min_conf:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)

                for r in roads:
                    if point_in_poly(cx, cy, r["poly"]):
                        print(f"[DETECT] 車輛 路段={r['name']} conf={conf:.2f}")
                        detections.append(
                            Detection(
                                points=np.array([[cx, cy]]),
                                scores=np.array([conf]),
                            )
                        )
                        break

            tracked = tracker.update(detections)

            for obj in tracked:
                cur = np.array(obj.estimate[0])

                # 第一次看到這個 id → 綁定路段
                if obj.id not in prev_positions:
                    prev_positions[obj.id] = cur

                    # 用目前位置決定屬於哪條路
                    for r in roads:
                        if point_in_poly(cur[0], cur[1], r["poly"]):
                            track_road[obj.id] = r["name"]
                            break

                    continue

                prev = np.array(prev_positions[obj.id])
                move = cur - prev
                dist = np.linalg.norm(move)
                prev_positions[obj.id] = cur

                if dist < move_min_pixels:
                    continue

                road_name = track_road.get(obj.id)
                if road_name is None:
                    continue

                road_dir = road_dir_by_name.get(road_name)
                if road_dir is None:
                    continue

                move_dir = move / dist
                dot = np.dot(move_dir, road_dir)

                print(f"[DIR] id={obj.id} road={road_name} dot={dot:.2f}")
                # Debug, 要改回<0
                if dot > 0:   # 之後記得改回 < 0
                    if obj.id in reported_ids:
                        continue

                    reported_ids.add(obj.id)

                    # ✅ 逆向觸發：用 tempfile 建立臨時截圖檔，回傳給上層
                    # ✅ 逆向觸發：複製當下畫面 + 畫「點 / ID / 行進方向箭頭」後再存檔
                    frame_copy = frame.copy()

                    # --- 1) 準備座標（目前點、上一點、移動方向） ---
                    p_cur = (int(cur[0]), int(cur[1]))
                    p_prev = (int(prev[0]), int(prev[1]))

                    # 箭頭長度（像素），你可以自行調整 30~80
                    ARROW_LEN = 60

                    # 如果 dist > 0 才畫箭頭（理論上你前面已經 dist < move_min_pixels continue 了）
                    end_x = int(cur[0] + move_dir[0] * ARROW_LEN)
                    end_y = int(cur[1] + move_dir[1] * ARROW_LEN)
                    p_end = (end_x, end_y)

                    # --- 2) 畫點（中心點）---
                    cv2.circle(frame_copy, p_cur, 6, (0, 255, 255), -1)  # 黃色實心點
                    cv2.circle(frame_copy, p_cur, 10, (0, 0, 0), 2)      # 黑色外框

                    # （可選）畫上一個位置，方便看軌跡
                    cv2.circle(frame_copy, p_prev, 4, (255, 255, 255), -1)  # 白點
                    cv2.line(frame_copy, p_prev, p_cur, (255, 255, 255), 2)

                    # --- 3) 畫行進方向箭頭 ---
                    cv2.arrowedLine(frame_copy, p_cur, p_end, (0, 0, 255), 3, tipLength=0.35)  # 紅色箭頭

                    # --- 4) 畫 ID（和一些 debug 資訊）---
                    label = f"ID:{obj.id} road:{road_name} dot:{dot:.2f}"
                    # 文字位置：點的右上方
                    text_pos = (p_cur[0] + 12, p_cur[1] - 12)

                    # 先畫黑色描邊，讓字在任何背景都清楚
                    cv2.putText(frame_copy, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame_copy, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    # --- 5) 存成臨時圖檔 ---
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        img_path = tmp_file.name

                    ok = cv2.imwrite(img_path, frame_copy)
                    print(f"[DEBUG] imwrite={ok} path={img_path}")
                    if not ok:
                        continue

                    print(f"[REVERSE] 車輛 ID={obj.id}, 路段={road_name}")
                    yield img_path, ["reverse"]


    except Exception as e:
        await on_error(str(e))
    finally:
        try:
            cap.release()
        except:
            pass
