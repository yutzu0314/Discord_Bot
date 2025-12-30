import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

# -----------------------------
# 參數
# -----------------------------
MIN_CONF = 0.5
MOVE_MIN_PIXELS = 5

# -----------------------------
# Norfair distance function
# -----------------------------
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

# -----------------------------
# Tracker 設定
# -----------------------------
tracker = Tracker(
    distance_function=distance,
    distance_threshold=30,
    hit_counter_max=8,
    initialization_delay=2,
    pointwise_hit_counter_max=3,
    detection_threshold=MIN_CONF
)

# -----------------------------
# YOLOv8 模型
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# 攝影機
# -----------------------------
url = "rtsp://rtsp:rtsp1234@140.128.124.58:7177/cam/realmonitor?channel=27&subtype=0"
cap = cv2.VideoCapture(url)

# -----------------------------
# ✅ 單一路 polygon & 方向（只剩 main）
# -----------------------------
main_road = np.array([
    [1121, 513],
    [1271, 512],
    [1438, 682],
    [1136, 738]
])

# ✅ 正向方向（請依實際車流方向調整）
dir_main = np.array([0, 1])
dir_main = dir_main / np.linalg.norm(dir_main)

road_dir_by_name = {
    "main": dir_main,
}

# -----------------------------
# 檢查點是否在 polygon 中
# -----------------------------
def point_in_poly(x, y, poly):
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0

# -----------------------------
# 紀錄上一幀位置
# -----------------------------
prev_positions = {}

# -----------------------------
# 主迴圈
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("讀不到畫面")
        break

    results = model(frame)[0]
    detections = []

    # -----------------------------
    # 只偵測車輛(cls=2)
    # -----------------------------
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)

        if cls != 2 or conf < MIN_CONF:
            continue

        x1, y1, x2, y2 = box.xyxy[0]
        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)

        # ✅ 只判斷 main road
        if point_in_poly(cx, cy, main_road):
            detections.append(
                Detection(
                    points=np.array([[cx, cy]]),
                    scores=np.array([conf]),
                    label="main",
                )
            )

    # -----------------------------
    # 更新 Tracker
    # -----------------------------
    tracked_objects = tracker.update(detections)

    # -----------------------------
    # 繪製結果 + 逆向判斷
    # -----------------------------
    for obj in tracked_objects:
        if obj.live_points is not None and len(obj.live_points) > 0:
            if not obj.live_points[0]:
                continue

        cx, cy = obj.estimate[0]
        cx, cy = int(cx), int(cy)

        road_name = obj.label
        if road_name not in road_dir_by_name:
            continue

        road_dir = road_dir_by_name[road_name]

        # 計算移動方向
        if obj.id in prev_positions:
            prev = np.array(prev_positions[obj.id])
            move_vec = np.array([cx, cy]) - prev
            dist = np.linalg.norm(move_vec)

            if dist > MOVE_MIN_PIXELS:
                move_dir = move_vec / dist

                # 畫移動箭頭
                end = (int(cx + move_dir[0] * 40), int(cy + move_dir[1] * 40))
                cv2.arrowedLine(frame, (cx, cy), end, (0, 255, 255), 2)

                # ✅ 逆向判斷
                dot = float(np.dot(move_dir, road_dir))
                if dot < 0:
                    cv2.putText(
                        frame,
                        "reverse!",
                        (cx, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

        prev_positions[obj.id] = [cx, cy]

        # 畫中心點 & ID
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"ID {obj.id}",
            (cx, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # 顯示畫面
    cv2.imshow("tracking", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
