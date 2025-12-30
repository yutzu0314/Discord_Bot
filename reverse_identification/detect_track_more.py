import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

# -----------------------------
# 參數
# -----------------------------
MIN_CONF = 0.5          # YOLO 信心值門檻
MOVE_MIN_PIXELS = 5     # 箭頭啟動的最小位移（像素）

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
# Tracker（調小 inertia，並啟用 pointwise_hit_counter）
# -----------------------------
tracker = Tracker(
    distance_function=distance,
    distance_threshold=30,
    hit_counter_max=8,          # 預設 15，縮短存活時間
    initialization_delay=2,     # 快速出現 ID
    pointwise_hit_counter_max=3,  # 點沒被 match 幾幀後就視為不 live
    detection_threshold=MIN_CONF # 低於這個分數的 detection 直接忽略
)

# -----------------------------
# YOLOv8
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# 攝影機
# -----------------------------
#url = "https://cctv-ss04.thb.gov.tw/T63-6K+400"
url = "https://tcnvr3.taichung.gov.tw/f3949e40"
cap = cv2.VideoCapture(url)

# -----------------------------
# 多條路 polygon & 方向
# -----------------------------
main_road = np.array([
    #[2, 197], [205, 68], [281, 77], [261, 210]
    [251, 73], [3, 205], [288, 359], [312, 69]

])
dir_main = np.array([0, 1])
dir_main = dir_main / np.linalg.norm(dir_main)

side_road = np.array([
    #[148, 47], [2, 76], [2, 58], [117, 36]
    [344, 65], [505, 357], [637, 160], [413, 66]
])
dir_side = np.array([0, -1])
dir_side = dir_side / np.linalg.norm(dir_side)

roads = [
    {"poly": main_road, "dir": dir_main, "name": "main"},
    {"poly": side_road, "dir": dir_side, "name": "side"},
]

# 方便用名稱查方向
road_dir_by_name = {
    "main": dir_main,
    "side": dir_side,
}

# -----------------------------
# 檢查點是否在 polygon
# -----------------------------
def point_in_poly(x, y, poly):
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0

# -----------------------------
# 紀錄上一幀中心點
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

    # 偵測車輛（只用 cls = 2 = car）
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)

        # 類別 + 信心值篩選
        if cls != 2:
            continue
        if conf < MIN_CONF:
            continue

        x1, y1, x2, y2 = box.xyxy[0]
        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)

        # 判斷落在哪條路，順便把路名塞進 label
        for r in roads:
            if point_in_poly(cx, cy, r["poly"]):
                detections.append(
                    Detection(
                        points=np.array([[cx, cy]]),
                        scores=np.array([conf]),  # 給 tracker 用 detection_threshold / live_points
                        label=r["name"],
                    )
                )
                break  # 一台車只屬於一條路

    # 更新 tracker
    tracked_objects = tracker.update(detections)

    # 畫結果
    for obj in tracked_objects:
        # 只畫「live」的點，避免 ghost node
        # 我們只追蹤一個點，所以看 live_points[0]
        if obj.live_points is not None and len(obj.live_points) > 0:
            if not obj.live_points[0]:
                continue

        # obj.estimate 是 Kalman filter 預測的位置
        cx, cy = obj.estimate[0]
        cx = int(cx)
        cy = int(cy)

        # 取得道路名稱 & 方向（label 是我們在 Detection 裡塞的）
        road_name = obj.label
        if road_name not in road_dir_by_name:
            continue
        road_dir = road_dir_by_name[road_name]

        # 計算移動方向
        if obj.id in prev_positions:
            prev = np.array(prev_positions[obj.id])
            move_vec = np.array([cx, cy]) - prev
            dist = np.linalg.norm(move_vec)

            # 位移量要大於 MOVE_MIN_PIXELS 才畫箭頭，減少紅燈抖動
            if dist > MOVE_MIN_PIXELS:
                move_dir = move_vec / dist

                # 畫移動方向箭頭
                end = (int(cx + move_dir[0] * 40), int(cy + move_dir[1] * 40))
                cv2.arrowedLine(frame, (cx, cy), end, (0, 255, 255), 2)

                # 判斷逆向
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

        # 更新上一幀位置
        prev_positions[obj.id] = [cx, cy]

        # 畫中心點 & ID
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"{road_name} ID {obj.id}",
            (cx, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("tracking", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
