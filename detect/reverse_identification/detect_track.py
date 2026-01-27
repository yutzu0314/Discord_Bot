import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

# --- 1. Norfair 距離函式 ---
'''
Norfair 會把「新偵測到的點」跟「前一幀的追蹤點」配對
判斷「這顆點是不是同一台車？」

points = 新的 Detection
estimate = 已存在的追蹤對象
np.linalg.norm(p1 - p2) = 兩點距離
'''
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


# --- 2. 設定追蹤器 ---
'''
新偵測到的點，距離舊點多遠以內才算「同一台車」？

監視器 車很小: 20~30
車子就在面前: 40~80
高解析: 50~100
'''
tracker = Tracker(
    distance_function=distance,
    distance_threshold=30,
)

# --- 3. YOLOv8 ---
'''
yolov8n: 最快但不準
yolov8s
yolov8m 
yolov8l: 超準但最慢
'''
model = YOLO("yolov8n.pt")

# --- 4. 道路方向 ---
'''
[a, b]
左/右/上下開始  a=1/-1/0
斜率水平/向下/向上  b=0/1/-1
越多數字增加, 如向右1格、向上2格=[1, -2]
'''
ROAD_DIR = np.array([1, -1])
ROAD_DIR = ROAD_DIR / np.linalg.norm(ROAD_DIR)

# --- 5. 用來記錄每台車上一幀的位置 ---
prev_positions = {}

# --- 6. 讀取攝影機 ---
url = "https://cctv-ss04.thb.gov.tw/T63-6K+400"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("讀不到畫面")
        break

    results = model(frame)[0]

    # --- YOLO 偵測結果轉成 Norfair Detection ---
    detections = []
    for box in results.boxes:
        cls = int(box.cls)
        if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            x1, y1, x2, y2 = box.xyxy[0]
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            detections.append(Detection(points=np.array([[cx, cy]])))

    # 用 Norfair 追蹤
    tracked_objects = tracker.update(detections)

    # --- 7. 處理每台追蹤到的車 ---
    for obj in tracked_objects:
        obj_id = obj.id
        cx, cy = obj.estimate[0]
        current = np.array([cx, cy])

        # 畫中心點與 ID
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"ID {obj_id}", (int(cx), int(cy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- 若有上一幀的位置，就計算移動方向 ---
        if obj_id in prev_positions:
            prev = prev_positions[obj_id]
            move_vec = current - prev

            # 如果移動過小視為停車，不判斷方向
            if np.linalg.norm(move_vec) > 2:
                move_dir = move_vec / np.linalg.norm(move_vec)
                dot = np.dot(move_dir, ROAD_DIR)

                # --- 畫方向箭頭 ---
                end = (int(cx + move_dir[0] * 40), int(cy + move_dir[1] * 40))
                cv2.arrowedLine(frame, (int(cx), int(cy)), end, (0, 255, 255), 2)

                # --- 判斷逆向 ---
                if dot < 0:
                    cv2.putText(frame, "Reverse!", (int(cx)+20, int(cy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # 更新位置
        prev_positions[obj_id] = current

    cv2.imshow("tracking", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
