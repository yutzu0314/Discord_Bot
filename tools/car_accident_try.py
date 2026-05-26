import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

# =========================
# 1. Norfair 距離函式
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

# =========================
# 2. Tracker
# =========================
tracker = Tracker(
    distance_function=distance,
    distance_threshold=30,
)

# =========================
# 3. YOLO
# =========================
model = YOLO("/home/inf431/Discord_Bot/runs/detect/car_v2_train/weights/best.pt")
#model = YOLO("best.pt")

# =========================
# 4. 參數
# =========================
FPS = 30
SPEED_HIGH = 60
SPEED_LOW = 5
MAX_SPEED = 200

MOVE_MIN = 6
DIR_VAR_TH = 0.05

HISTORY_LEN = 5
NEAR_FRAMES = 3

# --- 分層：Near / Mid / Far (以 bbox size 分段) ---
SIZE_NEAR = 40     # >= 40 近景
SIZE_MID  = 25     # 25~40 中景
# < 25 遠景

# --- 追蹤可靠度 ---
MIN_TRACK_AGE = 8  # 至少存在幾幀後才允許判斷事故（減少 ID swap 誤判）

# --- Mid 分數制 ---
MID_SCORE_TH = 3          # 分數門檻
MID_SCORE_WINDOW = int(1.2 * FPS)  # 分數記錄視窗（秒） -> 幀數
MID_HOLD_FRAMES = 3       # 達標需連續幾幀才觸發

# --- Far 群體熱區（網格） ---
GRID_COLS = 8  # 橫向切 8 格
GRID_ROWS = 6  # 直向切 6 格
HOT_WINDOW = int(1.5 * FPS)    # 熱區統計窗長（秒）
HOT_SUDDEN_STOP_TH = 3         # 同格 sudden_stop >= 3
HOT_COMBO_STOP = 2             # or sudden_stop >=2 且 dir_var >=1
HOT_DIRVAR_TH = 1
HOT_HOLD = int(2.0 * FPS)      # 觸發後維持顯示（秒）

# =========================
# 5. 狀態
# =========================
prev_positions = {}
track_history = {}
prev_speed = {}
dir_history = {}
near_counter = {}
last_seen = {}

# --- 新增：追蹤年齡、Mid 分數、熱區統計 ---
track_age = {}          # obj_id -> 生存幀數
mid_score_hist = {}     # obj_id -> list[(frame_idx, score_delta)]
mid_hold = {}           # obj_id -> 連續達標計數

# 熱區統計：cell -> list[(frame_idx, event_type)]
# event_type: "stop" / "dir"
hot_events = {}
hot_active = {}         # cell -> active_until_frame

frame_idx = 0

# =========================
# 6. 讀取影片
# =========================
cap = cv2.VideoCapture(r"/home/inf431/Discord_Bot/car_accident.mp4")

# =========================
# 工具：滑動平均速度
# =========================
def calc_avg_speed(history, fps):
    if len(history) < 2:
        return 0

    dist = 0.0
    for i in range(1, len(history)):
        dist += np.linalg.norm(history[i][1] - history[i-1][1])

    return min((dist / (len(history) - 1)) * fps, MAX_SPEED)

def get_layer(size):
    if size >= SIZE_NEAR:
        return "near"
    elif size >= SIZE_MID:
        return "mid"
    else:
        return "far"

def grid_cell(cx, cy, w, h):
    # clamp
    cx = max(0, min(w - 1, int(cx)))
    cy = max(0, min(h - 1, int(cy)))
    col = int(cx / (w / GRID_COLS))
    row = int(cy / (h / GRID_ROWS))
    col = max(0, min(GRID_COLS - 1, col))
    row = max(0, min(GRID_ROWS - 1, row))
    return (row, col)

def hot_add_event(cell, frame_idx, event_type):
    hot_events.setdefault(cell, []).append((frame_idx, event_type))
    # 清掉窗外
    cutoff = frame_idx - HOT_WINDOW
    hot_events[cell] = [(f, t) for (f, t) in hot_events[cell] if f >= cutoff]

def hot_check_and_activate(cell, frame_idx):
    events = hot_events.get(cell, [])
    stop_cnt = sum(1 for _, t in events if t == "stop")
    dir_cnt  = sum(1 for _, t in events if t == "dir")

    if stop_cnt >= HOT_SUDDEN_STOP_TH or (stop_cnt >= HOT_COMBO_STOP and dir_cnt >= HOT_DIRVAR_TH):
        hot_active[cell] = frame_idx + HOT_HOLD
        return True, stop_cnt, dir_cnt
    return False, stop_cnt, dir_cnt

def is_hot_active(cell, frame_idx):
    until = hot_active.get(cell, -1)
    if frame_idx <= until:
        return True
    return False

def mid_add_score(obj_id, frame_idx, delta):
    if delta == 0:
        return
    mid_score_hist.setdefault(obj_id, []).append((frame_idx, delta))
    cutoff = frame_idx - MID_SCORE_WINDOW
    mid_score_hist[obj_id] = [(f, d) for (f, d) in mid_score_hist[obj_id] if f >= cutoff]

def mid_sum_score(obj_id):
    return sum(d for _, d in mid_score_hist.get(obj_id, []))

# =========================
# 主迴圈
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    h, w = frame.shape[:2]
    results = model(frame)[0]

    # =========================
    # YOLO → Norfair
    # =========================
    detections = []
    for box in results.boxes:
        cls = int(box.cls)
        if cls in [0]:  # car
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

    # =========================
    # 處理每台車
    # =========================
    for obj in tracked_objects:
        obj_id = obj.id
        cx, cy = obj.estimate[0]
        current = np.array([cx, cy], dtype=np.float32)
        size = float(obj.last_detection.data.get("size", 0))

        last_seen[obj_id] = frame_idx
        track_age[obj_id] = track_age.get(obj_id, 0) + 1

        layer = get_layer(size)
        cell = grid_cell(cx, cy, w, h)

        # --- 畫 ID ---
        cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"ID {obj_id} [{layer}]", (int(cx), int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- 歷史 ---
        track_history.setdefault(obj_id, []).append((frame_idx, current))
        if len(track_history[obj_id]) > HISTORY_LEN:
            track_history[obj_id].pop(0)

        speed = calc_avg_speed(track_history[obj_id], FPS)
        cv2.putText(frame, f"v={int(speed)}", (int(cx), int(cy) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # =========================
        # 方向歷史（防抖）
        # =========================
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

        # =========================
        # 事故判斷（分層）
        # =========================
        accident = False
        reasons = []

        # 先算 dir_var（near/mid 用；far 只用來加熱區事件）
        if obj_id in dir_history and len(dir_history[obj_id]) >= 3:
            dirs = np.array(dir_history[obj_id])
            var = float(np.var(dirs, axis=0).sum())
            if var > DIR_VAR_TH and speed > SPEED_LOW:
                dir_var_hit = True

        # sudden stop（全部都可算，但 far 只加到熱區）
        sudden_stop_hit = False
        if obj_id in prev_speed:
            if prev_speed[obj_id] > SPEED_HIGH and speed < SPEED_LOW:
                sudden_stop_hit = True

        # ---------- Far：群體熱區 ----------
        if layer == "far":
            # 遠景不做單車 accident；只把事件丟到格子裡
            if sudden_stop_hit and track_age[obj_id] >= MIN_TRACK_AGE:
                hot_add_event(cell, frame_idx, "stop")
            if dir_var_hit and track_age[obj_id] >= MIN_TRACK_AGE:
                hot_add_event(cell, frame_idx, "dir")

        # ---------- Mid：分數制（不做 collision） ----------
        elif layer == "mid":
            if track_age[obj_id] >= MIN_TRACK_AGE:
                # 分數：stop +2, dir +1, 大幅降速(Δv) +1
                delta = 0
                if sudden_stop_hit:
                    delta += 2
                if dir_var_hit:
                    delta += 1
                # 額外：速度下降幅度（更穩，避免只看 v < low）
                dv = float(prev_speed.get(obj_id, speed) - speed)
                if dv > 20:  # 可調
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

        # ---------- Near：原本規則 ----------
        else:  # near
            if track_age[obj_id] >= MIN_TRACK_AGE:
                if dir_var_hit:
                    accident = True
                    reasons.append("dir_var")

                if sudden_stop_hit:
                    accident = True
                    reasons.append("sudden_stop")

                # collision（只在 near 做）
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

        # =========================
        # 畫事故框（單車事故）
        # =========================
        if accident:
            w_box = int(size)
            h_box = int(size * 0.6)
            x1 = int(cx - w_box / 2)
            y1 = int(cy - h_box / 2)
            x2 = int(cx + w_box / 2)
            y2 = int(cy + h_box / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "ACCIDENT:" + ",".join(reasons),
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

            print(f"[ACCIDENT] ID {obj_id} frame {frame_idx} {reasons}")

    # =========================
    # 畫熱區（Far 群體事故）
    # =========================
    cell_w = w / GRID_COLS
    cell_h = h / GRID_ROWS

    # 每幀檢查所有有事件的 cell
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
        cv2.putText(frame, f"HOTSPOT stop={stop_cnt} dir={dir_cnt}",
                    (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if activated:
            print(f"[HOTSPOT] cell={cell} frame={frame_idx} stop={stop_cnt} dir={dir_cnt}")

    cv2.imshow("tracking", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
