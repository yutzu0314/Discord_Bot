import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker


class CarAccidentDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

        # =========================
        # 參數
        # =========================
        self.FPS = 30
        self.SPEED_HIGH = 60
        self.SPEED_LOW = 5
        self.MAX_SPEED = 200

        self.MOVE_MIN = 6
        self.DIR_VAR_TH = 0.05

        self.HISTORY_LEN = 5
        self.NEAR_FRAMES = 3

        # --- 分層：Near / Mid / Far ---
        self.SIZE_NEAR = 40
        self.SIZE_MID = 25

        # --- 追蹤可靠度 ---
        self.MIN_TRACK_AGE = 8

        # --- Mid 分數制 ---
        self.MID_SCORE_TH = 3
        self.MID_SCORE_WINDOW = int(1.2 * self.FPS)
        self.MID_HOLD_FRAMES = 3

        # --- Far 群體熱區（網格） ---
        self.GRID_COLS = 8
        self.GRID_ROWS = 6
        self.HOT_WINDOW = int(1.5 * self.FPS)
        self.HOT_SUDDEN_STOP_TH = 3
        self.HOT_COMBO_STOP = 2
        self.HOT_DIRVAR_TH = 1
        self.HOT_HOLD = int(2.0 * self.FPS)

        # 避免同一事件連續狂發
        self.EVENT_COOLDOWN = int(2.0 * self.FPS)

        self.tracker = Tracker(
            distance_function=self.distance,
            distance_threshold=30,
        )

        self.reset_state()

    def reset_state(self):
        self.prev_positions = {}
        self.track_history = {}
        self.prev_speed = {}
        self.dir_history = {}
        self.near_counter = {}
        self.last_seen = {}

        self.track_age = {}
        self.mid_score_hist = {}
        self.mid_hold = {}

        self.hot_events = {}
        self.hot_active = {}

        self.last_accident_frame = {}   # obj_id -> last frame
        self.last_hotspot_frame = {}    # cell -> last frame

        self.frame_idx = 0

    def distance(self, a, b):
        if hasattr(a, "points"):
            p1 = np.array(a.points[0])
        else:
            p1 = np.array(a.estimate[0])

        if hasattr(b, "points"):
            p2 = np.array(b.points[0])
        else:
            p2 = np.array(b.estimate[0])

        return np.linalg.norm(p1 - p2)

    def calc_avg_speed(self, history):
        if len(history) < 2:
            return 0

        dist = 0.0
        for i in range(1, len(history)):
            dist += np.linalg.norm(history[i][1] - history[i - 1][1])

        return min((dist / (len(history) - 1)) * self.FPS, self.MAX_SPEED)

    def get_layer(self, size):
        if size >= self.SIZE_NEAR:
            return "near"
        elif size >= self.SIZE_MID:
            return "mid"
        else:
            return "far"

    def grid_cell(self, cx, cy, w, h):
        cx = max(0, min(w - 1, int(cx)))
        cy = max(0, min(h - 1, int(cy)))
        col = int(cx / (w / self.GRID_COLS))
        row = int(cy / (h / self.GRID_ROWS))
        col = max(0, min(self.GRID_COLS - 1, col))
        row = max(0, min(self.GRID_ROWS - 1, row))
        return (row, col)

    def hot_add_event(self, cell, frame_idx, event_type):
        self.hot_events.setdefault(cell, []).append((frame_idx, event_type))
        cutoff = frame_idx - self.HOT_WINDOW
        self.hot_events[cell] = [(f, t) for (f, t) in self.hot_events[cell] if f >= cutoff]

    def hot_check_and_activate(self, cell, frame_idx):
        events = self.hot_events.get(cell, [])
        stop_cnt = sum(1 for _, t in events if t == "stop")
        dir_cnt = sum(1 for _, t in events if t == "dir")

        if stop_cnt >= self.HOT_SUDDEN_STOP_TH or (
            stop_cnt >= self.HOT_COMBO_STOP and dir_cnt >= self.HOT_DIRVAR_TH
        ):
            self.hot_active[cell] = frame_idx + self.HOT_HOLD
            return True, stop_cnt, dir_cnt

        return False, stop_cnt, dir_cnt

    def is_hot_active(self, cell, frame_idx):
        until = self.hot_active.get(cell, -1)
        return frame_idx <= until

    def mid_add_score(self, obj_id, frame_idx, delta):
        if delta == 0:
            return
        self.mid_score_hist.setdefault(obj_id, []).append((frame_idx, delta))
        cutoff = frame_idx - self.MID_SCORE_WINDOW
        self.mid_score_hist[obj_id] = [
            (f, d) for (f, d) in self.mid_score_hist[obj_id] if f >= cutoff
        ]

    def mid_sum_score(self, obj_id):
        return sum(d for _, d in self.mid_score_hist.get(obj_id, []))

    def run(self, video_path: str, output_path: str = None):
        self.reset_state()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片：{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 1:
            self.FPS = fps

        writer = None
        events = []

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("影片讀取失敗，無法取得第一幀")

        h, w = first_frame.shape[:2]

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, self.FPS, (w, h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_idx += 1
            h, w = frame.shape[:2]
            results = self.model(frame, verbose=False)[0]

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

            tracked_objects = self.tracker.update(detections)

            for obj in tracked_objects:
                obj_id = obj.id
                cx, cy = obj.estimate[0]
                current = np.array([cx, cy], dtype=np.float32)
                size = float(obj.last_detection.data.get("size", 0))

                self.last_seen[obj_id] = self.frame_idx
                self.track_age[obj_id] = self.track_age.get(obj_id, 0) + 1

                layer = self.get_layer(size)
                cell = self.grid_cell(cx, cy, w, h)

                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                cv2.putText(
                    frame, f"ID {obj_id} [{layer}]", (int(cx), int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                self.track_history.setdefault(obj_id, []).append((self.frame_idx, current))
                if len(self.track_history[obj_id]) > self.HISTORY_LEN:
                    self.track_history[obj_id].pop(0)

                speed = self.calc_avg_speed(self.track_history[obj_id])
                cv2.putText(
                    frame, f"v={int(speed)}", (int(cx), int(cy) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
                )

                dir_var_hit = False
                if obj_id in self.prev_positions:
                    move = current - self.prev_positions[obj_id]
                    move_norm = float(np.linalg.norm(move))
                    if move_norm > self.MOVE_MIN:
                        move_dir = move / move_norm
                        self.dir_history.setdefault(obj_id, []).append(move_dir)
                        if len(self.dir_history[obj_id]) > 5:
                            self.dir_history[obj_id].pop(0)

                self.prev_positions[obj_id] = current

                accident = False
                reasons = []

                if obj_id in self.dir_history and len(self.dir_history[obj_id]) >= 3:
                    dirs = np.array(self.dir_history[obj_id])
                    var = float(np.var(dirs, axis=0).sum())
                    if var > self.DIR_VAR_TH and speed > self.SPEED_LOW:
                        dir_var_hit = True

                sudden_stop_hit = False
                if obj_id in self.prev_speed:
                    if self.prev_speed[obj_id] > self.SPEED_HIGH and speed < self.SPEED_LOW:
                        sudden_stop_hit = True

                if layer == "far":
                    if sudden_stop_hit and self.track_age[obj_id] >= self.MIN_TRACK_AGE:
                        self.hot_add_event(cell, self.frame_idx, "stop")
                    if dir_var_hit and self.track_age[obj_id] >= self.MIN_TRACK_AGE:
                        self.hot_add_event(cell, self.frame_idx, "dir")

                elif layer == "mid":
                    if self.track_age[obj_id] >= self.MIN_TRACK_AGE:
                        delta = 0
                        if sudden_stop_hit:
                            delta += 2
                        if dir_var_hit:
                            delta += 1

                        dv = float(self.prev_speed.get(obj_id, speed) - speed)
                        if dv > 20:
                            delta += 1

                        self.mid_add_score(obj_id, self.frame_idx, delta)
                        score = self.mid_sum_score(obj_id)

                        if score >= self.MID_SCORE_TH:
                            self.mid_hold[obj_id] = self.mid_hold.get(obj_id, 0) + 1
                        else:
                            self.mid_hold[obj_id] = 0

                        if self.mid_hold.get(obj_id, 0) >= self.MID_HOLD_FRAMES:
                            accident = True
                            reasons.append(f"mid_score={score}")
                            if sudden_stop_hit:
                                reasons.append("sudden_stop")
                            if dir_var_hit:
                                reasons.append("dir_var")

                else:
                    if self.track_age[obj_id] >= self.MIN_TRACK_AGE:
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
                            if self.track_age.get(other_id, 0) < self.MIN_TRACK_AGE:
                                continue

                            pair = tuple(sorted([obj_id, other_id]))
                            ox, oy = other.estimate[0]
                            dist = float(np.linalg.norm(current - np.array([ox, oy], dtype=np.float32)))

                            s1 = size
                            s2 = float(other.last_detection.data.get("size", 0))
                            adaptive_dist = 0.6 * min(s1, s2)

                            if adaptive_dist < 10:
                                self.near_counter[pair] = 0
                                continue

                            if dist < adaptive_dist and speed < self.SPEED_LOW:
                                self.near_counter[pair] = self.near_counter.get(pair, 0) + 1
                            else:
                                self.near_counter[pair] = 0

                            if self.near_counter[pair] >= self.NEAR_FRAMES:
                                accident = True
                                reasons.append("collision")
                                break

                self.prev_speed[obj_id] = speed

                if accident:
                    last_frame = self.last_accident_frame.get(obj_id, -999999)
                    if self.frame_idx - last_frame >= self.EVENT_COOLDOWN:
                        self.last_accident_frame[obj_id] = self.frame_idx
                        timestamp_sec = self.frame_idx / self.FPS
                        events.append({
                            "type": "accident",
                            "frame": self.frame_idx,
                            "time_sec": round(timestamp_sec, 2),
                            "obj_id": int(obj_id),
                            "layer": layer,
                            "reasons": reasons,
                        })

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

            cell_w = w / self.GRID_COLS
            cell_h = h / self.GRID_ROWS

            for cell in list(self.hot_events.keys()):
                activated, stop_cnt, dir_cnt = self.hot_check_and_activate(cell, self.frame_idx)
                active = self.is_hot_active(cell, self.frame_idx)

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
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

                if activated:
                    last_frame = self.last_hotspot_frame.get(cell, -999999)
                    if self.frame_idx - last_frame >= self.EVENT_COOLDOWN:
                        self.last_hotspot_frame[cell] = self.frame_idx
                        timestamp_sec = self.frame_idx / self.FPS
                        events.append({
                            "type": "hotspot",
                            "frame": self.frame_idx,
                            "time_sec": round(timestamp_sec, 2),
                            "cell": cell,
                            "stop_cnt": stop_cnt,
                            "dir_cnt": dir_cnt,
                        })

            if writer:
                writer.write(frame)

        cap.release()
        if writer:
            writer.release()

        return {
            "success": True,
            "video_path": video_path,
            "output_path": output_path,
            "event_count": len(events),
            "events": events,
        }