import os
import json
import asyncio
import tempfile
import time
import cv2
import sys

TRACKGUARD_ROOT = "/home/inf431/Discord_Bot/trackguard"
TRACKGUARD_MAIN = os.path.join(TRACKGUARD_ROOT, "main.py")
EVENT_DIR = "/home/inf431/Discord_Bot/trackguard_events"

TRACKGUARD_ROOT = "/home/inf431/Discord_Bot/trackguard"

if TRACKGUARD_ROOT not in sys.path:
    sys.path.insert(0, TRACKGUARD_ROOT)

from trackguard.class_config import TRACKGUARD_MODEL

def print_trackguard_debug():
    print("=== DEBUG TRACKGUARD ===")
    print(f"TrackGuard root: {TRACKGUARD_ROOT}")
    print(f"TrackGuard main: {TRACKGUARD_MAIN}")
    print(f"TrackGuard model: {TRACKGUARD_MODEL}")
    print(f"TrackGuard events dir: {EVENT_DIR}")
    print(f"TrackGuard results dir: {os.path.join(TRACKGUARD_ROOT, 'results')}")
    print(f"TrackGuard main exists: {os.path.exists(TRACKGUARD_MAIN)}")
    print(f"TrackGuard model exists: {os.path.exists(TRACKGUARD_MODEL)}")
    print(f"TrackGuard events dir exists: {os.path.exists(EVENT_DIR)}")

    try:
        if TRACKGUARD_ROOT not in sys.path:
            sys.path.insert(0, TRACKGUARD_ROOT)

        from trackguard.class_config import (
            TRACKGUARD_MODEL_PROFILE,
            TARGET_CLASSES,
            CLASS_IDS,
            get_person_classes,
            get_wrong_way_vehicle_classes,
            get_collision_participant_classes,
        )

        print("TrackGuard class_config: OK")
        print(f"▶ Model profile: {TRACKGUARD_MODEL_PROFILE}")
        print(f"▶ Target classes: {TARGET_CLASSES}")
        print(f"▶ Class IDs: {CLASS_IDS}")
        print(f"▶ Person classes excluded from wrong_way: {sorted(get_person_classes())}")
        print(f"▶ Wrong-way vehicle classes: {sorted(get_wrong_way_vehicle_classes())}")
        print(f"▶ Collision participant classes: {sorted(get_collision_participant_classes())}")

    except Exception as e:
        print(f"TrackGuard class_config: ERROR {repr(e)}")

    print("========================")

def cleanup_old_events_for_video(event_dir: str, video_path: str):
    if not os.path.exists(event_dir):
        return

    abs_video = os.path.abspath(video_path)

    for filename in os.listdir(event_dir):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(event_dir, filename)

        try:
            with open(path, "r", encoding="utf-8") as f:
                event = json.load(f)

            event_video = event.get("source_video_abs") or event.get("source_video") or ""

            if event_video and os.path.abspath(event_video) == abs_video:
                os.remove(path)
                print(f"[EVENT CLEANUP] removed old event: {filename}")

        except Exception as e:
            print(f"[EVENT CLEANUP ERROR] {filename}: {repr(e)}")
            
async def run_trackguard_process(video_path: str, detect_type: str, on_event=None):
    
    cleanup_old_events_for_video(EVENT_DIR, video_path)
    
    output_dir = os.path.join(TRACKGUARD_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = int(time.time())
    output_path = os.path.join(
        output_dir,
        f"dcbot_{detect_type}_{timestamp}.mp4"
    )
    
    cmd = [
        "/home/inf431/Discord_Bot/venv/bin/python",
        "/home/inf431/Discord_Bot/trackguard/main.py",
        "--video", video_path,
        "--model", TRACKGUARD_MODEL,
        "--physics",
        "--detect", detect_type,
        "--conf", "70",
        "--min-hits", "10",
        "--output", output_path,
        "--show",
    ]

    # 只有逆向或全部偵測才顯示方向場
    if detect_type in ("wrong_way", "all"):
        cmd.append("--show-direction-field")

    print("[TrackGuard runner] cmd =", " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    


    def make_event_snapshot(event: dict) -> dict:
        """
        如果 TrackGuard event 沒有 image_path，就從 source_video 的 frame_id 截一張圖，
        並畫上 bbox / collision label，讓 Discord 可以送圖。
        """
        if event.get("image_path") and os.path.exists(event.get("image_path")):
            return event

        source_video = event.get("source_video_abs") or event.get("source_video")
        frame_id = int(event.get("frame_id", 0))
        
        print(
            f"[SNAPSHOT DEBUG] behaviour={event.get('behaviour_type') or event.get('event')} "
            f"frame_id={frame_id} source={source_video}"
        )

        if not source_video or not os.path.exists(source_video):
            print(f"[SNAPSHOT] source video not found: {source_video}")
            return event

        cap = cv2.VideoCapture(source_video)
        if not cap.isOpened():
            print(f"[SNAPSHOT] cannot open video: {source_video}")
            return event

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print(f"[SNAPSHOT] failed to read frame={frame_id} from {source_video}")
            return event

        bbox = event.get("bbox") or event.get("raw_event", {}).get("bbox")
        confidence = event.get("confidence", 0)
        behaviour_type = event.get("behaviour_type") or event.get("event") or "trackguard_event"

        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(
                frame,
                f"{behaviour_type} {confidence:.1f}%",
                (max(0, x1), max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        out_path = os.path.join(
            tempfile.gettempdir(),
            f"trackguard_{event.get('event_id', 'event')[:8]}_f{frame_id}.jpg"
        )

        ok = cv2.imwrite(out_path, frame)
        print(f"[SNAPSHOT] imwrite={ok} path={out_path}")

        if ok:
            event["image_path"] = out_path
            event["annotated_image_path"] = out_path

        return event

    def event_matches_detect_type(event: dict, detect_type: str) -> bool:
        behaviour_type = (
            event.get("behaviour_type")
            or event.get("event")
            or event.get("raw_event", {}).get("behaviour_type")
            or ""
        )

        behaviour_type = str(behaviour_type)

        if detect_type == "collision":
            return behaviour_type == "collision"

        if detect_type == "wrong_way":
            return behaviour_type == "wrong_way"

        if detect_type == "all":
            return behaviour_type in {
                "collision",
                "wrong_way",
                "motorcycle_fallen",
                "fallen",
                "turn",
                "brake",
                "decelerating",
            }

        return behaviour_type == detect_type

    def format_collision_debug(event: dict) -> str:
        raw = event.get("raw_event", {}) or {}

        def get_value(key, default="N/A"):
            return event.get(key, raw.get(key, default))

        behaviour_type = get_value("behaviour_type")
        confidence = get_value("confidence", 0)
        confidence_label = get_value("confidence_label")
        alert_level = get_value("alert_level")
        detection_mode = get_value("detection_mode")
        state = get_value("state")
        tier = get_value("tier")
        persist_count = get_value("persist_count")

        track_id = get_value("track_id")
        track_id_secondary = get_value("track_id_secondary")

        class_primary = get_value("class_primary")
        class_secondary = get_value("class_secondary")

        iou = get_value("iou_overlap")
        energy_i = get_value("energy_loss_primary")
        energy_j = get_value("energy_loss_secondary")
        ar_i = get_value("ar_change_i")
        ar_j = get_value("ar_change_j")
        area_i = get_value("area_change_i")
        area_j = get_value("area_change_j")
        ars_i = get_value("ars_zscore_i")
        ars_j = get_value("ars_zscore_j")

        frame_id = get_value("frame_id")
        event_id = get_value("event_id")

        return (
            "```text\n"
            "[TrackGuard Collision Debug]\n"
            f"event_id: {event_id}\n"
            f"frame_id: {frame_id}\n"
            f"behaviour: {behaviour_type}\n"
            f"tracks: {track_id} <-> {track_id_secondary}\n"
            f"classes: {class_primary} <-> {class_secondary}\n"
            f"confidence: {confidence}\n"
            f"confidence_label: {confidence_label}\n"
            f"alert_level: {alert_level}\n"
            f"state: {state}\n"
            f"persist_count: {persist_count}\n"
            f"tier: {tier}\n"
            f"detection_mode: {detection_mode}\n"
            "\n"
            "[Evidence]\n"
            f"iou_overlap: {iou}\n"
            f"energy_loss_primary: {energy_i}\n"
            f"energy_loss_secondary: {energy_j}\n"
            f"ar_change_i: {ar_i}\n"
            f"ar_change_j: {ar_j}\n"
            f"area_change_i: {area_i}\n"
            f"area_change_j: {area_j}\n"
            f"ars_zscore_i: {ars_i}\n"
            f"ars_zscore_j: {ars_j}\n"
            "```"
        )

    async def read_event_files():
        print(f"[EVENT WATCHER] started, dir={EVENT_DIR}")

        # 記錄本次 TrackGuard 開始監看的時間
        start_ns = time.time_ns()

        # key: filename, value: mtime_ns
        processed = {}

        while proc.returncode is None:
            try:
                if not os.path.exists(EVENT_DIR):
                    print(f"[EVENT WATCHER] dir not exists: {EVENT_DIR}")
                    await asyncio.sleep(0.5)
                    continue

                files = [
                    f for f in os.listdir(EVENT_DIR)
                    if f.endswith(".json")
                ]

                for filename in sorted(files):
                    path = os.path.join(EVENT_DIR, filename)

                    try:
                        stat = os.stat(path)
                    except FileNotFoundError:
                        continue

                    mtime_ns = stat.st_mtime_ns

                    # 只處理本次 runner 啟動後才建立/更新的檔案
                    if mtime_ns < start_ns:
                        continue

                    # 同一個檔案同一個修改時間已處理過，就跳過
                    if processed.get(filename) == mtime_ns:
                        continue

                    processed[filename] = mtime_ns

                    # 避免 TrackGuard 還沒寫完 JSON
                    await asyncio.sleep(0.05)

                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            event = json.load(f)

                        event_video = event.get("source_video_abs") or event.get("source_video") or ""

                        # 避免讀到其他影片或其他 camera 的事件
                        if event_video and os.path.abspath(event_video) != os.path.abspath(video_path):
                            continue
                        
                        if not event_matches_detect_type(event, detect_type):
                            print(
                                f"[EVENT WATCHER] skip event by detect_type={detect_type}, "
                                f"event={event.get('event')}, behaviour={event.get('behaviour_type')}"
                            )
                            continue
                        
                        print(f"[PARSED EVENT FILE] {filename}")
                        print(event)


                        print(
                            f"[EVENT IMAGE DEBUG BEFORE SNAPSHOT] "
                            f"behaviour={event.get('behaviour_type') or event.get('event')} "
                            f"image_path={event.get('image_path')} "
                            f"annotated_image_path={event.get('annotated_image_path')} "
                            f"source={event.get('source_video_abs') or event.get('source_video')}"
                        )
                        
                        event = make_event_snapshot(event)
                        
                        print(
                            f"[EVENT IMAGE DEBUG AFTER SNAPSHOT] "
                            f"image_path={event.get('image_path')} "
                            f"annotated_image_path={event.get('annotated_image_path')}"
                        )

                        if (
                            event.get("behaviour_type") == "collision"
                            or event.get("event") == "collision"
                            or event.get("raw_event", {}).get("behaviour_type") == "collision"
                        ):
                            event["discord_debug_text"] = format_collision_debug(event)

                        print("[PARSED EVENT]", event)
                        if on_event is not None:
                            await on_event(event)

                    except Exception as e:
                        print(f"[EVENT FILE ERROR] {path}: {repr(e)}")

            except Exception as e:
                print(f"[EVENT WATCHER ERROR] {repr(e)}")

            await asyncio.sleep(0.2)
            
    async def read_stdout():
        IMPORTANT_KEYWORDS = [
            "SUDDEN DECELERATION",
            "COLLISION EVIDENCE DETECTED",
            "Output video saved successfully",
            "FINAL STATISTICS",
            "Processing completed successfully",
            "Collision:",
        ]

        SKIP_KEYWORDS = [
            "[DEBUG",
            "[DEBUG FALLBACK]",
            "FALLBACK CHECK",
            "DISAPPEARANCE COLLISION CHECK",
            "SKIP COLLISION",
            "SKIP EVASIVE COLLISION",
            "EVASIVE MANEUVER DETECTED",
            "Average Processing Time",
            "Average FPS",
            "Active Tracks",
            "Ghost Reidentifications",
            "File size:",
            "Location:",
            "Filename ",
            "does not match standard format",
        ]

        last_printed = None

        while True:
            line = await proc.stdout.readline()
            if not line:
                break

            text = line.decode("utf-8", errors="ignore").strip()
            if not text:
                continue

            try:
                event = json.loads(text)
                
                if (
                    event.get("type") == "trackguard_event"
                    or event.get("event_source") == "trackguard"
                    or "image_path" in event
                    or "annotated_image_path" in event
                ):
                    
                    if event_matches_detect_type(event, detect_type):
                        print("[PARSED EVENT]", event)
                        if on_event is not None:
                            await on_event(event)
                    else:
                        print(
                            f"[STDOUT EVENT SKIP] detect_type={detect_type}, "
                            f"event={event.get('event')}, behaviour={event.get('behaviour_type')}"
                        )
                    continue
            
            except json.JSONDecodeError:
                pass

            if any(skip_word in text for skip_word in SKIP_KEYWORDS):
                continue

            if any(keyword in text for keyword in IMPORTANT_KEYWORDS):
                if text != last_printed:
                    print("[TrackGuard stdout]", text)
                    last_printed = text

    async def read_stderr():
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            print("[TrackGuard stderr]", line.decode("utf-8", errors="ignore").strip())

    try:
        await asyncio.gather(read_stdout(), read_stderr(), read_event_files())
        await proc.wait()
        return proc

    except asyncio.CancelledError:
        print("[TrackGuard runner] cancelled, terminating subprocess...")

        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

        raise