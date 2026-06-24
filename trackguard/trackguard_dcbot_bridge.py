# /home/inf431/trackguard-lte-04 (ver 2)/trackguard_dcbot_bridge.py

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

TRACKGUARD_ROOT = "/home/inf431/Discord_Bot/trackguard"
DCBOT_ROOT = "/home/inf431/Discord_Bot"
EVENT_DIR = os.path.join(DCBOT_ROOT, "trackguard_events")
EVENT_INDEX_FILE = os.path.join(EVENT_DIR, "events.jsonl")
DISABLED_BEHAVIOUR_TYPES = {
    "motorcycle_fallen",
    "fallen",
}
os.makedirs(EVENT_DIR, exist_ok=True)


def is_stream_source(path: str) -> bool:
    return str(path).startswith(("rtsp://", "http://", "https://"))


def normalize_source_video_abs(source_video: str) -> str:
    if not source_video:
        return ""

    if is_stream_source(source_video):
        return source_video

    return os.path.abspath(source_video)

def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _build_event_id(event: Dict[str, Any]) -> str:
    """
    依 collision 事件核心欄位建立穩定 event_id，用來去重。
    """
    frame_id = _safe_int(event.get("frame_id", 0))
    track_a = _safe_int(event.get("track_id", 0))
    track_b = _safe_int(event.get("track_id_secondary", 0))
    behaviour_type = str(event.get("behaviour_type", "unknown"))
    confidence = round(_safe_float(event.get("confidence", 0.0)), 2)

    raw = f"{behaviour_type}|{frame_id}|{min(track_a, track_b)}|{max(track_a, track_b)}|{confidence}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _build_payload(event: Dict[str, Any], source_video: Optional[str] = None) -> Dict[str, Any]:
    """
    把 TrackGuard 的 detection dict 轉成 DCBot 可用的事件格式。
    支援 collision / wrong_way。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    behaviour_type = event.get("behaviour_type", "collision")

    # DCBot 分類用
    if behaviour_type == "wrong_way":
        category = "traffic_violation"
        event_name = "wrong_way"
        title = "逆向偵測"
        note_prefix = "TrackGuard wrong_way"
    elif behaviour_type == "collision":
        category = "accident"
        event_name = "collision"
        title = "車禍偵測"
        note_prefix = "TrackGuard collision"
    elif behaviour_type in ("motorcycle_fallen", "fallen"):
        category = "accident"
        event_name = "motorcycle_fallen"
        title = "機車倒地偵測"
        note_prefix = "TrackGuard motorcycle_fallen"
    else:
        category = "traffic_event"
        event_name = behaviour_type
        title = behaviour_type
        note_prefix = f"TrackGuard {behaviour_type}"

    track_id = _safe_int(event.get("track_id", 0))
    track_id_secondary = _safe_int(event.get("track_id_secondary", 0))
    frame_id = _safe_int(event.get("frame_id", 0))
    confidence = _safe_float(event.get("confidence", 0.0))

    source_video_rel = source_video or event.get("source_video") or ""
    source_video_abs = (
        normalize_source_video_abs(source_video_rel)
        if source_video_rel
        else event.get("source_video_abs", "")
    )

    payload = {
        "event_id": _build_event_id(event),
        "event_source": "trackguard",

        # 這幾個很重要，給 DCBot 篩選用
        "event": event_name,
        "category": category,
        "behaviour_type": behaviour_type,
        "title": title,

        "timestamp": now,

        "source_video": source_video_rel,
        "source_video_abs": source_video_abs,

        # RTSP / stream 事件當下截圖，由 main.py 存好後傳進來
        "image_path": event.get("image_path"),
        "annotated_image_path": event.get("annotated_image_path"),

        "frame_id": frame_id,
        "track_id": track_id,
        "track_id_secondary": track_id_secondary,

        "class_primary": (
            event.get("class_primary")
            or event.get("class_name")
            or ("motorcycle" if behaviour_type in ("motorcycle_fallen", "fallen") else "unknown")
        ),
        
        "class_secondary": event.get("class_secondary", "unknown"),

        "confidence": confidence,
        "confidence_label": event.get("confidence_label", ""),
        "alert_level": event.get("alert_level", ""),
        "severity": event.get("severity", ""),

        "detection_mode": event.get("detection_mode", ""),
        "tier": event.get("tier", ""),

        "angle_diff_deg": _safe_float(event.get("angle_diff_deg", 0.0)),
        "expected_direction": event.get("expected_direction", []),
        "actual_direction": event.get("actual_direction", []),

        "iou_overlap": _safe_float(event.get("iou_overlap", 0.0)),
        "energy_loss_primary": _safe_float(event.get("energy_loss_primary", 0.0)),
        "energy_loss_secondary": _safe_float(event.get("energy_loss_secondary", 0.0)),

        "collision_point": event.get("collision_point", []),
        "bbox": event.get("bbox", []),
        "bbox_primary": event.get("bbox_primary", []),
        "bbox_secondary": event.get("bbox_secondary", []),
        "impact_direction": event.get("impact_direction", []),

        "raw_event": event,
        "status": "new",
        "bridge_created_at_unix": time.time(),
        "note": (
            f"{note_prefix} | "
            f"confidence={confidence:.1f} | "
            f"mode={event.get('detection_mode', '')} | "
            f"tier={event.get('tier', '')}"
        ),
    }

    return payload

def report_collision_event(
    event: Dict[str, Any],
    source_video: Optional[str] = None
) -> Optional[str]:

    behaviour_type = str(
        event.get("behaviour_type")
        or event.get("event")
        or ""
    ).lower()

    if behaviour_type in DISABLED_BEHAVIOUR_TYPES:
        print(
            f"[DCBOT BRIDGE] disabled event skipped: "
            f"{behaviour_type}"
        )
        return None

    payload = _build_payload(event, source_video=source_video)
    payload = _json_safe(payload)

    event_id = payload["event_id"]
    frame_id = _safe_int(payload.get("frame_id", 0))
    track_a = _safe_int(payload.get("track_id", 0))
    track_b = _safe_int(payload.get("track_id_secondary", 0))

    behaviour_type = payload.get("behaviour_type", "collision")
    filename = (
        f"{behaviour_type}_f{frame_id}_t"
        f"{track_a}_{track_b}_{event_id[:8]}.json"
    )

def report_collision_event(event: Dict[str, Any], source_video: Optional[str] = None) -> str:
    payload = _build_payload(event, source_video=source_video)
    payload = _json_safe(payload)

    event_id = payload["event_id"]
    frame_id = _safe_int(payload.get("frame_id", 0))
    track_a = _safe_int(payload.get("track_id", 0))
    track_b = _safe_int(payload.get("track_id_secondary", 0))

    behaviour_type = payload.get("behaviour_type", "collision")
    filename = f"{behaviour_type}_f{frame_id}_t{track_a}_{track_b}_{event_id[:8]}.json"
    
    output_path = os.path.join(EVENT_DIR, filename)
    tmp_path = output_path + ".tmp"

    if os.path.exists(output_path):
        print(f"[DCBOT BRIDGE] duplicate skipped: {output_path}")
        return output_path

    # 先寫 tmp，寫完再原子替換
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, output_path)

    with open(EVENT_INDEX_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

    print(
        f"[DCBOT BRIDGE] event saved | "
        f"frame={frame_id} | tracks={track_a}<->{track_b} | "
        f"confidence={payload['confidence']:.1f} | file={output_path}"
    )

    return output_path

def _json_safe(obj):
    """
    避免 NaN / Infinity / numpy 型別造成 JSON 問題
    """
    import math
    import numpy as np

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj

def report_trackguard_event(
    event: Dict[str, Any],
    source_video: Optional[str] = None
) -> Optional[str]:
    return report_collision_event(event, source_video=source_video)