import json
import os
import time
from typing import Dict, Any, Optional

OUTPUT_DIR = "/home/inf431/Discord_Bot/trackguard/bridge_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def emit_trackguard_event(
    detection: Dict,
    image_path: Optional[str] = None,
    annotated_image_path: Optional[str] = None,
    source_video: Optional[str] = None
):
    event = {
        "type": "trackguard_event",
        "event": "collision",
        "timestamp": time.time(),
        "frame_id": detection.get("frame_id"),
        "confidence": float(detection.get("confidence", 0.0)),
        "track_id_a": detection.get("track_id"),
        "track_id_b": detection.get("track_id_secondary"),
        "class_primary": detection.get("class_primary", "unknown"),
        "class_secondary": detection.get("class_secondary", "unknown"),
        "image_path": image_path,
        "annotated_image_path": annotated_image_path,
        "source_video": source_video,
    }

    print(json.dumps(event, ensure_ascii=False), flush=True)