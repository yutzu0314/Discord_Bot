import asyncio
import json
import os


TRACKGUARD_ROOT = "/home/inf431/trackguard"
TRACKGUARD_MAIN = os.path.join(TRACKGUARD_ROOT, "main.py")


async def run_trackguard_process(video_path: str, on_event):
    proc = await asyncio.create_subprocess_exec(
        "python3",
        TRACKGUARD_MAIN,
        "--video", video_path,
        "--physics",
        "--detect", "collision",
        "--tracker", "bytetrack",
        cwd=TRACKGUARD_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

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
                    print("[PARSED EVENT]", event)
                    await on_event(event)
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

    await asyncio.gather(read_stdout(), read_stderr())
    await proc.wait()
    return proc