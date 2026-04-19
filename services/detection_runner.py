# services/detection_runner.py

import asyncio
from typing import Callable, Awaitable


class DetectionRunner:
    def __init__(self, cog):
        self.cog = cog

    async def run_camera_job(
        self,
        camera: dict,
        detect_type: str,
        stop_event: asyncio.Event,
        send_result: Callable[[str, list[str]], Awaitable[None]],
    ):
        """
        camera 至少包含:
        {
            "id": 1,
            "name": "中山路口",
            "stream_url": "rtsp://..." 或 mp4/url
        }
        """
        stream_url = camera["stream_url"]

        try:
            if detect_type == "violation":
                await self.run_violation_detection(
                    stream_url=stream_url,
                    stop_event=stop_event,
                    send_result=send_result,
                )

            elif detect_type == "reverse":
                await self.run_reverse_detection(
                    stream_url=stream_url,
                    stop_event=stop_event,
                    send_result=send_result,
                    camera=camera,
                )

            elif detect_type == "accident":
                await self.run_accident_detection(
                    stream_url=stream_url,
                    stop_event=stop_event,
                    send_result=send_result,
                )

        except asyncio.CancelledError:
            print(f"🛑 camera={camera['name']} job cancelled")
            raise
        except Exception as e:
            print(f"❌ camera={camera['name']} job error: {e}")

    async def run_violation_detection(self, stream_url, stop_event, send_result):
        """
        這裡接你原本 detect_video_live() 或 violation 流程
        """
        from detect.detector import detect_video_live

        async def on_error(msg: str):
            print(f"❌ violation error: {msg}")

        async for img_path, class_names in detect_video_live(stream_url, on_error, interval=10):
            if stop_event.is_set():
                break
            await send_result(img_path, class_names)

    async def run_reverse_detection(self, stream_url, stop_event, send_result, camera):
        """
        這裡接你原本 reverse detector 的流程
        """
        while not stop_event.is_set():
            await asyncio.sleep(1)
            # TODO: 接你原本的逆向偵測
            # 偵測到後呼叫:
            # await send_result(img_path, ["reverse"])

    async def run_accident_detection(self, stream_url, stop_event, send_result):
        """
        這裡接你 TrackGuard / 車禍流程
        """
        while not stop_event.is_set():
            await asyncio.sleep(1)
            # TODO: 接你原本車禍偵測
            # await send_result(img_path, ["accident"])