# services/detection_manager.py

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Set


@dataclass
class DetectionJob:
    camera_id: int
    camera_name: str
    stream_url: str
    detect_type: str
    owner_id: int
    task: asyncio.Task | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    watcher_ids: Set[int] = field(default_factory=set)


class DetectionManager:
    """
    第一版先用 camera_id 當 key：
    = 同一個鏡頭同時只能跑一個偵測流程
    之後若要支援同鏡頭多違規項目，再改成 (camera_id, detect_type)
    """
    def __init__(self):
        self.jobs: Dict[int, DetectionJob] = {}
        self._lock = asyncio.Lock()

    async def get_job(self, camera_id: int) -> Optional[DetectionJob]:
        async with self._lock:
            return self.jobs.get(camera_id)

    async def add_job(self, job: DetectionJob) -> bool:
        async with self._lock:
            if job.camera_id in self.jobs:
                return False
            self.jobs[job.camera_id] = job
            return True

    async def remove_job(self, camera_id: int):
        async with self._lock:
            self.jobs.pop(camera_id, None)

    async def stop_job(self, camera_id: int) -> bool:
        async with self._lock:
            job = self.jobs.get(camera_id)
            if not job:
                return False

            job.stop_event.set()

            if job.task and not job.task.done():
                job.task.cancel()

            return True

    async def list_jobs(self) -> list[DetectionJob]:
        async with self._lock:
            return list(self.jobs.values())