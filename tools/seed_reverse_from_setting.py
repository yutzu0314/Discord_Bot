# tools/seed_reverse_from_setting.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # ✅ 讓 tools/ 能 import 到專案根模組

import json
import asyncio

from services.camera_service import list_active_cameras
from services.reverse_service import (
    disable_all_profiles,
    upsert_profile,
    upsert_zone,
)

async def main():
    with open("setting.json", "r", encoding="utf-8") as f:
        jdata = json.load(f)

    roads = jdata.get("roads", [])

    # 你也可以直接寫死 guild_id，不想每次輸入就固定
    guild_id = int(input("guild_id? ").strip())

    cameras = await list_active_cameras(guild_id)
    cam_by_name = {c["name"]: c for c in cameras}

    ok = 0
    skip = 0

    for r in roads:
        if not r.get("reverse_enabled", False):
            continue

        name = r.get("name")
        cam = cam_by_name.get(name)
        if not cam:
            print(f"[SKIP] cameras 找不到同名路段：{name}")
            skip += 1
            continue

        reverse_cfg = r.get("reverse_config") or {}
        profile_name = reverse_cfg.get("profile", "more")
        min_conf = float(reverse_cfg.get("min_conf", 0.5))
        move_min_pixels = int(reverse_cfg.get("move_min_pixels", 5))
        zones = reverse_cfg.get("roads") or []

        # 啟用：先關全部，再把這個 profile 開起來
        await disable_all_profiles(cam["id"])
        pid = await upsert_profile(
            camera_id=cam["id"],
            profile_name=profile_name,
            min_conf=min_conf,
            move_min_pixels=move_min_pixels,
            is_enabled=True,
        )

        for z in zones:
            await upsert_zone(
                profile_id=pid,
                zone_name=z["name"],
                poly=z["poly"],
                dir_x=int(z["dir"][0]),
                dir_y=int(z["dir"][1]),
            )

        print(f"[OK] 匯入 reverse_config：{name} (camera_id={cam['id']}, profile={profile_name})")
        ok += 1

    print(f"Done. ok={ok}, skip={skip}")

if __name__ == "__main__":
    asyncio.run(main())