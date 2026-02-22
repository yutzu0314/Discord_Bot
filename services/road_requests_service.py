# services/road_requests_service.py
from __future__ import annotations

from db import aexec, afetchall
from services.camera_service import get_or_create_camera


async def save_road_request(
    *,
    guild_id: int | None,
    admin_channel_id: int | None,
    message_id: int | None,
    reporter_id: int,
    road_name: str,
    image_url: str | None = None,  # 你 modal 的「監視器網址」
    note: str | None = None,
    status: str = "pending",
) -> None:
    """
    寫入 road_requests（申請單）
    """
    sql = """
    INSERT INTO road_requests
      (guild_id, admin_channel_id, message_id, reporter_id,
       road_name, image_url, note, status)
    VALUES
      (:gid, :acid, :mid, :rid,
       :road, :img, :note, :st)
    """
    params = {
        "gid": guild_id,
        "acid": admin_channel_id,
        "mid": message_id,
        "rid": reporter_id,
        "road": road_name,
        "img": image_url,
        "note": note,
        "st": status,
    }
    await aexec(sql, params)


async def get_request_by_message_id(message_id: int) -> dict | None:
    """
    透過管理員頻道那則審核訊息的 message_id 找到申請單
    """
    sql = "SELECT * FROM road_requests WHERE message_id = :mid LIMIT 1"
    rows = await afetchall(sql, {"mid": message_id})
    return rows[0] if rows else None


async def update_request_status_by_message_id(
    *,
    message_id: int,
    status: str,
    reviewed_by: int | None = None,
) -> None:
    """
    更新申請單狀態 + 審核者/審核時間
    status: pending / approved / rejected / need_edit
    """
    sql = """
    UPDATE road_requests
    SET status = :st,
        reviewed_by = :rb,
        reviewed_at = NOW()
    WHERE message_id = :mid
    """
    await aexec(sql, {"st": status, "rb": reviewed_by, "mid": message_id})


async def approve_request_create_camera(
    *,
    message_id: int,
    reviewed_by: int | None,
) -> int | None:
    """
    核准申請：
    1) 找申請單
    2) 立即建立/取得 camera（get_or_create_camera）
    3) 將 road_requests.status=approved 並寫回 camera_id
    回傳 camera_id
    """
    req = await get_request_by_message_id(message_id)
    if not req:
        return None

    guild_id = req.get("guild_id")
    admin_channel_id = req.get("admin_channel_id")
    road_name = req.get("road_name")
    image_url = req.get("image_url")  # 你目前填的是監視器網址

    # 這邊用「申請的監視器網址」當作 camera.stream_url
    # （你的 cameras 表欄位叫 stream_url，剛好可用）
    stream_url = image_url

    camera_id = None
    if guild_id is not None and admin_channel_id is not None and road_name:
        camera_id = await get_or_create_camera(
            guild_id=guild_id,
            channel_id=admin_channel_id,  # 目前你有在申請單存 channel_id 的話可換成那個
            name=road_name,
            road_name=road_name,
            latitude=req.get("latitude"),
            longitude=req.get("longitude"),
            stream_url=stream_url,
        )

    sql = """
    UPDATE road_requests
    SET status = 'approved',
        reviewed_by = :rb,
        reviewed_at = NOW(),
        camera_id = :cam_id
    WHERE message_id = :mid
    """
    await aexec(sql, {"rb": reviewed_by, "cam_id": camera_id, "mid": message_id})
    return camera_id