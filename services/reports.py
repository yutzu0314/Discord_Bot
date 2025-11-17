# services/reports.py
from db import aexec, afetchall
from services.camera_service import get_or_create_camera  # ⬅️ 新增這行 import

async def save_report(
    *,
    guild_id=None,
    channel_id=None,
    message_id=None,
    reporter_id=None,
    road_name=None,
    latitude=None,
    longitude=None,
    image_url=None,
    category=None,
    note=None,
    status='pending',
    stream_url=None,          # ⬅️ 新增一個參數（有預設值，所以舊呼叫不會壞）
):
    # 先預設 camera_id 為 None（避免沒資料時出錯）
    camera_id = None

    # 如果基本資訊齊全，就嘗試綁定/建立一台 camera
    if guild_id is not None and channel_id is not None and road_name:
        camera_id = await get_or_create_camera(
            guild_id=guild_id,
            channel_id=channel_id,
            name=road_name,      # 用 road_name 當 camera 顯示名稱
            road_name=road_name,
            latitude=latitude,
            longitude=longitude,
            stream_url=stream_url,
        )

    sql = """
    INSERT INTO reports
      (guild_id, channel_id, message_id, reporter_id,
       camera_id,
       road_name, latitude, longitude,
       image_url, category, note, status)
    VALUES
      (:gid, :cid, :mid, :rid,
       :cam_id,
       :road, :lat, :lng,
       :img, :cat, :note, :st)
    """
    params = {
        "gid": guild_id,
        "cid": channel_id,
        "mid": message_id,
        "rid": reporter_id,
        "cam_id": camera_id,
        "road": road_name,
        "lat": latitude,
        "lng": longitude,
        "img": image_url,
        "cat": category,
        "note": note,
        "st": status,
    }
    await aexec(sql, params)

async def list_reports_with_camera(
    *,
    status: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """
    取得最近的違規紀錄，順便把 camera 資訊一起帶出來
    """
    sql = """
    SELECT
      r.id,
      r.created_at,
      r.guild_id,
      r.channel_id,
      r.message_id,
      r.reporter_id,
      r.image_url,
      r.category,
      r.note,
      r.status,

      r.camera_id,
      c.name      AS camera_name,
      c.road_name AS camera_road_name,
      c.latitude  AS camera_latitude,
      c.longitude AS camera_longitude,
      c.stream_url AS camera_stream_url
    FROM reports r
    LEFT JOIN cameras c ON r.camera_id = c.id
    WHERE (:st IS NULL OR r.status = :st)
    ORDER BY r.created_at DESC
    LIMIT :limit
    """
    rows = await afetchall(
        sql,
        {
            "st": status,
            "limit": limit,
        },
    )
    return rows