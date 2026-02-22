# services/reports.py
from db import aexec, afetchall
from services.camera_service import get_or_create_camera  # ⬅️ 新增這行 import

async def get_report_by_message_id(message_id: int) -> dict | None:
    sql = "SELECT * FROM reports WHERE message_id = :mid LIMIT 1"
    rows = await afetchall(sql, {"mid": message_id})
    return rows[0] if rows else None

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

async def get_weekly_summary(
    days: int = 7,
) -> dict:
    """
    取得最近 N 天的違規統計：
    - total：總筆數
    - by_camera：每個路口的違規數
    - by_category：每種類型的違規數
    """
    # 各路口統計
    sql_by_camera = """
    SELECT
      COALESCE(c.name, '未知地點') AS camera_name,
      COUNT(*) AS total
    FROM reports r
    LEFT JOIN cameras c ON r.camera_id = c.id
    WHERE r.created_at >= NOW() - INTERVAL :days DAY
    GROUP BY camera_name
    ORDER BY total DESC
    """
    by_camera = await afetchall(
        sql_by_camera,
        {"days": days},
    )

    # 各類型統計
    sql_by_category = """
    SELECT
      COALESCE(r.category, '未標註') AS category,
      COUNT(*) AS total
    FROM reports r
    WHERE r.created_at >= NOW() - INTERVAL :days DAY
    GROUP BY category
    ORDER BY total DESC
    """
    by_category = await afetchall(
        sql_by_category,
        {"days": days},
    )

    total = sum(row["total"] for row in by_camera)

    return {
        "total": total,
        "by_camera": by_camera,
        "by_category": by_category,
        "days": days,
    }

async def get_weekly_camera_category_counts(
    days: int = 7,
) -> list[dict]:
    """
    回傳格式：
    [
      {
        "camera_name": "約農力行路口",
        "camera_name_en": "Yuenong-Lixing Intersection",
        "category": "bike",
        "total": 10
      },
      ...
    ]
    """
    sql = """
    SELECT
      COALESCE(c.name, '未知地點') AS camera_name,
      COALESCE(c.name_en, c.name, 'unknown') AS camera_name_en,
      COALESCE(r.category, '未標註') AS category,
      COUNT(*) AS total
    FROM reports r
    LEFT JOIN cameras c ON r.camera_id = c.id
    WHERE r.created_at >= NOW() - INTERVAL :days DAY
    GROUP BY camera_name, camera_name_en, category
    ORDER BY camera_name_en, category
    """
    rows = await afetchall(sql, {"days": days})
    return rows

