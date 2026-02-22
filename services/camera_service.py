# services/camera_service.py
from typing import Optional
from db import afetchall, aexec

async def list_active_cameras(guild_id: int):
    rows = await afetchall(
        """
        SELECT id, name, stream_url, latitude, longitude
        FROM cameras
        WHERE guild_id = :gid
          AND is_active = 1
        ORDER BY name
        """,
        {"gid": guild_id},
    )
    return rows

async def get_or_create_camera(
    *,
    guild_id: int,
    channel_id: int,
    name: str,
    road_name: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    stream_url: Optional[str] = None,
) -> int:
    """
    依照 guild + channel + name 找 camera
    若不存在則建立一筆，回傳 camera.id
    """
    # 1️⃣ 先看看是否已存在
    rows = await afetchall(
        """
        SELECT id FROM cameras
        WHERE guild_id = :guild_id
          AND channel_id = :channel_id
          AND name = :name
        LIMIT 1
        """,
        {
            "guild_id": guild_id,
            "channel_id": channel_id,
            "name": name,
        },
    )
    if rows:
        return rows[0]["id"]

    # 2️⃣ 沒有就新增一筆
    await aexec(
        """
        INSERT INTO cameras (
            guild_id, channel_id,
            name, road_name, latitude, longitude, stream_url
        ) VALUES (
            :guild_id, :channel_id,
            :name, :road_name, :latitude, :longitude, :stream_url
        )
        """,
        {
            "guild_id": guild_id,
            "channel_id": channel_id,
            "name": name,
            "road_name": road_name,
            "latitude": latitude,
            "longitude": longitude,
            "stream_url": stream_url,
        },
    )

    # 3️⃣ 再查一次拿 id（確保拿到的是剛剛那筆）
    rows = await afetchall(
        """
        SELECT id FROM cameras
        WHERE guild_id = :guild_id
          AND channel_id = :channel_id
          AND name = :name
        ORDER BY id DESC
        LIMIT 1
        """,
        {
            "guild_id": guild_id,
            "channel_id": channel_id,
            "name": name,
        },
    )
    return rows[0]["id"]
