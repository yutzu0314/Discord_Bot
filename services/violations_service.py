# services/violations_service.py
from __future__ import annotations

from db import aexec, afetchall


async def save_violation(
    *,
    guild_id: int | None = None,
    channel_id: int | None = None,
    camera_id: int | None = None,
    category: str | None = None,
    confidence: float | None = None,
    image_url: str | None = None,
    note: str | None = None,
) -> None:
    """
    寫入 violations（實際違規事件）
    """
    sql = """
    INSERT INTO violations
      (guild_id, channel_id, camera_id, category, confidence, image_url, note)
    VALUES
      (:gid, :cid, :cam, :cat, :conf, :img, :note)
    """
    await aexec(
        sql,
        {
            "gid": guild_id,
            "cid": channel_id,
            "cam": camera_id,
            "cat": category,
            "conf": confidence,
            "img": image_url,
            "note": note,
        },
    )


async def get_weekly_summary(days: int = 7) -> dict:
    """
    取得最近 N 天的違規統計（只看 violations）
    """
    sql_by_camera = """
    SELECT
      COALESCE(c.name, '未知地點') AS camera_name,
      COUNT(*) AS total
    FROM violations v
    LEFT JOIN cameras c ON v.camera_id = c.id
    WHERE v.created_at >= NOW() - INTERVAL :days DAY
    GROUP BY camera_name
    ORDER BY total DESC
    """
    by_camera = await afetchall(sql_by_camera, {"days": days})

    sql_by_category = """
    SELECT
      COALESCE(v.category, '未標註') AS category,
      COUNT(*) AS total
    FROM violations v
    WHERE v.created_at >= NOW() - INTERVAL :days DAY
    GROUP BY category
    ORDER BY total DESC
    """
    by_category = await afetchall(sql_by_category, {"days": days})

    total = sum(row["total"] for row in by_camera)

    return {
        "total": total,
        "by_camera": by_camera,
        "by_category": by_category,
        "days": days,
    }


async def get_weekly_camera_category_counts(days: int = 7) -> list[dict]:
    """
    回傳：每個 camera（英文名）× category 的統計（畫圖用）
    """
    sql = """
    SELECT
      COALESCE(c.name, '未知地點') AS camera_name,
      COALESCE(c.name_en, c.name, 'unknown') AS camera_name_en,
      COALESCE(v.category, '未標註') AS category,
      COUNT(*) AS total
    FROM violations v
    LEFT JOIN cameras c ON v.camera_id = c.id
    WHERE v.created_at >= NOW() - INTERVAL :days DAY
    GROUP BY camera_name, camera_name_en, category
    ORDER BY camera_name_en, category
    """
    return await afetchall(sql, {"days": days})