# services/reverse_service.py
import json
from db import afetchall, aexec


async def get_active_reverse_config(camera_id: int) -> dict | None:
    """
    回傳 detect_reverse_live 需要的 config（跟你原本 setting.json reverse_config 格式對齊）：
    {
      "profile": "more",
      "min_conf": 0.5,
      "move_min_pixels": 5,
      "roads": [
         {"name":"main","poly":[[x,y],...],"dir":[0,1]},
         ...
      ]
    }
    """
    prof_rows = await afetchall(
        """
        SELECT id, profile_name, min_conf, move_min_pixels
        FROM camera_reverse_profiles
        WHERE camera_id = :cid AND is_enabled = 1
        LIMIT 1
        """,
        {"cid": camera_id},
    )
    if not prof_rows:
        return None

    prof = prof_rows[0]
    zones = await afetchall(
        """
        SELECT zone_name, polygon_json, dir_x, dir_y
        FROM camera_reverse_zones
        WHERE profile_id = :pid
        ORDER BY zone_name
        """,
        {"pid": prof["id"]},
    )

    roads = []
    for z in zones:
        poly = z["polygon_json"]
        if isinstance(poly, str):
            poly = json.loads(poly)
        roads.append(
            {
                "name": z["zone_name"],
                "poly": poly,
                "dir": [int(z["dir_x"]), int(z["dir_y"])],
            }
        )

    return {
        "profile": prof["profile_name"],
        "min_conf": float(prof["min_conf"]),
        "move_min_pixels": int(prof["move_min_pixels"]),
        "roads": roads,
    }


async def upsert_profile(
    *,
    camera_id: int,
    profile_name: str,
    min_conf: float,
    move_min_pixels: int,
    is_enabled: bool,
) -> int:
    await aexec(
        """
        INSERT INTO camera_reverse_profiles
          (camera_id, profile_name, is_enabled, min_conf, move_min_pixels)
        VALUES
          (:cid, :pname, :ena, :conf, :mmp)
        ON DUPLICATE KEY UPDATE
          is_enabled = VALUES(is_enabled),
          min_conf = VALUES(min_conf),
          move_min_pixels = VALUES(move_min_pixels)
        """,
        {
            "cid": camera_id,
            "pname": profile_name,
            "ena": 1 if is_enabled else 0,
            "conf": min_conf,
            "mmp": move_min_pixels,
        },
    )

    rows = await afetchall(
        """
        SELECT id FROM camera_reverse_profiles
        WHERE camera_id=:cid AND profile_name=:pname
        LIMIT 1
        """,
        {"cid": camera_id, "pname": profile_name},
    )
    return rows[0]["id"]


async def disable_all_profiles(camera_id: int) -> None:
    await aexec(
        "UPDATE camera_reverse_profiles SET is_enabled=0 WHERE camera_id=:cid",
        {"cid": camera_id},
    )


async def upsert_zone(
    *,
    profile_id: int,
    zone_name: str,
    poly: list,
    dir_x: int,
    dir_y: int,
) -> None:
    await aexec(
        """
        INSERT INTO camera_reverse_zones
          (profile_id, zone_name, polygon_json, dir_x, dir_y)
        VALUES
          (:pid, :zname, :poly, :dx, :dy)
        ON DUPLICATE KEY UPDATE
          polygon_json = VALUES(polygon_json),
          dir_x = VALUES(dir_x),
          dir_y = VALUES(dir_y)
        """,
        {
            "pid": profile_id,
            "zname": zone_name,
            "poly": json.dumps(poly, ensure_ascii=False),
            "dx": int(dir_x),
            "dy": int(dir_y),
        },
    )