# services/reports.py
from db import aexec

async def save_report(
    *,
    guild_id=None, channel_id=None, message_id=None, reporter_id=None,
    road_name=None, latitude=None, longitude=None,
    image_url=None, category=None, note=None, status='pending'
):
    sql = """
    INSERT INTO reports
      (guild_id, channel_id, message_id, reporter_id,
       road_name, latitude, longitude, image_url, category, note, status)
    VALUES
      (:gid, :cid, :mid, :rid, :road, :lat, :lng, :img, :cat, :note, :st)
    """
    params = {
        "gid": guild_id, "cid": channel_id, "mid": message_id, "rid": reporter_id,
        "road": road_name, "lat": latitude, "lng": longitude,
        "img": image_url, "cat": category, "note": note, "st": status
    }
    await aexec(sql, params)
