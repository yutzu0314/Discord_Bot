import asyncio
from services.reports_service import save_report, list_reports_with_camera

async def main():
    print("=== 建一筆測試 report（會自動建 camera） ===")
    await save_report(
        guild_id=123456789,
        channel_id=987654321,
        message_id=111222333,
        reporter_id=444555666,
        road_name="約農力行路口",
        latitude=24.123456,
        longitude=120.654321,
        image_url="https://example.com/test.jpg",
        category="bike",
        note="這是一筆測試資料",
        status="pending",
        stream_url="rtmp://example.com/live/test",
    )

    print("=== 查詢最近的 reports（附 camera 資訊） ===")
    rows = await list_reports_with_camera(
        status=None,  # None = 不過濾狀態
        limit=5,
    )

    for r in rows:
        print("-" * 40)
        print(f"report_id      = {r['id']}")
        print(f"created_at     = {r['created_at']}")
        print(f"status         = {r['status']}")
        print(f"image_url      = {r['image_url']}")
        print(f"category       = {r['category']}")
        print(f"note           = {r['note']}")
        print()
        print(f"camera_id      = {r['camera_id']}")
        print(f"camera_name    = {r['camera_name']}")
        print(f"camera_road    = {r['camera_road_name']}")
        print(f"camera_lat/lng = {r['camera_latitude']}, {r['camera_longitude']}")
        print(f"stream_url     = {r['camera_stream_url']}")

if __name__ == "__main__":
    asyncio.run(main())
