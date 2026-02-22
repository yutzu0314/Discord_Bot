import asyncio
from services.reports_service import save_report

async def main():
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
        note="測試用",
        status="pending",
        stream_url="rtmp://example.com/live/test",
    )

if __name__ == "__main__":
    asyncio.run(main())
