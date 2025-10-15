import asyncio
from services.reports import save_report

async def main():
    await save_report(
        road_name="測試路口",
        latitude=25.0418,
        longitude=121.536,
        image_url="http://example.com/test.jpg",
        category="違停",
        note="這是模擬 YOLO 偵測結果",
        status="pending"
    )

if __name__ == "__main__":
    asyncio.run(main())
