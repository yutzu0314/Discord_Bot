# db.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DB_URL")

engine = create_engine(
    DB_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    future=True,
)

def exec(sql: str, params: dict | None = None) -> None:
    """同步：執行 INSERT/UPDATE/DELETE"""
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})

def fetchall(sql: str, params: dict | None = None) -> list[dict]:
    """同步：查詢多列"""
    with engine.begin() as conn:
        res = conn.execute(text(sql), params or {})
        return [dict(r._mapping) for r in res]

# --- 非同步包裝：把同步 I/O 丟到 thread pool，避免卡住事件 loop ---
import asyncio
async def aexec(sql: str, params: dict | None = None) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, exec, sql, params)

async def afetchall(sql: str, params: dict | None = None) -> list[dict]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fetchall, sql, params)
