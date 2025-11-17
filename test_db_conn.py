import os
from sqlalchemy import text
from db import engine

print("DB_URL from env =", os.getenv("DB_URL"))

with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print(result.scalar())
