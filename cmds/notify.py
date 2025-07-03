from discord.ext import commands
from core.classes import Cog_Extension
from detect.detector import detect_video_live
import discord
import os
from datetime import datetime
import json
from functools import partial
import asyncio

with open("setting.json", "r", encoding="utf-8") as f:
    jdata = json.load(f)

class Notify(Cog_Extension):

    @commands.command()
    async def 偵測影片(self, ctx):
        road_name = jdata.get("road_name", "未知路段")
        channel = ctx.bot.get_channel(int(jdata['違規車輛_channel']))

        # 📤 定義當有違規圖時的傳送行為
        async def send_violation(img_path):
            now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = (
                f"🚨 偵測到違規車輛\n"
                f"📷 路段：{road_name}\n"
                f"🕒 時間：{now_time}"
            )

            await channel.send(msg, file=discord.File(img_path))
            os.remove(img_path)  # 🗑️ 傳完即刪

        await channel.send("🎥 開始進行即時影片偵測...")
        await self.run_live_detection(jdata["video_path"], send_violation)
        await channel.send("✅ 偵測結束。")

    # ✅ 包裝非同步偵測流程
    async def run_live_detection(self, video_path, send_fn):
        await detect_video_live(video_path, send_fn, 10)

    @commands.command()
    async def 偵測串流(self, ctx):
        road_name = jdata.get("road_name", "未知路段")
        channel = ctx.bot.get_channel(int(jdata['違規車輛_channel']))

        async def send_violation(img_path):
            now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = (
                f"🚨 偵測到違規車輛\n"
                f"📷 路段：{road_name}\n"
                f"🕒 時間：{now_time}"
            )
            await channel.send(msg, file=discord.File(img_path))
            os.remove(img_path)

        await channel.send("📡 開始即時串流偵測...")
        await self.run_live_detection(jdata["stream_url"], send_violation)
        await channel.send("📴 偵測結束。")

    async def run_live_detection(self, video_path, send_fn):
        await detect_video_live(video_path, send_fn, 10)



async def setup(bot):
    await bot.add_cog(Notify(bot))
