from pydoc import classname
from discord.ext import commands
from core.classes import Cog_Extension
from detect.detector import detect_video_live
from detect.github_sync import update_violation_to_github
import os
from datetime import datetime
import json
import tempfile
import asyncio
import discord

with open("setting.json", "r", encoding="utf-8") as f:
    jdata = json.load(f)

# 選單元件：路段選擇
class RoadSelect(discord.ui.Select):
    def __init__(self, road_names, ctx, parent_view):
        self.ctx = ctx
        self.parent_view = parent_view
        options = [
            discord.SelectOption(label=name, description="選擇此路段進行偵測")
            for name in road_names
        ]
        super().__init__(placeholder="請選擇要偵測的路段", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.parent_view.owner_id:
            await interaction.response.send_message("❌ 你不是這個選單的使用者，無法操作。", ephemeral=True)
            return

        selected_road = self.values[0]
        index = jdata["road_name"].index(selected_road)
        stream_url = jdata["stream_url"][index]

        view = StopDetectionView(self.parent_view.cog, interaction.user.id)
        view.set_stop_state(False)

        channel = interaction.client.get_channel(int(jdata["違規車輛_channel"]))

        async def send_violation(img_path, class_names):
            now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vehicle_str = ", ".join(class_names) if class_names else "unknown"

            msg = (
                f"🚨 偵測到違規車輛\n"
                f"🛵 類別：{vehicle_str}\n"
                f"📷 路段：{selected_road}\n"
                f"🕒 時間：{now_time}"
            )
            # 傳送訊息並取得訊息物件
            sent_msg = await channel.send(msg, file=discord.File(img_path))

            # 從 Discord 訊息附件取得圖片網址
            image_url = sent_msg.attachments[0].url

            # 呼叫 GitHub 更新函式
            try:
                for vehicle in class_names:
                    update_violation_to_github(selected_road, vehicle, image_url)
            except Exception as e:
                print(f"⚠️ GitHub 更新失敗：{e}")

            # 刪除本地圖片釋放空間
            for _ in range(5):
                try:
                    os.remove(img_path)
                    break
                except PermissionError:
                    await asyncio.sleep(0.5)

        await interaction.response.send_message(
            f"📡 開始偵測 `{selected_road}` 路段...", view=view, ephemeral=True
        )

        async def detection_task():
            await self.parent_view.cog.run_live_detection(stream_url, send_violation, view)
            await channel.send("✅ 偵測結束。")

        view.task = asyncio.create_task(detection_task())

# 路段選單 View
class RoadSelectView(discord.ui.View):
    def __init__(self, road_names, ctx, cog):
        super().__init__(timeout=60)
        self.cog = cog
        self.owner_id = ctx.author.id
        self.add_item(RoadSelect(road_names, ctx, self))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("❌ 你不是這個選單的使用者，無法操作。", ephemeral=True)
            return False
        return True

# 停止偵測按鈕
class StopButton(discord.ui.Button):
    def __init__(self, parent_view):
        super().__init__(label="中止偵測", style=discord.ButtonStyle.danger)
        self.parent_view = parent_view

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.parent_view.owner_id:
            await interaction.response.send_message("❌ 你無權按下這個按鈕。", ephemeral=True)
            return

        # 強制中止任務
        task: asyncio.Task = getattr(self.parent_view, "task", None)
        if task and not task.done():
            task.cancel()

        self.parent_view.set_stop_state(True)
        self.disabled = True
        await interaction.response.edit_message(view=self.parent_view)
        await interaction.followup.send("中止偵測！", ephemeral=True)

# 停止偵測按鈕 View
class StopDetectionView(discord.ui.View):
    def __init__(self, cog, owner_id):
        super().__init__(timeout=None)
        self.cog = cog
        self.stop_flag = False
        self.owner_id = owner_id
        self.task = None
        self.add_item(StopButton(self))

    def set_stop_state(self, value: bool):
        self.stop_flag = value

    def get_stop_state(self) -> bool:
        return self.stop_flag

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("❌ 你無權按下這個按鈕。", ephemeral=True)
            return False
        return True

class Notify(Cog_Extension):

    @commands.command()
    async def 偵測串流(self, ctx):
        road_names = jdata.get("road_name", [])
        if not road_names:
            await ctx.send("❌ 未找到任何可用路段設定。")
            return

        view = RoadSelectView(road_names, ctx, self)
        await ctx.send("請選擇要進行偵測的路段：", view=view, ephemeral=True)

    async def run_live_detection(self, video_path, send_fn, view: StopDetectionView, interval=10):
        async def on_error(error_msg: str):
            channel = self.bot.get_channel(int(jdata["違規車輛_channel"]))
            await channel.send(f"⚠️ 錯誤：{error_msg}")

        try:
            async for img_path, class_names in detect_video_live(video_path, on_error, interval):
                await send_fn(img_path, class_names)
                if view.get_stop_state():
                    break
        except asyncio.CancelledError:
            pass  # 被取消不算錯誤
        except Exception as e:
            channel = self.bot.get_channel(int(jdata["違規車輛_channel"]))
            await channel.send(f"🚫 偵測中斷錯誤：{str(e)}")

async def setup(bot):
    await bot.add_cog(Notify(bot))