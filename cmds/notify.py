from discord.ext import commands
from core.classes import Cog_Extension
from detect.detector import detect_video_live, detect_accident_live
import os
from datetime import datetime
import json
import asyncio
import discord
from services.violations_service import save_violation
from services.camera_service import list_active_cameras
from services.reverse_service import get_active_reverse_config
from detect.reverse_identification.reverse_detector import detect_reverse_live


with open("setting.json", "r", encoding="utf-8") as f:
    jdata = json.load(f)

# ============================
# UI 元件：路段選單
# ============================

class RoadSelect(discord.ui.Select):
    def __init__(self, cameras, ctx, parent_view):
        self.ctx = ctx
        self.parent_view = parent_view

        options = [
            discord.SelectOption(
                label=cam["name"],
                value=str(cam["id"]),
                description="選擇此路段進行偵測"
            )
            for cam in cameras
        ]

        super().__init__(
            placeholder="請選擇要偵測的路段",
            min_values=1,
            max_values=1,
            options=options
        )

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.parent_view.owner_id:
            await interaction.response.send_message("❌ 你不是這個選單的使用者", ephemeral=True)
            return

        camera_id = int(self.values[0])
        selected_camera = next(
            (c for c in self.parent_view.cameras if c["id"] == camera_id),
            None
        )

        if not selected_camera:
            await interaction.response.send_message("❌ 找不到路段資料", ephemeral=True)
            return

        type_view = DetectTypeView(
            owner_id=interaction.user.id,
            cog=self.parent_view.cog,
            camera=selected_camera
        )

        await interaction.response.send_message(
            f"✅ 已選擇路段：`{selected_camera['name']}`\n請選擇要偵測的類型：",
            view=type_view
        )


class RoadSelectView(discord.ui.View):
    def __init__(self, cameras, ctx, cog):
        super().__init__(timeout=60)
        self.cog = cog
        self.owner_id = ctx.author.id
        self.cameras = cameras
        self.add_item(RoadSelect(cameras, ctx, self))

# ============================
# UI 元件：偵測類型選單（違規 / 逆向）
# ============================

class DetectTypeSelect(discord.ui.Select):
    def __init__(self, parent_view):
        self.parent_view = parent_view
        options = [
            discord.SelectOption(label="違規偵測", value="violation", description="使用 YOLO 偵測違規車輛"),
            discord.SelectOption(label="逆向偵測", value="reverse", description="偵測逆向行駛（需該路段開啟 reverse_enabled）"),
            discord.SelectOption(label="車禍偵測", value="accident", description="偵測車禍 / 碰撞異常"),
        ]
        super().__init__(placeholder="請選擇偵測類型", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.parent_view.owner_id:
            await interaction.response.send_message("❌ 你不是這個選單的使用者，無法操作。", ephemeral=True)
            return

        detect_type = self.values[0]
        await self.parent_view.cog.start_detection(interaction, self.parent_view, detect_type)


class DetectTypeView(discord.ui.View):
    def __init__(self, owner_id: int, cog, camera: dict):
        super().__init__(timeout=60)
        self.cog = cog
        self.owner_id = owner_id
        self.camera = camera
        self.add_item(DetectTypeSelect(self))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("❌ 你不是這個選單的使用者，無法操作。", ephemeral=True)
            return False
        return True


# ============================
# 停止偵測按鈕
# ============================

class StopButton(discord.ui.Button):
    def __init__(self, parent_view):
        super().__init__(label="中止偵測", style=discord.ButtonStyle.danger)
        self.parent_view = parent_view

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.parent_view.owner_id:
            await interaction.response.send_message("❌ 你無權按下這個按鈕。", ephemeral=True)
            return

        task: asyncio.Task = getattr(self.parent_view, "task", None)
        if task and not task.done():
            task.cancel()

        self.parent_view.set_stop_state(True)
        self.disabled = True
        await interaction.response.edit_message(view=self.parent_view)
        await interaction.followup.send("中止偵測！")


class StopDetectionView(discord.ui.View):
    def __init__(self, cog, owner_id):
        super().__init__(timeout=None)
        self.cog = cog
        self.stop_flag = False
        self.owner_id = owner_id
        self.task = None
        self.violations = []
        self.flush_task = None
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


# ============================
# Notify Cog
# ============================
async def get_report_channel(client, channel_id: int):
    """優先用快取 get_channel，拿不到就 fetch_channel；並確保是可發訊息的頻道物件。"""
    ch = client.get_channel(channel_id)
    if ch is None:
        ch = await client.fetch_channel(channel_id)  # 這步失敗會直接丟例外給外層
    return ch

class Notify(Cog_Extension):

    @commands.command()
    async def 偵測串流(self, ctx):
        cameras = await list_active_cameras(ctx.guild.id)

        if not cameras:
            await ctx.send("❌ 沒有可用路段（cameras 為空）")
            return

        view = RoadSelectView(cameras, ctx, self)
        await ctx.send("請選擇要進行偵測的路段：", view=view)

    async def start_detection(self, interaction: discord.Interaction, type_view: DetectTypeView, detect_type: str):
        camera = type_view.camera
        camera_id = camera["id"]
        selected_road = camera["name"]
        stream_url = camera.get("stream_url")
        lat = camera.get("latitude")
        lng = camera.get("longitude")

        if not stream_url:
            await interaction.response.send_message(
                f"❌ `{selected_road}` 沒有設定 stream_url，請先到資料庫 cameras 補上。",
                ephemeral=True
            )
            return

        reverse_cfg = None
        if detect_type == "reverse":
            reverse_cfg = await get_active_reverse_config(camera_id)
            if not reverse_cfg:
                await interaction.response.send_message(
                    f"⚠️ `{selected_road}` 尚未在資料庫啟用逆向設定（camera_reverse_profiles / zones）。",
                    ephemeral=True
                )
                return

        view = StopDetectionView(self, interaction.user.id)
        view.set_stop_state(False)
        view.violations = []
        view.flush_task = None

        report_channel_id = int(jdata["違規車輛_channel"])
        try:
            channel = await get_report_channel(interaction.client, report_channel_id)
        except Exception as e:
            print(f"[ERROR] 無法取得回報頻道 channel_id={report_channel_id}：{repr(e)}")
            try:
                await interaction.followup.send("❌ Bot 無法取得回報頻道（頻道ID或權限有問題）。", ephemeral=True)
            except:
                pass
            return

        async def send_violation(img_path, class_names):
            now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vehicle_str = ", ".join(class_names) if class_names else "unknown"
            if detect_type == "violation":
                type_text = "違規偵測"
            elif detect_type == "reverse":
                type_text = "逆向偵測"
            elif detect_type == "accident":
                type_text = "車禍偵測"
            else:
                type_text = detect_type

            msg = (
                f"🚨 偵測到事件（{type_text}）\n"
                f"🛵 類別：{vehicle_str}\n"
                f"📷 路段：{selected_road}\n"
                f"🕒 時間：{now_time}"
            )

            if not os.path.exists(img_path):
                print(f"[ERROR] 檔案不存在，無法送出：{img_path}")
                return

            try:
                with open(img_path, "rb") as f:
                    head = f.read(16)
                    if not head:
                        print(f"[ERROR] 檔案為空：{img_path}")
                        return

                sent_msg = await channel.send(msg, file=discord.File(img_path))
            except discord.Forbidden as e:
                print(f"[ERROR] Discord Forbidden: {repr(e)}")
                try:
                    await interaction.followup.send("❌ 權限不足：Bot 可能沒有在該頻道『附加檔案/傳送訊息』權限。", ephemeral=True)
                except:
                    pass
                return
            except discord.HTTPException as e:
                print(f"[ERROR] Discord HTTPException: {repr(e)}")
                return
            except Exception as e:
                print(f"[ERROR] 送圖未知錯誤: {repr(e)}")
                return

            image_url = sent_msg.attachments[0].url
            note_text = f"{detect_type} stream detection"

            # ✅ 寫入 violations（一張圖可能多個類別 -> 多筆）
            for vehicle in (class_names or ["unknown"]):
                await save_violation(
                    guild_id=interaction.guild.id if interaction.guild else None,
                    channel_id=channel.id,
                    camera_id=camera_id,
                    category=vehicle,
                    confidence=None,
                    image_url=image_url,
                    note=note_text,
                )

                # 你原本的 GitHub bulk 也保留
                view.violations.append({
                    "road_name": selected_road,
                    "vehicle": vehicle,
                    "image_url": image_url,
                    "time": now_time
                })

            if view.flush_task is None or view.flush_task.done():
                view.flush_task = asyncio.create_task(self.flush_violations_later(view))

            # 刪除本地檔
            for _ in range(5):
                try:
                    os.remove(img_path)
                    break
                except PermissionError:
                    await asyncio.sleep(0.5)

        type_label_map = {
            "violation": "違規",
            "reverse": "逆向",
            "accident": "車禍",
        }

        await interaction.response.send_message(
            f"📡 開始偵測 `{selected_road}`（{type_label_map.get(detect_type, detect_type)}）...",
            view=view
        )

        async def detection_task():
            if detect_type == "violation":
                await self.run_live_detection(stream_url, send_violation, view)
            elif detect_type == "reverse":
                await self.run_reverse_detection(stream_url, send_violation, view, reverse_cfg)
            elif detect_type == "accident":
                await self.run_accident_detection(stream_url, send_violation, view)
            else:
                await channel.send(f"❌ 未知偵測類型：{detect_type}")
                return

            if view.violations:
                await self.flush_violations_later(view, delay=0)

            await channel.send("✅ 偵測結束。")

        view.task = asyncio.create_task(detection_task())

    async def run_reverse_detection(self, video_path, send_fn, view: StopDetectionView, reverse_cfg: dict, interval=1):
        async def on_error(error_msg: str):
            channel = self.bot.get_channel(int(jdata["違規車輛_channel"]))
            await channel.send(f"⚠️ 逆向偵測錯誤：{error_msg}")

        profile = reverse_cfg.get("profile", "more")
        model_path = jdata.get("yolo_model", "detect/reverse_identification/yolov8n.pt")
        config = reverse_cfg  # ✅ DB 回來的格式就是 {profile,min_conf,move_min_pixels,roads}

        try:
            async for img_path, class_names in detect_reverse_live(
                video_path,
                on_error=on_error,
                interval=interval,
                profile=profile,
                model_path=model_path,
                config=config
            ):
                print("[BOT] 收到逆向截圖:", img_path, class_names)
                await send_fn(img_path, class_names)
                if view.get_stop_state():
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            channel = self.bot.get_channel(int(jdata["違規車輛_channel"]))
            await channel.send(f"🚫 逆向偵測中斷錯誤：{str(e)}")


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
            pass
        except Exception as e:
            channel = self.bot.get_channel(int(jdata["違規車輛_channel"]))
            await channel.send(f"🚫 偵測中斷錯誤：{str(e)}")
    
    async def run_accident_detection(self, video_path, send_fn, view: StopDetectionView, interval=1):
        async def on_error(error_msg: str):
            channel = self.bot.get_channel(int(jdata["違規車輛_channel"]))
            await channel.send(f"⚠️ 車禍偵測錯誤：{error_msg}")

        try:
            async for img_path, class_names in detect_accident_live(video_path, on_error, interval):
                print("[BOT] 收到車禍截圖:", img_path, class_names)
                await send_fn(img_path, class_names)
                if view.get_stop_state():
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            channel = self.bot.get_channel(int(jdata["違規車輛_channel"]))
            await channel.send(f"🚫 車禍偵測中斷錯誤：{str(e)}")

    async def flush_violations_later(self, view, delay=300):
        await asyncio.sleep(delay)

        if view.violations:
            from detect.github_sync import update_violation_to_github_bulk
            update_violation_to_github_bulk(view.violations)
            view.violations.clear()
            #channel = self.bot.get_channel(int(jdata["違規車輛_channel"]))
            #await channel.send("✅ 自動更新 GitHub 完成")

        view.flush_task = None


async def setup(bot):
    await bot.add_cog(Notify(bot))