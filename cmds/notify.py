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


with open("setting.json", "r", encoding="utf-8") as f:
    jdata = json.load(f)

# ============================
# UI 元件：路段選單
# ============================

class VideoPathModal(discord.ui.Modal, title="輸入測試影片路徑"):

    video_path = discord.ui.TextInput(
        label="影片路徑",
        placeholder="/home/inf431/datasets/sources/video/xxx.mp4",
        default="/home/inf431/datasets/sources/video/",
        required=True
    )

    def __init__(self, cog, owner_id):
        super().__init__()
        self.cog = cog
        self.owner_id = owner_id

    async def on_submit(self, interaction: discord.Interaction):
        path = str(self.video_path).strip()

        if not os.path.isabs(path):
            path = os.path.join(VIDEO_BASE_DIR, path)

        if not path:
            await interaction.response.send_message("❌ 請輸入影片路徑", ephemeral=True)
            return

        if not os.path.exists(path):
            await interaction.response.send_message(f"❌ 找不到影片：`{path}`", ephemeral=True)
            return

        fake_camera = {
            "id": -1,
            "name": f"測試影片：{os.path.basename(path)}",
            "stream_url": f"file://{path}",
            "latitude": None,
            "longitude": None,
            "channel_id": "1496769328727724172",  # 測試影片先走 fallback
        }

        type_view = DetectTypeView(
            owner_id=self.owner_id,
            cog=self.cog,
            camera=fake_camera
        )

        await interaction.response.send_message(
            f"🎬 已載入測試影片：`{path}`\n請選擇要偵測的類型：",
            view=type_view,
            ephemeral=True
        )

class OpenVideoPathModalButton(discord.ui.Button):
    def __init__(self, cog, owner_id):
        super().__init__(
            label="輸入影片路徑",
            style=discord.ButtonStyle.primary
        )
        self.cog = cog
        self.owner_id = owner_id

    async def callback(self, interaction: discord.Interaction):
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("❌ 你不是這個選單的使用者", ephemeral=True)
            return

        await interaction.response.send_modal(
            VideoPathModal(
                cog=self.cog,
                owner_id=self.owner_id
            )
        )


class OpenVideoPathModalView(discord.ui.View):
    def __init__(self, cog, owner_id):
        super().__init__(timeout=60)
        self.add_item(OpenVideoPathModalButton(cog, owner_id))

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

        options.append(
            discord.SelectOption(
                label="其他（測試影片）",
                value="__custom__",
                description="手動輸入影片路徑"
            )
        )

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

        selected_value = self.values[0]

        if selected_value == "__custom__":
            tree_text = build_video_tree()

            await interaction.response.send_message(
                f"📂 目前測試影片資料夾：`{VIDEO_BASE_DIR}`\n"
                f"```text\n{tree_text}\n```\n"
                f"請點下面按鈕輸入影片路徑。",
                view=OpenVideoPathModalView(
                    cog=self.parent_view.cog,
                    owner_id=interaction.user.id
                ),
                ephemeral=True
            )
            return
        camera_id = int(selected_value)

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
            view=type_view,
            ephemeral=True
        )

class RoadSelectView(discord.ui.View):
    def __init__(self, cameras, ctx, cog):
        super().__init__(timeout=60)
        self.cog = cog
        self.owner_id = ctx.author.id
        self.cameras = cameras
        self.add_item(RoadSelect(cameras, ctx, self))


# ============================
# UI 元件：偵測類型選單（違規 / 逆向 / 車禍）
# ============================

class DetectTypeSelect(discord.ui.Select):
    def __init__(self, parent_view):
        self.parent_view = parent_view
        options = [
            discord.SelectOption(
                label="違規偵測",
                value="violation",
                description="使用 YOLO 偵測違規車輛"
            ),
            discord.SelectOption(
                label="逆向偵測",
                value="trackguard_wrong_way",
                description="使用新版 TrackGuard wrong_way 偵測"
            ),
            discord.SelectOption(
                label="車禍偵測",
                value="accident",
                description="TrackGuard collision 偵測"
            ),
            discord.SelectOption(
                label="TrackGuard 全部偵測",
                value="trackguard_all",
                description="TrackGuard collision + wrong_way + 其他行為"
            ),
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
# Helper
# ============================

VIDEO_BASE_DIR = "/home/inf431/datasets/sources/video"
RUNNING_TRACKGUARD_TASKS = {}

def build_video_tree(base_dir=VIDEO_BASE_DIR, max_chars=1600):
    lines = ["."]
    
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()
        files.sort()

        rel_root = os.path.relpath(root, base_dir)
        if rel_root == ".":
            level = 0
        else:
            level = rel_root.count(os.sep) + 1

        indent = "│   " * max(level - 1, 0)

        if rel_root != ".":
            dirname = os.path.basename(root)
            lines.append(f"{indent}├── {dirname}")

        file_indent = "│   " * level
        for filename in files:
            if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                lines.append(f"{file_indent}├── {filename}")

    text = "\n".join(lines)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n...（檔案太多，已截斷）"

    return text

async def get_report_channel(client, channel_id: int):
    """優先用快取 get_channel，拿不到就 fetch_channel。"""
    ch = client.get_channel(channel_id)
    if ch is None:
        ch = await client.fetch_channel(channel_id)
    return ch

def get_camera_channel_id(camera: dict) -> int:
    """
    優先使用 camera.channel_id
    找不到時 fallback 到 setting.json 的 違規車輛_channel，再不行用 通知_channel
    """
    camera_channel_id = camera.get("channel_id")

    if camera_channel_id:
        return int(camera_channel_id)

    fallback_id = jdata.get("違規車輛_channel") or jdata["通知_channel"]
    return int(fallback_id)


# ============================
# Notify Cog
# ============================

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
        selected_road = camera["name"].strip()
        stream_url = camera.get("stream_url")
        lat = camera.get("latitude")
        lng = camera.get("longitude")

        print("[DEBUG] selected camera =", camera)

        if not stream_url:
            await interaction.response.send_message(
                f"❌ `{selected_road}` 沒有設定 stream_url，請先到資料庫 cameras 補上。",
                ephemeral=True
            )
            return

        view = StopDetectionView(self, interaction.user.id)
        view.set_stop_state(False)
        view.violations = []
        view.flush_task = None

        report_channel_id = get_camera_channel_id(camera)
        print(f"[DEBUG] report_channel_id = {report_channel_id}")

        try:
            print("[DEBUG] before get_report_channel")
            channel = await get_report_channel(interaction.client, report_channel_id)
            print(f"[DEBUG] after get_report_channel, channel = {channel}")
        except Exception as e:
            print(f"[ERROR] 無法取得回報頻道 channel_id={report_channel_id}：{repr(e)}")
            try:
                await interaction.followup.send(
                    f"❌ Bot 無法取得 `{selected_road}` 的回報頻道（channel_id={report_channel_id}）。",
                    ephemeral=True
                )
            except Exception:
                pass
            return

        async def send_violation(original_img_path, class_names, annotated_img_path=None):
            now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vehicle_str = ", ".join(class_names) if class_names else "unknown"

            if detect_type == "violation":
                type_text = "違規偵測"
            elif detect_type == "trackguard_wrong_way":
                type_text = "逆向偵測"
            elif detect_type == "accident":
                type_text = "車禍偵測"
            elif detect_type == "trackguard_all":
                type_text = "TrackGuard 全部偵測"
            else:
                type_text = detect_type

            msg = (
                f"🚨 偵測到事件（{type_text}）\n"
                f"🛵 類別：{vehicle_str}\n"
                f"📷 路段：{selected_road}\n"
                f"🕒 時間：{now_time}\n"
                f"🧠 偵測標註畫面"
            )

            send_path = None

            if annotated_img_path:
                for _ in range(10):
                    if os.path.exists(annotated_img_path):
                        send_path = annotated_img_path
                        break
                    await asyncio.sleep(0.2)

            if not send_path and original_img_path and os.path.exists(original_img_path):
                send_path = original_img_path

            if not send_path:
                print(f"[ERROR] 無可傳送圖片，original={original_img_path}, annotated={annotated_img_path}")
                return

            print(f"[SEND] channel={channel} id={channel.id}")
            print(f"[SEND] send_path={send_path}")
            print(f"[SEND] exists={os.path.exists(send_path) if send_path else False}")
            print(f"[SEND] class_names={class_names}")

            try:
                sent_msg = await channel.send(
                    content=msg,
                    file=discord.File(send_path)
                )
                print(f"[SEND] success message_id={sent_msg.id}")

            except Exception as e:
                print(f"[SEND ERROR] {repr(e)}")
                try:
                    await channel.send(f"🚫 圖片傳送失敗：{repr(e)}")
                except Exception as e2:
                    print(f"[SEND ERROR 2] {repr(e2)}")
                return

            image_url = sent_msg.attachments[0].url
            note_text = f"{detect_type} stream detection"

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

                view.violations.append({
                    "road_name": selected_road,
                    "vehicle": vehicle,
                    "image_url": image_url,
                    "time": now_time
                })

            for path in [original_img_path, annotated_img_path]:
                if not path:
                    continue
                for _ in range(5):
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                        break
                    except PermissionError:
                        await asyncio.sleep(0.5)

        type_label_map = {
            "violation": "違規",
            "trackguard_wrong_way": "逆向",
            "accident": "車禍",
            "trackguard_all": "TrackGuard 全部",
        }

        print("[DEBUG] before interaction.response.send_message")
        await interaction.response.send_message(
            f"📡 開始偵測 `{selected_road}`（{type_label_map.get(detect_type, detect_type)}）...\n"
            f"📨 回報頻道：{channel.mention}",
            view=view
        )
        print("[DEBUG] after interaction.response.send_message")

        async def detection_task():
            print(f"[DEBUG] detection_task started, detect_type={detect_type}, stream_url={stream_url}")

            try:
                if detect_type == "violation":
                    print("[DEBUG] enter violation")
                    await self.run_live_detection(stream_url, send_violation, view, channel)
                
                elif detect_type == "trackguard_wrong_way":
                    print("[DEBUG] enter trackguard_wrong_way")

                    if stream_url.startswith("file://"):
                        video_path = stream_url.replace("file://", "")
                    else:
                        video_path = stream_url

                    print(f"[DEBUG] wrong_way video_path={video_path}")
                    await self.run_trackguard_detection(video_path, "wrong_way", send_violation, view, channel)

                elif detect_type == "accident":
                    print("[DEBUG] enter accident")
                    if stream_url.startswith("file://"):
                        video_path = stream_url.replace("file://", "")
                    else:
                        video_path = stream_url

                    print(f"[DEBUG] accident video_path={video_path}")
                    await self.run_trackguard_detection(video_path, "collision", send_violation, view, channel)

                elif detect_type in ("trackguard_all", "all"):
                    print("[DEBUG] enter trackguard_all")
                    if stream_url.startswith("file://"):
                        video_path = stream_url.replace("file://", "")
                    else:
                        video_path = stream_url

                    print(f"[DEBUG] all video_path={video_path}")
                    await self.run_trackguard_detection(video_path, "all", send_violation, view, channel)

                else:
                    await channel.send(f"❌ 未知偵測類型：{detect_type}")
                    return

                if view.violations:
                    await self.flush_violations_later(view, delay=0)

                await channel.send("✅ 偵測結束。")

            except asyncio.CancelledError:
                print("[DEBUG] detection_task cancelled inside")
                raise

            except Exception as e:
                print(f"[ERROR] detection_task inner crashed: {repr(e)}")
                await channel.send(f"🚫 偵測任務發生錯誤：{str(e)}")
                raise


        task_key = f"{camera_id}:{detect_type}"

        old_task = RUNNING_TRACKGUARD_TASKS.get(task_key)
        if old_task and not old_task.done():
            await channel.send(
                f"⚠️ `{selected_road}` 的 `{type_label_map.get(detect_type, detect_type)}` 正在執行中，請先按「中止偵測」。"
            )
            return


        def task_done_callback(task: asyncio.Task):
            RUNNING_TRACKGUARD_TASKS.pop(task_key, None)
            print(f"[DEBUG] cleared running task: {task_key}")

            try:
                task.result()
            except asyncio.CancelledError:
                print("[DEBUG] detection_task cancelled")
            except Exception as e:
                print(f"[ERROR] detection_task crashed: {repr(e)}")


        print("[DEBUG] before create detection_task")
        view.task = asyncio.create_task(detection_task())
        RUNNING_TRACKGUARD_TASKS[task_key] = view.task
        view.task.add_done_callback(task_done_callback)
        print("[DEBUG] after create detection_task")
        
    async def run_live_detection(self, video_path, send_fn, view: StopDetectionView, report_channel, interval=10):
        async def on_error(error_msg: str):
            await report_channel.send(f"⚠️ 錯誤：{error_msg}")

        try:
            async for img_path, class_names in detect_video_live(video_path, on_error, interval):
                await send_fn(img_path, class_names)
                if view.get_stop_state():
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await report_channel.send(f"🚫 偵測中斷錯誤：{str(e)}")

    async def run_accident_detection(self, video_path, send_fn, view: StopDetectionView, report_channel, interval=1):
        async def on_error(error_msg: str):
            await report_channel.send(f"⚠️ 車禍偵測錯誤：{error_msg}")

        try:
            async for img_path, class_names in detect_accident_live(video_path, on_error, interval):
                print("[BOT] 收到車禍截圖:", img_path, class_names)
                await send_fn(img_path, class_names)
                if view.get_stop_state():
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await report_channel.send(f"🚫 車禍偵測中斷錯誤：{str(e)}")

    async def run_trackguard_detection(self, video_path, trackguard_detect_type, send_fn, view: StopDetectionView, report_channel):
        from services.trackguard_runner import run_trackguard_process

        async def on_event(event: dict):
            image_path = event.get("image_path")
            annotated_image_path = event.get("annotated_image_path")

            behaviour_type = event.get("behaviour_type") or event.get("event") or trackguard_detect_type

            if behaviour_type == "wrong_way":
                class_names = [
                    f"{event.get('class_primary', event.get('class_name', 'unknown'))} wrong_way"
                ]
            elif behaviour_type == "collision":
                class_names = [
                    f"{event.get('class_primary', 'unknown')} vs {event.get('class_secondary', 'unknown')}"
                ]
            else:
                class_names = [
                    f"{event.get('class_primary', event.get('class_name', 'unknown'))} {behaviour_type}"
                ]

            if image_path and os.path.exists(image_path):
                await send_fn(image_path, class_names, annotated_image_path)
            elif annotated_image_path and os.path.exists(annotated_image_path):
                await send_fn(annotated_image_path, class_names, annotated_image_path)
            else:
                print("[TRACKGUARD] 事件收到，但找不到可傳送的圖片")

            if view.get_stop_state():
                return

        try:
            print(f"[DEBUG] TrackGuard video_path={video_path}")
            print(f"[DEBUG] TrackGuard detect_type={trackguard_detect_type}")

            await run_trackguard_process(video_path, trackguard_detect_type, on_event)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await report_channel.send(f"🚫 TrackGuard 偵測中斷錯誤：{str(e)}")

    async def flush_violations_later(self, view, delay=300):
        await asyncio.sleep(delay)

        if view.violations:
            from detect.github_sync import update_violation_to_github_bulk
            update_violation_to_github_bulk(view.violations)
            view.violations.clear()

        view.flush_task = None


async def setup(bot):
    try:
        from services.trackguard_runner import print_trackguard_debug
        print_trackguard_debug()
    except Exception as e:
        print(f"⚠️ TrackGuard debug init failed: {repr(e)}")

    await bot.add_cog(Notify(bot))