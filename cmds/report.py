import json
import os
import discord
from discord.ext import commands
from discord import app_commands
from core.classes import Cog_Extension
from db import aexec
from services.reports import save_report

with open('setting.json', 'r', encoding='utf8') as jfile:
    jdata = json.load(jfile)

ADMIN_CHANNEL_ID  = int(jdata["道路管理_channel"])
ENTRY_CHANNEL_ID  = int(jdata["道路申請_channel"])  # ← 申請入口要貼在這
NOTIFY_CHANNEL_ID = int(jdata["通知_channel"])       # ← 審核結果發這裡

STORE = "reports.json"

def load_store():
    if not os.path.exists(STORE):
        return {}
    with open(STORE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_store(data: dict):
    with open(STORE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ========== Modal ==========
class ReportModal(discord.ui.Modal, title="回報違規路段"):
    camera = discord.ui.TextInput(label="監視器網址", style=discord.TextStyle.short, required=True)
    location = discord.ui.TextInput(label="地點", style=discord.TextStyle.short, required=True)
    desc = discord.ui.TextInput(label="補充說明", style=discord.TextStyle.paragraph, required=False)

    def __init__(self, bot, reporter: discord.User):
        super().__init__()
        self.bot = bot
        self.reporter = reporter

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.send_message("✅ 已收到回報，管理員將審核。", ephemeral=True)

        embed = discord.Embed(title="🚨 新違規回報", color=discord.Color.red())
        embed.add_field(name="地點", value=self.location.value, inline=True)
        embed.add_field(name="監視器", value=self.camera.value, inline=True)
        embed.add_field(name="補充說明", value=self.desc.value or "無", inline=False)
        embed.set_footer(text=f"回報者: {interaction.user}")
        embed.timestamp = discord.utils.utcnow()

        # 發送到管理員頻道
        view = ManageView()  # persistent view
        channel = self.bot.get_channel(ADMIN_CHANNEL_ID)
        if channel:
            admin_msg = await channel.send(embed=embed, view=view)

            # 存檔到 JSON（原本流程）
            data = load_store()
            data[str(admin_msg.id)] = {
                "reporter_id": interaction.user.id
            }
            save_store(data)

            # 👉 新增：存到 MySQL
            await save_report(
                guild_id=interaction.guild.id if interaction.guild else None,
                channel_id=channel.id,
                message_id=admin_msg.id,
                reporter_id=interaction.user.id,
                road_name=self.location.value,
                image_url=self.camera.value,   # 你這邊填「監視器網址」欄位
                note=self.desc.value,
                category=None,                 # 目前沒有分類欄位，可以先傳 None
                status='pending'
            )



# ========== 管理員操作 UI ==========
class ManageView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)  # persistent view 不帶狀態

    async def _notify(self, interaction: discord.Interaction, result_text: str):
        data = load_store()
        key = str(interaction.message.id)
        info = data.get(key)
        if not info:
            return  # 找不到對應就略過（可能已被清掉）

        reporter_id = info.get("reporter_id")

        channel = interaction.client.get_channel(NOTIFY_CHANNEL_ID)  # ← 固定通知頻道
        if channel and reporter_id:
            # 發通知到「申請者來源頻道」
            if result_text == "核准":
                await channel.send(f"<@{reporter_id}> 你的申請已被 **{result_text}** ✅")
            elif result_text == "拒絕":
                await channel.send(f"<@{reporter_id}> 你的申請已被 **{result_text}** ❌")
            else:
                await channel.send(f"<@{reporter_id}> 你的申請需要 **{result_text}** ✏️")

        # 清理這筆紀錄，避免檔案無限成長
        if key in data:
            del data[key]
            save_store(data)

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green, custom_id="report_approve")
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(
            content=f"✅ 已核准（由 {interaction.user} 操作）",
            view=None
        )
        await self._notify(interaction, "核准")

        # update SQL
        await aexec("""
            UPDATE reports SET atatus='approved'
            WHERE message_id = :mid
            """, {"mid": interaction.message.id})

    @discord.ui.button(label="Reject", style=discord.ButtonStyle.red, custom_id="report_reject")
    async def reject(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(
            content=f"❌ 已拒絕（由 {interaction.user} 操作）",
            view=None
        )
        await self._notify(interaction, "拒絕")

        # update SQL
        await aexec("""
            UPDATE reports SET atatus='rejected'
            WHERE message_id = :mid
            """, {"mid": interaction.message.id})

    @discord.ui.button(label="Request Edit", style=discord.ButtonStyle.gray, custom_id="report_request_edit")
    async def request_edit(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(
            content=f"✏️ 請回報者補充/修改（由 {interaction.user} 操作）",
            view=None
        )
        await self._notify(interaction, "補充/修改")

        # update SQL
        await aexec("""
            UPDATE reports SET atatus='pending'
            WHERE message_id = :mid
            """, {"mid": interaction.message.id})



# ========== 使用者入口按鈕 ==========
class ApplyView(discord.ui.View):
    def __init__(self, bot):
        super().__init__(timeout=None)
        self.bot = bot

    @discord.ui.button(label="📋 填寫道路申請", style=discord.ButtonStyle.primary, custom_id="report_apply")
    async def apply(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = ReportModal(self.bot, interaction.user)
        await interaction.response.send_modal(modal)



# ========== Cog ==========
class Report(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="report")
    @commands.has_permissions(administrator=True)
    async def report(self, ctx: commands.Context):
        if ctx.channel.id != ENTRY_CHANNEL_ID:
            await ctx.send("⚠️ 這個指令只能在指定的 #違規道路申請 頻道使用！", delete_after=5)
            return

        view = ApplyView(self.bot)
        embed = discord.Embed(
            title="🚦 道路違規申請入口",
            description="請點下方按鈕，填寫回報表單。\n(只有管理員能看到審核結果)",
            color=discord.Color.blue()
        )
        await ctx.send(embed=embed, view=view)


async def setup(bot: commands.Bot):
    await bot.add_cog(Report(bot))
