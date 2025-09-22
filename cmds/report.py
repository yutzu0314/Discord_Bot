import json
import discord
from discord.ext import commands
from discord import app_commands
from core.classes import Cog_Extension

with open('setting.json', 'r', encoding='utf8') as jfile:
    jdata = json.load(jfile)

ADMIN_CHANNEL_ID = int(jdata["道路管理_channel"])
APPLY_CHANNEL_ID = int(jdata["道路申請_channel"])

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
        view = ManageView(reporter_id=interaction.user.id)
        channel = self.bot.get_channel(ADMIN_CHANNEL_ID)
        if channel:
            await channel.send(embed=embed, view=view)


# ========== 管理員操作 UI ==========
class ManageView(discord.ui.View):
    def __init__(self, reporter_id: int):
        super().__init__(timeout=None)
        self.reporter_id = reporter_id

    async def notify_user(self, bot, result: str):
        """在道路申請頻道通知申請者"""
        channel = bot.get_channel(APPLY_CHANNEL_ID)
        if channel:
            if({result} == "核准"):
                await channel.send(f"<@{self.reporter_id}> 你的申請已被 **{result}** ✅")
            elif({result} == "拒絕"):
                await channel.send(f"<@{self.reporter_id}> 你的申請已被 **{result}** ❌")
            else:
                await channel.send(f"<@{self.reporter_id}> 你的申請需要 **{result}** ✏️")

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.green)
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(
            content=f"✅ 已核准（由 {interaction.user} 操作）",
            view=None
        )
        await self.notify_user(interaction.client, "核准")

    @discord.ui.button(label="Reject", style=discord.ButtonStyle.red)
    async def reject(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(
            content=f"❌ 已拒絕（由 {interaction.user} 操作）",
            view=None
        )
        await self.notify_user(interaction.client, "拒絕")

    @discord.ui.button(label="Request Edit", style=discord.ButtonStyle.gray)
    async def request_edit(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(
            content=f"✏️ 請回報者補充/修改（由 {interaction.user} 操作）",
            view=None
        )
        await self.notify_user(interaction.client, "請補充/修改")


# ========== 使用者入口按鈕 ==========
class ApplyView(discord.ui.View):
    def __init__(self, bot):
        super().__init__(timeout=None)
        self.bot = bot

    @discord.ui.button(label="📋 填寫道路申請", style=discord.ButtonStyle.primary)
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
        if ctx.channel.id != APPLY_CHANNEL_ID:
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
