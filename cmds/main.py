print("✅ cmds.main 正在被載入")

import discord
from discord.ext import commands
from core.classes import Cog_Extension

# ✅ 設定 intents 並開啟 members 權限
intents = discord.Intents.default()
intents.members = True  # <<< 必須開啟這個
intents.message_content = True  # 若還有文字指令也建議開啟

class Main(Cog_Extension):

    """Ping 指令，測量 bot 延遲"""
    @commands.command()
    async def ping(self, ctx):
        latency = self.bot.latency
        if latency is not None:
            await ctx.send(f'{round(latency * 1000)} ms')
        else:
            await ctx.send("延遲測不到 🤔")

async def setup(bot):
    print("📥 setup() 開始執行")
    await bot.add_cog(Main(bot))
    print("✅ Cog 'Main' 已載入")
