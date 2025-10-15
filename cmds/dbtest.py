from discord.ext import commands
from services.reports import save_report

class DBTest(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def dbtest(self, ctx):
        await save_report(
            guild_id=ctx.guild.id if ctx.guild else None,
            channel_id=ctx.channel.id,
            reporter_id=ctx.author.id,
            image_url="https://example.com/test.jpg",
            note="這是測試紀錄",
            category="test"
        )
        await ctx.send("✅ 已寫入一筆測試資料到 MySQL！")

async def setup(bot):
    await bot.add_cog(DBTest(bot))
