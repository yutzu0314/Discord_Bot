import discord
import random
import json
from discord.ext import commands
from core.classes import Cog_Extension


with open('setting.json', 'r', encoding='utf8') as jfile:
    jdata = json.load(jfile)

class Reacts(Cog_Extension):

    """隨機送出一張圖片"""
    @commands.command()
    async def 圖片(self, ctx):
        random_pic = random.choice(jdata['pic'])
        pic = discord.File(random_pic)
        await ctx.send(file=pic)

async def setup(bot):
    await bot.add_cog(Reacts(bot))

