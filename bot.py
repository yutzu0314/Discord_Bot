import asyncio
import random
import discord
import json
import os
from discord.ext import commands

with open('setting.json', 'r', encoding='utf8') as jfile:
    jdata = json.load(jfile)

# intents
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix='[', intents=intents)

@bot.event
async def on_ready():
    print(">> Bot is online <<")

# 成員變動
@bot.event
async def on_member_join(member):
    channel = bot.get_channel(int(jdata['成員變更_channel']))
    await channel.send(f'{member} 加入伺服器！🎉')

@bot.event
async def on_member_remove(member):
    channel = bot.get_channel(int(jdata['成員變更_channel']))
    await channel.send(f'{member} 離開了伺服器... 😢')

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("🚫 你沒有權限使用這個指令！")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("⚠️ 指令參數不完整，請重新輸入。")
    elif isinstance(error, commands.CommandNotFound):
        await ctx.send("❓ 找不到這個指令，請輸入 `[help` 查看可用指令。")
    else:
        raise error  # 其他錯誤照原本拋出（方便除錯）

# No Category
@bot.command()
@commands.has_permissions(administrator=True) #指令只能給管理員用
async def load(ctx, extension):
    await bot.load_extension(f'cmds.{extension}')
    await ctx.send(f'Loaded {extension} done.')

@bot.command()
@commands.has_permissions(administrator=True)
async def unload(ctx, extension):
    await bot.unload_extension(f'cmds.{extension}')
    await ctx.send(f'Unloaded {extension} done.')

@bot.command()
@commands.has_permissions(administrator=True)
async def reload(ctx, extension):
    await bot.reload_extension(f'cmds.{extension}')
    await ctx.send(f'Reloaded {extension} done.')

# 自動載入 cmds 內的 Cogs
async def main():
    for filename in os.listdir('./cmds'):
        if filename.endswith('.py') and filename != '__init__.py':
            await bot.load_extension(f'cmds.{filename[:-3]}')
            print(f"✅ 已載入模組：{filename}")

    await bot.start(jdata['TOKEN'])

if __name__ == "__main__":
    asyncio.run(main())