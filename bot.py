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

@bot.event #成員變動
async def on_member_join(member):
    channel = bot.get_channel(int(jdata['成員變更_channel']))
    await channel.send(f'{member} 加入伺服器！🎉')

@bot.event #成員變動
async def on_member_remove(member):
    channel = bot.get_channel(int(jdata['成員變更_channel']))
    await channel.send(f'{member} 離開了伺服器... 😢')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!大屁股楊景棠')

    await bot.process_commands(message)

# No Category
@bot.command()
async def load(ctx, extension):
    await bot.load_extension(f'cmds.{extension}')
    await ctx.send(f'Loaded {extension} done.')

@bot.command()
async def unload(ctx, extension):
    await bot.unload_extension(f'cmds.{extension}')
    await ctx.send(f'Unloaded {extension} done.')

@bot.command()
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