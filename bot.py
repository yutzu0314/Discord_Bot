from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")  # 讓 .env 生效（如果你沒在別處做過）

import matplotlib
from matplotlib import font_manager

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# 把字型檔加進 matplotlib
font_manager.fontManager.addfont(font_path)

# 取得 matplotlib 真正辨識到的 family name（避免手打 TC 找不到）
prop = font_manager.FontProperties(fname=font_path)
font_name = prop.get_name()
print("=== DEBUG FONT ===")
print("✅ Matplotlib font resolved as:", font_name)

matplotlib.rcParams["font.family"] = font_name
matplotlib.rcParams["axes.unicode_minus"] = False

 
import asyncio
import random
import discord
import json
import os
from discord.ext import commands, tasks
from datetime import datetime, timedelta, time
import matplotlib.pyplot as plt
from io import BytesIO
from services.violations_service import get_weekly_summary, get_weekly_camera_category_counts
from cmds.violation_request import ApplyView, ManageView
from playwright.async_api import async_playwright

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ["TORCH_CUDA_ARCH_LIST"] = ""


import sys
print("=== DEBUG PYTHON ===")
print("sys.executable:", sys.executable)
print("cwd:", os.getcwd())

with open('setting.json', 'r', encoding='utf8') as jfile:
    jdata = json.load(jfile)

print()

# intents
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix='[', intents=intents)
WEEKLY_REPORT_CHANNEL_ID = 1419250669272961041  # ←改成「每周報表」頻道 ID

async def capture_grafana_dashboard(output_path="grafana_weekly.png"):
    grafana_url = os.getenv("GRAFANA_URL")
    grafana_user = os.getenv("GRAFANA_USER")
    grafana_password = os.getenv("GRAFANA_PASSWORD")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        page = await browser.new_page(
            viewport={"width": 1920, "height": 1080}
        )

        # 先進登入頁
        await page.goto("http://localhost:3000/login")

        await page.fill('input[name="user"]', grafana_user)
        await page.fill('input[name="password"]', grafana_password)

        await page.click('button[type="submit"]')

        # 等登入完成
        await page.wait_for_timeout(3000)

        # 開 Dashboard
        await page.goto(grafana_url)

        # 等圖表畫出來
        await page.wait_for_timeout(8000)

        await page.screenshot(
            path=output_path,
            full_page=True
        )

        await browser.close()

    return output_path

@tasks.loop(time=time(hour=0, minute=0)) #每天早上 08:00 執行一次 (設定時間UTC=台灣時間減8小時)
async def weekly_report_task():
    await bot.wait_until_ready()

    # 只有星期一才送
    if datetime.now().weekday() != 0: #星期一是0，二是1，以此類推
        return
    
    channel = bot.get_channel(WEEKLY_REPORT_CHANNEL_ID)
    if channel is None:
        print("⚠ 找不到每周報表頻道，請確認 WEEKLY_REPORT_CHANNEL_ID 是否正確")
        return

    # 文字統計
    summary = await get_weekly_summary(days=7)
    total = summary["total"]
    by_camera = summary["by_camera"]
    by_category = summary["by_category"]

    # 日期範圍
    now = datetime.now()
    start = now - timedelta(days=7)
    title = f"每周違規報表（{start:%m/%d}–{now:%m/%d}）"

    if total == 0:
        embed = discord.Embed(
            title=title,
            description="這週（過去 7 天）沒有任何違規紀錄 ✅",
            color=0x2ecc71,
        )
        await channel.send(embed=embed)
        return

    desc = f"過去 7 天共有 **{total}** 筆違規紀錄。"

    embed = discord.Embed(
        title=title,
        description=desc,
        color=0x3498db,
    )

    embed.set_footer(text="自動產生｜資料來源：reports / cameras")
    
        # === Grafana Dashboard 截圖 ===
    screenshot_path = await capture_grafana_dashboard()

    if screenshot_path:
        file = discord.File(screenshot_path, filename="grafana_weekly.png")
        embed.set_image(url="attachment://grafana_weekly.png")
        await channel.send(embed=embed, file=file)
    else:
        await channel.send(embed=embed)



# --- Bot 啟動 ---
@bot.event
async def on_ready():
    print(">> Bot is online <<")
    if not weekly_report_task.is_running():
        weekly_report_task.start()
        print("▶ 每周報表排程啟動")

    # 重新註冊所有會出現在訊息上的 View
    bot.add_view(ManageView())       # 審核按鈕
    bot.add_view(ApplyView(bot))     # 申請入口按鈕

# --- 指令錯誤處理 ---
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

# --- 全域錯誤攔截 ---
@bot.event
async def on_error(event, *args, **kwargs):
    import traceback
    traceback.print_exc()



# --- 管理指令 (只限管理員) ---
@bot.command()
@commands.has_permissions(administrator=True)
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

@bot.command(name="每周報表", help="立即產生本週違規統計報表")
async def weekly_report_cmd(ctx: commands.Context):
    channel = ctx.channel  # 用使用者當下那個頻道

    summary = await get_weekly_summary(days=7)
    total = summary["total"]
    by_camera = summary["by_camera"]
    by_category = summary["by_category"]

    now = datetime.now()
    start = now - timedelta(days=7)
    title = f"每周違規報表（{start:%m/%d}–{now:%m/%d}）"

    if total == 0:
        embed = discord.Embed(
            title=title,
            description="這週（過去 7 天）沒有任何違規紀錄 ✅",
            color=0x2ecc71,
        )
        await channel.send(embed=embed)
        return

    desc = f"過去 7 天共有 **{total}** 筆違規紀錄。"

    embed = discord.Embed(
        title=title,
        description=desc,
        color=0x3498db,
    )

    embed.set_footer(text="自動產生｜資料來源：reports / cameras")

    # === Grafana Dashboard 截圖 ===
    screenshot_path = await capture_grafana_dashboard()

    if screenshot_path:
        file = discord.File(screenshot_path, filename="grafana_weekly.png")
        embed.set_image(url="attachment://grafana_weekly.png")
        await channel.send(embed=embed, file=file)
    else:
        await channel.send(embed=embed)


# --- 自動載入 cmds 內的 Cogs ---
async def main():
    for filename in os.listdir('./cmds'):
        if filename.endswith('.py') and filename != '__init__.py':
            await bot.load_extension(f'cmds.{filename[:-3]}')
            print(f"✅ 已載入模組：{filename}")

    await bot.start(jdata['TOKEN'])

if __name__ == "__main__":
    asyncio.run(main())
