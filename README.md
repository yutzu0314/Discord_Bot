# 🎥 Discord\_Bot

本專案為一套整合即時違規車輛偵測與地圖展示的校園交通監控系統。
前端使用 Discord 機器人進行即時偵測與通報，並可搭配地圖網頁查看當日違規紀錄。

---

## 🚀 功能特色

### 🎥 Discord\_Bot

* 即時監控選定路段的串流，偵測違規車輛。
* 自動截圖違規畫面。
* 自動傳送違規資訊（地點、時間、圖片、車種）到指定的 Discord 頻道。
* 支援互動選單選擇路段、開始/中止偵測。
* 提供查詢歷史紀錄功能。

### 🗺️ map (地圖頁面)

* 圖形化展示當日違規紀錄與標註位置。
* 資料來源為後端自動更新的 JSON 檔。

---

## 📦 安裝與部署

### 安裝步驟

1️⃣ 克隆本專案：

```bash
git clone https://github.com/你的帳號/DC_bot.git
cd DC_bot
```

2️⃣ 安裝所需套件：

```bash
pip install -r requirements.txt
```

3️⃣ 設定環境變數或設定檔
範例：

```json
{
  "GITHUB_TOKEN": "",
  "TOKEN": "",
  "成員變更_channel": "",
  "違規車輛_channel": "",
  "心跳_channel": "",
  "pic": [],
  "yolo_model": "",
  "road_name": [],
  "stream_url": []
}
```

4️⃣ 啟動機器人：

```
debug bot.py
```

---

## 🧭 使用方式

### 在 Discord 中操作

1️⃣ 加入 Discord 群組後，先到 **#rules** 頻道閱讀規則並了解指令用法。

2️⃣ 在頻道輸入：

```
[help
```

查看所有可用指令。

3️⃣ 啟動偵測：

```
[偵測串流
```

選擇要偵測的路段。

4️⃣ 系統會開始監控，若偵測到違規車輛會自動：

* 📸 截圖
* 📝 發送違規訊息到 **#違規車輛** 頻道

5️⃣ 若要停止偵測，可點選「中止偵測」按鈕。

---

## 🗺️ 地圖展示

### 當日違規地圖

若想以地圖方式查看違規紀錄，可透過以下網址：
🌐 [違規地圖](https://yijean333.github.io/map.github.io/map.html)

⚠️ 地圖僅顯示**當日**違規紀錄。
過往歷史紀錄請至 Discord 群組的 **#違規車輛** 頻道搜尋。

---

## 🔗 相關資源

* 地圖前端原始碼：[map.github.io](https://github.com/yijean333/map.github.io)
* 地圖後端 JSON 資料來源：[backend](https://github.com/yijean333/backend)
* 當日 JSON 資料檔案：[violations.json](https://yijean333.github.io/backend/violations.json)

歡迎提 issue、PR 或討論建議！
