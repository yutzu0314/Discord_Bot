# 🚦 交通路況異常即時偵測系統

**AI-Based Real-Time Traffic Anomaly Detection System**

本專案提出一套基於 **深度學習與車輛行為分析** 的交通監控系統，可從監控影像中即時偵測交通異常事件，並透過 **Discord Bot 即時通報**。

系統整合：

* 🚗 **車輛物件偵測（YOLO）**
* 🎯 **多目標追蹤（Norfair）**
* 📊 **車輛行為分析**
* 🤖 **Discord Bot 通報**
* 🗺️ **交通事件地圖視覺化**

目前可偵測事件包含：

* 🚗 違規車輛
* ↩️ 逆向行駛
* 💥 車禍事件

當偵測到異常事件時，系統會自動：

* 📸 擷取事件畫面
* 📩 發送 Discord 通知
* 🗂️ 儲存事件紀錄
* 🗺️ 同步至交通地圖系統

---

# 🎯 研究目標

本研究希望建立一套 **低成本、可即時運行的交通異常偵測系統**。

主要目標：

1️⃣ 自動化交通監控
2️⃣ 即時交通異常偵測
3️⃣ 降低人工監控成本
4️⃣ 支援智慧城市交通管理

本系統可應用於：

* 校園交通監控
* 城市路口監控
* 智慧交通管理系統

---

# 📊 系統架構

```
Video Stream / CCTV
        │
        ▼
YOLO Object Detection
        │
        ▼
Multi Object Tracking (Norfair)
        │
        ▼
Trajectory Analysis
        │
        ▼
Anomaly Detection
├─ 違規偵測
├─ 逆向偵測
└─ 車禍偵測
        │
        ▼
Discord Bot Notification
        │
        ▼
Database + JSON
        │
        ▼
Traffic Map Visualization
```

---

# 🧠 方法

本系統採用 **Computer Vision + Trajectory Analysis** 方法偵測交通事件。

主要分為三個模組：

## 1️⃣ 車輛偵測

使用 **YOLO11 (Ultralytics)** 進行車輛物件偵測。

訓練資料來源：

* 自行標註資料
* BDD100K dataset
* UA-DETRAC dataset

目前訓練資料量：

```
870 images
+ augmentation
≈ 1300 images
```

資料增強包含：

* rotation
* flip
* brightness
* noise

---

## 2️⃣ 車輛追蹤

本系統使用 **Norfair Multi Object Tracking**。

追蹤資訊包含：

* 車輛 ID
* 車輛位置
* 車輛速度
* 行駛方向

追蹤結果可用於分析：

* 車輛行為
* 車輛互動
* 異常事件

---

# 💥 車禍偵測演算法

車禍偵測採用 **多特徵分析方法（Multi-feature Accident Detection）**。

主要特徵包含：

---

### 1️⃣ sudden stop detection

車輛速度由高速突然下降：

```
v_previous > threshold_high
v_current < threshold_low
```

此情況常見於碰撞或急煞。

---

### 2️⃣ direction variance detection

若車輛方向突然劇烈變化：

```
Var(direction) > threshold
```

可能表示車輛受到撞擊或失控。

---

### 3️⃣ collision detection

當兩車距離小於動態閾值：

```
distance < adaptive_threshold
```

並持續多幀時，判定為碰撞。

---

### 4️⃣ hotspot detection

對遠距離車輛使用 **區域異常分析**。

若同一區域出現：

* 多次 sudden stop
* 多次方向異常

則標記為 **事故熱點 (Hotspot)**。

---

# 🤖 Discord Bot

Discord Bot 為本系統的主要操作介面。

使用者可透過指令：

```
!偵測串流
```

依序選擇：

```
路段 → 偵測模式
```

偵測模式：

| 模式   | 說明        |
| ---- | --------- |
| 違規偵測 | YOLO車輛偵測  |
| 逆向偵測 | 車輛方向分析    |
| 車禍偵測 | 車輛軌跡與速度分析 |

若偵測到事件：

Bot 會自動：

```
📸 擷取畫面
📩 發送Discord通知
🗂️ 儲存事件資料
```

---

# 🗺️ 交通事件地圖

所有偵測事件會同步至地圖系統。

地圖可顯示：

* 發生地點
* 發生時間
* 車輛類型
* 事件圖片

地圖網址：

```
https://yijean333.github.io/map.github.io/map.html
```

---

# 🧰 技術棧

| 技術         | 用途     |
| ---------- | ------ |
| Python     | 系統核心   |
| PyTorch    | AI模型   |
| YOLO11     | 車輛偵測   |
| Norfair    | 車輛追蹤   |
| OpenCV     | 影像處理   |
| Discord.py | Bot操作  |
| MySQL      | 資料庫    |
| GitHub API | JSON同步 |

---

# 📁 專案結構

```
Discord_Bot
│
├─ bot.py
│
├─ cmds
│   ├─ notify.py
│   └─ violation_request.py
│
├─ detect
│   ├─ detector.py
│   ├─ reverse_identification
│   └─ accident_detection
│
├─ services
│   ├─ camera_service.py
│   ├─ violations_service.py
│   └─ reverse_service.py
│
├─ core
│   └─ classes.py
│
├─ setting.json
└─ requirements.txt
```

---

# ⚙️ 安裝與部署

## 1️⃣ Clone repository

```
git clone https://github.com/你的帳號/DC_bot.git
cd DC_bot
```

---

## 2️⃣ 安裝套件

```
pip install -r requirements.txt
```

---

## 3️⃣ 設定設定檔

```
setting.json
```

範例：

```
{
 "TOKEN": "",
 "GITHUB_TOKEN": "",
 "違規車輛_channel": "",
 "yolo_model": "",
 "accident_yolo_model": ""
}
```

---

## 4️⃣ 啟動系統

```
python bot.py
```

---

# 📈 未來研究方向

未來可進一步發展：

* Transformer-based detection
* 深度行為分析
* 交通流量預測
* 多攝影機融合偵測
* AI交通風險預測

---

# 🤝 貢獻

歡迎提出：

* Issue
* Pull Request
* Feature 建議

共同改進交通監控系統。
