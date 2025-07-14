import base64
import json
import requests
from datetime import datetime
import json

with open("setting.json", "r", encoding="utf-8") as f:
    jdata = json.load(f)

GITHUB_TOKEN = jdata["GITHUB_TOKEN"]

REPO_OWNER = "yijean333"
REPO_NAME = "flask_backend"
FILE_PATH = "violations.json"

API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def update_violation_to_github(road_name, vehicle, image_url):
    try:
        # 取得原始 JSON 資料
        res = requests.get(API_URL, headers=HEADERS)
        if res.status_code != 200:
            print("❌ 無法取得 JSON：", res.text)
            return

        content = res.json()
        sha = content["sha"]
        data = json.loads(base64.b64decode(content["content"]).decode())

        # 準備新資料
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {
            "time": now,
            "vehicle": vehicle,
            "image_url": image_url,
            "camera_name": road_name
        }

        if road_name not in data:
            data[road_name] = []

        data[road_name].append(new_entry)

        # 重新 encode 後送出
        encoded = base64.b64encode(json.dumps(data, ensure_ascii=False, indent=2).encode()).decode()
        commit_data = {
            "message": f"Add violation: {road_name} @ {now}",
            "content": encoded,
            "sha": sha
        }

        put_res = requests.put(API_URL, headers=HEADERS, data=json.dumps(commit_data))
        if put_res.status_code in [200, 201]:
            print("✅ GitHub 更新成功")
        else:
            print("❌ GitHub 更新失敗：", put_res.text)
    except Exception as e:
        print(f"⚠️ GitHub 更新異常：{e}")

def update_violation_to_github_bulk(violations):
    try:
        # 取得現有資料
        res = requests.get(API_URL, headers=HEADERS)
        if res.status_code != 200:
            print("❌ 無法取得 JSON：", res.text)
            return

        content = res.json()
        sha = content["sha"]
        data = json.loads(base64.b64decode(content["content"]).decode())

        # 把 violations 一次更新進去
        for v in violations:
            road_name = v["road_name"]
            new_entry = {
                "time": v["time"],
                "vehicle": v["vehicle"],
                "image_url": v["image_url"],
                "camera_name": road_name
            }
            if road_name not in data:
                data[road_name] = []
            data[road_name].append(new_entry)

        # 重新 encode 並提交
        encoded = base64.b64encode(json.dumps(data, ensure_ascii=False, indent=2).encode()).decode()
        commit_data = {
            "message": f"Add {len(violations)} violations (bulk)",
            "content": encoded,
            "sha": sha
        }

        put_res = requests.put(API_URL, headers=HEADERS, data=json.dumps(commit_data))
        if put_res.status_code in [200, 201]:
            print("✅ GitHub 批次更新成功")
        else:
            print("❌ GitHub 批次更新失敗：", put_res.text)

    except Exception as e:
        print(f"⚠️ GitHub 批次更新異常：{e}")
