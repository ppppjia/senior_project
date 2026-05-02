import json
import os
import subprocess
import sys


def extract_and_save_standard(video_path, json_path):
    try:
        print("正在執行骨架擷取程式...")
        result = subprocess.run(
            [sys.executable, 'extract_standard.py'],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"標準動作資料已儲存至 {json_path}")
            return True
        else:
            print(f"執行失敗：{result.stderr}")
            return False
    except Exception as e:
        print(f"產生標準動作資料失敗：{e}")
        return False


def check_and_update_standard(video_path='teacher_dance.mp4', json_path='dance_standard.json'):
    if not os.path.exists(video_path):
        print(f"錯誤：找不到影片檔案 {video_path}")
        return False

    if not os.path.exists(json_path):
        print(f"找不到 {json_path}，正在產生標準動作資料...")
        return extract_and_save_standard(video_path, json_path)

    video_mtime = os.path.getmtime(video_path)
    json_mtime = os.path.getmtime(json_path)

    if video_mtime > json_mtime:
        print(f"偵測到 {video_path} 已更新，正在重新產生骨架資料...")
        return extract_and_save_standard(video_path, json_path)
    else:
        print("標準動作資料已是最新版本。")
        return True


def load_standard_pose_data(json_path='dance_standard.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
