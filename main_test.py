# =============================================================================
# main_test.py - AI Dance Teaching System with On-Screen Speed Controller
# =============================================================================

import cv2
import mediapipe as mp
import json
import time
import numpy as np
import os

from pose_utils import normalize_pose, calculate_pose_error
import config

# 初始化 MediaPipe
mp_pose = mp.solutions.pose

# =============================================================================
# 倍速設定
# =============================================================================
MIN_SPEED = 0.25
MAX_SPEED = 2.0
current_speed = 1.0

# =============================================================================
# On-Screen Speed Controller 設定
# =============================================================================
show_controller = True
controller_x, controller_y = 250, 15    #(控制速度位置, 不知道)
controller_w, controller_h = 330, 100   #(控制器邊框長度,控制器高度)

# 按鈕定義: (文字, 速度變化值或目標值, x_offset, y_offset, width, height)
buttons = [
    #("-0.25", -0.25, 20, 55, 65, 38),
    #("-0.10", -0.10, 95, 55, 55, 38),
    #("+0.10", 0.10, 160, 55, 55, 38),
    #("+0.25", 0.25, 225, 55, 65, 38),

    ("0.5x", 0.5, 20, 55, 60, 38),
    ("1.0x", 1.0, 90, 55, 60, 38),
    ("1.5x", 1.5, 160, 55, 60, 38),
    ("2.0x", 2.0, 230, 55, 60, 38),
]


def draw_speed_controller(img, speed):
    """繪製速度控制器面板"""
    overlay = img.copy()

    # 半透明黑色背景
    cv2.rectangle(overlay, (controller_x, controller_y),
                  (controller_x + controller_w, controller_y + controller_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

    # 白色邊框
    cv2.rectangle(img, (controller_x, controller_y),
                  (controller_x + controller_w, controller_y + controller_h),
                  (255, 255, 255), 1)

    # 標題
    cv2.putText(img, f"Speed Controller : {speed:.2f}x",
                (controller_x + 25, controller_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

    # 繪製所有按鈕
    for text, value, dx, dy, bw, bh in buttons:
        bx = controller_x + dx
        by = controller_y + dy
        # 按鈕背景
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
        # 按鈕文字
        cv2.putText(img, text, (bx + 10, by + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1)


def mouse_callback(event, x, y, flags, param):
    """滑鼠點擊回調函式"""
    global current_speed, show_controller

    if event == cv2.EVENT_LBUTTONDOWN and show_controller:
        for text, value, dx, dy, bw, bh in buttons:
            bx = controller_x + dx
            by = controller_y + dy
            if (bx <= x <= bx + bw) and (by <= y <= by + bh):
                if text.startswith(('+', '-')):  # 微調按鈕
                    new_speed = round(current_speed + value, 2)
                else:  # 直接設定按鈕
                    new_speed = value

                current_speed = max(MIN_SPEED, min(MAX_SPEED, new_speed))
                print(f"速度已調整為: {current_speed:.2f}x")
                break


# =============================================================================
# 檢查並更新標準動作資料
# =============================================================================
def check_and_update_standard():
    video_path = 'teacher_dance.mp4'
    json_path = 'dance_standard.json'

    if not os.path.exists(video_path):
        print(f"錯誤：找不到影片檔案 {video_path}")
        return False

    if not os.path.exists(json_path):
        print(f"找不到 {json_path}，正在產生標準動作資料...")
        return extract_and_save_standard(video_path, json_path)

    video_mtime = os.path.getmtime(video_path)
    json_mtime = os.path.getmtime(json_path)

    if video_mtime > json_mtime:
        print(f"偵測到 teacher_dance.mp4 已更新，正在重新產生骨架資料...")
        return extract_and_save_standard(video_path, json_path)
    else:
        print("標準動作資料已是最新版本。")
        return True


def extract_and_save_standard(video_path, json_path):
    import subprocess
    try:
        print("正在執行骨架擷取程式...")
        result = subprocess.run(
            ['python', 'extract_standard.py'],
            capture_output=True, text=True, timeout=300
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


# =============================================================================
# 主程式開始
# =============================================================================
print("檢查標準動作資料...")
if not check_and_update_standard():
    print("警告：無法自動更新標準動作資料，將嘗試載入現有資料")

print("載入標準動作資料...")
with open('dance_standard.json', 'r', encoding='utf-8') as f:
    standard_pose_data = json.load(f)

cap_webcam = cv2.VideoCapture(0)
cap_teacher = cv2.VideoCapture('teacher_dance.mp4')

# 取得影片資訊
teacher_preview = None
video_duration_ms = 0
video_frame_count = 0

if cap_teacher.isOpened():
    fps = cap_teacher.get(cv2.CAP_PROP_FPS) or 30
    video_frame_count = int(cap_teacher.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration_ms = (video_frame_count / fps) * 1000 if fps > 0 else 0
    cap_teacher.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success_preview, teacher_preview = cap_teacher.read()
    if success_preview:
        teacher_preview = teacher_preview.copy()
    cap_teacher.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 系統狀態
is_playing = False
is_paused = False
start_time = 0
pause_start_time = 0
total_pause_time = 0
pause_elapsed_ms = 0
current_score = 0

print("\n=== AI Dance Teaching System 已啟動 ===")
print("操作說明：")
print("  's'     → 開始播放與評分")
print("  空白鍵  → 暫停 / 繼續")
print("  'C'     → 顯示 / 隱藏速度控制器（滑鼠點擊調整）")
print("  'q'     → 離開程式\n")

# 建立視窗並註冊滑鼠回調
window_name = 'AI Dance Teaching System - Ghost Shadow'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)
cv2.setMouseCallback(window_name, mouse_callback)

# =============================================================================
# 主迴圈
# =============================================================================
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap_webcam.isOpened():
        success_webcam, image_webcam = cap_webcam.read()
        if not success_webcam:
            break

        image_webcam = cv2.flip(image_webcam, 1)
        h, w, _ = image_webcam.shape

        display_image = None
        elapsed_ms = 0
        success_teacher = False
        image_teacher = None

        # 播放時間計算
        if is_playing:
            if is_paused:
                elapsed_ms = pause_elapsed_ms
            else:
                raw_elapsed = (time.time() - start_time) * 1000
                elapsed_ms = int(raw_elapsed * current_speed) - total_pause_time
                pause_elapsed_ms = elapsed_ms

            if video_duration_ms > 0 and elapsed_ms >= video_duration_ms:
                success_teacher = False
            else:
                cap_teacher.set(cv2.CAP_PROP_POS_MSEC, elapsed_ms)
                success_teacher, image_teacher = cap_teacher.read()
                if success_teacher:
                    elapsed_ms = cap_teacher.get(cv2.CAP_PROP_POS_MSEC)
                    if int(cap_teacher.get(cv2.CAP_PROP_POS_FRAMES) or 0) >= video_frame_count:
                        success_teacher = False

        # 選擇顯示畫面
        if success_teacher and image_teacher is not None:
            display_image = cv2.resize(image_teacher, (w, h))
        elif teacher_preview is not None:
            display_image = cv2.resize(teacher_preview, (w, h))
        else:
            display_image = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(display_image, "Teacher video unavailable", (50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if is_playing and not success_teacher:
            cv2.putText(display_image, "FINISH! Press 's' to replay", (50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            is_playing = False

        # 學生姿態估測
        image_rgb = cv2.cvtColor(image_webcam, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # 鬼影效果與評分（保持你原本的邏輯）
        if is_playing and success_teacher:
            target_frame = None
            for frame_data in standard_pose_data:
                if frame_data['timestamp_ms'] >= elapsed_ms:
                    target_frame = frame_data
                    break

            ghost_scale = 0.35
            ghost_w = int(w * ghost_scale)
            ghost_h = int(h * ghost_scale)
            ghost_x = w - ghost_w - 20
            ghost_y = h - ghost_h - 20

            skeleton_canvas = np.zeros((ghost_h, ghost_w, 3), dtype=np.uint8)

            ghost_joint = (180, 180, 180)
            ghost_line = (150, 150, 150)
            student_joint = (0, 255, 0)
            student_line = (0, 200, 0)

            # 老師骨架
            if target_frame is not None:
                for lm in target_frame['landmarks']:
                    if lm.get('v', 0) > 0.5:
                        px = int(lm['x'] * ghost_w)
                        py = int(lm['y'] * ghost_h)
                        cv2.circle(skeleton_canvas, (px, py), 3, ghost_joint, -1)

                for conn in config.POSE_CONNECTIONS:
                    start_idx, end_idx = conn
                    if start_idx < len(target_frame['landmarks']) and end_idx < len(target_frame['landmarks']):
                        start_lm = target_frame['landmarks'][start_idx]
                        end_lm = target_frame['landmarks'][end_idx]
                        if start_lm.get('v', 0) > 0.5 and end_lm.get('v', 0) > 0.5:
                            sx = int(start_lm['x'] * ghost_w)
                            sy = int(start_lm['y'] * ghost_h)
                            ex = int(end_lm['x'] * ghost_w)
                            ey = int(end_lm['y'] * ghost_h)
                            cv2.line(skeleton_canvas, (sx, sy), (ex, ey), ghost_line, 2)

            # 學生骨架
            if results.pose_landmarks:
                live_landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'v': lm.visibility}
                                  for lm in results.pose_landmarks.landmark]

                for lm in live_landmarks:
                    if lm['v'] > 0.5:
                        px = int(lm['x'] * ghost_w)
                        py = int(lm['y'] * ghost_h)
                        cv2.circle(skeleton_canvas, (px, py), 3, student_joint, -1)

                for conn in config.POSE_CONNECTIONS:
                    start_idx, end_idx = conn
                    if start_idx < len(live_landmarks) and end_idx < len(live_landmarks):
                        start_lm = live_landmarks[start_idx]
                        end_lm = live_landmarks[end_idx]
                        if start_lm['v'] > 0.5 and end_lm['v'] > 0.5:
                            sx = int(start_lm['x'] * ghost_w)
                            sy = int(start_lm['y'] * ghost_h)
                            ex = int(end_lm['x'] * ghost_w)
                            ey = int(end_lm['y'] * ghost_h)
                            cv2.line(skeleton_canvas, (sx, sy), (ex, ey), student_line, 2)

            # 混合鬼影
            roi = display_image[ghost_y:ghost_y + ghost_h, ghost_x:ghost_x + ghost_w]
            if roi.shape[:2] == skeleton_canvas.shape[:2]:
                blended = cv2.addWeighted(roi, 0.55, skeleton_canvas, 0.45, 0)
                display_image[ghost_y:ghost_y + ghost_h, ghost_x:ghost_x + ghost_w] = blended

            # 邊框與標籤
            cv2.rectangle(display_image, (ghost_x - 4, ghost_y - 4),
                          (ghost_x + ghost_w + 4, ghost_y + ghost_h + 4), (180, 180, 180), 1)
            cv2.putText(display_image, "Teacher Ghost", (ghost_x + 8, ghost_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            cv2.putText(display_image, "Student", (ghost_x + 8, ghost_y + ghost_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, student_joint, 1)

            # 計算分數
            if target_frame is not None and results.pose_landmarks:
                norm_live = normalize_pose(live_landmarks)
                norm_target = normalize_pose(target_frame['landmarks'])
                error = calculate_pose_error(norm_live, norm_target)
                raw_score = 100 - (error * 40)
                current_score = max(0, min(100, int(raw_score)))

        # 顯示即時資訊
        if is_playing:
            score_color = (0, 255, 0) if current_score > 80 else (0, 165, 255) if current_score > 60 else (0, 0, 255)
            cv2.putText(display_image, f"Score: {current_score}", (10, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, score_color, 1)
            cv2.putText(display_image, f"Time: {int(elapsed_ms / 1000)}s", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)
            cv2.putText(display_image, f"Speed: {current_speed:.2f}x", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

            if is_paused:
                cv2.putText(display_image, "PAUSED", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 1)
        else:
            cv2.putText(display_image, "Press 's' to start playback", (50, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        # 顯示控制器
        if show_controller and display_image is not None:
            draw_speed_controller(display_image, current_speed)

        cv2.imshow(window_name, display_image)

        key = cv2.waitKey(1) & 0xFF

        # 視窗關閉偵測
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("使用者已關閉視窗")
            break

        # 鍵盤操作
        if key == ord('q'):
            print("按下 'q' 離開系統")
            break
        elif key == ord('s') and not is_playing:
            print("開始播放與評分！")
            is_playing = True
            start_time = time.time()
            total_pause_time = 0
            cap_teacher.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif key == 32:  # 空白鍵
            if is_playing:
                if is_paused:
                    pause_duration = (time.time() - pause_start_time) * 1000
                    total_pause_time += int(pause_duration)
                    is_paused = False
                    print(f"▶ 恢復播放 (Speed: {current_speed:.2f}x)")
                else:
                    pause_start_time = time.time()
                    is_paused = True
                    print("⏸ 暫停")
        elif key == ord('c') or key == ord('C'):
            show_controller = not show_controller
            print("速度控制器", "已顯示" if show_controller else "已隱藏")

# =============================================================================
# 資源釋放
# =============================================================================
cap_webcam.release()
cap_teacher.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("程式已結束。")