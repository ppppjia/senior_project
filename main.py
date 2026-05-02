# =============================================================================
# main_test.py - AI Dance Teaching System with Ghost Shadow & DTW 評分
# =============================================================================

import cv2
import mediapipe as mp
import time
import numpy as np

from ghost_visualizer import create_ghost_panel, create_teacher_panel
from dtw_scoring import create_student_buffer, get_teacher_window, update_student_buffer, compute_dtw_score
from speed_controller import draw_speed_controller, get_speed_from_click
from standard_data import check_and_update_standard, load_standard_pose_data

# 初始化 MediaPipe
mp_pose = mp.solutions.pose

# 全域狀態
current_speed = 1.0
show_controller = True


def mouse_callback(event, x, y, flags, param):
    global current_speed
    if event == cv2.EVENT_LBUTTONDOWN and show_controller:
        new_speed = get_speed_from_click(x, y, current_speed)
        if new_speed != current_speed:
            current_speed = new_speed
            print(f"速度已調整為: {current_speed:.2f}x")


# =============================================================================
# 主程式開始
# =============================================================================
print("檢查標準動作資料...")
if not check_and_update_standard():
    print("警告：無法自動更新標準動作資料，將嘗試載入現有資料")

print("載入標準動作資料...")
standard_pose_data = load_standard_pose_data()

cap_webcam = cv2.VideoCapture(0)
cap_teacher = cv2.VideoCapture('teacher_dance.mp4')

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

is_playing = False
is_paused = False
start_time = 0
pause_start_time = 0
pause_elapsed_ms = 0
total_pause_time = 0
current_score = 0
student_buffer = create_student_buffer()

print("\n=== AI Dance Teaching System 已啟動 ===")
print("操作說明：")
print("  's'     → 開始播放與評分")
print("  空白鍵  → 暫停 / 繼續")
print("  'C'     → 顯示 / 隱藏速度控制器（滑鼠點擊調整）")
print("  'q'     → 離開程式\n")

window_name = 'AI Dance Teaching System - Ghost Shadow'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)
cv2.setMouseCallback(window_name, mouse_callback)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap_webcam.isOpened():
        success_webcam, image_webcam = cap_webcam.read()
        if not success_webcam:
            break

        image_webcam = cv2.flip(image_webcam, 1)
        h, w, _ = image_webcam.shape
        panel_w = w // 2
        target_frame = None
        live_landmarks = None
        elapsed_ms = 0
        success_teacher = False
        image_teacher = None

        if is_playing:
            if is_paused:
                elapsed_ms = pause_elapsed_ms
            else:
                raw_elapsed = (time.time() - start_time) * 1000
                elapsed_ms = int(raw_elapsed * current_speed) - total_pause_time
                pause_elapsed_ms = elapsed_ms

            if video_duration_ms > 0 and elapsed_ms >= video_duration_ms:
                success_teacher = False
                is_playing = False
            else:
                cap_teacher.set(cv2.CAP_PROP_POS_MSEC, elapsed_ms)
                success_teacher, image_teacher = cap_teacher.read()
                if success_teacher:
                    elapsed_ms = cap_teacher.get(cv2.CAP_PROP_POS_MSEC)
                    if int(cap_teacher.get(cv2.CAP_PROP_POS_FRAMES) or 0) >= video_frame_count:
                        success_teacher = False

        finish_message = None
        if is_playing and not success_teacher:
            finish_message = "FINISH! Press 's' to replay"

        teacher_panel = create_teacher_panel(image_teacher if success_teacher else None,
                                            teacher_preview, panel_w, h,
                                            finish_message=finish_message)

        image_rgb = cv2.cvtColor(image_webcam, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            live_landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'v': lm.visibility}
                              for lm in results.pose_landmarks.landmark]

        if is_playing and success_teacher:
            for frame_data in standard_pose_data:
                if frame_data['timestamp_ms'] >= elapsed_ms:
                    target_frame = frame_data
                    break

            if live_landmarks is not None:
                buffer_ready = update_student_buffer(student_buffer, live_landmarks)
                if buffer_ready:
                    teacher_windows = get_teacher_window(standard_pose_data, elapsed_ms)
                    score = compute_dtw_score(student_buffer, teacher_windows)
                    if score is not None:
                        current_score = score

        ghost_panel = create_ghost_panel(image_webcam, target_frame, live_landmarks,
                                        panel_w, h, is_playing)

        combined_image = np.hstack((teacher_panel, ghost_panel))

        if is_playing:
            score_color = (0, 255, 0) if current_score > 80 else (0, 165, 255) if current_score > 60 else (0, 0, 255)
            cv2.putText(combined_image, f"Score: {current_score}", (panel_w + 10, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, score_color, 2)
            cv2.putText(combined_image, f"Time: {int(elapsed_ms / 1000)}s", (panel_w + 10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(combined_image, f"Speed: {current_speed:.2f}x", (panel_w + 10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            if is_paused:
                cv2.putText(combined_image, "PAUSED", (panel_w + 10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 2)
        #else:
        #    cv2.putText(combined_image, "Press 's' to start playback", (panel_w + 10, h - 80),
        #                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if show_controller:
            draw_speed_controller(combined_image, current_speed)

        cv2.imshow(window_name, combined_image)
        key = cv2.waitKey(1) & 0xFF

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("使用者已關閉視窗")
            break

        if key == ord('q'):
            print("按下 'q' 離開系統")
            break
        elif key == ord('s') and not is_playing:
            print("開始播放與評分！")
            is_playing = True
            start_time = time.time()
            total_pause_time = 0
            pause_elapsed_ms = 0
            is_paused = False
            student_buffer.clear()
            current_score = 0
            cap_teacher.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif key == 32:
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

cap_webcam.release()
cap_teacher.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("程式已結束。")
