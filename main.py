# =============================================================================
# main_test.py - AI Dance Teaching System with Ghost Shadow & DTW 評分
# =============================================================================

import cv2
import mediapipe as mp
import time
import numpy as np

import config
from ghost_visualizer import create_ghost_panel, create_teacher_panel
from dtw_scoring import create_student_buffer, update_student_buffer, ScoreTracker
from speed_controller import draw_speed_controller, get_speed_from_click, get_controller_rect, is_inside_controller_header
from standard_data import check_and_update_standard, load_standard_pose_data

# 初始化 MediaPipe
mp_pose = mp.solutions.pose

# 全域狀態
current_speed = 1.0
show_controller = True
current_frame_width = 0
controller_dragging = False
controller_origin = None
controller_drag_offset = (0, 0)
controller_history = []


def record_controller_position(x, y):
    if not controller_history or controller_history[-1]['x'] != x or controller_history[-1]['y'] != y:
        controller_history.append({'x': x, 'y': y})


def mouse_callback(event, x, y, flags, param):
    global current_speed, current_frame_width, controller_dragging, controller_origin, controller_drag_offset
    if not show_controller or current_frame_width <= 0:
        return

    controller_x, controller_y, controller_w, controller_h = get_controller_rect(current_frame_width, controller_origin)
    inside_controller = (controller_x <= x <= controller_x + controller_w and
                         controller_y <= y <= controller_y + controller_h)
    inside_header = is_inside_controller_header(x, y, current_frame_width, controller_origin)

    if event == cv2.EVENT_LBUTTONDOWN and inside_header:
        controller_dragging = True
        controller_drag_offset = (x - controller_x, y - controller_y)
        controller_origin = [controller_x, controller_y]
        record_controller_position(controller_x, controller_y)
    elif event == cv2.EVENT_LBUTTONDOWN and inside_controller:
        new_speed = get_speed_from_click(x, y, current_speed, current_frame_width, controller_origin)
        if new_speed != current_speed:
            current_speed = new_speed
            print(f"速度已調整為: {current_speed:.2f}x")
    elif event == cv2.EVENT_MOUSEMOVE and controller_dragging:
        new_x = x - controller_drag_offset[0]
        new_y = y - controller_drag_offset[1]
        new_x = max(0, min(new_x, current_frame_width - controller_w))
        if new_x != controller_origin[0] or new_y != controller_origin[1]:
            controller_origin = [new_x, new_y]
            record_controller_position(new_x, new_y)
    elif event == cv2.EVENT_LBUTTONUP and controller_dragging:
        controller_dragging = False
        new_x = x - controller_drag_offset[0]
        new_y = y - controller_drag_offset[1]
        new_x = max(0, min(new_x, current_frame_width - controller_w))
        controller_origin = [new_x, new_y]
        record_controller_position(new_x, new_y)


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
teacher_width = None
teacher_height = None
video_duration_ms = 0
video_frame_count = 0

if cap_teacher.isOpened():
    fps = cap_teacher.get(cv2.CAP_PROP_FPS) or 30
    teacher_width = int(cap_teacher.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    teacher_height = int(cap_teacher.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Teacher video opened: {teacher_width}x{teacher_height} @ {fps:.2f} fps")
    video_frame_count = int(cap_teacher.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration_ms = (video_frame_count / fps) * 1000 if fps > 0 else 0
    cap_teacher.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success_preview, teacher_preview = cap_teacher.read()
    if success_preview:
        teacher_preview = teacher_preview.copy()
        teacher_height, teacher_width = teacher_preview.shape[:2]
    cap_teacher.set(cv2.CAP_PROP_POS_FRAMES, 0)

is_playing = False
is_paused = False
start_time = 0
pause_start_time = 0
pause_elapsed_ms = 0
total_pause_time = 0
current_score = 100
student_buffer = create_student_buffer()
score_tracker = ScoreTracker()

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

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_pose.Pose(static_image_mode=False, enable_segmentation=True,
                  min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_teacher:
    while cap_webcam.isOpened():
        success_webcam, image_webcam = cap_webcam.read()
        if not success_webcam:
            break

        image_webcam = cv2.flip(image_webcam, 1)
        h, w, _ = image_webcam.shape
        teacher_panel_h = teacher_height if teacher_height else h
        teacher_panel_w = teacher_width if teacher_width else max(1, int(w * (1.0 - config.GHOST_PANEL_RATIO)))
        student_panel_h = teacher_panel_h
        student_panel_w = int((w / h) * student_panel_h) if h else w
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
                                            teacher_preview, teacher_panel_w, teacher_panel_h,
                                            finish_message=finish_message)

        teacher_seg_mask = None
        if is_playing and success_teacher and image_teacher is not None:
            teacher_rgb = cv2.cvtColor(image_teacher, cv2.COLOR_BGR2RGB)
            teacher_results = pose_teacher.process(teacher_rgb)
            if teacher_results.segmentation_mask is not None:
                # mask 值 0.0~1.0，>0.5 為人形區域
                teacher_seg_mask = (teacher_results.segmentation_mask > 0.5).astype(np.uint8)

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

        ghost_panel, out_indices = create_ghost_panel(image_webcam, target_frame, live_landmarks,
                                                      student_panel_w, student_panel_h, is_playing,
                                                      teacher_seg_mask=teacher_seg_mask)

        # 計分：有超出才扣，不加分
        if is_playing and live_landmarks is not None:
            current_score = score_tracker.update(out_indices, total_joints=len(live_landmarks))

        combined_image = np.hstack((teacher_panel, ghost_panel))
        current_frame_width = combined_image.shape[1]

        if is_playing:
            score_color = (0, 255, 0) if current_score > 80 else (0, 165, 255) if current_score > 60 else (0, 0, 255)
            cv2.putText(combined_image, f"Score: {current_score}", (teacher_panel_w + 10, 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, score_color, 2)
            cv2.putText(combined_image, f"Time: {int(elapsed_ms / 1000)}s", (teacher_panel_w + 10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(combined_image, f"Speed: {current_speed:.2f}x", (teacher_panel_w + 10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            if is_paused:
                cv2.putText(combined_image, "PAUSED", (teacher_panel_w + 10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 2)
        #else:
        #    cv2.putText(combined_image, "Press 's' to start playback", (teacher_panel_w + 10, h - 80),
        #                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if show_controller:
            draw_speed_controller(combined_image, current_speed, history_positions=controller_history, origin=controller_origin)

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
            current_score = 100
            score_tracker.reset()
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
