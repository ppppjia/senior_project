import cv2
import mediapipe as mp
import json
import time
import numpy as np
from collections import deque
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from pose_utils import normalize_pose

# --- 1. 載入資料庫 ---
print("載入標準動作資料庫...")
try:
    with open('dance_standard.json', 'r', encoding='utf-8') as f:
        standard_pose_data = json.load(f)
except FileNotFoundError:
    print("❌ 找不到 dance_standard.json！請先執行擷取腳本建立題庫。")
    exit()

# --- 2. 初始化 AI 與攝影機 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap_webcam = cv2.VideoCapture(0)
cap_teacher = cv2.VideoCapture('teacher_dance.mp4')

# 建立可自由縮放的視窗
window_name = 'AI Dance Teaching System'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 480)

# 系統狀態變數
is_playing = False
start_time = 0
current_score = 0

# --- DTW 參數設定 ---
BUFFER_SIZE = 15  # 收集 15 幀 (約 0.5 秒) 的歷史紀錄來算 DTW
student_buffer = deque(maxlen=BUFFER_SIZE)

print("✅ 系統啟動完成！請點擊彈出的視窗，確認輸入法為英文，然後按下 's' 鍵開始。")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap_webcam.isOpened():
        # 讀取 WebCam
        success_webcam, image_webcam = cap_webcam.read()
        if not success_webcam: break
        image_webcam = cv2.flip(image_webcam, 1)
        h, w, _ = image_webcam.shape

        # 預設老師畫面為黑畫面
        image_teacher_display = np.zeros((h, w, 3), dtype=np.uint8)

        # ====== 核心邏輯 ======
        if is_playing:
            elapsed_ms = (time.time() - start_time) * 1000
            
            # 同步播放老師影片
            cap_teacher.set(cv2.CAP_PROP_POS_MSEC, elapsed_ms)
            success_teacher, image_teacher_raw = cap_teacher.read()

            if success_teacher:
                # 調整老師畫面高度以利拼接
                th, tw, _ = image_teacher_raw.shape
                new_tw = int(tw * (h / th))
                image_teacher_display = cv2.resize(image_teacher_raw, (new_tw, h))

                # 進行學生骨架辨識
                image_rgb = cv2.cvtColor(image_webcam, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    # 畫出骨架
                    mp_drawing.draw_landmarks(image_webcam, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # 正規化學生骨架
                    live_landmarks = [{'id': i, 'x': lm.x, 'y': lm.y, 'z': lm.z, 'v': lm.visibility} for i, lm in enumerate(results.pose_landmarks.landmark)]
                    norm_live = normalize_pose(live_landmarks)
                    
                    # 將上半身主要關節 (11肩 ~ 24髖) 攤平存入 Buffer 給 DTW 用
                    flat_live = []
                    for i in range(11, 25):
                        flat_live.extend([norm_live[i]['x'], norm_live[i]['y']])
                    student_buffer.append(flat_live)

                    # 當 Buffer 收集滿 0.5 秒後，開始計算分數
                    if len(student_buffer) == BUFFER_SIZE:
                        teacher_buffer = []
                        start_time_target = elapsed_ms - (BUFFER_SIZE * 33.3) # 往前推算
                        
                        # 抓出 JSON 裡面對應時間段的老師資料
                        for frame_data in standard_pose_data:
                            if start_time_target <= frame_data['timestamp_ms'] <= elapsed_ms + 200:
                                norm_target = normalize_pose(frame_data['landmarks'])
                                flat_target = []
                                for i in range(11, 25):
                                    flat_target.extend([norm_target[i]['x'], norm_target[i]['y']])
                                teacher_buffer.append(flat_target)

                        # 執行 DTW 演算法
                        if len(teacher_buffer) > 0:
                            # 算出時間扭曲後的最短距離
                            distance, _ = fastdtw(student_buffer, teacher_buffer, dist=euclidean)
                            
                            # 【分數調校區】如果覺得太難，把 25 改大；如果覺得太簡單，把 25 改小
                            raw_score = 100 - (distance / 25)
                            current_score = max(0, min(100, int(raw_score)))

                # 繪製 UI (分數與時間)
                color = (0, 255, 0) if current_score > 80 else (0, 165, 255) if current_score > 60 else (0, 0, 255)
                cv2.putText(image_webcam, f"Score: {current_score}", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
                cv2.putText(image_webcam, f"Time: {int(elapsed_ms/1000)}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(image_teacher_display, "FINISH!", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                is_playing = False
        else:
            cv2.putText(image_teacher_display, "Waiting to Start (Press 's')", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- 3. 畫面拼接與顯示 ---
        if not is_playing:
            image_teacher_display = cv2.resize(image_teacher_display, (w, h))
            
        combined_image = np.hstack((image_teacher_display, image_webcam))
        cv2.imshow(window_name, combined_image)

        # --- 4. 鍵盤控制 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not is_playing:
            is_playing = True
            start_time = time.time()
            cap_teacher.set(cv2.CAP_PROP_POS_FRAMES, 0) # 影片回歸原點
            student_buffer.clear() # 清空歷史資料
            current_score = 0

cap_webcam.release()
cap_teacher.release()
cv2.destroyAllWindows()
