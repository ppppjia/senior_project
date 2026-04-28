import cv2
import mediapipe as mp
import json
import time
from pose_utils import normalize_pose, calculate_pose_error
import numpy as np

# 1. 載入標準動作 JSON
print("載入標準動作資料庫...")
with open('dance_standard.json', 'r', encoding='utf-8') as f:
    standard_pose_data = json.load(f)

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 2. 同時開啟兩支攝影機/影片串流
cap_webcam = cv2.VideoCapture(0)
cap_teacher = cv2.VideoCapture('teacher_dance.mp4') # 載入老師的影片

# 系統狀態變數
is_playing = False
start_time = 0
current_score = 0

print("系統啟動完成！請站在鏡頭前。")
print("按下 's' 鍵開始播放影片並評分，按下 'q' 鍵離開。")

# --- 建立可縮放的視窗 ---
window_name = 'AI Dance Teaching System - Split Screen'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # 允許視窗自由縮放

# 設定一個合理的初始視窗大小 (例如寬 1280, 高 480)
# 這樣一啟動就不會整個爆出螢幕
cv2.resizeWindow(window_name, 1280, 480)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap_webcam.isOpened():
        # 讀取學生的 WebCam 畫面
        success_webcam, image_webcam = cap_webcam.read()
        if not success_webcam:
            break

        image_webcam = cv2.flip(image_webcam, 1) # 鏡像翻轉
        
        # 預設老師的畫面為全黑 (在還沒按 S 開始前)
        # 取得 webcam 的長寬，用來建立相同大小的黑畫面
        h, w, _ = image_webcam.shape
        image_teacher_display = cv2.resize(cv2.imread('teacher_dance.mp4') if False else image_webcam, (w, h)) 
        image_teacher_display[:] = (0, 0, 0) # 塗黑
        cv2.putText(image_teacher_display, "Waiting to Start (Press 's')", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # ====== 核心評分與影片同步邏輯 ======
        if is_playing:
            elapsed_ms = (time.time() - start_time) * 1000
            
            # --- 重點 1：根據經過的時間，手動設定老師影片的播放位置 ---
            cap_teacher.set(cv2.CAP_PROP_POS_MSEC, elapsed_ms)
            success_teacher, image_teacher_raw = cap_teacher.read()
            
            if success_teacher:
                # 將老師的影片縮放到跟 WebCam 一樣的高度，以利水平拼接
                th, tw, _ = image_teacher_raw.shape
                ratio = h / th
                new_tw = int(tw * ratio)
                image_teacher_display = cv2.resize(image_teacher_raw, (new_tw, h))
                
                # --- 重點 2：執行 AI 骨架辨識與評分 (針對學生) ---
                image_rgb = cv2.cvtColor(image_webcam, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image_webcam, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # 尋找對應的 JSON 標準數據
                    target_frame = None
                    for frame_data in standard_pose_data:
                        if frame_data['timestamp_ms'] >= elapsed_ms:
                            target_frame = frame_data
                            break 
                    
                    if target_frame:
                        live_landmarks = [{'id': i, 'x': lm.x, 'y': lm.y, 'z': lm.z, 'v': lm.visibility} for i, lm in enumerate(results.pose_landmarks.landmark)]
                        norm_live = normalize_pose(live_landmarks)
                        norm_target = normalize_pose(target_frame['landmarks'])
                        
                        error = calculate_pose_error(norm_live, norm_target)
                        raw_score = 100 - (error * 40) 
                        current_score = max(0, min(100, int(raw_score)))

                # 在學生的畫面上繪製分數與時間
                color = (0, 255, 0) if current_score > 80 else (0, 165, 255) if current_score > 60 else (0, 0, 255)
                cv2.putText(image_webcam, f"Score: {current_score}", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
                cv2.putText(image_webcam, f"Time: {int(elapsed_ms/1000)}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
            else:
                # 影片播完了
                cv2.putText(image_teacher_display, "FINISH!", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                is_playing = False

        # --- 重點 3：畫面拼接 (Concatenation) ---
        # 如果老師畫面的寬度跟學生不一樣 (因為比例縮放)，為了用 hconcat，我們強制把它 Resize 到一樣寬度
        # 實務上更好的做法是用 numpy 建立一個大畫布把兩張圖貼上去，這裡為了簡潔先強制縮放
        # image_teacher_display = cv2.resize(image_teacher_display, (w, h))
        
        # 將老師畫面(左)與學生畫面(右)水平拼接在一起
        # combined_image = cv2.hconcat([image_teacher_display, image_webcam])

        # 顯示最終的雙拼畫面
        # cv2.imshow('AI Dance Teaching System - Split Screen', combined_image)

        # --- 改良版的畫面拼接 (保持正確的長寬比) ---
        if is_playing and success_teacher:
            # 老師畫面的高度已經縮放成跟 WebCam 一樣 (h)
            # 我們保留它算出來的正確寬度 (new_tw)，不強制拉伸
            pass # 前面已經處理過 image_teacher_display 了
        else:
            # 如果還沒開始，確保黑畫面的高度也是 h
            # 這裡我們給黑畫面一個預設寬度，例如跟 WebCam 一樣寬
            image_teacher_display = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(image_teacher_display, "Waiting to Start (Press 's')", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 使用 numpy.hstack 進行水平拼接。
        # 好處是只要兩個矩陣的高度 (h) 相同，寬度不同也可以拼在一起！
        combined_image = np.hstack((image_teacher_display, image_webcam))

        # 顯示最終畫面 (請確保視窗名稱與前面 namedWindow 設定的一致)
        cv2.imshow('AI Dance Teaching System - Split Screen', combined_image)

        # 鍵盤監聽事件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not is_playing:
            print("開始播放與評分！")
            is_playing = True
            start_time = time.time()
            # 確保影片回到開頭
            cap_teacher.set(cv2.CAP_PROP_POS_FRAMES, 0)

cap_webcam.release()
cap_teacher.release()
cv2.destroyAllWindows()