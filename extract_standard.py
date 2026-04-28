import cv2
import mediapipe as mp
import json
import time

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose

# 設定影片路徑與輸出的 JSON 檔名
VIDEO_PATH = 'teacher_dance.mp4'
OUTPUT_JSON = 'dance_standard.json'

cap = cv2.VideoCapture(VIDEO_PATH)

# 取得影片的幀率 (FPS)，用來計算每個影格的時間點
fps = cap.get(cv2.CAP_PROP_FPS)

# 用來存放所有骨架資料的陣列
standard_pose_data = []

print(f"開始處理影片: {VIDEO_PATH} (FPS: {fps})")
start_time = time.time()

with mp_pose.Pose(
    static_image_mode=False,      # 設定為 False 表示處理連續影片
    model_complexity=1,           # 複雜度 0, 1, 2 (1 是速度與精度的良好平衡)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    
    frame_index = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("影片讀取完畢。")
            break

        # 將 BGR 轉換為 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # 如果該幀有偵測到人體骨架
        if results.pose_landmarks:
            
            # 計算該幀的影片時間戳記 (毫秒)
            timestamp_ms = (frame_index / fps) * 1000 if fps > 0 else 0
            
            frame_data = {
                "frame": frame_index,
                "timestamp_ms": round(timestamp_ms, 2),
                "landmarks": []
            }
            
            # 遍歷 33 個關鍵點，並儲存 x, y, z 與可見度 v
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                frame_data["landmarks"].append({
                    "id": i,
                    "x": round(landmark.x, 5),
                    "y": round(landmark.y, 5),
                    "z": round(landmark.z, 5),
                    "v": round(landmark.visibility, 5) # 可見度，可用來判斷關節是否被遮擋
                })
                
            standard_pose_data.append(frame_data)
            
        frame_index += 1

cap.release()

# 將資料寫入 JSON 檔案
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(standard_pose_data, f, indent=4)

end_time = time.time()
print(f"處理完成！共儲存了 {len(standard_pose_data)} 幀的骨架資料。")
print(f"資料已匯出至 {OUTPUT_JSON}。耗時: {round(end_time - start_time, 2)} 秒")