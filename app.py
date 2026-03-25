import cv2
import mediapipe as mp

# 初始化 MediaPipe Pose 模型與繪圖工具
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 開啟視訊鏡頭 (0 通常是電腦的內建預設攝影機)
cap = cv2.VideoCapture(0)

# 設定 Pose 模型參數
with mp_pose.Pose(
    min_detection_confidence=0.5, # 偵測到人體的信心閾值
    min_tracking_confidence=0.5   # 追蹤骨架的信心閾值
) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("無法讀取攝影機畫面")
            break

        # OpenCV 預設讀取 BGR 格式，MediaPipe 需要 RGB 格式，所以先轉換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 為了提升效能，可以將影像標記為不可寫入
        image_rgb.flags.writeable = False
        
        # 進行姿態辨識
        results = pose.process(image_rgb)
        
        # 將影像轉回 BGR 以便 OpenCV 顯示
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 如果有偵測到骨架，將其繪製在畫面上
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # 關節點顏色
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # 骨架線條顏色
            )
            
            # 這裡可以擷取特定關節的座標，例如列出右肩膀 (RIGHT_SHOULDER, index 12) 的座標
            # right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            # print(f"右肩 X: {right_shoulder.x}, Y: {right_shoulder.y}")

        # 顯示影像，視窗名稱為 'AI Dance Tracker'
        cv2.imshow('AI Dance Tracker PoC', image)

        # 按下 'q' 鍵離開迴圈
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 釋放資源並關閉視窗
cap.release()
cv2.destroyAllWindows()