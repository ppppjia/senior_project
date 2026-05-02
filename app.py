import cv2
import mediapipe as mp

# ====================== 初始化 ======================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 增加這些設定可以提高相機穩定性
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ 無法開啟攝影機！請檢查：")
    print("   1. 攝影機是否被其他程式占用")
    print("   2. 是否有權限存取攝影機")
    print("   3. 試試改成 cv2.VideoCapture(1) 或其他數字")
    exit()

print("✅ 攝影機開啟成功")

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # 可改成 0 加速
) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("⚠️ 無法讀取影像畫面")
            break

        # 轉 RGB 並處理
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = pose.process(image_rgb)

        # 轉回 BGR 準備顯示
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # 繪製骨架
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # 顯示影像
        cv2.imshow('AI Dance Tracker PoC', image)

        # 關鍵：使用較長的 waitKey + 檢查視窗是否被關閉
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27 or cv2.getWindowProperty('AI Dance Tracker PoC', cv2.WND_PROP_VISIBLE) < 1:
            break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
print("程式結束")