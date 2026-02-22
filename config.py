# config.py
import mediapipe as mp

# 模型與路徑設定
MODEL_PATH = 'pose_landmarker_heavy.task'

# 辨識信心度微調
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# 繪圖設定
SKELETON_COLOR = (255, 255, 255)  # 白色線條
JOINT_COLOR = (0, 255, 0)        # 綠色關節點
LINE_THICKNESS = 2

# 骨架連線定義 (維持你們目前的定義)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 12), (12, 24), (24, 23), (23, 11), (23, 25), (25, 27), (27, 29),
    (27, 31), (29, 31), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]