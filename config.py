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

# Ghost 相關設定
VISIBILITY_THRESHOLD = 0.5        # 關節可見度門檻
GHOST_ALPHA = 0.45                # 老師 ghost 透明度 (0.0~1.0)
GHOST_HULL_ALPHA = 0.15           # 安全區域填充透明度
GHOST_PANEL_RATIO = 0.60          # ghost panel 佔畫面寬度比例
JOINT_DISTANCE_THRESHOLD = 0.10   # 關節距離門檻 (normalized，可調 0.08~0.15)

# 計分設定
PENALTY_PER_FRAME = 0.5           # 每幀全部超出時最大扣分（超出比例 × 此值）

# Ghost 顏色
GHOST_JOINT_COLOR = (50, 50, 50)      # 老師 ghost 關節（黑色）
GHOST_LINE_COLOR  = (30, 30, 30)      # 老師 ghost 骨架線（黑色）
OUT_OF_BOUNDS_COLOR = (0, 0, 255)     # 超出範圍標紅

# 骨架連線定義 (維持你們目前的定義)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 12), (12, 24), (24, 23), (23, 11), (23, 25), (25, 27), (27, 29),
    (27, 31), (29, 31), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]