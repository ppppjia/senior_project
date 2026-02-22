# visualizer.py
import cv2
import config

def draw_skeleton(canvas, landmarks, w, h):
    points = []
    # 座標轉換
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        cv2.circle(canvas, (px, py), 4, config.JOINT_COLOR, -1)

    # 畫連線
    for conn in config.POSE_CONNECTIONS:
        start_idx, end_idx = conn
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(canvas, points[start_idx], points[end_idx], 
                     config.SKELETON_COLOR, config.LINE_THICKNESS)