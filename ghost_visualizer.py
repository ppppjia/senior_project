import cv2
import numpy as np
import config

# MediaPipe 33 點人形輪廓連接順序（順時針繞一圈）
# 頭部 → 右肩 → 右手 → 右腰 → 右腳 → 左腳 → 左腰 → 左手 → 左肩 → 回頭
SILHOUETTE_OUTLINE = [
    # 頭（用鼻子 + 耳朵近似）
    7, 3, 1, 4, 8,
    # 右半身
    12, 14, 16, 18, 20, 16, 22,
    # 右腰到腳
    12, 24, 26, 28, 32, 30, 28,
    # 跨到左腳
    27, 29, 31, 27,
    # 左腰回身
    23, 25, 27, 23,
    # 左半身手臂
    11, 23, 21, 19, 15, 17, 15, 13,
    # 左肩回頭
    11, 7
]


# =============================================================================
# 基礎工具
# =============================================================================

def create_panel(image, panel_w, panel_h, placeholder_text=None):
    if image is None:
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        if placeholder_text:
            cv2.putText(panel, placeholder_text, (20, panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return panel
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.resize(image, (panel_w, panel_h))


# =============================================================================
# 剪影工具
# =============================================================================

def landmarks_to_silhouette_mask(landmarks, w, h):
    """
    從 landmarks 產生人形剪影 mask（白底黑字，uint8）。
    做法：
      1. 取所有可見點的 convex hull 作為身體輪廓
      2. 另外對頭部區域畫橢圓補充（MediaPipe 頭部點較少）
    回傳 shape (h, w) 的 mask，人形區域=255，背景=0
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    # --- 身體 convex hull ---
    body_indices = list(range(11, 33))   # 11~32：肩膀以下全身
    pts = []
    for i in body_indices:
        if i < len(landmarks) and landmarks[i].get('v', 0) > config.VISIBILITY_THRESHOLD:
            pts.append([int(landmarks[i]['x'] * w), int(landmarks[i]['y'] * h)])

    if len(pts) >= 3:
        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

    # --- 頭部橢圓（用鼻子 + 耳朵估算位置與大小）---
    head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    head_pts = []
    for i in head_indices:
        if i < len(landmarks) and landmarks[i].get('v', 0) > config.VISIBILITY_THRESHOLD:
            head_pts.append([landmarks[i]['x'] * w, landmarks[i]['y'] * h])

    if head_pts:
        hx = [p[0] for p in head_pts]
        hy = [p[1] for p in head_pts]
        cx = int(np.mean(hx))
        cy = int(np.mean(hy))
        # 以耳朵間距估算頭寬，高度約 1.3 倍
        ear_width = max(int((max(hx) - min(hx)) * 0.8), 20)
        ear_height = max(int(ear_width * 1.3), 25)
        cv2.ellipse(mask, (cx, cy), (ear_width, ear_height), 0, 0, 360, 255, -1)

    # 膨脹讓輪廓稍微圓潤
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def draw_silhouette(canvas, mask, color, alpha=1.0):
    """
    把 mask 區域填上 color，疊到 canvas 上。
    """
    overlay = canvas.copy()
    overlay[mask == 255] = color
    if alpha < 1.0:
        return cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)
    return overlay


# =============================================================================
# 超出範圍偵測（改成 mask 判斷）
# =============================================================================

def build_teacher_hull(teacher_landmarks, w, h):
    """convex hull，供 get_out_of_bounds_indices 使用"""
    pts = []
    for lm in teacher_landmarks:
        if lm.get('v', 0) > config.VISIBILITY_THRESHOLD:
            pts.append([int(lm['x'] * w), int(lm['y'] * h)])
    if len(pts) < 3:
        return None
    return cv2.convexHull(np.array(pts, dtype=np.int32))


def get_out_of_bounds_indices(student_landmarks, teacher_mask, teacher_landmarks, w, h):
    """
    回傳超出範圍的學生關節 index 集合。
    超出條件（OR）：
      1. 對應像素在 teacher_mask 外（mask==0）
      2. 與對應 teacher 關節距離超過 JOINT_DISTANCE_THRESHOLD
    """
    out_indices = set()

    for i, s_lm in enumerate(student_landmarks):
        if s_lm.get('v', 0) <= config.VISIBILITY_THRESHOLD:
            continue

        sx = int(np.clip(s_lm['x'] * w, 0, w - 1))
        sy = int(np.clip(s_lm['y'] * h, 0, h - 1))

        # --- 判斷 1：mask ---
        if teacher_mask is not None and teacher_mask[sy, sx] == 0:
            out_indices.add(i)
            continue

        # --- 判斷 2：距離 ---
        if i < len(teacher_landmarks):
            t_lm = teacher_landmarks[i]
            if t_lm.get('v', 0) > config.VISIBILITY_THRESHOLD:
                dx = s_lm['x'] - t_lm['x']
                dy = s_lm['y'] - t_lm['y']
                if (dx**2 + dy**2) ** 0.5 > config.JOINT_DISTANCE_THRESHOLD:
                    out_indices.add(i)

    return out_indices


# =============================================================================
# 骨架繪製（保留，學生仍用骨架顯示）
# =============================================================================

def draw_pose(canvas, landmarks, joint_color, line_color, alpha=1.0,
              highlight_indices=None, highlight_color=None):
    if highlight_color is None:
        highlight_color = config.OUT_OF_BOUNDS_COLOR

    overlay = canvas.copy()
    h, w = canvas.shape[:2]

    for i, lm in enumerate(landmarks):
        if lm.get('v', 0) > config.VISIBILITY_THRESHOLD:
            px = int(lm['x'] * w)
            py = int(lm['y'] * h)
            color = highlight_color if (highlight_indices and i in highlight_indices) else joint_color
            radius = 7 if (highlight_indices and i in highlight_indices) else 5
            cv2.circle(overlay, (px, py), radius, color, -1)

    for conn in config.POSE_CONNECTIONS:
        start_idx, end_idx = conn
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            s = landmarks[start_idx]
            e = landmarks[end_idx]
            if s.get('v', 0) > config.VISIBILITY_THRESHOLD and e.get('v', 0) > config.VISIBILITY_THRESHOLD:
                sx2, sy2 = int(s['x'] * w), int(s['y'] * h)
                ex2, ey2 = int(e['x'] * w), int(e['y'] * h)
                seg_out = highlight_indices and (start_idx in highlight_indices or end_idx in highlight_indices)
                color = highlight_color if seg_out else line_color
                thickness = 3 if seg_out else 2
                cv2.line(overlay, (sx2, sy2), (ex2, ey2), color, thickness)

    if alpha < 1.0:
        return cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)
    return overlay


# =============================================================================
# 紅燈警示
# =============================================================================

def draw_warning_light(canvas, active):
    """右上角圓形指示燈：超出=紅，正常=綠"""
    h, w = canvas.shape[:2]
    center = (w - 30, 30)
    color = (0, 0, 220) if active else (0, 180, 0)
    cv2.circle(canvas, center, 18, color, -1)
    cv2.circle(canvas, center, 18, (255, 255, 255), 2)


# =============================================================================
# Panel 建立
# =============================================================================

def create_teacher_panel(image_teacher, teacher_preview, panel_w, panel_h, finish_message=None):
    if image_teacher is not None:
        panel = create_panel(image_teacher, panel_w, panel_h)
    elif teacher_preview is not None:
        panel = create_panel(teacher_preview, panel_w, panel_h)
    else:
        panel = create_panel(None, panel_w, panel_h, placeholder_text="Teacher video unavailable")

    if finish_message:
        cv2.putText(panel, finish_message, (20, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return panel


def create_ghost_panel(webcam_image, target_frame, student_landmarks,
                        panel_w, panel_h, is_playing, ghost_alpha=None,
                        teacher_seg_mask=None):
    """
    回傳：(panel, out_indices)
    """
    if ghost_alpha is None:
        ghost_alpha = config.GHOST_ALPHA

    panel = create_panel(webcam_image, panel_w, panel_h,
                          placeholder_text="Student camera unavailable")
    h, w = panel.shape[:2]
    out_indices = set()
    teacher_mask = None

    if target_frame is not None:
        # 1. 產生老師剪影 mask：優先用 seg_mask，沒有才用 landmarks
        if teacher_seg_mask is not None:
            # seg_mask 是原始 webcam 尺寸，需縮放到 panel 尺寸
            teacher_mask = cv2.resize(teacher_seg_mask, (w, h),
                                      interpolation=cv2.INTER_NEAREST) * 255
        else:
            teacher_mask = landmarks_to_silhouette_mask(target_frame['landmarks'], w, h)

        # 2. 畫老師剪影（深灰黑，半透明）
        panel = draw_silhouette(panel, teacher_mask, color=(40, 40, 40), alpha=ghost_alpha)

        # 3. 計算學生超出範圍
        if student_landmarks is not None:
            out_indices = get_out_of_bounds_indices(
                student_landmarks, teacher_mask, target_frame['landmarks'], w, h
            )

    # 4. 畫學生骨架（超出部分標紅）
    if student_landmarks is not None:
        panel = draw_pose(panel, student_landmarks,
                          joint_color=(0, 255, 0),
                          line_color=(0, 200, 0),
                          alpha=1.0,
                          highlight_indices=out_indices)

        cv2.putText(panel, "Student", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 5. 右上角紅燈
    draw_warning_light(panel, active=bool(out_indices))

    # 6. 超出警告文字
    if out_indices:
        msg = f"Out of bounds: {len(out_indices)} joints"
        cv2.putText(panel, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.OUT_OF_BOUNDS_COLOR, 2)

    if not is_playing:
        cv2.putText(panel, "Press 's' to start", (int(w * 0.15), h - 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return panel, out_indices
