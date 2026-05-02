import cv2
import numpy as np
import config


def create_panel(image, panel_w, panel_h, placeholder_text=None):
    if image is None:
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        if placeholder_text:
            cv2.putText(panel, placeholder_text, (20, panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return panel

    return cv2.resize(image, (panel_w, panel_h))


def draw_pose(canvas, landmarks, joint_color, line_color, alpha=1.0):
    overlay = canvas.copy()
    h, w = canvas.shape[:2]

    for lm in landmarks:
        if lm.get('v', 0) > 0.5:
            px = int(lm['x'] * w)
            py = int(lm['y'] * h)
            cv2.circle(overlay, (px, py), 4, joint_color, -1)

    for conn in config.POSE_CONNECTIONS:
        start_idx, end_idx = conn
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            if start_lm.get('v', 0) > 0.5 and end_lm.get('v', 0) > 0.5:
                sx = int(start_lm['x'] * w)
                sy = int(start_lm['y'] * h)
                ex = int(end_lm['x'] * w)
                ey = int(end_lm['y'] * h)
                cv2.line(overlay, (sx, sy), (ex, ey), line_color, 2)

    if alpha < 1.0:
        return cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)
    return overlay


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


def create_ghost_panel(webcam_image, target_frame, student_landmarks, panel_w, panel_h, is_playing):
    panel = create_panel(webcam_image, panel_w, panel_h, placeholder_text="Student camera unavailable")

    if target_frame is not None:
        panel = draw_pose(panel, target_frame['landmarks'], (0, 0,0), (0, 0, 0))
        cv2.putText(panel, "Teacher Ghost", (100, panel_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2)

    if student_landmarks is not None:
        panel = draw_pose(panel, student_landmarks, (0, 255, 0), (0, 200, 0), alpha=1.0)
        cv2.putText(panel, "Student", (10, panel_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

    if not is_playing:
        cv2.putText(panel, "Press 's' to start", (70, panel_h - 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return panel
