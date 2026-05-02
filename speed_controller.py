import cv2

MIN_SPEED = 0.25
MAX_SPEED = 2.0
CONTROLLER_X = 250
CONTROLLER_Y = 15
CONTROLLER_W = 330
CONTROLLER_H = 100
BUTTONS = [
    ("0.5x", 0.5, 20, 55, 60, 38),
    ("1.0x", 1.0, 90, 55, 60, 38),
    ("1.5x", 1.5, 160, 55, 60, 38),
    ("2.0x", 2.0, 230, 55, 60, 38),
]


def draw_speed_controller(img, speed):
    overlay = img.copy()
    cv2.rectangle(overlay, (CONTROLLER_X, CONTROLLER_Y),
                  (CONTROLLER_X + CONTROLLER_W, CONTROLLER_Y + CONTROLLER_H),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

    cv2.rectangle(img, (CONTROLLER_X, CONTROLLER_Y),
                  (CONTROLLER_X + CONTROLLER_W, CONTROLLER_Y + CONTROLLER_H),
                  (255, 255, 255), 1)

    cv2.putText(img, f"Speed Controller : {speed:.2f}x",
                (CONTROLLER_X + 25, CONTROLLER_Y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

    for text, value, dx, dy, bw, bh in BUTTONS:
        bx = CONTROLLER_X + dx
        by = CONTROLLER_Y + dy
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
        cv2.putText(img, text, (bx + 10, by + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1)


def get_speed_from_click(x, y, current_speed):
    for text, value, dx, dy, bw, bh in BUTTONS:
        bx = CONTROLLER_X + dx
        by = CONTROLLER_Y + dy
        if bx <= x <= bx + bw and by <= y <= by + bh:
            if text.startswith(('+', '-')):
                new_speed = round(current_speed + value, 2)
            else:
                new_speed = value
            return max(MIN_SPEED, min(MAX_SPEED, new_speed))
    return current_speed
