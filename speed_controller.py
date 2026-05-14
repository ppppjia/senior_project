import cv2

MIN_SPEED = 0.25
MAX_SPEED = 2.0
CONTROLLER_MARGIN_X = 15
CONTROLLER_MARGIN_Y = 15
CONTROLLER_W = 400
CONTROLLER_H = 120
CONTROLLER_HEADER_H = 28
BUTTONS = [
    ("0.5x", 0.5, 20, 55, 75, 45),#(內容,x,y,方框大小寬度,方框大小高度)
    ("1.0x", 1.0, 115, 55, 75, 45),
    ("1.5x", 1.5,210, 55, 75, 45),
    ("2.0x", 2.0, 300, 55, 75, 45),
]
KNOB_COLOR = (100, 100, 255)


def get_default_controller_origin(img_width):
    controller_x = max(CONTROLLER_MARGIN_X, img_width - CONTROLLER_MARGIN_X - CONTROLLER_W)
    controller_y = CONTROLLER_MARGIN_Y
    return controller_x, controller_y


def get_controller_origin(img_width, origin=None):
    if origin is not None:
        return int(origin[0]), int(origin[1])
    return get_default_controller_origin(img_width)


def get_controller_rect(img_width, origin=None):
    controller_x, controller_y = get_controller_origin(img_width, origin)
    return controller_x, controller_y, CONTROLLER_W, CONTROLLER_H


def is_inside_controller_header(x, y, img_width, origin=None):
    controller_x, controller_y = get_controller_origin(img_width, origin)
    return (controller_x <= x <= controller_x + CONTROLLER_W and
            controller_y <= y <= controller_y + CONTROLLER_HEADER_H)


def speed_to_slider_x(speed, img_width, origin=None):
    controller_x, controller_y = get_controller_origin(img_width, origin)
    slider_left = controller_x + 20
    slider_right = controller_x + CONTROLLER_W - 20
    t = (speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
    return int(slider_left + t * (slider_right - slider_left)), controller_y + 70


def get_speed_from_position(x, img_width, origin=None):
    controller_x, controller_y, controller_w, controller_h = get_controller_rect(img_width, origin)
    slider_left = controller_x + 20
    slider_right = controller_x + controller_w - 20
    clamped_x = max(slider_left, min(x, slider_right))
    t = (clamped_x - slider_left) / max(1, slider_right - slider_left)
    return max(MIN_SPEED, min(MAX_SPEED, MIN_SPEED + t * (MAX_SPEED - MIN_SPEED)))


def draw_speed_controller(img, speed, history_positions=None, origin=None):
    img_h, img_w = img.shape[:2]
    controller_x, controller_y = get_controller_origin(img_w, origin)
    overlay = img.copy()
    cv2.rectangle(overlay, (controller_x, controller_y),
                  (controller_x + CONTROLLER_W, controller_y + CONTROLLER_H),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

    cv2.rectangle(img, (controller_x, controller_y),
                  (controller_x + CONTROLLER_W, controller_y + CONTROLLER_H),
                  (255, 255, 255), 1)
    cv2.rectangle(img, (controller_x, controller_y),
                  (controller_x + CONTROLLER_W, controller_y + CONTROLLER_HEADER_H),
                  (255, 255, 255), 1)

    cv2.putText(img, f"Speed Controller : {speed:.2f}x",
                (controller_x + 15, controller_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    for text, value, dx, dy, bw, bh in BUTTONS:
        bx = controller_x + dx
        by = controller_y + dy
        if abs(value - speed) < 0.001:
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (100, 100, 100), -1)
        else:
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
        cv2.putText(img, text, (bx + 10, by + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)


def get_speed_from_click(x, y, current_speed, img_width, origin=None):
    controller_x, controller_y, controller_w, controller_h = get_controller_rect(img_width, origin)
    for text, value, dx, dy, bw, bh in BUTTONS:
        bx = controller_x + dx
        by = controller_y + dy
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return max(MIN_SPEED, min(MAX_SPEED, value))

    slider_left = controller_x + 20
    slider_right = controller_x + controller_w - 20
    slider_y = controller_y + 70
    if slider_left <= x <= slider_right and slider_y - 10 <= y <= slider_y + 10:
        return get_speed_from_position(x, img_width, origin)

    return current_speed
