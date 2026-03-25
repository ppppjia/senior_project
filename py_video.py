import cv2
import numpy as np
import mediapipe as mp
import time
mppose = mp.solutions.pose
pose = mppose.Pose()
h = 0
w = 0
status = False
countdown_seconds = 30
start_time = 0

def calc_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def get_landmark(landmarks, part_name):
    return [
        landmarks[mppose.PoseLandmark[part_name].value].x,
        landmarks[mppose.PoseLandmark[part_name].value].y,
        landmarks[mppose.PoseLandmark[part_name].value].z,
    ]


def get_visibility(landmarks):
    if landmarks[mppose.PoseLandmark["RIGHT_HIP"].value].visibility < 0.8 or \
            landmarks[mppose.PoseLandmark["LEFT_HIP"].value].visibility < 0.8:
        return False
    else:
        return True


def get_body_ratio(landmarks):
    r_body = abs(landmarks[mppose.PoseLandmark["RIGHT_SHOULDER"].value].y
                 - landmarks[mppose.PoseLandmark["RIGHT_HIP"].value].y)
    l_body = abs(landmarks[mppose.PoseLandmark["LEFT_SHOULDER"].value].y
                 - landmarks[mppose.PoseLandmark["LEFT_HIP"].value].y)
    avg_body = (r_body + l_body) / 2
    r_leg = abs(landmarks[mppose.PoseLandmark["RIGHT_HIP"].value].y
                - landmarks[mppose.PoseLandmark["RIGHT_ANKLE"].value].y)
    l_leg = abs(landmarks[mppose.PoseLandmark["LEFT_HIP"].value].y
                - landmarks[mppose.PoseLandmark["LEFT_ANKLE"].value].y)
    if r_leg > l_leg:
        return r_leg / avg_body
    else:
        return l_leg / avg_body


def get_knee_angle(landmarks):
    r_hip = get_landmark(landmarks, "RIGHT_HIP")
    l_hip = get_landmark(landmarks, "LEFT_HIP")

    r_knee = get_landmark(landmarks, "RIGHT_KNEE")
    l_knee = get_landmark(landmarks, "LEFT_KNEE")

    r_ankle = get_landmark(landmarks, "RIGHT_ANKLE")
    l_ankle = get_landmark(landmarks, "LEFT_ANKLE")

    r_angle = calc_angles(r_hip, r_knee, r_ankle)
    l_angle = calc_angles(l_hip, l_knee, l_ankle)


    return [r_angle, l_angle]

video1 = cv2.VideoCapture('dancing.mp4')  
video1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'XVID'))
video1.set(cv2.CAP_PROP_FPS, 30)
video1.set(cv2.CAP_PROP_POS_FRAMES, 0)


video2 = cv2.VideoCapture(0)  


if not video1.isOpened() or not video2.isOpened():
    print("無法打開影片")
    exit()

frame_width2 = 1280
frame_height2 = 960
print(frame_height2) #480
print(frame_width2) #640

scaled_width = frame_width2 // 6
scaled_height = frame_height2 // 6


cv2.namedWindow('Combined Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Combined Video', frame_width2, frame_height2)

while True:
    if not status:
        start_time = time.time()
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    if not ret1:
        video1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    if not ret2:
        break
    results = pose.process(frame2)
    if results.pose_landmarks is not None:
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = frame2.copy()
        mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mppose.POSE_CONNECTIONS)
        knee_angles = get_knee_angle(results.pose_landmarks.landmark)
        body_ratio = get_body_ratio(results.pose_landmarks.landmark)
        avg_angle = (knee_angles[0] + knee_angles[1]) // 2
        if 115<knee_angles[0] < 140 and 160<knee_angles[1]< 180:
            cv2.putText(annotated_image, "Left: {:.1f}".format(knee_angles[0]), (10, 230)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Right: {:.1f}".format(knee_angles[1]), (10, 260)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Good! Keep going!", (10, 290)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            status= True
        elif 115<knee_angles[1] < 140 and  160<knee_angles[0]< 180:
            cv2.putText(annotated_image, "Left: {:.1f}".format(knee_angles[0]), (10, 230)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Right: {:.1f}".format(knee_angles[1]), (10, 260)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA) 
            cv2.putText(annotated_image, "Good! Keep going!", (10, 290)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)            
            status= True   
        else:
            cv2.putText(annotated_image, "Left: {:.1f}".format(knee_angles[0]), (10, 230)
                            , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Right: {:.1f}".format(knee_angles[1]), (10, 260)
                            , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Squat down!", (10, 290)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA) 
            status= False               
        if status:
            current_time = time.time()
            elapsed_time = int(current_time - start_time)
            remaining_seconds = max(0, countdown_seconds - elapsed_time)

            text = f" {remaining_seconds} second"
            cv2.putText(annotated_image, text, (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if remaining_seconds == 0:
                status = False


    frame1_resized = cv2.resize(frame1, (scaled_width, scaled_height))
    annotated_image[0:scaled_height, 0:scaled_width] = frame1_resized
    cv2.imshow('Combined Video', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video1.release()
video2.release()
cv2.destroyAllWindows()
