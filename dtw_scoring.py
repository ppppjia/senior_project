from collections import deque
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from pose_utils import normalize_pose

BUFFER_SIZE = 15


def create_student_buffer():
    return deque(maxlen=BUFFER_SIZE)


def build_flat_pose(landmarks, index_range=range(11, 25)):
    normalized = normalize_pose(landmarks)
    flat = []
    for i in index_range:
        flat.extend([normalized[i]['x'], normalized[i]['y']])
    return flat


def update_student_buffer(student_buffer, live_landmarks):
    if not live_landmarks or len(live_landmarks) != 33:
        return False

    flat_live = build_flat_pose(live_landmarks)
    student_buffer.append(flat_live)
    return len(student_buffer) == BUFFER_SIZE


def get_teacher_window(standard_pose_data, elapsed_ms, lookback_ms=500, lookahead_ms=200):
    start_ts = max(0, elapsed_ms - lookback_ms)
    end_ts = elapsed_ms + lookahead_ms
    return [frame for frame in standard_pose_data if start_ts <= frame['timestamp_ms'] <= end_ts]


def compute_dtw_score(student_buffer, teacher_windows, scale=25):
    if len(student_buffer) < BUFFER_SIZE or not teacher_windows:
        return None

    teacher_buffer = []
    for frame_data in teacher_windows:
        flat_target = build_flat_pose(frame_data['landmarks'])
        teacher_buffer.append(flat_target)

    if not teacher_buffer:
        return None

    distance, _ = fastdtw(student_buffer, teacher_buffer, dist=euclidean)
    raw_score = 100 - (distance / scale)
    return max(0, min(100, int(raw_score)))
