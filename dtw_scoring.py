# dtw_scoring.py
# 計分邏輯：從 100 分開始，學生骨架超出老師鬼影範圍時扣分，不加分。

import config


# =============================================================================
# 分數狀態
# =============================================================================

class ScoreTracker:
    """
    持有目前分數，每幀呼叫 update() 傳入 out_indices。
    - 分數從 100 開始
    - 每幀有超出關節 → 依超出數量扣分
    - 不會加分
    """

    def __init__(self):
        self.score = 100.0

    def reset(self):
        self.score = 100.0

    def update(self, out_indices: set, total_joints: int = 33) -> int:
        """
        out_indices : get_out_of_bounds_indices() 回傳的超出關節集合
        total_joints: 用於計算超出比例（預設 33，MediaPipe 全身）
        回傳目前整數分數。
        """
        if not out_indices:
            return int(self.score)   # 沒超出 → 不動分數

        # 超出比例 0.0 ~ 1.0
        out_ratio = len(out_indices) / total_joints

        # 每幀扣分量：比例 × 每幀最大扣分
        penalty = out_ratio * config.PENALTY_PER_FRAME

        self.score = max(0.0, self.score - penalty)
        return int(self.score)


# =============================================================================
# 保留原本 buffer 工具（其他地方若有用到不會壞）
# =============================================================================

from collections import deque

BUFFER_SIZE = 15


def create_student_buffer():
    return deque(maxlen=BUFFER_SIZE)


def update_student_buffer(student_buffer, live_landmarks):
    """保留介面，目前計分不再用 DTW，此函式可留作未來擴充。"""
    if not live_landmarks or len(live_landmarks) != 33:
        return False
    student_buffer.append(live_landmarks)
    return len(student_buffer) == BUFFER_SIZE
