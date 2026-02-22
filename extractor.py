# extractor.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config

class PoseExtractor:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=config.MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def get_landmarks(self, frame_rgb, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if result.pose_landmarks:
            return result.pose_landmarks[0]  # 只取第一個人
        return None