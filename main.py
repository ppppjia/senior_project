# main.py
import cv2
import numpy as np
import json
import config
from extractor import PoseExtractor
from visualizer import draw_skeleton

def main(input_video, output_mp4, output_json):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 原本少打了 FRAME_
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 原本少打了 FRAME_
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))
    
    extractor = PoseExtractor()
    results_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(frame_idx * 1000 / fps)
        
        # 1. 提取
        landmarks = extractor.get_landmarks(frame_rgb, timestamp_ms)
        
        # 2. 處理畫布與繪製
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if landmarks:
            # 存檔數據
            frame_data = [{"x": lm.x, "y": lm.y, "z": lm.z, "v": lm.visibility} for lm in landmarks]
            results_data.append({"frame": frame_idx, "landmarks": frame_data})
            
            # 視覺化
            draw_skeleton(canvas, landmarks, w, h)

        out.write(canvas)
        frame_idx += 1
        if frame_idx % 100 == 0: print(f"已處理 {frame_idx} 幀...")

    # 釋放與存檔
    cap.release()
    out.release()
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    print("處理完成！")

if __name__ == "__main__":
    main('test2(5p).mp4', 'output_skeleton.mp4', 'data.json')