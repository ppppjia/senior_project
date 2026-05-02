import math

def get_center_point(p1, p2):
    """計算兩個點的中心點"""
    return {
        'x': (p1['x'] + p2['x']) / 2,
        'y': (p1['y'] + p2['y']) / 2,
        'z': (p1['z'] + p2['z']) / 2
    }

def calculate_distance(p1, p2):
    """計算三維空間中兩點的歐幾里得距離"""
    return math.sqrt(
        (p2['x'] - p1['x'])**2 + 
        (p2['y'] - p1['y'])**2 + 
        (p2['z'] - p1['z'])**2
    )

def normalize_pose(landmarks):
    """
    將傳入的 33 個骨架節點進行平移(對齊骨盆)與縮放(根據軀幹長度)
    landmarks: 包含 x, y, z, v 的字典陣列
    """
    # 1. 取得左右髖關節 (23: 左髖, 24: 右髖) 並計算骨盆中心
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    pelvis_center = get_center_point(left_hip, right_hip)

    # 2. 取得左右肩膀 (11: 左肩, 12: 右肩) 並計算肩膀中心
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    shoulder_center = get_center_point(left_shoulder, right_shoulder)

    # 3. 計算軀幹長度 (作為縮放的基準值)
    torso_length = calculate_distance(shoulder_center, pelvis_center)
    
    # 避免除以零的極端情況
    if torso_length == 0:
        torso_length = 0.0001 

    # 4. 針對每一個關節點進行平移與縮放
    normalized_landmarks = []
    for lm in landmarks:
        norm_x = (lm['x'] - pelvis_center['x']) / torso_length
        norm_y = (lm['y'] - pelvis_center['y']) / torso_length
        norm_z = (lm['z'] - pelvis_center['z']) / torso_length
        
        normalized_landmarks.append({
            'id': lm.get('id', 0),
            'x': norm_x,
            'y': norm_y,
            'z': norm_z,
            'v': lm.get('v', 0)
        })

    return normalized_landmarks

def calculate_pose_error(normalized_student, normalized_teacher):
    """計算兩組正規化骨架的誤差值"""
    total_error = 0.0
    
    # MediaPipe 有 33 個點，我們可以只挑選重要的四肢點來評分 (例如手肘、手腕、膝蓋、腳踝)
    # 這裡我們為了簡單，先計算所有 33 個點的誤差
    for i in range(33):
        stu = normalized_student[i]
        tea = normalized_teacher[i]
        
        # 只在該關節可見度 (Visibility) 高於 0.5 時才計算，避免遮擋造成的誤判
        if stu['v'] > 0.5 and tea['v'] > 0.5:
            # 計算該關節的空間距離誤差
            error = calculate_distance(stu, tea)
            total_error += error
            
    return total_error