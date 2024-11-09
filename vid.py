import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def extract_keyframes(video_path, base_threshold=0.9, min_interval=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    
    keyframes, prev_frame_features, frame_count = [], None, 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_resized = cv2.resize(frame, (100, 100))
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_features = frame_gray.flatten().astype(np.float32)
        frame_features /= np.linalg.norm(frame_features)
        
        if prev_frame_features is not None:
            similarity = cosine_similarity(
                frame_features.reshape(1, -1), 
                prev_frame_features.reshape(1, -1)
            )[0][0]
            adaptive_threshold = base_threshold - (0.05 if frame_count % min_interval == 0 else 0)
            if similarity < adaptive_threshold:
                keyframes.append(frame)
        
        prev_frame_features = frame_features
        frame_count += 1
    
    cap.release()
    return keyframes
