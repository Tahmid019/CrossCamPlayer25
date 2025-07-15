import cv2
from collections import defaultdict
from .features import extract_cnn_features

def detect_players(video_path, model, conf=0.3, max_frames=None, start = 0, stride = 1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    count = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(max_frames or total_frames, total_frames)

    detections = defaultdict(list)
    frames = {}

    while cap.isOpened():
        if max_frames is not None and count >= max_frames:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        if (idx - start) % stride != 0:
            idx += 1
            continue
        
        results = model(frame, verbose=False)
        frames[idx] = frame.copy()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            if cls_id == 2 and conf_score > conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    feature = extract_cnn_features(crop)
                    detections[idx].append({'bbox': [x1, y1, x2, y2], 'feature': feature})
        idx += 1
        count += 1
        
    cap.release()
    return detections, frames
