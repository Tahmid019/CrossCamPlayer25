import cv2
from collections import defaultdict
from .features import extract_cnn_features

def detect_players(video_path, model, conf=0.3, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(max_frames or total_frames, total_frames)

    detections = defaultdict(list)
    frames = {}

    for idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
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
    cap.release()
    return detections, frames
