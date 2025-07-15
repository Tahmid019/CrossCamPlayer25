import cv2

def draw_boxes(frame, detections, prefix=""):
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        label = f"{prefix}_{i}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame
