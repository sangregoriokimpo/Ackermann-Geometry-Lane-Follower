# lane_detection.py
import cv2
import numpy as np

def detect_lanes(frame):
    h, w = frame.shape[:2]
    roi = frame[int(h / 2):, :]  # Use bottom half for lane detection

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=100)

    direction = "forward"
    if lines is not None:
        left, right = 0, 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.5:
                left += 1
            elif slope > 0.5:
                right += 1
        if left > right:
            direction = "left"
        elif right > left:
            direction = "right"
    
    return direction, edges  # return debug view optionally
